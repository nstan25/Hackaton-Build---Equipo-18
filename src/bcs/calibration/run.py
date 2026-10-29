"""T11: distribución + calibración + conformal sobre las predicciones de val de T09.

`python -m bcs.calibration.run` (solo val, nunca test):

- Cross-fitting por `group_id` dentro de val (k=5): cada variante se ajusta en 4 partes y predice
  la quinta. Esas predicciones (con `fold = cf<k>`) son las que mide `make eval`; así ECE y
  cobertura no salen optimistas aunque val ya se usó para elegir el checkpoint.
- Variantes: la referencia de T09 (σ global, conformal marginal) y la distribución ajustada por
  máxima verosimilitud con grado 0/1/2 y conformal marginal o condicional a la clase.
- La variante elegida (`CHOSEN`) se re-ajusta con todo val → `calibration.json` (producción, T12).

Salidas: `$BCS_ARTIFACTS_DIR/T11/{predictions_val.csv, predictions_val_<variante>.csv,
calibration.json}` y `runs/T11/<ts>/summary.md`.

T28: `--scale dairy_zen --task T28 --predictions <val del modelo de T25>` recalibra el modelo nuevo
en la grilla 3.0–5.0 con val de `sdb_bcs` + `zen_rgbd` (mismo procedimiento; el resumen agrega el
desglose por fuente).
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from bcs.calibration.regression import (
    SEED,
    CalibratedRegression,
    GlobalSigma,
    cross_fit,
    snap_index,
)
from bcs.eval.metrics import confidence_from_probs, evaluate_with_intervals
from bcs.models.efficientnet import origin
from bcs.paths import artifacts_dir, runs_dir
from bcs.scale import grid_key, load_scale

TASK = "T11"
K_FOLDS = 5
CHOSEN = "ml_deg0_class"
# T28: en dairy_zen las colas (3.0, 4.5–5.0) tienen 9–43 imágenes de val; con menos de 20 casos de
# calibración, una clase usa el q̂ marginal (sin esto el intervalo cubre toda la grilla).
MIN_CLASS_N = 20


def variants(grid: tuple[float, ...]) -> dict[str, Callable[[], CalibratedRegression]]:
    return {
        "t09_global_marginal": lambda: GlobalSigma(grid),
        "ml_deg0_marginal": lambda: CalibratedRegression(grid, 0, by_class=False),
        "ml_deg0_class": lambda: CalibratedRegression(grid, 0),
        f"ml_deg0_class_min{MIN_CLASS_N}": lambda: CalibratedRegression(
            grid, 0, min_class_n=MIN_CLASS_N
        ),
        "ml_deg1_class": lambda: CalibratedRegression(grid, 1),
        "ml_deg2_class": lambda: CalibratedRegression(grid, 2),
    }


def predictions_frame(
    df: pd.DataFrame, probs: np.ndarray, iv: np.ndarray, folds: np.ndarray, grid: tuple[float, ...]
) -> pd.DataFrame:
    """Formato de `make eval` (`tasks/_contracts.md`), con `value` de T09 al final."""
    out = pd.DataFrame({c: df[c].to_numpy() for c in ("image_id", "group_id", "source", "bcs")})
    for j, g in enumerate(grid):
        out[f"p_{grid_key(g)}"] = probs[:, j]
    out["fold"] = [f"cf{f}" for f in folds]
    out["interval_lo"] = iv[:, 0]
    out["interval_hi"] = iv[:, 1]
    out["value"] = df["value"].to_numpy()
    return out


def summarize(
    df: pd.DataFrame, probs: np.ndarray, iv: np.ndarray, folds: np.ndarray, grid: tuple[float, ...]
) -> dict[str, Any]:
    """Métricas como las calcula `make eval` (promedio de los folds) + desgloses."""
    y = df["bcs"].to_numpy(dtype=float)
    grp, src = df["group_id"].to_numpy(str), df["source"].to_numpy(str)
    per_fold = [
        evaluate_with_intervals(probs[m], y[m], grid, grp[m].tolist(), src[m].tolist(), iv[m])
        for m in (folds == f for f in range(K_FOLDS))
    ]
    keys = ("ece", "coverage_90", "interval_width_steps", "nll", "mae", "mean_confidence")
    out: dict[str, Any] = {k: float(np.mean([m[k] for m in per_fold])) for k in keys}
    out["ece_folds"] = [m["ece"] for m in per_fold]
    out["coverage_folds"] = [m["coverage_90"] for m in per_fold]

    y_idx = snap_index(y, grid)
    conf = confidence_from_probs(probs)
    hit = (np.abs(probs.argmax(axis=1) - y_idx) <= 1).astype(float)
    cov = ((y >= iv[:, 0] - 1e-6) & (y <= iv[:, 1] + 1e-6)).astype(float)
    width = (iv[:, 1] - iv[:, 0]) / (grid[1] - grid[0])
    expected = probs @ np.asarray(grid)
    t = pd.DataFrame(
        {"conf": conf, "hit": hit, "cov": cov, "width": width, "bias_exp": expected - y, "y": y}
    )
    t["origin"] = origin(list(df["image_id"]))
    t["source"] = df["source"].to_numpy(str)
    # valor predicho → clase de la grilla más cercana
    t["v_bin"] = [f"≈{grid_key(grid[i])}" for i in snap_index(df["value"].to_numpy(float), grid)]
    agg = {"n": ("cov", "size"), "conf": ("conf", "mean"), "hit_pm1": ("hit", "mean")}
    agg |= {"coverage": ("cov", "mean"), "width": ("width", "mean"), "bias": ("bias_exp", "mean")}
    for name, col in (
        ("by_class", "y"),
        ("by_origin", "origin"),
        ("by_source", "source"),
        ("by_value", "v_bin"),
    ):
        out[name] = t.groupby(col).agg(**agg).round(4).to_dict(orient="index")  # type: ignore[call-overload]
    return out


def _table(rows: dict[Any, dict[str, Any]], label: str) -> list[str]:
    lines = [
        f"| {label} | n | confianza | acierto ±1 | cobertura | ancho (pasos) | sesgo E[p] |",
        "|---|---|---|---|---|---|---|",
    ]
    for k, r in rows.items():
        name = grid_key(k) if isinstance(k, float) else k
        lines.append(
            f"| {name} | {r['n']} | {r['conf']:.3f} | {r['hit_pm1']:.3f} | {r['coverage']:.3f} "
            f"| {r['width']:.2f} | {r['bias']:+.3f} |"
        )
    return lines


def write_summary(
    path: Path,
    results: dict[str, dict[str, Any]],
    prod: dict[str, Any],
    n: int,
    n_groups: int,
    chosen: str = CHOSEN,
) -> None:
    lines = [
        f"# T11 · Distribución, calibración y conformal sobre T09 (val, n={n}, {n_groups} grupos)",
        "",
        f"Cross-fitting por `group_id` dentro de val (k={K_FOLDS}, semilla {SEED}): cada variante "
        "se ajusta en 4/5 de los grupos y se mide en el 1/5 restante; las métricas son el promedio "
        "de los 5 folds (como `make eval`). Dentro de cada ajuste, la distribución y la "
        "temperatura usan la mitad de los grupos y el conformal la otra mitad (salvo la "
        "referencia T09, que usa las mismas filas). Los parámetros de producción "
        "(`calibration.json`) se ajustan con todo val.",
        "",
        "## Variantes (promedio de folds)",
        "",
        "| variante | ECE | cobertura 90% | ancho (pasos) | NLL | MAE | confianza media |",
        "|---|---|---|---|---|---|---|",
    ]
    for name, r in results.items():
        mark = " **(elegida)**" if name == chosen else ""
        lines.append(
            f"| {name}{mark} | {r['ece']:.4f} | {r['coverage_90']:.3f} | "
            f"{r['interval_width_steps']:.2f} | {r['nll']:.4f} | {r['mae']:.4f} | "
            f"{r['mean_confidence']:.3f} |"
        )
    lines += ["", "ECE y cobertura por fold:", ""]
    for name, r in results.items():
        ece = ", ".join(f"{x:.3f}" for x in r["ece_folds"])
        cov = ", ".join(f"{x:.3f}" for x in r["coverage_folds"])
        lines.append(f"- {name}: ECE [{ece}] · cobertura [{cov}]")
    for name in ("t09_global_marginal", chosen):
        r = results[name]
        lines += ["", f"## {name}: por clase real", "", *_table(r["by_class"], "clase")]
        lines += ["", f"## {name}: por valor predicho", "", *_table(r["by_value"], "valor")]
        lines += ["", f"## {name}: por origen", "", *_table(r["by_origin"], "origen")]
        lines += ["", f"## {name}: por fuente", "", *_table(r["by_source"], "fuente")]
    lines += [
        "",
        "## Parámetros de producción (todo val)",
        "",
        "```json",
        json.dumps(prod, indent=2),
        "```",
        "",
    ]
    path.write_text("\n".join(lines))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--predictions",
        type=Path,
        default=artifacts_dir() / "T09" / "predictions_val.csv",
        help="predicciones de val de T09 (con la columna `value`)",
    )
    parser.add_argument("--task", default=TASK, help="carpeta de salida (artefactos y runs)")
    parser.add_argument("--out", type=Path, default=None, help="default: $BCS_ARTIFACTS_DIR/<task>")
    parser.add_argument("--runs-dir", type=Path, default=runs_dir())
    parser.add_argument("--scale", default="dairy_sdb", help="grilla del modelo calibrado")
    parser.add_argument("--chosen", default=CHOSEN, help="variante que va a calibration.json")
    args = parser.parse_args(argv)
    chosen: str = args.chosen
    args.out = args.out or artifacts_dir() / args.task

    grid = load_scale(args.scale).grid
    df = pd.read_csv(args.predictions, dtype={"image_id": str, "group_id": str})
    folds_in = set(df["fold"].astype(str)) if "fold" in df.columns else {"val"}
    if folds_in != {"val"}:
        raise SystemExit(f"se esperaban solo predicciones de val y hay folds {sorted(folds_in)}")
    v, y, grp = (
        df["value"].to_numpy(dtype=float),
        df["bcs"].to_numpy(dtype=float),
        df["group_id"].to_numpy(str),
    )

    args.out.mkdir(parents=True, exist_ok=True)
    results: dict[str, dict[str, Any]] = {}
    for name, factory in variants(grid).items():
        probs, iv, folds = cross_fit(factory, v, y, grp, K_FOLDS)
        pred = predictions_frame(df, probs, iv, folds, grid)
        pred.to_csv(args.out / f"predictions_val_{name}.csv", index=False)
        if name == chosen:
            pred.to_csv(args.out / "predictions_val.csv", index=False)
        results[name] = summarize(df, probs, iv, folds, grid)
        r = results[name]
        print(
            f"{name}: ECE {r['ece']:.4f} · cobertura {r['coverage_90']:.3f} · "
            f"ancho {r['interval_width_steps']:.2f} · NLL {r['nll']:.4f}"
        )

    model = variants(grid)[chosen]().fit(v, y, grp)
    model.info |= {
        "variant": chosen,
        "source_predictions": str(args.predictions),
        "fit_on": "val (todo)",
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    model.save(args.out / "calibration.json")

    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_dir = args.runs_dir / args.task / ts
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "crossfit.json").write_text(json.dumps(results, indent=2, default=str) + "\n")
    write_summary(run_dir / "summary.md", results, model.to_dict(), len(df), len(set(grp)), chosen)
    print(f"calibración en {args.out / 'calibration.json'} · resumen en {run_dir / 'summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
