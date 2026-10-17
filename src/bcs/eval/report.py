"""`report.json` de una evaluación (SPEC §3.2, §4.1.5): métricas por fold y en promedio,
leave-one-source-out, baseline tonto y comparación contra `eval/thresholds.yaml[STAGE]`.

Archivo de predicciones (CSV, una fila por imagen evaluada):
    image_id, group_id, source, bcs        BCS real, valor de la grilla
    p_<clave>                              una columna por valor de la grilla (`p_3.25`, ...)
    fold          (opcional, default "val") los folds `loso:<fuente>` son leave-one-source-out
    interval_lo, interval_hi  (opcionales)  intervalo conformal al 90%

Lo escribe quien entrena (`bcs.train`, T09–T11, T16); `build_report` es la misma función que usa
`eval/evaluate.py`, así todos los `report.json` tienen el mismo esquema.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from bcs.eval.metrics import evaluate_with_intervals, snap_to_grid
from bcs.eval.plots import plot_confusion, plot_reliability
from bcs.paths import data_dir, holdout_dir
from bcs.scale import Scale

SCHEMA_VERSION = 1
LOSO_PREFIX = "loso:"
FORBIDDEN_FOLDS = {"test", "holdout"}
REQUIRED_COLUMNS = ("image_id", "group_id", "source", "bcs")


class EvalInputError(ValueError):
    """Predicciones inválidas o que violan una pared (holdout, test, fuga de train)."""


def prob_columns(scale: Scale) -> list[str]:
    return [f"p_{k}" for k in scale.keys]


def load_predictions(path: Path, scale: Scale) -> pd.DataFrame:
    path = Path(path).expanduser().resolve()
    if path.is_relative_to(holdout_dir().expanduser().resolve()):
        raise EvalInputError(f"{path} está bajo $BCS_HOLDOUT_DIR: no se evalúa ahí (SPEC §4.1)")
    df = pd.read_csv(path, dtype={"image_id": str, "group_id": str, "source": str})
    return validate_predictions(df, scale)


def validate_predictions(df: pd.DataFrame, scale: Scale) -> pd.DataFrame:
    cols = list(REQUIRED_COLUMNS) + prob_columns(scale)
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise EvalInputError(f"faltan columnas en las predicciones: {missing}")
    df = df.copy()
    df["fold"] = df["fold"].astype(str) if "fold" in df.columns else "val"
    bad = sorted(set(df["fold"].str.lower()) & FORBIDDEN_FOLDS)
    if bad:
        raise EvalInputError(f"folds {bad}: test es el holdout y ningún worker lo evalúa (G0)")
    if bool(df.duplicated(["fold", "image_id"]).any()):
        raise EvalInputError("image_id duplicado dentro de un fold")
    has_lo, has_hi = "interval_lo" in df.columns, "interval_hi" in df.columns
    if has_lo != has_hi:
        raise EvalInputError("interval_lo e interval_hi van juntas")
    return df


def check_walls(df: pd.DataFrame, data: Path | None = None) -> None:
    """Grupos del holdout o imágenes de train en las predicciones → error. Lee solo archivos
    visibles de `$BCS_DATA_DIR` (`holdout_groups.txt`, `splits.csv`), nunca `$BCS_HOLDOUT_DIR`."""
    data = data or data_dir()
    hg = data / "holdout_groups.txt"
    if hg.exists():
        holdout = {ln.strip() for ln in hg.read_text().splitlines() if ln.strip()}
        leaked = sorted(set(df["group_id"]) & holdout)
        if leaked:
            raise EvalInputError(f"{len(leaked)} group_id del holdout en las predicciones")
    sp = data / "splits.csv"
    if sp.exists():
        splits = pd.read_csv(sp, dtype=str)
        bad = set(splits.loc[splits["split"] != "val", "image_id"])
        folds = df["fold"].astype(str) if "fold" in df.columns else pd.Series("val", df.index)
        own = df.loc[~folds.str.startswith(LOSO_PREFIX), "image_id"]
        n_bad = int(own.isin(bad).sum())
        if n_bad:
            raise EvalInputError(f"{n_bad} imágenes de las predicciones no son de val")


def train_labels(data: Path | None = None) -> np.ndarray | None:
    """BCS de las imágenes de train (`labels.csv` ⨝ `splits.csv`), para el baseline tonto."""
    data = data or data_dir()
    labels, splits = data / "labels.csv", data / "splits.csv"
    if not (labels.exists() and splits.exists()):
        return None
    lab = pd.read_csv(labels, usecols=["image_id", "bcs"], dtype={"image_id": str})
    sp = pd.read_csv(splits, dtype=str)
    train = lab.merge(sp, on="image_id")
    y = train.loc[train["split"] == "train", "bcs"].to_numpy(dtype=float)
    return y if len(y) else None


def dummy_baseline(y_eval: np.ndarray, grid: Sequence[float], y_fit: np.ndarray | None) -> dict:
    """Baseline tonto (§2.3): siempre la moda o siempre la media (llevada a la grilla).
    Se ajusta en train; si no hay labels de train, en el propio set evaluado (más exigente)."""
    g = np.asarray(grid, dtype=float)
    fit_on = "train"
    if y_fit is None:
        y_fit, fit_on = y_eval, "eval_set"
    idx = snap_to_grid(y_fit, g)
    mode = float(g[np.bincount(idx, minlength=len(g)).argmax()])
    mean = float(g[np.abs(g - g[idx].mean()).argmin()])
    y = g[snap_to_grid(y_eval, g)]
    mae_mode = float(np.abs(y - mode).mean())
    mae_mean = float(np.abs(y - mean).mean())
    return {
        "fit_on": fit_on,
        "value_mode": mode,
        "value_mean": mean,
        "mae_mode": mae_mode,
        "mae_mean": mae_mean,
        "mae": min(mae_mode, mae_mean),
    }


def _metrics_for(df: pd.DataFrame, scale: Scale) -> dict[str, Any]:
    intervals = (
        df[["interval_lo", "interval_hi"]].to_numpy(dtype=float)
        if "interval_lo" in df.columns
        else None
    )
    return evaluate_with_intervals(
        df[prob_columns(scale)].to_numpy(dtype=float),
        df["bcs"].to_numpy(dtype=float),
        scale.grid,
        df["group_id"].tolist(),
        df["source"].tolist(),
        intervals,
    )


def _mean_scalars(per_fold: dict[str, dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {"n_folds": len(per_fold)}
    keys = next(iter(per_fold.values())).keys()
    for k in keys:
        vals = [m[k] for m in per_fold.values()]
        if all(isinstance(v, int | float) and not isinstance(v, bool) for v in vals):
            out[k] = float(np.mean(vals))
        elif all(v is None for v in vals):
            out[k] = None
    return out


def build_report(
    df: pd.DataFrame,
    scale: Scale,
    *,
    model: str,
    stage: str | None = None,
    task: str | None = None,
    thresholds: dict[str, Any] | None = None,
    quality: dict[str, Any] | None = None,
    y_train: np.ndarray | None = None,
) -> dict[str, Any]:
    df = validate_predictions(df, scale)
    by_fold = {str(f): df.loc[df["fold"] == f] for f in sorted(set(df["fold"]))}
    loso_folds = {f: d for f, d in by_fold.items() if f.startswith(LOSO_PREFIX)}
    if len(loso_folds) == len(by_fold):
        raise EvalInputError("no hay predicciones fuera de los folds leave-one-source-out")
    main = df.loc[~df["fold"].str.startswith(LOSO_PREFIX)]

    folds = {f: _metrics_for(d, scale) for f, d in by_fold.items() if f not in loso_folds}
    mean = _mean_scalars(folds)
    pooled = _metrics_for(main, scale)
    sources = sorted(set(main["source"]))
    if loso_folds:
        loso: dict[str, Any] | None = {
            f.removeprefix(LOSO_PREFIX): _metrics_for(d, scale) for f, d in loso_folds.items()
        }
        loso_note = "entrenado sin la fuente, evaluado en ella"
    else:
        loso = None
        loso_note = (
            "una sola fuente: leave-one-source-out no aplica"
            if len(sources) < 2
            else "sin folds loso:<fuente> en las predicciones"
        )

    dummy = dummy_baseline(main["bcs"].to_numpy(dtype=float), scale.grid, y_train)
    improvement = 1.0 - mean["mae"] / dummy["mae"] if dummy["mae"] > 0 else None

    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "task": task or model,
        "model": model,
        "stage": stage,
        "scale": {"name": scale.name, "step": scale.step, "grid": list(scale.grid)},
        "sources": sources,
        "mean": mean,
        "pooled": pooled,
        "folds": folds,
        "loso": loso,
        "loso_note": loso_note,
        "dummy": dummy,
        "mae_improvement_vs_dummy": improvement,
        "quality": quality,
        "thresholds": None,
        "plots": {},
    }
    if thresholds is not None and stage is not None:
        report["thresholds"] = check_thresholds(report, stage, thresholds)
    return report


_WITHIN = re.compile(r"^acc_within_(\d+)_(\d+)$")


def _resolve(report: dict[str, Any], name: str) -> float | None:
    """Valor de una métrica de `thresholds.yaml` en el reporte (promedio de folds)."""
    mean = report["mean"]
    if name.startswith("quality_filter."):
        q = report.get("quality") or {}
        v = q.get(name.split(".", 1)[1])
        return None if v is None else float(v)
    if name == "mae_improvement_vs_dummy":
        return report["mae_improvement_vs_dummy"]
    if (m := _WITHIN.match(name)) is not None:
        # tolerancia absoluta en unidades de BCS → pasos de la grilla activa
        steps = math.floor(float(f"{m[1]}.{m[2]}") / report["scale"]["step"] + 1e-9)
        key = "accuracy" if steps == 0 else f"acc_within_{steps}"
        return mean.get(key)
    if name in mean and isinstance(mean[name], int | float):
        return float(mean[name])
    return None


def _flatten(d: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out.update(_flatten(v, f"{prefix}{k}."))
        else:
            out[f"{prefix}{k}"] = v
    return out


def check_thresholds(report: dict[str, Any], stage: str, thresholds: dict) -> dict[str, Any]:
    """Cada clave `<métrica>_min` exige ≥ y `<métrica>_max` exige ≤. Una métrica que falta o
    que no se reconoce cuenta como no aprobada."""
    checks = []
    for key, limit in _flatten(thresholds).items():
        if key.endswith("_min"):
            metric, op = key.removesuffix("_min"), ">="
        elif key.endswith("_max"):
            metric, op = key.removesuffix("_max"), "<="
        else:
            checks.append(
                {"key": key, "metric": key, "op": "?", "threshold": limit, "value": None,
                 "passed": False, "note": "clave sin sufijo _min/_max"}
            )  # fmt: skip
            continue
        value = _resolve(report, metric)
        ok = value is not None and (value >= limit if op == ">=" else value <= limit)
        check = {"key": key, "metric": metric, "op": op, "threshold": limit, "value": value}
        check["passed"] = bool(ok)
        if value is None:
            check["note"] = "métrica no disponible en el reporte"
        checks.append(check)
    return {"stage": stage, "passed": all(c["passed"] for c in checks), "checks": checks}


def write_run(
    report: dict[str, Any], run_dir: Path, config: dict[str, Any] | None = None
) -> dict[str, Path]:
    """Escribe `report.json`, `config.json` y los PNG en `run_dir` (`runs/<task>/<ts>/`)."""
    run_dir.mkdir(parents=True, exist_ok=True)
    title = f"{report['model']} · {report['stage'] or ''}".strip(" ·")
    pooled = report["pooled"]
    report["plots"] = {
        "reliability": plot_reliability(pooled, run_dir / "reliability.png", title).name,
        "confusion": plot_confusion(pooled, run_dir / "confusion.png", title).name,
    }
    paths = {"report": run_dir / "report.json", "config": run_dir / "config.json"}
    paths["report"].write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    paths["config"].write_text(json.dumps(config or {}, indent=2, ensure_ascii=False) + "\n")
    return paths


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def synthetic_predictions(
    y: Sequence[float] | np.ndarray, probs: np.ndarray, scale: Scale, **columns: Sequence[Any]
) -> pd.DataFrame:
    """Arma un DataFrame de predicciones con el formato de arriba (tests y `bcs.train`).
    Sin `image_id`/`group_id`/`source` se usa un id por fila y la fuente "synthetic"."""
    n = len(y)
    df = pd.DataFrame(
        {
            "image_id": [f"img_{i}" for i in range(n)],
            "group_id": [f"g_{i}" for i in range(n)],
            "source": ["synthetic"] * n,
            "bcs": list(y),
        }
    )
    df[prob_columns(scale)] = np.asarray(probs, dtype=float)
    for col, values in columns.items():
        df[col] = list(values)
    return df
