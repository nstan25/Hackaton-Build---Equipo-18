"""`make eval-final` (G4): evaluación final del modelo de producción sobre test (el holdout).

Solo la corre el humano. Los agentes la prueban con un CSV de val (`--labels`), nunca con el
default (`$BCS_DATA_DIR/holdout_labels_sdb_bcs.csv`).

Modelo: el pipeline de G3 (`bcs.inference.model_pipeline`): EfficientNet-B0 del equipo + la
calibración de `models/calibration_efficientnet_b0.json` (distribución, temperatura y conformal de
T11). Dos modos en la misma corrida, sobre las mismas filas:

1. **recortes**: el recorte de `labels.csv` (`images/sdb_bcs/`) va directo al scorer, sin filtro
   de calidad ni detector. Comparable con las métricas de val de T09/T11.
2. **punta a punta**: la foto cruda (`raw/sdb_bcs/dataset/<clase>/<archivo>.jpg`) entra a
   `run_pipeline` como en producción (filtro sin CLIP + detector a 0,15 + EfficientNet +
   calibración + status). El archivo se busca por nombre en todas las carpetas de clase: la
   etiqueta no se usa para encontrarlo. Las métricas son sobre las aceptadas; se reporta la tasa
   de rechazo.

Métricas con `bcs.eval.metrics` (MAE, exacta, ±1/±2 pasos, ECE, cobertura 90%) por origen
(GS / YM / L-R) y por clase, el subgrupo GS pintadas vs. no pintadas (rojo > 0,5% del recorte,
la medida de T06/T09) y la comparación con val (predicciones cross-fitted de T11) y con
`eval/thresholds.yaml` m3 y m4.

Salida: `runs/<task>/<ts>/` (default `runs/G4/`) con `report.json`, `report.md`, `config.json`
y los PNG de confusión y fiabilidad de cada modo. Exit 0 si escribió el reporte (el veredicto
contra los umbrales está en el reporte), 2 si la entrada es inválida.

T28 (rama `dev`): el modelo es el de `BCS_MODEL` (default `zen`, el de T25 en `dairy_zen`) y por
defecto se evalúan **las dos fuentes**: `holdout_labels_sdb_bcs.csv` + `holdout_labels_zen_rgbd.csv`
(`--labels` acepta varios CSV). Mismas métricas, además **por fuente** (`by_source` en el reporte y
una tabla en `report.md`). La foto cruda de `zen_rgbd` es el frame entero de la sala
(`raw/zen_rgbd/rgb/<rgb_path>` de `plan.csv`). Val de referencia: las predicciones cross-fitted de
la calibración del modelo (`T28/predictions_val.csv` para `zen`, `T11/` para `sdb`); el filtro de
m4 del modelo nuevo (detector + OOD) sale de `$BCS_ARTIFACTS_DIR/efficientnet_b0_zen/
quality_metrics.json` si existe.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from PIL import Image

from bcs.eval.metrics import evaluate_with_intervals, snap_to_grid
from bcs.eval.plots import plot_confusion, plot_reliability
from bcs.eval.report import check_thresholds, prob_columns, sha256_file
from bcs.models.baseline import PAINT_THRESHOLD, red_mask
from bcs.models.efficientnet import origin
from bcs.paths import REPO_ROOT, artifacts_dir, data_dir, runs_dir
from bcs.scale import Scale

THRESHOLDS = Path(__file__).with_name("thresholds.yaml")
HOLDOUT_LABELS = "holdout_labels_sdb_bcs.csv"
HOLDOUT_LABELS_ZEN = "holdout_labels_zen_rgbd.csv"
ZEN = "zen_rgbd"
TASK = "G4"
SEED = 42
CHUNK = 64
SOURCE = "sdb_bcs"
RAW_DIR = Path("raw") / SOURCE / "dataset"
REQUIRED = ("image_id", "path", "group_id", "bcs")
MODES = ("crops", "e2e")
MODE_NAMES = {"crops": "recortes", "e2e": "punta a punta"}
COW_ID = "eval-final"  # viaja con la predicción; nunca llega al modelo
# Filtro de producción sin CLIP a 0,15 (tasks/G3.md §5): no se puede medir sobre test.
G3_QUALITY = {
    "reject_negatives": 0.85,
    "accept_rear": None,
    "note": "reject_negatives: set de rechazo de T08 (40 img), sin CLIP, detector a 0,15 "
    "(tasks/G3.md §5); accept_rear sin medir, aceptado en G3 §2; reject_val = rechazo punta "
    "a punta de esta corrida",
}
# Rechazo punta a punta sobre fotos crudas de val (T13, R10; tasks/G3.md §5).
VAL_E2E_REJECT = 0.003


class FinalEvalError(ValueError):
    """Entrada inválida (exit 2)."""


# ---------------------------------------------------------------- entrada


def load_labels(path: Path, limit: int | None = None, data: Path | None = None) -> pd.DataFrame:
    """Etiquetas con el esquema de `labels.csv`; con `limit`, N filas al azar (semilla fija).
    Rechaza imágenes de train (`splits.csv`)."""
    if not path.exists():
        raise FinalEvalError(f"no existe {path}")
    df = pd.read_csv(path, dtype={"image_id": str, "group_id": str, "path": str})
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise FinalEvalError(f"faltan columnas {missing} en {path}")
    if df.empty:
        raise FinalEvalError(f"{path} no tiene filas")
    if bool(df["image_id"].duplicated().any()):
        raise FinalEvalError("image_id duplicado")
    # `splits.csv` (sdb_bcs) y `splits_<fuente>.csv` (T28: zen_rgbd)
    for sp in sorted((data or data_dir()).glob("splits*.csv")):
        splits = pd.read_csv(sp, dtype=str)
        train = set(splits.loc[splits["split"] == "train", "image_id"])
        n_train = int(df["image_id"].isin(train).sum())
        if n_train:
            raise FinalEvalError(f"{n_train} imágenes de train en {path}")
    if "source" not in df.columns:
        df["source"] = SOURCE
    if limit is not None and limit < len(df):
        df = df.sample(n=limit, random_state=SEED).sort_index()
    return df.reset_index(drop=True)


def load_all_labels(
    paths: Sequence[Path], limit: int | None = None, data: Path | None = None
) -> pd.DataFrame:
    """`load_labels` de cada CSV, juntos (con `limit`, N filas al azar del total)."""
    df = pd.concat([load_labels(p, None, data) for p in paths], ignore_index=True)
    if bool(df["image_id"].duplicated().any()):
        raise FinalEvalError("image_id duplicado entre los CSV de etiquetas")
    if limit is not None and limit < len(df):
        df = df.sample(n=limit, random_state=SEED).sort_index()
    return df.reset_index(drop=True)


def _zen_raw_paths(data: Path) -> dict[str, str]:
    plan = data / "raw" / ZEN / "plan.csv"
    if not plan.exists():
        return {}
    df = pd.read_csv(plan, usecols=["image_id", "rgb_path"], dtype=str)
    return dict(zip(df["image_id"], df["rgb_path"], strict=True))


def raw_path(data: Path, image_id: str) -> Path | None:
    """La foto cruda de `image_id`: en `sdb_bcs`, buscada por nombre en todas las carpetas de
    clase; en `zen_rgbd`, el frame de `plan.csv` (sin mirar la etiqueta)."""
    if image_id.startswith(f"{ZEN}_"):
        rel = _zen_raw_paths(data).get(image_id)
        path = data / "raw" / ZEN / "rgb" / rel if rel else None
        return path if path is not None and path.exists() else None
    name = f"{image_id.removeprefix(f'{SOURCE}_')}.jpg"
    root = data / RAW_DIR
    dirs = sorted(d for d in root.iterdir() if d.is_dir()) if root.is_dir() else []
    return next((d / name for d in dirs if (d / name).exists()), None)


def paint_fraction(img: Image.Image) -> float:
    return float(red_mask(img).mean())


# ---------------------------------------------------------------- predicción


def _origin(image_ids: pd.Series, sources: pd.Series) -> np.ndarray:
    """GS / YM / L-R en `sdb_bcs`; el nombre de la fuente en las demás."""
    src = sources.astype(str).to_numpy()
    return np.where(src == SOURCE, origin(image_ids), src).astype(str)


def _prob_frame(df: pd.DataFrame, scale: Scale) -> pd.DataFrame:
    out = pd.DataFrame(
        {
            "image_id": df["image_id"],
            "group_id": df["group_id"],
            "source": df["source"] if "source" in df.columns else SOURCE,
            "origin": _origin(
                df["image_id"],
                df["source"] if "source" in df.columns else pd.Series(SOURCE, df.index),
            ),
            "bcs": df["bcs"].astype(float),
        }
    )
    for c in [*prob_columns(scale), "interval_lo", "interval_hi"]:
        out[c] = np.nan
    return out


def run_crops(df: pd.DataFrame, pipe: Any, data: Path) -> pd.DataFrame:
    """Recortes de `labels.csv` → scorer → calibración → intervalo (sin filtro ni detector)."""
    out = _prob_frame(df, pipe.scale)
    out["red_frac"] = np.nan
    cols = prob_columns(pipe.scale)
    for k in range(0, len(df), CHUNK):
        rows = df.iloc[k : k + CHUNK]
        images = []
        for p in rows["path"]:
            with Image.open(data / p) as im:
                images.append(im.convert("RGB"))
        probs = pipe.calibrator.transform(pipe.scorer.predict_proba(images))
        idx = out.index[k : k + len(rows)]
        out.loc[idx, cols] = probs
        out.loc[idx, ["interval_lo", "interval_hi"]] = np.asarray(pipe.interval.interval(probs))
        out.loc[idx, "red_frac"] = [paint_fraction(im) for im in images]
        print(f"[recortes] {k + len(rows)}/{len(df)}", file=sys.stderr)
    return out


def run_e2e(df: pd.DataFrame, pipe: Any, data: Path) -> pd.DataFrame:
    """Foto cruda → `run_pipeline` de producción (filtro, detector, modelo, calibración, status)."""
    from bcs.inference import run_pipeline

    out = _prob_frame(df, pipe.scale)
    out["status"], out["issues"] = "", ""
    cols = prob_columns(pipe.scale)
    for i, image_id in enumerate(df["image_id"]):
        path = raw_path(data, image_id)
        if path is None:
            out.loc[i, ["status", "issues"]] = ["missing", "no está la foto cruda"]
            continue
        pred = run_pipeline(pipe, path.read_bytes(), COW_ID)
        out.loc[i, ["status", "issues"]] = [pred.status, "|".join(pred.quality.issues)]
        if pred.status != "rejected":
            out.loc[i, cols] = list(pred.distribution.values())
            if pred.interval is not None:
                out.loc[i, ["interval_lo", "interval_hi"]] = list(pred.interval)
        if (i + 1) % 20 == 0 or i + 1 == len(df):
            print(f"[punta a punta] {i + 1}/{len(df)}", file=sys.stderr)
    return out


# ---------------------------------------------------------------- métricas


def class_table(probs: np.ndarray, y: np.ndarray, grid: Sequence[float]) -> dict[str, dict]:
    """Por clase real: n, MAE, sesgo (predicho − real), exacta y ±1 paso (argmax)."""
    g = np.asarray(grid, dtype=float)
    y_idx = snap_to_grid(y, g)
    pred_idx = probs.argmax(axis=1)
    out = {}
    for c, v in enumerate(g):
        m = y_idx == c
        if not m.any():
            continue
        d = pred_idx[m] - c
        out[f"{v:.2f}"] = {
            "n": int(m.sum()),
            "mae": float(np.abs(g[pred_idx[m]] - v).mean()),
            "bias": float((g[pred_idx[m]] - v).mean()),
            "accuracy": float((d == 0).mean()),
            "acc_within_1": float((np.abs(d) <= 1).mean()),
        }
    return out


def _metrics(preds: pd.DataFrame, scale: Scale, sources: Sequence[str]) -> dict[str, Any]:
    cols = prob_columns(scale)
    probs = preds[cols].to_numpy(dtype=float)
    iv = preds[["interval_lo", "interval_hi"]].to_numpy(dtype=float)
    out = evaluate_with_intervals(
        probs,
        preds["bcs"].to_numpy(dtype=float),
        scale.grid,
        preds["group_id"].astype(str).tolist(),
        list(sources),
        None if np.isnan(iv).any() else iv,
    )
    out["by_class"] = class_table(probs, preds["bcs"].to_numpy(dtype=float), scale.grid)
    return out


def mode_metrics(
    preds: pd.DataFrame, scale: Scale, red_frac: pd.Series | None = None
) -> dict[str, Any]:
    """Métricas sobre las filas con distribución (en punta a punta, las aceptadas); `by_source`
    es por origen. `red_frac` (por `image_id`) agrega GS pintadas vs. no pintadas."""
    n = len(preds)
    status = preds["status"] if "status" in preds.columns else pd.Series("ok", preds.index)
    scored = preds[preds[prob_columns(scale)].notna().all(axis=1)]
    out: dict[str, Any] = {
        "n_total": n,
        "n_rejected": int((status == "rejected").sum()),
        "n_missing": int((status == "missing").sum()),
        "n_low_confidence": int((status == "low_confidence").sum()),
        "n_scored": len(scored),
    }
    found = n - out["n_missing"]
    out["reject_rate"] = out["n_rejected"] / found if found else None
    out["low_confidence_rate"] = out["n_low_confidence"] / found if found else None
    if "issues" in preds.columns:
        issues = preds.loc[status == "rejected", "issues"].str.split("|").explode()
        out["reject_issues"] = {str(k): int(v) for k, v in issues.value_counts().items()}
    if scored.empty:
        out["metrics"] = None
        return out
    out["metrics"] = _metrics(scored, scale, scored["origin"].tolist())
    out["paint"] = paint_metrics(scored, scale, red_frac)
    return out


def paint_metrics(
    preds: pd.DataFrame, scale: Scale, red_frac: pd.Series | None
) -> dict[str, Any] | None:
    """GS pintadas (rojo > `PAINT_THRESHOLD` del recorte) vs. no pintadas; None sin medida."""
    if red_frac is None:
        return None
    gs = preds[preds["origin"] == "GS"]
    frac = red_frac.reindex(gs["image_id"]).to_numpy(dtype=float)
    gs, frac = gs[~np.isnan(frac)], frac[~np.isnan(frac)]
    if gs.empty:
        return None
    label = np.where(frac > PAINT_THRESHOLD, "pintadas", "no pintadas")
    out: dict[str, Any] = {"threshold": PAINT_THRESHOLD, "n": len(gs)}
    for name in ("pintadas", "no pintadas"):
        part = gs[label == name]
        out[name] = None if part.empty else _metrics(part, scale, [name] * len(part))
        if out[name] is not None:
            out[name].pop("by_source")
    return out


def val_reference(path: Path, scale: Scale) -> dict[str, Any] | None:
    """Métricas de val con las predicciones cross-fitted de la calibración (las de `make eval`
    m4); con `source`, también por fuente (`by_dataset`)."""
    if not path.exists():
        return None
    df = pd.read_csv(path, dtype={"image_id": str, "group_id": str})
    src = df["source"] if "source" in df.columns else pd.Series(SOURCE, df.index)
    df["origin"] = _origin(df["image_id"], src)
    out = _metrics(df, scale, df["origin"].tolist())
    if src.nunique() > 1:
        out["by_dataset"] = {
            str(s): _metrics(df[src == s], scale, df.loc[src == s, "origin"].tolist())
            for s in sorted(set(src))
        }
    return out


def threshold_checks(
    metrics: dict[str, Any] | None, scale: Scale, thresholds: dict, quality: dict | None
) -> dict[str, Any] | None:
    if metrics is None:
        return None
    report = {"mean": metrics, "scale": {"step": scale.step}, "quality": quality}
    return {stage: check_thresholds(report, stage, thresholds[stage]) for stage in ("m3", "m4")}


def build_final_report(
    crops: pd.DataFrame,
    e2e: pd.DataFrame,
    scale: Scale,
    thresholds: dict,
    val: dict[str, Any] | None,
    meta: dict[str, Any],
    base_quality: dict[str, Any] | None = None,
) -> dict[str, Any]:
    red = crops.set_index("image_id")["red_frac"] if "red_frac" in crops.columns else None
    modes = {"crops": mode_metrics(crops, scale, red), "e2e": mode_metrics(e2e, scale, red)}
    quality = {**(base_quality or G3_QUALITY), "reject_val": modes["e2e"]["reject_rate"]}
    sources = sorted(set(crops["source"].astype(str))) if "source" in crops.columns else []
    by_source: dict[str, Any] = {}
    if len(sources) > 1:
        for s in sources:
            sub = {
                "crops": mode_metrics(crops[crops["source"] == s], scale, red),
                "e2e": mode_metrics(e2e[e2e["source"] == s], scale, red),
            }
            q = {**quality, "reject_val": sub["e2e"]["reject_rate"]}
            by_source[s] = {
                "modes": sub,
                "thresholds": {
                    m: threshold_checks(sub[m]["metrics"], scale, thresholds, q) for m in MODES
                },
            }
    return {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        **meta,
        "scale": {"name": scale.name, "step": scale.step, "grid": list(scale.grid)},
        "modes": modes,
        "val": val,
        "val_e2e_reject_rate": VAL_E2E_REJECT,
        "quality": quality,
        "thresholds": {
            "crops": threshold_checks(modes["crops"]["metrics"], scale, thresholds, quality),
            "e2e": threshold_checks(modes["e2e"]["metrics"], scale, thresholds, quality),
        },
        "by_source": by_source,
    }


# ---------------------------------------------------------------- report.md


def _f(v: Any, nd: int = 3) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "n/d"
    return f"{v:+.{-nd}f}" if nd < 0 else f"{v:.{nd}f}"


def _pct(v: Any) -> str:
    return "n/d" if v is None else f"{100 * v:.1f}%"


ROWS = (
    ("n", "n", 0),
    ("MAE", "mae", 3),
    ("exacta", "accuracy", 3),
    ("±1 paso", "acc_within_1", 3),
    ("±2 pasos", "acc_within_2", 3),
    ("sesgo", "bias", 3),
    ("ECE", "ece", 3),
    ("cobertura 90%", "coverage_90", 3),
    ("ancho intervalo (pasos)", "interval_width_steps", 2),
    ("confianza media", "mean_confidence", 3),
)


def report_markdown(report: dict[str, Any]) -> str:
    val = report["val"] or {}
    crops, e2e = report["modes"]["crops"], report["modes"]["e2e"]
    mc, me = crops["metrics"] or {}, e2e["metrics"] or {}
    limit = f", límite {report['limit']}" if report["limit"] else ""
    lines = [
        f"# {report['task']} · evaluación final del modelo de producción",
        "",
        f"Modelo: `{report['model_version']}` · etiquetas: `{report['labels']}` "
        f"({report['n_labels']} filas{limit})"
        f" · commit `{report.get('git_commit') or 'n/d'}`.",
        "",
        f"Val = predicciones cross-fitted de T11 sobre los recortes de val (n={val.get('n', 0)}, "
        "todas juntas); val fue usado para elegir el checkpoint y calibrar, así que es optimista. "
        "Punta a punta: métricas sobre las fotos aceptadas. El filtro de calidad (m4) es el de "
        "producción en los dos modos.",
        "",
        *(
            []
            if report["is_holdout"]
            else [
                "**Prueba: estas etiquetas no son las de test; las columnas «test» no son G4.**",
                "",
            ]
        ),
        "## Val vs. test",
        "",
        "| métrica | val (T11) | test recortes | test punta a punta |",
        "|---|---|---|---|",
    ]
    for name, key, nd in ROWS:
        signed = key == "bias"
        lines.append(
            f"| {name} | {_f(val.get(key), -3 if signed else nd)} | "
            f"{_f(mc.get(key), -3 if signed else nd)} | {_f(me.get(key), -3 if signed else nd)} |"
        )
    lines += [
        f"| rechazo | {_pct(report['val_e2e_reject_rate'])} (fotos crudas, T13) | — | "
        f"{_pct(e2e['reject_rate'])} ({e2e['n_rejected']}/{e2e['n_total'] - e2e['n_missing']}) |",
        f"| low_confidence | — | — | {_pct(e2e['low_confidence_rate'])} |",
        "",
    ]
    if e2e["n_missing"]:
        lines += [f"Fotos crudas que no se encontraron: {e2e['n_missing']}.", ""]
    if e2e.get("reject_issues"):
        issues = ", ".join(f"{k}: {v}" for k, v in e2e["reject_issues"].items())
        lines += [f"Motivos de rechazo: {issues}.", ""]

    lines += ["## Por clase real (sesgo = predicho − real)", ""]
    lines += [
        "| clase | val n | val MAE | val sesgo | rec. n | rec. MAE | rec. sesgo | rec. ±1 "
        "| p.a.p. n | p.a.p. MAE | p.a.p. sesgo |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for c in [f"{v:.2f}" for v in report["scale"]["grid"]]:
        v, r, e = (d.get("by_class", {}).get(c, {}) for d in (val, mc, me))
        lines.append(
            f"| {c} | {v.get('n', 0)} | {_f(v.get('mae'))} | {_f(v.get('bias'), -3)} "
            f"| {r.get('n', 0)} | {_f(r.get('mae'))} | {_f(r.get('bias'), -3)} "
            f"| {_f(r.get('acc_within_1'))} | {e.get('n', 0)} | {_f(e.get('mae'))} "
            f"| {_f(e.get('bias'), -3)} |"
        )
    lines += ["", "## Por origen", ""]
    lines += [
        "| origen | val n | val MAE | val ±1 | rec. n | rec. MAE | rec. ±1 | rec. cobertura "
        "| p.a.p. n | p.a.p. MAE | p.a.p. ±1 |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    origins = sorted({*val.get("by_source", {}), *mc.get("by_source", {})})
    for o in origins:
        v, r, e = (d.get("by_source", {}).get(o, {}) for d in (val, mc, me))
        lines.append(
            f"| {o} | {v.get('n', 0)} | {_f(v.get('mae'))} | {_f(v.get('acc_within_1'))} "
            f"| {r.get('n', 0)} | {_f(r.get('mae'))} | {_f(r.get('acc_within_1'))} "
            f"| {_f(r.get('coverage_90'))} | {e.get('n', 0)} | {_f(e.get('mae'))} "
            f"| {_f(e.get('acc_within_1'))} |"
        )

    if report.get("by_source"):
        vds = val.get("by_dataset", {})
        lines += ["", "## Por fuente", ""]
        lines += [
            "| fuente | modo | n | MAE | exacta | ±1 | sesgo | ECE | cobertura 90% | rechazo "
            "| low_confidence | m3 | m4 |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
        ]
        for s, b in report["by_source"].items():
            v = vds.get(s, {})
            lines.append(
                f"| {s} | val | {v.get('n', 0)} | {_f(v.get('mae'))} | {_f(v.get('accuracy'))} "
                f"| {_f(v.get('acc_within_1'))} | {_f(v.get('bias'), -3)} | {_f(v.get('ece'))} "
                f"| {_f(v.get('coverage_90'))} | — | — | — | — |"
            )
            for mode in MODES:
                mm = b["modes"][mode]
                m = mm["metrics"] or {}
                th = (b["thresholds"].get(mode) or {}) if b.get("thresholds") else {}
                ok = {st: ("sí" if r["passed"] else "**no**") for st, r in (th or {}).items()}
                lines.append(
                    f"| {s} | {MODE_NAMES[mode]} | {m.get('n', 0)} | {_f(m.get('mae'))} "
                    f"| {_f(m.get('accuracy'))} | {_f(m.get('acc_within_1'))} "
                    f"| {_f(m.get('bias'), -3)} | {_f(m.get('ece'))} | {_f(m.get('coverage_90'))} "
                    f"| {_pct(mm['reject_rate'])} | {_pct(mm['low_confidence_rate'])} "
                    f"| {ok.get('m3', 'n/d')} | {ok.get('m4', 'n/d')} |"
                )
        lines.append("")

    lines += ["", "## GS pintadas vs. no pintadas (recortes)", ""]
    paint = crops.get("paint")
    if paint is None:
        lines.append("Sin medida de pintura (no hay imágenes de GS): se omite.")
    else:
        lines += [
            f"Pintada = más de {100 * paint['threshold']:.1f}% de píxeles rojo-naranja en el "
            "recorte (T06/T09). Referencia de val (G2): 4.25 pintadas MAE 0,48 vs. 0,25.",
            "",
            "| subgrupo | n | MAE | exacta | ±1 | sesgo | MAE en 4.25 |",
            "|---|---|---|---|---|---|---|",
        ]
        for name in ("pintadas", "no pintadas"):
            m = paint[name]
            if m is None:
                lines.append(f"| {name} | 0 | n/d | n/d | n/d | n/d | n/d |")
                continue
            top = m["by_class"].get(f"{report['scale']['grid'][-1]:.2f}", {})
            lines.append(
                f"| {name} | {m['n']} | {_f(m['mae'])} | {_f(m['accuracy'])} "
                f"| {_f(m['acc_within_1'])} | {_f(m['bias'], -3)} | {_f(top.get('mae'))} |"
            )

    lines += ["", "## Umbrales (`eval/thresholds.yaml`)", ""]
    lines += ["| modo | etapa | chequeo | valor | umbral | ok |", "|---|---|---|---|---|---|"]
    for mode, stages in report["thresholds"].items():
        for stage, res in (stages or {}).items():
            for c in res["checks"]:
                mark = "sí" if c["passed"] else "**no**"
                note = f" ({c['note']})" if c.get("note") else ""
                lines.append(
                    f"| {MODE_NAMES[mode]} | {stage} | {c['key']} | {_f(c['value'])}{note} "
                    f"| {c['op']} {c['threshold']} | {mark} |"
                )
    lines += [
        "",
        f"Filtro de calidad (m4): {report['quality']['note']}.",
        "",
        "## Notas",
        "",
        "- Fuga conocida del split (G0 §7): < 0,5% de casi-duplicados entre splits; las métricas "
        "de test pueden estar levemente infladas.",
        "- Recortes: el recorte del XML va directo al modelo (sin filtro ni detector), como en "
        "T09/T11. Punta a punta: foto cruda por el pipeline de producción de G3.",
        "",
        f"Gráficos: {', '.join(report['plots'].values())}.",
        "",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------- main


def write_outputs(report: dict[str, Any], run_dir: Path, config: dict[str, Any]) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    report["plots"] = {}
    for mode in MODES:
        m = report["modes"][mode]["metrics"]
        if m is None:
            continue
        title = f"{report['task']} · {MODE_NAMES[mode]}"
        report["plots"][f"{mode}_confusion"] = plot_confusion(
            m, run_dir / f"confusion_{mode}.png", title
        ).name
        report["plots"][f"{mode}_reliability"] = plot_reliability(
            m, run_dir / f"reliability_{mode}.png", title
        ).name
    path = run_dir / "report.json"
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    (run_dir / "report.md").write_text(report_markdown(report))
    (run_dir / "config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n")
    return path


def _git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, check=True
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return out.stdout.strip()


def main(argv: list[str] | None = None, pipe: Any = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--labels",
        type=Path,
        nargs="+",
        help="uno o más CSV con el esquema de labels.csv (default: $BCS_DATA_DIR/"
        "holdout_labels_sdb_bcs.csv y, con el modelo zen, holdout_labels_zen_rgbd.csv)",
    )
    parser.add_argument("--limit", type=int, help="evaluar solo N imágenes (pruebas)")
    parser.add_argument("--task", default=TASK, help="carpeta en runs/ (default: G4)")
    parser.add_argument("--runs-dir", type=Path, help="default: $BCS_RUNS_DIR o ~/bcs-shared/runs")
    parser.add_argument("--data-dir", type=Path, help="default: $BCS_DATA_DIR")
    parser.add_argument(
        "--val-predictions",
        type=Path,
        help="predicciones de val de referencia (default: $BCS_ARTIFACTS_DIR/T11/"
        "predictions_val.csv; con el modelo zen, T28/)",
    )
    parser.add_argument(
        "--quality",
        type=Path,
        help="quality_metrics.json del filtro (reject_negatives); default: el de G3 o, con el "
        "modelo zen, $BCS_ARTIFACTS_DIR/efficientnet_b0_zen/quality_metrics.json",
    )
    parser.add_argument("--thresholds", type=Path, default=THRESHOLDS)
    args = parser.parse_args(argv)
    if args.limit is not None and args.limit < 1:
        parser.error("--limit tiene que ser ≥ 1")

    data: Path = args.data_dir or data_dir()
    from bcs.inference import model_pipeline, model_spec, model_version

    zen = pipe is None and model_spec().name == "efficientnet_b0_zen"
    holdouts = [data / HOLDOUT_LABELS, *([data / HOLDOUT_LABELS_ZEN] if zen else [])]
    labels_paths: list[Path] = args.labels or holdouts
    val_path: Path = args.val_predictions or (
        artifacts_dir() / ("T28" if zen else "T11") / "predictions_val.csv"
    )
    quality_path: Path | None = args.quality or (
        artifacts_dir() / "efficientnet_b0_zen" / "quality_metrics.json" if zen else None
    )
    base_quality = None
    if quality_path is not None and quality_path.exists():
        q = json.loads(quality_path.read_text())
        base_quality = {
            "reject_negatives": q.get("reject_negatives"),
            "accept_rear": q.get("accept_rear"),
            "note": q.get("note") or f"reject_negatives de {quality_path}",
        }
    thresholds = yaml.safe_load(args.thresholds.read_text())
    try:
        df = load_all_labels(labels_paths, args.limit, data)
    except FinalEvalError as e:
        print(f"etiquetas inválidas: {e}", file=sys.stderr)
        return 2

    if pipe is None:
        pipe = model_pipeline()
        version = model_version()
    else:
        version = pipe.scorer.model_version
    scale: Scale = pipe.scale

    try:
        crops = run_crops(df, pipe, data)
        e2e = run_e2e(df, pipe, data)
    except ValueError as e:
        print(f"etiquetas inválidas: {e}", file=sys.stderr)
        return 2
    meta = {
        "task": args.task,
        "model_version": version,
        "labels": ", ".join(str(p) for p in labels_paths),
        "is_holdout": all(
            p.resolve() in {h.resolve() for h in (data / HOLDOUT_LABELS, data / HOLDOUT_LABELS_ZEN)}
            for p in labels_paths
        ),
        "labels_sha256": {str(p): sha256_file(p) for p in labels_paths},
        "n_labels": len(df),
        "limit": args.limit,
        "git_commit": _git_commit(),
    }
    report = build_final_report(
        crops, e2e, scale, thresholds, val_reference(val_path, scale), meta, base_quality
    )

    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_dir = (args.runs_dir or runs_dir()) / args.task / ts
    config = {
        "argv": sys.argv if argv is None else ["eval/evaluate_final.py", *argv],
        **meta,
        "data_dir": str(data),
        "val_predictions": str(val_path),
        "quality": str(quality_path) if quality_path is not None else None,
        "thresholds_file": str(args.thresholds),
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    crops.to_csv(run_dir / "predictions_crops.csv", index=False)
    e2e.to_csv(run_dir / "predictions_e2e.csv", index=False)
    path = write_outputs(report, run_dir, config)

    for mode, stages in report["thresholds"].items():
        for stage, res in (stages or {}).items():
            verdict = "PASA" if res["passed"] else "NO PASA"
            print(f"{MODE_NAMES[mode]} · {stage}: {verdict}", file=sys.stderr)
    print(f"reporte: {path} · {run_dir / 'report.md'}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
