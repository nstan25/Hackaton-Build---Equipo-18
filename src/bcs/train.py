"""`python -m bcs.train --config config/<modelo>.yaml --fold <val|all> [--smoke]` (SPEC §2.4).

Un solo código de entrenamiento para todos los backends (local, Modal, Kaggle, Tenki): cada
backend solo empaqueta los datos y llama a este módulo.

Datos (decisiones de G0): `labels.csv` (solo train+val) + `splits.csv` (`image_id,split`) en
`--data-dir` (por defecto `$BCS_DATA_DIR`, o un pack extraído de `make pack-data`). Hay un único
split: se entrena en `train` y se evalúa en `val`; `--fold all`, `val` y `0` son lo mismo. Las
filas de `test` y los grupos de `holdout_groups.txt` se descartan siempre, aunque aparezcan.

Config: `config/<modelo>.yaml` se aplica encima de `config/train.yaml`. `model.name` elige el
`Scorer`: un nombre del registro (`stub_uniform`) o `model.target: "paquete.modulo:Clase"`. La
clase se construye con `grid` y, si los acepta, `config` (el config completo ya resuelto).

`Scorer.fit(df)` recibe filas de `labels.csv` con `path` **absoluto**. El modelo nunca ve el
`cow_id` (no existe en los datos de entrenamiento).

Salidas: `report.json` (+ `config.json`, PNG y predicciones) en `runs/<task>/<ts>/`; pesos y
`predictions_val.csv` en `$BCS_ARTIFACTS_DIR/<task>/`. `report.json` lo arma
`bcs.eval.report.build_report` (el mismo esquema que `make eval`, T02) sobre val, y los datos de
la corrida (backend, device, tiempos, pack, n) van en `report["train"]`.

Sin `labels.csv` y con `--smoke`, corre sobre las fixtures sintéticas del repo y lo marca con
`"synthetic": true`: sirve para probar el pipeline, las métricas no significan nada.
"""

from __future__ import annotations

import argparse
import copy
import importlib
import inspect
import json
import os
import platform
import random
import subprocess
import sys
import time
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from PIL import Image

from bcs.data.schema import load_labels
from bcs.eval.report import EvalInputError, build_report, synthetic_predictions, write_run
from bcs.models.base import Scorer
from bcs.paths import artifacts_dir, config_dir, data_dir, fixtures_dir, runs_dir
from bcs.scale import load_scale

SINGLE_FOLD = "val"
FOLD_ALIASES = {"all", "val", "0"}
REGISTRY: dict[str, str] = {"stub_uniform": "bcs.models.stub:UniformScorer"}
EVAL_BATCH = 64


class TrainError(RuntimeError):
    pass


# ---------------------------------------------------------------- config


def _deep_merge(base: dict[str, Any], over: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_config(path: Path, model: str | None = None) -> dict[str, Any]:
    """`path` encima de `config/train.yaml`. `model` pisa `model.name` (para `stub_uniform`)."""
    base_path = config_dir() / "train.yaml"
    cfg: dict[str, Any] = yaml.safe_load(base_path.read_text()) or {}
    if path.resolve() != base_path.resolve():
        cfg = _deep_merge(cfg, yaml.safe_load(path.read_text()) or {})
    if model:
        cfg.setdefault("model", {})["name"] = model
    return cfg


def resolve_config_path(model: str) -> Path:
    """`config/<model>.yaml` si existe; si no, el base (`model.name` se pisa con `model`)."""
    candidate = config_dir() / f"{model}.yaml"
    return candidate if candidate.exists() else config_dir() / "train.yaml"


def build_scorer(cfg: dict[str, Any], grid: tuple[float, ...]) -> Scorer:
    model_cfg = cfg.get("model") or {}
    name = str(model_cfg.get("name") or "")
    target = model_cfg.get("target") or REGISTRY.get(name)
    if not target:
        raise TrainError(f"modelo desconocido {name!r}: poné model.target 'modulo:Clase'")
    module_name, _, cls_name = str(target).partition(":")
    cls = getattr(importlib.import_module(module_name), cls_name)
    params = inspect.signature(cls).parameters
    kwargs: dict[str, Any] = {"config": cfg} if "config" in params else {}
    scorer = cls(grid, **kwargs)
    if not isinstance(scorer, Scorer):
        raise TrainError(f"{target} no implementa el Protocol Scorer")
    return scorer


# ---------------------------------------------------------------- datos


def _read_holdout_groups(root: Path) -> set[str]:
    path = root / "holdout_groups.txt"
    if not path.exists():
        return set()
    return {ln.strip() for ln in path.read_text().splitlines() if ln.strip()}


def _rows(df: pd.DataFrame, mask: pd.Series) -> pd.DataFrame:
    return pd.DataFrame(df.loc[mask]).reset_index(drop=True)


def _synthetic_split(df: pd.DataFrame) -> list[str]:
    """Fixtures sin split: uno de cada tres grupos (orden alfabético) va a val."""
    groups = sorted(set(df["group_id"]))
    val = {g for i, g in enumerate(groups) if i % 3 == 1}
    return ["val" if g in val else "train" for g in df["group_id"]]


def load_dataset(root: Path, *, allow_synthetic: bool) -> tuple[pd.DataFrame, bool]:
    """Filas de train+val con columna `split` y `path` absoluto. Devuelve (df, synthetic)."""
    labels_path = root / "labels.csv"
    if not labels_path.exists():
        if not allow_synthetic:
            raise TrainError(f"no existe {labels_path} (corré make labels o pasá un pack)")
        df = load_labels(fixtures_dir() / "labels_fixture.csv")
        df["path"] = [str(fixtures_dir().parent / p) for p in df["path"]]
        df["split"] = _synthetic_split(df)
        return df, True

    df = load_labels(labels_path)
    splits_path = root / "splits.csv"
    if not splits_path.exists():
        raise TrainError(f"no existe {splits_path} (`image_id,split`, lo escribe make labels)")
    splits = pd.read_csv(splits_path, dtype=str, keep_default_na=False)
    df = df.merge(splits[["image_id", "split"]], on="image_id", how="left")
    if bool(df["split"].isna().any()) or bool((df["split"] == "").any()):
        missing = int(df["split"].isna().sum() + (df["split"] == "").sum())
        raise TrainError(f"{missing} imágenes de labels.csv sin split en splits.csv")
    holdout = _read_holdout_groups(root)
    keep = df["split"].isin(["train", "val"]) & ~df["group_id"].isin(sorted(holdout))
    df = _rows(df, keep)
    df["path"] = [str(root / p) for p in df["path"]]
    return df, False


def check_no_leakage(train: pd.DataFrame, val: pd.DataFrame) -> None:
    shared = set(train["group_id"]) & set(val["group_id"])
    if shared:
        raise TrainError(f"{len(shared)} group_id en train y val a la vez: {sorted(shared)[:5]}")


def smoke_subsample(df: pd.DataFrame, max_images: int, seed: int) -> pd.DataFrame:
    """A lo sumo `max_images` filas, proporcional por split, con al menos 1 de val."""
    if len(df) <= max_images:
        return df
    parts = []
    for _split, part in df.groupby("split", sort=True):
        n = max(1, round(max_images * len(part) / len(df)))
        parts.append(part.sample(n=min(n, len(part)), random_state=seed))
    return pd.concat(parts).sort_values("image_id").reset_index(drop=True)


def _load_images(paths: Sequence[str]) -> list[Image.Image]:
    out = []
    for p in paths:
        with Image.open(p) as im:
            out.append(im.convert("RGB"))
    return out


def predict_in_batches(scorer: Scorer, paths: Sequence[str]) -> np.ndarray:
    chunks = [
        scorer.predict_proba(_load_images(paths[i : i + EVAL_BATCH]))
        for i in range(0, len(paths), EVAL_BATCH)
    ]
    return np.concatenate(chunks, axis=0) if chunks else np.zeros((0, 0))


# ---------------------------------------------------------------- corrida


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:  # pragma: no cover - torch está en las deps
        pass


def _git_sha() -> str:
    if sha := os.environ.get("BCS_GIT_SHA"):
        return sha
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            cwd=Path(__file__).parent,
        )
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _device() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            return f"cuda:{torch.cuda.get_device_name(0)}"
    except ImportError:  # pragma: no cover
        pass
    return "cpu"


def _pack_hash(root: Path) -> str | None:
    manifest = root / "manifest.json"
    if manifest.exists():
        return json.loads(manifest.read_text()).get("pack_hash")
    return None


def run(
    config_path: Path,
    *,
    fold: str = "all",
    smoke: bool = False,
    model: str | None = None,
    task: str | None = None,
    data_root: Path | None = None,
    out_dir: Path | None = None,
    weights_dir: Path | None = None,
    backend: str | None = None,
) -> dict[str, Any]:
    """Entrena y evalúa sobre val. Devuelve el report (también escrito en `out_dir`)."""
    if fold not in FOLD_ALIASES:
        raise TrainError(f"--fold {fold!r}: hay un único split train/val (G0); usá all o val")
    t0 = time.monotonic()
    cfg = load_config(config_path, model)
    seed = int(cfg.get("seed", 42))
    if smoke:
        smoke_cfg = cfg.get("smoke") or {}
        cfg.setdefault("train", {})["epochs"] = int(smoke_cfg.get("epochs", 1))
    _seed_everything(seed)

    scale = load_scale(cfg.get("scale"))
    model_name = str((cfg.get("model") or {}).get("name"))
    task = task or os.environ.get("BCS_TASK") or model_name
    root = data_root or data_dir()
    df, synthetic = load_dataset(root, allow_synthetic=smoke)
    if smoke:
        df = smoke_subsample(df, int((cfg.get("smoke") or {}).get("max_images", 500)), seed)
    df["bcs"] = [scale.snap(float(b)) for b in df["bcs"]]
    train_df = _rows(df, df["split"] == "train").drop(columns="split")
    val_df = _rows(df, df["split"] == "val").drop(columns="split")
    if train_df.empty or val_df.empty:
        raise TrainError(f"split vacío: train={len(train_df)} val={len(val_df)}")
    check_no_leakage(train_df, val_df)

    scorer = build_scorer(cfg, scale.grid)
    t_fit = time.monotonic()
    scorer.fit(train_df)
    fit_seconds = time.monotonic() - t_fit

    probs = predict_in_batches(scorer, list(val_df["path"]))
    if probs.shape != (len(val_df), len(scale.grid)):
        raise TrainError(
            f"predict_proba devolvió {probs.shape}, esperaba {len(val_df)}x{len(scale.grid)}"
        )
    preds = synthetic_predictions(
        np.asarray(val_df["bcs"], dtype=float),
        probs,
        scale,
        image_id=list(val_df["image_id"]),
        group_id=list(val_df["group_id"]),
        source=list(val_df["source"]),
        fold=[SINGLE_FOLD] * len(val_df),
    )
    try:
        report = build_report(
            preds,
            scale,
            model=model_name,
            task=task,
            y_train=np.asarray(train_df["bcs"], dtype=float),
        )
    except EvalInputError as e:
        raise TrainError(f"predicciones inválidas: {e}") from e

    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    suffix = "_smoke" if smoke else ""
    out_dir = out_dir or runs_dir() / task / ts
    weights_dir = weights_dir or artifacts_dir() / task
    out_dir.mkdir(parents=True, exist_ok=True)
    weights_dir.mkdir(parents=True, exist_ok=True)
    weights_path = weights_dir / f"{model_name}_{SINGLE_FOLD}{suffix}.weights"
    scorer.save(weights_path)
    # `make eval MODEL=<task>` lee `$BCS_ARTIFACTS_DIR/<task>/predictions_val.csv` (T02).
    preds_name = f"predictions_{SINGLE_FOLD}{suffix}.csv"
    preds.to_csv(weights_dir / preds_name, index=False)
    preds.to_csv(out_dir / preds_name, index=False)

    report["train"] = {
        "model_version": scorer.model_version,
        "fold": SINGLE_FOLD,
        "smoke": smoke,
        "synthetic": synthetic,
        "backend": backend or os.environ.get("BCS_BACKEND", "local"),
        "device": _device(),
        "host": platform.node(),
        "git_sha": _git_sha(),
        "pack_hash": _pack_hash(root),
        "seed": seed,
        "n_train": len(train_df),
        "n_val": len(val_df),
        "n_train_groups": len(set(train_df["group_id"])),
        "n_val_groups": len(set(val_df["group_id"])),
        "fit_seconds": round(fit_seconds, 2),
        "seconds": round(time.monotonic() - t0, 2),
        "weights": str(weights_path),
        "predictions": str(weights_dir / preds_name),
    }
    if synthetic:
        report["train"]["warning"] = "fixtures sintéticas: las métricas no significan nada"
    paths = write_run(json.loads(json.dumps(report, default=_json_default)), out_dir, cfg)
    report = json.loads(paths["report"].read_text())
    report["report_path"] = str(paths["report"])
    return report


def _json_default(o: object) -> Any:
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"no serializable: {type(o).__name__}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="python -m bcs.train", description=(__doc__ or "").split("\n")[0]
    )
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--fold", default="all", help="all | val | 0 (único split train/val, G0)")
    p.add_argument("--smoke", action="store_true", help="1 época, ≤ 500 imágenes")
    p.add_argument("--model", default=None, help="pisa model.name del config")
    p.add_argument("--task", default=None, help="subcarpeta de runs/ y artifacts/")
    p.add_argument("--data-dir", type=Path, default=None, help="por defecto $BCS_DATA_DIR")
    p.add_argument("--out-dir", type=Path, default=None, help="por defecto runs/<task>/<ts>/")
    p.add_argument("--weights-dir", type=Path, default=None)
    p.add_argument("--backend", default=None)
    args = p.parse_args(argv)
    try:
        report = run(
            args.config,
            fold=args.fold,
            smoke=args.smoke,
            model=args.model,
            task=args.task,
            data_root=args.data_dir,
            out_dir=args.out_dir,
            weights_dir=args.weights_dir,
            backend=args.backend,
        )
    except TrainError as e:
        print(f"bcs.train: {e}", file=sys.stderr)
        return 2
    t, m = report["train"], report["mean"]
    print(
        f"bcs.train: {report['model']} fold={t['fold']} smoke={t['smoke']} "
        f"synthetic={t['synthetic']} n_train={t['n_train']} n_val={t['n_val']} "
        f"mae={m.get('mae')} (dummy {report['dummy']['mae']}) → {report['report_path']}",
        file=sys.stderr,
    )
    print(report["report_path"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
