"""La EfficientNet-B0 del equipo (`models/bcs_efficientnet_b0.pt`) como `Scorer` (SPEC §2.3, T09).

No se reentrena. Arquitectura, transformación (224×320, normalización ImageNet) y decodificación
(`bcs = 3.75 + 0.25 · salida`) salen del código del equipo (`train_cnn.build_model`, `EVAL_TF`,
`decode`), que se importa sin editarlo. TTA: promedio con el flip horizontal, como
`bcs_pipeline.py`. Como la decodificación es lineal, promediar logits o valores da lo mismo.

T28: un `.pt` con `letterbox: true` (el de T25, `models/bcs_efficientnet_b0_zen.pt`) se evalúa con
la transformación con la que se entrenó (`eval_transform`: letterbox a 224×320 con el gris de la
media, como `train_zen.make_transforms`) y se decodifica con su `center`/`scale`; el modelo viejo
sigue estirando. `predict_views_features` devuelve además el pooling (1.280-d) de la vista sin
flip, de la misma pasada, para el puntaje OOD.

Distribución sobre la grilla (provisoria, T11 la refina): normal discretizada centrada en el valor
continuo, con σ = RMSE de los residuos en las imágenes que recibe `fit()` (las de val si `df`
trae la columna `split`). Las clases de las puntas se quedan con las colas.

`python -m bcs.models.efficientnet` corre la evaluación de T09 sobre val (nunca test): predicciones
en `$BCS_ARTIFACTS_DIR/T09/`, reporte en `runs/T09/<ts>/` (ver `run_experiment`).
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import sys
import time
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image
from scipy.special import ndtr

from bcs.models.baseline import PAINT_THRESHOLD, desaturate_red, grid_index, paint_report
from bcs.paths import REPO_ROOT, artifacts_dir

WEIGHTS = "models/bcs_efficientnet_b0.pt"
CUTS = "models/cuts_efficientnet_b0.json"
BATCH = 32
SEED = 42
SIGMA_MAX_IMAGES = 2000
DEFAULT_SIGMA = 0.25


def team_module(name: str) -> Any:
    """Un módulo del código del equipo (raíz del repo: `train_cnn`, `metrics`, ...)."""
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    root = str(REPO_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    return importlib.import_module(name)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def discretized_normal(mu: np.ndarray, sigma: float, grid: Sequence[float]) -> np.ndarray:
    """[n, K]: masa de N(mu, σ) en el intervalo de cada clase (medio paso a cada lado; las
    puntas abiertas)."""
    g = np.asarray(grid, dtype=float)
    half = (g[1] - g[0]) / 2 if len(g) > 1 else 0.5
    lo = np.r_[-np.inf, g[1:] - half][None, :]
    hi = np.r_[g[:-1] + half, np.inf][None, :]
    m = np.asarray(mu, dtype=float)[:, None]
    p = ndtr((hi - m) / sigma) - ndtr((lo - m) / sigma)
    p = np.clip(p, 1e-9, None)
    return p / p.sum(axis=1, keepdims=True)


def eval_transform(ckpt: dict[str, Any]) -> Any:
    """La transformación de evaluación que pide el checkpoint: letterbox (T25) o la de
    `train_cnn.EVAL_TF` (estira a 224×320)."""
    if not ckpt.get("letterbox"):
        return team_module("train_cnn").EVAL_TF
    from torchvision import transforms as T

    from bcs.models.train_zen import MEAN, STD, Letterbox

    size = ckpt.get("size") or (224, 320)
    return T.Compose([Letterbox(size), T.ToTensor(), T.Normalize(MEAN, STD)])


def _resolve(path: str | Path) -> Path:
    p = Path(path).expanduser()
    return p if p.is_absolute() else REPO_ROOT / p


class EfficientNetScorer:
    """Config (`config/efficientnet_b0.yaml`): `model.weights`, `model.tta`, opcional
    `model.sigma` (fijo; si no, lo estima `fit()`)."""

    def __init__(self, grid: tuple[float, ...], config: dict[str, Any] | None = None) -> None:
        m = (config or {}).get("model") or {}
        self.grid = grid
        self.weights = _resolve(m.get("weights") or WEIGHTS)
        self.tta = bool(m.get("tta", True))
        sigma = m.get("sigma")
        self.sigma = float(sigma) if sigma is not None else DEFAULT_SIGMA
        self.sigma_fixed = sigma is not None
        self.sigma_info: dict[str, Any] = {"source": "config" if sigma is not None else "default"}
        self._model: Any = None
        self._ckpt: dict[str, Any] = {}
        self._tf: Any = None

    @property
    def model_version(self) -> str:
        return f"efficientnet_b0-team-{'tta' if self.tta else 'notta'}"

    def _load_model(self) -> Any:
        if self._model is None:
            import torch

            if not self.weights.exists():
                raise FileNotFoundError(f"no existe {self.weights}")
            train_cnn = team_module("train_cnn")
            ck = torch.load(self.weights, map_location="cpu")
            loss = ck.get("loss", "huber")
            model = train_cnn.build_model(ck["arch"], pretrained=False, loss=loss)
            model.load_state_dict(ck["state_dict"])
            self._model = model.eval()
            self._ckpt = {k: v for k, v in ck.items() if k != "state_dict"} | {"loss": loss}
            self._tf = eval_transform(self._ckpt)
        return self._model

    @property
    def checkpoint(self) -> dict[str, Any]:
        """Metadatos del `.pt` (sin los pesos); carga el modelo si hace falta."""
        self._load_model()
        return self._ckpt

    def _batch(self, images: Sequence[Image.Image]) -> Any:
        import torch

        return torch.stack([self._tf(im.convert("RGB")) for im in images])

    def _decode(self, out: Any) -> Any:
        ck = self._ckpt
        if ck["loss"] == "huber" and "center" in ck:
            return out.float().squeeze(1) * float(ck.get("scale", 0.25)) + float(ck["center"])
        return team_module("train_cnn").decode(out, ck["loss"])

    def predict_views_features(
        self, images: Sequence[Image.Image], flip: bool = True
    ) -> tuple[np.ndarray, np.ndarray]:
        """([n, 2] valor con la imagen tal cual y con el flip horizontal, [n, 1280] pooling global
        de la vista sin flip, float32). El valor sin flip sale de la cabeza sobre ese pooling: es
        el mismo `model(x)` (T27: dif. máx. 9e-16). Con `flip=False` no se corre la vista
        espejada y su columna queda en NaN."""
        import torch

        model = self._load_model()
        out, feats = [], []
        with torch.inference_mode():
            for i in range(0, len(images), BATCH):
                x = self._batch(images[i : i + BATCH])
                f = model.forward_head(model.forward_features(x), pre_logits=True)
                plain = self._decode(model.get_classifier()(f))
                mirror = (
                    self._decode(model(torch.flip(x, dims=[3])))
                    if flip
                    else torch.full_like(plain, float("nan"))
                )
                out.append(torch.stack([plain, mirror], 1).numpy())
                feats.append(f.numpy().astype(np.float32))
        if not out:
            return np.zeros((0, 2)), np.zeros((0, 0), dtype=np.float32)
        return np.concatenate(out).astype(float), np.concatenate(feats)

    def predict_views(self, images: Sequence[Image.Image]) -> np.ndarray:
        """[n, 2]: valor continuo con la imagen tal cual y con el flip horizontal."""
        return self.predict_views_features(images)[0]

    def predict_value_features(
        self, images: Sequence[Image.Image]
    ) -> tuple[np.ndarray, np.ndarray]:
        """BCS continuo con TTA y el pooling de la vista sin flip (una sola pasada)."""
        views, feats = self.predict_views_features(images)
        return views.mean(axis=1), feats

    def predict_value(self, images: Sequence[Image.Image]) -> np.ndarray:
        """BCS continuo (con TTA si `tta`)."""
        if not self.tta:
            import torch

            model = self._load_model()
            vals = []
            with torch.inference_mode():
                for i in range(0, len(images), BATCH):
                    vals.append(self._decode(model(self._batch(images[i : i + BATCH]))).numpy())
            return np.concatenate(vals).astype(float) if vals else np.zeros(0)
        return self.predict_views(images).mean(axis=1)

    def proba_from_values(self, values: np.ndarray) -> np.ndarray:
        return discretized_normal(values, self.sigma, self.grid)

    def predict_proba(self, images: Sequence[Image.Image]) -> np.ndarray:
        return self.proba_from_values(self.predict_value(images))

    def fit(self, df: pd.DataFrame) -> None:
        """No entrena: estima σ (RMSE de residuos) en las filas de val si las hay, si no en
        todas; a lo sumo `SIGMA_MAX_IMAGES` elegidas con semilla fija."""
        if self.sigma_fixed:
            return
        rows = df.loc[df["split"] == "val"] if "split" in df.columns else df
        rows = rows if len(rows) else df
        if len(rows) > SIGMA_MAX_IMAGES:
            rows = rows.sample(SIGMA_MAX_IMAGES, random_state=SEED)
        images = []
        for p in rows["path"]:
            with Image.open(p) as im:
                images.append(im.convert("RGB"))
        values = self.predict_value(images)
        self.set_sigma_from(values, rows["bcs"].to_numpy(dtype=float), "fit")

    def set_sigma_from(self, values: np.ndarray, y: np.ndarray, source: str) -> None:
        resid = np.asarray(values, dtype=float) - np.asarray(y, dtype=float)
        self.sigma = max(float(np.sqrt(np.mean(resid**2))), 1e-3)
        self.sigma_info = {"source": source, "n": len(resid), "bias": float(resid.mean())}

    def save(self, path: Path) -> None:
        meta = {
            "model_version": self.model_version,
            "grid": list(self.grid),
            "weights": str(self.weights),
            "weights_sha256": sha256_file(self.weights),
            "tta": self.tta,
            "sigma": self.sigma,
            "sigma_info": self.sigma_info,
        }
        Path(path).write_text(json.dumps(meta, indent=2))

    def load(self, path: Path) -> None:
        meta = json.loads(Path(path).read_text())
        self.grid = tuple(float(v) for v in meta["grid"])
        self.weights = _resolve(meta["weights"])
        if self.weights.exists() and sha256_file(self.weights) != meta["weights_sha256"]:
            raise ValueError(f"{self.weights} no es el mismo .pt que se guardó")
        self.tta = bool(meta["tta"])
        self.sigma, self.sigma_fixed = float(meta["sigma"]), True
        self.sigma_info = meta.get("sigma_info", {})
        self._model = None


# ---------------------------------------------------------------- evaluación de T09


def origin(image_ids: pd.Series | Sequence[str]) -> np.ndarray:
    """GS / YM / L-R a partir del `image_id` (`sdb_bcs_GS_1000_1`, `sdb_bcs_L-i10454`, ...)."""
    farm = pd.Series(list(image_ids), dtype=str).str.removeprefix("sdb_bcs_").str[:2]
    return np.where(farm.isin(["L-", "R-"]), "L-R", farm.str.rstrip("_-")).astype(str)


def value_metrics(y: np.ndarray, value: np.ndarray, pred: np.ndarray, step: float) -> dict:
    """`value` continuo, `pred` discreto (una de las discretizaciones)."""
    e = pred - y
    return {
        "n": len(y),
        "mae_continuous": float(np.abs(value - y).mean()),
        "bias_continuous": float((value - y).mean()),
        "mae": float(np.abs(e).mean()),
        "bias": float(e.mean()),
        "accuracy": float((np.abs(e) < 1e-6).mean()),
        "acc_within_1": float((np.abs(e) <= step + 1e-6).mean()),
    }


def breakdown(
    y: np.ndarray, value: np.ndarray, pred: np.ndarray, grid: Sequence[float], org: np.ndarray
) -> dict[str, Any]:
    g = np.asarray(grid, dtype=float)
    step = float(g[1] - g[0])
    out: dict[str, Any] = {
        "all": value_metrics(y, value, pred, step),
        "by_class": {},
        "by_origin": {},
    }
    for c in g:
        m = np.isclose(y, c)
        if m.any():
            out["by_class"][f"{c:.2f}"] = value_metrics(y[m], value[m], pred[m], step)
    for o in sorted(set(org)):
        m = org == o
        out["by_origin"][o] = value_metrics(y[m], value[m], pred[m], step)
    return out


def _views_cached(
    scorer: EfficientNetScorer, ids: list[str], paths: list[str], name: str, desat: bool = False
) -> pd.DataFrame:
    """Valores (plain, flip) por `image_id`, cacheados en `$BCS_ARTIFACTS_DIR/T09/views_<name>.csv`
    por tandas: una corrida cortada sigue donde quedó."""
    cache = artifacts_dir() / "T09" / f"views_{name}.csv"
    cache.parent.mkdir(parents=True, exist_ok=True)
    have = (
        pd.read_csv(cache, dtype={"image_id": str})
        if cache.exists()
        else pd.DataFrame(columns=["image_id", "plain", "flip"])
    )
    done = set(have["image_id"])
    todo = [(i, p) for i, p in zip(ids, paths, strict=True) if i not in done]
    t0 = time.monotonic()
    chunk = 512
    for k in range(0, len(todo), chunk):
        part = todo[k : k + chunk]
        images = []
        for _, p in part:
            with Image.open(p) as im:
                rgb = im.convert("RGB")
            images.append(desaturate_red(rgb) if desat else rgb)
        v = scorer.predict_views(images)
        new = pd.DataFrame({"image_id": [i for i, _ in part], "plain": v[:, 0], "flip": v[:, 1]})
        new.to_csv(cache, mode="a", header=not cache.exists(), index=False)
        have = pd.concat([have, new], ignore_index=True)
        rate = (k + len(part)) / max(time.monotonic() - t0, 1e-9)
        print(f"[T09 {name}] {k + len(part)}/{len(todo)} · {rate:.1f} img/s", file=sys.stderr)
    have = have.drop_duplicates("image_id", keep="last").set_index("image_id")
    return have.loc[ids].astype(float)


def latency(scorer: EfficientNetScorer, paths: Sequence[str], n: int = 40) -> dict[str, Any]:
    """Segundos por imagen en CPU, de a una (como la API), con y sin TTA."""
    import torch

    images = []
    for p in paths[:n]:
        with Image.open(p) as im:
            images.append(im.convert("RGB"))
    model = scorer._load_model()
    out: dict[str, Any] = {"threads": torch.get_num_threads(), "n": len(images)}
    for tta in (False, True):
        scorer.tta = tta
        scorer.predict_value(images[:2])  # calentar
        times = []
        for im in images:
            t = time.perf_counter()
            scorer.predict_value([im])
            times.append(time.perf_counter() - t)
        key = "tta" if tta else "no_tta"
        out[key] = {"median_s": float(np.median(times)), "p95_s": float(np.quantile(times, 0.95))}
    scorer.tta = True
    out["params"] = int(sum(p.numel() for p in model.parameters()))
    out["weights_mb"] = round(scorer.weights.stat().st_size / 2**20, 2)
    return out


def run_experiment(task: str = "T09", config: Path | None = None) -> dict[str, Any]:
    from bcs.eval.report import build_report, synthetic_predictions, write_run
    from bcs.paths import data_dir, runs_dir
    from bcs.scale import load_scale
    from bcs.train import check_no_leakage, load_config, load_dataset, resolve_config_path

    t0 = time.monotonic()
    cfg = load_config(config or resolve_config_path("efficientnet_b0"))
    scale = load_scale(cfg.get("scale"))
    grid = scale.grid
    g = np.asarray(grid, dtype=float)
    scorer = EfficientNetScorer(grid, cfg)
    out_dir = artifacts_dir() / task
    out_dir.mkdir(parents=True, exist_ok=True)

    df, _ = load_dataset(data_dir(), allow_synthetic=False)
    df["bcs"] = [scale.snap(float(b)) for b in df["bcs"]]
    tr = pd.DataFrame(df.loc[df["split"] == "train"]).reset_index(drop=True)
    va = pd.DataFrame(df.loc[df["split"] == "val"]).reset_index(drop=True)
    check_no_leakage(tr, va)
    ids, y = list(va["image_id"]), va["bcs"].to_numpy(dtype=float)
    y_tr = tr["bcs"].to_numpy(dtype=float)
    org = origin(list(va["image_id"]))

    # 1. nuestros recortes (texto tapado), lo que ve el modelo en producción
    views = _views_cached(scorer, ids, list(va["path"]), "covered")
    value = views.mean(axis=1).to_numpy()
    value_plain = views["plain"].to_numpy()
    scorer.set_sigma_from(value, y, "val")
    probs = scorer.proba_from_values(value)
    pred_argmax = g[probs.argmax(axis=1)]
    meta_cols = {
        "image_id": ids,
        "group_id": list(va["group_id"]),
        "source": list(va["source"]),
        "fold": ["val"] * len(va),
    }
    preds = synthetic_predictions(y, probs, scale, **meta_cols)
    preds["value"] = value
    preds.to_csv(out_dir / "predictions_val.csv", index=False)
    scorer.save(out_dir / "scorer.json")

    # baseline tonto: la frecuencia de clases de train para toda imagen
    prior = np.bincount(grid_index(y_tr, grid), minlength=len(g)) / len(y_tr)
    dummy = synthetic_predictions(y, np.tile(prior, (len(y), 1)), scale, **meta_cols)
    dummy.to_csv(out_dir / "predictions_val_dummy.csv", index=False)

    # 2. texto sobreimpreso: los recortes del README sin tapar (solo val)
    crops_val = data_dir() / "raw" / "sdb_bcs_pre" / "crops" / "val"
    uncovered = {p.stem: p for p in crops_val.glob("*/*.jpg")}
    keys = [i.removeprefix("sdb_bcs_") for i in ids]
    has_unc = np.array([k in uncovered for k in keys])
    unc_ids = [i for i, h in zip(ids, has_unc, strict=True) if h]
    unc_paths = [str(uncovered[k]) for k, h in zip(keys, has_unc, strict=True) if h]
    views_unc = _views_cached(scorer, unc_ids, unc_paths, "uncovered")
    value_unc = np.full(len(ids), np.nan)
    value_unc[has_unc] = views_unc.mean(axis=1).to_numpy()
    probs_unc = scorer.proba_from_values(np.where(has_unc, value_unc, value))
    synthetic_predictions(
        y[has_unc],
        probs_unc[has_unc],
        scale,
        **{k: list(np.asarray(v)[has_unc]) for k, v in meta_cols.items()},
    ).to_csv(out_dir / "predictions_val_uncovered.csv", index=False)

    # 3. pintura roja en GS: la misma imagen con la franja desaturada
    paint = pd.read_csv(runs_dir() / "T06" / "paint_frac.csv", dtype={"image_id": str})
    red_of = dict(zip(paint["image_id"], paint["red_frac"].astype(float), strict=True))
    red = np.array([red_of.get(i, np.nan) for i in ids], dtype=float)
    gs = org == "GS"
    has_red = gs & (red > 0)
    red_ids = [i for i, h in zip(ids, has_red, strict=True) if h]
    red_paths = [p for p, h in zip(va["path"], has_red, strict=True) if h]
    views_d = _views_cached(scorer, red_ids, red_paths, "desat", desat=True)
    value_d = value.copy()
    value_d[has_red] = views_d.mean(axis=1).to_numpy()
    painted = red > PAINT_THRESHOLD

    # 4. latencia y peso
    lat = latency(scorer, list(va["path"]))

    # discretizaciones: argmax de la normal (lo que mide make eval), redondeo y cortes del equipo
    cuts = np.sort(json.loads(_resolve(CUTS).read_text())["cuts"])
    disc = {
        "argmax": pred_argmax,
        "nearest": g[grid_index(value, grid)],
        "team_cuts": g[np.searchsorted(cuts, value)],
    }
    report = build_report(preds, scale, model="efficientnet_b0", task=task, y_train=y_tr)
    m = has_unc
    text: dict[str, Any] = {
        "n": int(m.sum()),
        "n_missing_uncovered": int((~m).sum()),
        "covered": breakdown(y[m], value[m], g[grid_index(value[m], grid)], grid, org[m]),
        "uncovered": breakdown(y[m], value_unc[m], g[grid_index(value_unc[m], grid)], grid, org[m]),
        "delta_mean": float(np.mean(value_unc[m] - value[m])),
        "delta_abs_mean": float(np.mean(np.abs(value_unc[m] - value[m]))),
        "delta_abs_p95": float(np.quantile(np.abs(value_unc[m] - value[m]), 0.95)),
        "argmax_eval_uncovered": build_report(
            pd.read_csv(out_dir / "predictions_val_uncovered.csv"),
            scale,
            model="efficientnet_b0_uncovered",
            y_train=y_tr,
        )["mean"],
    }
    report["t09"] = {
        "model": {
            "weights": str(scorer.weights.relative_to(REPO_ROOT)),
            "weights_sha256": sha256_file(scorer.weights),
            "checkpoint": {k: v for k, v in scorer._ckpt.items() if not isinstance(v, dict)},
            "tta": True,
            "sigma": scorer.sigma,
            "sigma_info": scorer.sigma_info,
            "distribution": "normal discretizada centrada en el valor continuo (provisorio, T11)",
        },
        "n_train": len(tr),
        "n_val": len(va),
        "metrics": {name: breakdown(y, value, d, grid, org) for name, d in disc.items()},
        "no_tta_nearest": value_metrics(y, value_plain, g[grid_index(value_plain, grid)], 0.25),
        "dummy": report["dummy"],
        "text_overlay": text,
        "paint_gs_val": {
            "n_gs": int(gs.sum()),
            "n_painted": int((gs & painted).sum()),
            "argmax": paint_report(
                y[gs], pred_argmax[gs], value[gs], value_d[gs], painted[gs], grid
            ),
            "nearest": paint_report(
                y[gs], disc["nearest"][gs], value[gs], value_d[gs], painted[gs], grid
            ),
        },
        "latency_cpu": lat,
        "seconds": round(time.monotonic() - t0, 1),
    }
    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_dir = runs_dir() / task / ts
    paths = write_run(report, run_dir, cfg)
    (run_dir / "summary.md").write_text(summary_markdown(report))
    print(f"[T09] reporte → {paths['report']}", file=sys.stderr)
    report["report_path"] = str(paths["report"])
    return report


def summary_markdown(report: dict[str, Any]) -> str:
    t = report["t09"]

    def row(name: str, m: dict[str, Any]) -> str:
        return (
            f"| {name} | {m['n']} | {m['mae_continuous']:.4f} | {m['mae']:.4f} | "
            f"{m['accuracy']:.3f} | {m['acc_within_1']:.3f} | {m['bias']:+.3f} |"
        )

    head = "| | n | MAE cont. | MAE | exacta | ±1 paso | sesgo |\n|---|---|---|---|---|---|---|"
    d = t["dummy"]
    lines = [
        f"# T09 · EfficientNet-B0 del equipo sobre val (n={t['n_val']})",
        "",
        f"σ = {t['model']['sigma']:.4f} (RMSE en val). Tonto: MAE {d['mae']:.4f} "
        f"(moda {d['value_mode']}, media {d['value_mean']}). Mejora (argmax, `make eval`): "
        f"{report['mae_improvement_vs_dummy']:.3f}.",
        "",
    ]
    for name, b in t["metrics"].items():
        lines += [f"## Discretización: {name}", "", head, row("todo", b["all"])]
        lines += [row(f"clase {c}", m) for c, m in b["by_class"].items()]
        lines += [row(f"origen {o}", m) for o, m in b["by_origin"].items()]
        lines.append("")
    lines += ["## Sin TTA (redondeo)", "", head, row("todo", t["no_tta_nearest"]), ""]
    x = t["text_overlay"]
    lines += [
        "## Texto sobreimpreso (val, redondeo)",
        "",
        head,
        *[
            row(f"{label} · {part}", m)
            for part in ("all", *x["covered"]["by_origin"], *x["covered"]["by_class"])
            for label, key in (("tapado", "covered"), ("sin tapar", "uncovered"))
            for m in [
                x[key]["all"]
                if part == "all"
                else x[key]["by_origin"].get(part) or x[key]["by_class"][part]
            ]
        ],
        "",
        f"Δ valor (sin tapar - tapado): media {x['delta_mean']:+.4f}, |Δ| media "
        f"{x['delta_abs_mean']:.4f}, p95 {x['delta_abs_p95']:.4f}.",
        "",
    ]
    p = t["paint_gs_val"]["nearest"]

    def f(v: Any) -> str:
        return "—" if v is None else f"{v:+.4f}"

    lines += [
        f"## Pintura roja en GS (val, n={t['paint_gs_val']['n_gs']}, "
        f"pintadas={t['paint_gs_val']['n_painted']}, redondeo)",
        "",
        "| clase | pint. n | pint. MAE | pint. sesgo | pint. ±1 | no pint. n | no pint. MAE "
        "| no pint. sesgo | no pint. ±1 |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for c, b in [
        ("todas", {"painted": p["painted"], "unpainted": p["unpainted"]}),
        *p["by_class"].items(),
    ]:
        cells = []
        for k in ("painted", "unpainted"):
            m = b[k]
            cells += (
                [str(m["n"]), f"{m['mae']:.4f}", f(m["bias"]), f"{m['acc_within_1']:.3f}"]
                if m["n"]
                else ["0", "—", "—", "—"]
            )
        lines.append(f"| {c} | " + " | ".join(cells) + " |")
    lines += [
        "",
        f"Δ (desaturada - original) en pintadas: media {f(p['delta_painted_mean'])}, |Δ| media "
        f"{f(p['delta_painted_abs_mean'])}, p95 {f(p['delta_painted_abs_p95'])}; cambia la "
        f"clase en {p['pred_changed_painted']:.3f} de las pintadas.",
        "",
    ]
    lat = t["latency_cpu"]

    def ms(d: dict[str, float]) -> str:
        return f"{d['median_s'] * 1000:.0f} ms (p95 {d['p95_s'] * 1000:.0f})"

    lines += [
        "## Latencia (CPU, de a una imagen)",
        "",
        f"Sin TTA {ms(lat['no_tta'])}, con TTA {ms(lat['tta'])}; "
        f"{lat['threads']} hilos. {lat['params'] / 1e6:.2f} M parámetros, {lat['weights_mb']} MB.",
    ]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m bcs.models.efficientnet", description="T09")
    ap.add_argument("--task", default="T09")
    ap.add_argument(
        "--config", type=Path, default=None, help="default: config/efficientnet_b0.yaml"
    )
    args = ap.parse_args(argv)
    run_experiment(args.task, args.config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
