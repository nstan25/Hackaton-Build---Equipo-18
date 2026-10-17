"""Baselines de T09 (SPEC §2.3 pasos 1–2): tonto (moda/media) y cabezas lineales sobre
embeddings congelados de DINOv2 (`bcs.models.embeddings`): regresión ridge y regresión logística
ordinal (odds proporcionales).

Reglas (G0/G1):
- Un único split train/val. Los hiperparámetros (`alpha`, `lam`) se eligen con Group K-Fold
  (3 folds por `group_id`) **dentro de train**, nunca mirando val.
- Peso por grupo ∝ 1/√(imágenes del grupo) en train (los modelos lineales no iteran por épocas).
- Mitigación de la pintura roja: con `red_paint_desaturate = p`, cada imagen de train entra dos
  veces, original (peso 1 - p) y desaturada (peso p): es la esperanza de la augmentation. Val se
  evalúa siempre original.

Salida de probabilidades sobre la grilla:
- tonto: la frecuencia de cada clase en train (argmax = moda).
- ridge: normal discretizada centrada en la predicción, con σ = desvío de los residuos fuera de
  fold de la búsqueda de hiperparámetros.
- ordinal: P(y ≤ k) = σ(θ_k - x·w), directo.

`python -m bcs.models.baseline` corre el experimento completo (cachea embeddings, entrena todo,
escribe predicciones en `$BCS_ARTIFACTS_DIR/T09/` y el reporte en `runs/T09/<ts>/`).
`EmbeddingScorer` es el `Scorer` (SPEC §2.0) para `bcs.train` (`config/train_m2.yaml`).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import pandas as pd
from PIL import Image
from scipy.special import expit, ndtr

from bcs.models.embeddings import Embedder, EmbeddingCache, task_dir

SEED = 42
INNER_FOLDS = 3
RIDGE_ALPHAS = (1.0, 10.0, 100.0, 1e3, 1e4, 1e5)
ORDINAL_LAMS = (1e-4, 1e-3, 1e-2, 1e-1)
PAINT_THRESHOLD = 0.005  # G1: "pintada" = > 0,5% de píxeles rojo-naranja (T06)


# ---------------------------------------------------------------- utilidades


def grid_index(y: np.ndarray, grid: Sequence[float] | np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=float)
    return np.abs(np.asarray(y, dtype=float)[:, None] - g[None, :]).argmin(axis=1)


def group_weights(groups: Sequence[str]) -> np.ndarray:
    """Peso ∝ 1/√(tamaño del grupo), normalizado a media 1 (G1 punto 3)."""
    s = pd.Series(list(groups))
    w = 1.0 / np.sqrt(s.map(s.value_counts()).to_numpy(dtype=float))
    return w / w.mean()


def inner_folds(groups: Sequence[str], k: int = INNER_FOLDS, seed: int = SEED) -> np.ndarray:
    """Fold (0..k-1) de cada fila: los grupos se barajan con semilla fija y se reparten."""
    uniq = np.array(sorted(set(groups)))
    rng = np.random.default_rng(seed)
    rng.shuffle(uniq)
    fold_of = {g: i % k for i, g in enumerate(uniq)}
    return np.array([fold_of[g] for g in groups])


class Head(Protocol):
    def fit(self, x: np.ndarray, y: np.ndarray, w: np.ndarray) -> Head: ...

    def predict_proba(self, x: np.ndarray) -> np.ndarray: ...

    def state(self) -> dict[str, np.ndarray]: ...


class _Standardize:
    mean: np.ndarray
    scale: np.ndarray

    def _fit_scaler(self, x: np.ndarray, w: np.ndarray) -> np.ndarray:
        self.mean = np.average(x, axis=0, weights=w)
        self.scale = np.sqrt(np.average((x - self.mean) ** 2, axis=0, weights=w)) + 1e-6
        return self._z(x)

    def _z(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / self.scale


# ---------------------------------------------------------------- cabezas


class PriorHead:
    """Baseline tonto: la distribución de clases de train para toda imagen (argmax = moda)."""

    def __init__(self, grid: Sequence[float]) -> None:
        self.grid = np.asarray(grid, dtype=float)
        self.prior = np.full(len(self.grid), 1.0 / len(self.grid))

    def fit(self, x: np.ndarray, y: np.ndarray, w: np.ndarray) -> PriorHead:
        counts = np.bincount(grid_index(y, self.grid), minlength=len(self.grid))
        self.prior = counts / counts.sum()
        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        return np.tile(self.prior, (len(x), 1))

    def state(self) -> dict[str, np.ndarray]:
        return {"prior": self.prior}


class RidgeHead(_Standardize):
    """Ridge ponderada sobre embeddings estandarizados; BCS continuo → normal discretizada."""

    def __init__(self, grid: Sequence[float], alpha: float = 100.0, sigma: float = 0.25) -> None:
        self.grid = np.asarray(grid, dtype=float)
        self.alpha = alpha
        self.sigma = sigma

    def fit(self, x: np.ndarray, y: np.ndarray, w: np.ndarray) -> RidgeHead:
        z = self._fit_scaler(x, w)
        self.intercept = float(np.average(y, weights=w))
        zw = z * w[:, None]
        a = z.T @ zw + self.alpha * len(z) / w.sum() * np.eye(z.shape[1])
        self.coef = np.linalg.solve(a, zw.T @ (y - self.intercept))
        return self

    def predict_value(self, x: np.ndarray) -> np.ndarray:
        return self._z(x) @ self.coef + self.intercept

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        mu = self.predict_value(x)[:, None]
        half = (self.grid[1] - self.grid[0]) / 2 if len(self.grid) > 1 else 0.5
        lo = np.r_[-np.inf, self.grid[1:] - half][None, :]
        hi = np.r_[self.grid[:-1] + half, np.inf][None, :]
        p = ndtr((hi - mu) / self.sigma) - ndtr((lo - mu) / self.sigma)
        p = np.clip(p, 1e-9, None)
        return p / p.sum(axis=1, keepdims=True)

    def state(self) -> dict[str, np.ndarray]:
        return {
            "mean": self.mean,
            "scale": self.scale,
            "coef": self.coef,
            "intercept": np.array([self.intercept]),
            "sigma": np.array([self.sigma]),
            "alpha": np.array([self.alpha]),
        }


class OrdinalHead(_Standardize):
    """Logística ordinal de odds proporcionales: P(y ≤ k | x) = σ(θ_k - x·w), θ crecientes.

    Pérdida: NLL ponderada (promedio) + lam/2·‖w‖². L-BFGS (torch, float64)."""

    def __init__(self, grid: Sequence[float], lam: float = 1e-3, max_iter: int = 300) -> None:
        self.grid = np.asarray(grid, dtype=float)
        self.lam = lam
        self.max_iter = max_iter

    def fit(self, x: np.ndarray, y: np.ndarray, w: np.ndarray) -> OrdinalHead:
        import torch

        torch.manual_seed(SEED)
        k = len(self.grid)
        z = torch.from_numpy(self._fit_scaler(x, w)).double()
        yi = torch.from_numpy(grid_index(y, self.grid)).long()
        wt = torch.from_numpy(w / w.sum()).double()
        # arranque: umbrales en los logits de la frecuencia acumulada
        cum = np.cumsum(np.bincount(yi.numpy(), minlength=k))[:-1] / len(yi)
        cum = np.clip(cum, 1e-3, 1 - 1e-3)
        th0 = np.log(cum / (1 - cum))
        b0 = torch.tensor([th0[0]], dtype=torch.float64, requires_grad=True)
        d0 = np.log(np.maximum(np.diff(th0), 1e-3))
        logd = torch.tensor(d0, dtype=torch.float64, requires_grad=True)
        coef = torch.zeros(z.shape[1], dtype=torch.float64, requires_grad=True)
        opt = torch.optim.LBFGS(
            [b0, logd, coef], max_iter=self.max_iter, line_search_fn="strong_wolfe"
        )

        def closure() -> torch.Tensor:
            opt.zero_grad()
            theta = torch.cat([b0, b0 + torch.cumsum(torch.exp(logd), 0)])
            p = self._probs_t(theta, z @ coef)
            nll = -(wt * torch.log(p[torch.arange(len(yi)), yi].clamp_min(1e-12))).sum()
            loss = nll + 0.5 * self.lam * (coef * coef).sum()
            loss.backward()
            return loss

        opt.step(closure)  # type: ignore[arg-type]
        with torch.no_grad():
            theta = torch.cat([b0, b0 + torch.cumsum(torch.exp(logd), 0)])
        self.theta = theta.numpy().copy()
        self.coef = coef.detach().numpy().copy()
        return self

    @staticmethod
    def _probs_t(theta, eta):
        import torch

        cdf = torch.sigmoid(theta[None, :] - eta[:, None])
        ones = torch.ones(len(eta), 1, dtype=cdf.dtype)
        zeros = torch.zeros(len(eta), 1, dtype=cdf.dtype)
        return torch.cat([cdf, ones], 1) - torch.cat([zeros, cdf], 1)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        eta = self._z(x) @ self.coef
        cdf = expit(self.theta[None, :] - eta[:, None])
        n = len(eta)
        p = np.concatenate([cdf, np.ones((n, 1))], 1) - np.concatenate([np.zeros((n, 1)), cdf], 1)
        p = np.clip(p, 1e-9, None)
        return p / p.sum(axis=1, keepdims=True)

    def state(self) -> dict[str, np.ndarray]:
        return {
            "mean": self.mean,
            "scale": self.scale,
            "coef": self.coef,
            "theta": self.theta,
            "lam": np.array([self.lam]),
        }


HEADS = ("prior", "ridge", "ordinal")


def make_head(name: str, grid: Sequence[float], **params: float) -> Head:
    if name == "prior":
        return PriorHead(grid)
    if name == "ridge":
        return RidgeHead(grid, alpha=params.get("alpha", 100.0), sigma=params.get("sigma", 0.25))
    if name == "ordinal":
        return OrdinalHead(grid, lam=params.get("lam", 1e-3))
    raise ValueError(f"cabeza desconocida {name!r}: {HEADS}")


def head_from_state(name: str, grid: Sequence[float], st: dict[str, np.ndarray]) -> Head:
    if name == "prior":
        h = PriorHead(grid)
        h.prior = st["prior"]
        return h
    if name == "ridge":
        r = RidgeHead(grid, alpha=float(st["alpha"][0]), sigma=float(st["sigma"][0]))
        r.mean, r.scale, r.coef = st["mean"], st["scale"], st["coef"]
        r.intercept = float(st["intercept"][0])
        return r
    if name == "ordinal":
        o = OrdinalHead(grid, lam=float(st["lam"][0]))
        o.mean, o.scale, o.coef, o.theta = st["mean"], st["scale"], st["coef"], st["theta"]
        return o
    raise ValueError(f"cabeza desconocida {name!r}: {HEADS}")


# ---------------------------------------------------------------- entrenamiento


def training_set(
    x: np.ndarray, x_desat: np.ndarray | None, y: np.ndarray, w: np.ndarray, p_desat: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Original con peso (1 - p) + desaturada con peso p (esperanza de la augmentation)."""
    if x_desat is None or p_desat <= 0:
        return x, y, w
    return (
        np.concatenate([x, x_desat]),
        np.concatenate([y, y]),
        np.concatenate([w * (1 - p_desat), w * p_desat]),
    )


def _mae_argmax(p: np.ndarray, y: np.ndarray, grid: Sequence[float]) -> float:
    g = np.asarray(grid, dtype=float)
    return float(np.abs(g[p.argmax(axis=1)] - y).mean())


def select_and_fit(
    name: str,
    grid: Sequence[float],
    x: np.ndarray,
    x_desat: np.ndarray | None,
    y: np.ndarray,
    groups: Sequence[str],
    p_desat: float,
    params: dict[str, float] | None = None,
) -> tuple[Head, dict[str, Any]]:
    """Elige hiperparámetros con Group K-Fold dentro de train (MAE del argmax fuera de fold,
    sobre imágenes originales) y reentrena con todo train. `params` fijos salta la búsqueda."""
    w = group_weights(groups)
    info: dict[str, Any] = {"head": name, "p_desat": p_desat, "search": []}
    t0 = time.monotonic()
    if name == "prior":
        head = make_head(name, grid).fit(x, y, w)
        info["fit_seconds"] = round(time.monotonic() - t0, 2)
        return head, info

    key = "alpha" if name == "ridge" else "lam"
    candidates = (RIDGE_ALPHAS if name == "ridge" else ORDINAL_LAMS) if not params else ()
    best: dict[str, float] = dict(params or {})
    if candidates:
        folds = inner_folds(groups)
        scores = []
        for c in candidates:
            maes, resid = [], []
            for f in range(INNER_FOLDS):
                tr, te = folds != f, folds == f
                xs, ys, ws = training_set(
                    x[tr], None if x_desat is None else x_desat[tr], y[tr], w[tr], p_desat
                )
                h = make_head(name, grid, **{key: c}).fit(xs, ys, ws)
                maes.append(_mae_argmax(h.predict_proba(x[te]), y[te], grid))
                if isinstance(h, RidgeHead):
                    resid.append(h.predict_value(x[te]) - y[te])
            row: dict[str, Any] = {key: c, "mae_inner": float(np.mean(maes))}
            if resid:
                row["sigma"] = float(np.std(np.concatenate(resid)))
            info["search"].append(row)
            scores.append(row)
        top = min(scores, key=lambda r: r["mae_inner"])
        best = {key: float(top[key])}
        if "sigma" in top:
            best["sigma"] = float(top["sigma"])
    if name == "ridge" and "sigma" not in best:
        best["sigma"] = 0.25
    xs, ys, ws = training_set(x, x_desat, y, w, p_desat)
    head = make_head(name, grid, **best).fit(xs, ys, ws)
    info["params"] = best
    info["fit_seconds"] = round(time.monotonic() - t0, 2)
    return head, info


# ---------------------------------------------------------------- Scorer (SPEC §2.0)


class EmbeddingScorer:
    """Cabeza lineal sobre embeddings congelados. Config (`config/train_m2.yaml`):
    `model.head` (prior | ridge | ordinal), `model.backbone`, `model.img_size`, opcional
    `model.params` (fija hiperparámetros); `augment.red_paint_desaturate`."""

    def __init__(self, grid: tuple[float, ...], config: dict[str, Any] | None = None) -> None:
        cfg = config or {}
        m = cfg.get("model") or {}
        self.grid = grid
        self.head_name = str(m.get("head", "ordinal"))
        self.params: dict[str, float] = {k: float(v) for k, v in (m.get("params") or {}).items()}
        self.p_desat = float((cfg.get("augment") or {}).get("red_paint_desaturate", 0.0))
        self.embedder = Embedder(
            str(m.get("backbone") or "vit_small_patch14_dinov2.lvd142m"),
            int(m.get("img_size") or 224),
        )
        self.head: Head | None = None
        self.info: dict[str, Any] = {}

    @property
    def model_version(self) -> str:
        return f"t09-{self.head_name}-{self.embedder.name}"

    def cache(self) -> EmbeddingCache:
        return EmbeddingCache(task_dir() / "emb" / self.embedder.name)

    def fit(self, df: pd.DataFrame) -> None:
        cache = self.cache()
        ids, paths = list(df["image_id"]), list(df["path"])
        x = cache.get(ids, paths, "orig", self.embedder.embed)
        xd = cache.get(ids, paths, "desat", self.embedder.embed) if self.p_desat > 0 else None
        y = np.asarray(df["bcs"], dtype=float)
        self.head, self.info = select_and_fit(
            self.head_name, self.grid, x, xd, y, list(df["group_id"]), self.p_desat, self.params
        )

    def predict_proba(self, images: Sequence[Image.Image]) -> np.ndarray:
        if self.head is None:
            raise RuntimeError("EmbeddingScorer sin entrenar: llamá fit() o load()")
        return self.head.predict_proba(self.embedder.embed(images))

    def save(self, path: Path) -> None:
        if self.head is None:
            raise RuntimeError("EmbeddingScorer sin entrenar")
        meta = {
            "model_version": self.model_version,
            "head": self.head_name,
            "grid": list(self.grid),
            "backbone": self.embedder.backbone,
            "img_size": self.embedder.img_size,
            "p_desat": self.p_desat,
            "info": self.info,
        }
        with Path(path).open("wb") as f:
            arrays: dict[str, Any] = {"meta": np.array(json.dumps(meta)), **self.head.state()}
            np.savez(f, **arrays)

    def load(self, path: Path) -> None:
        with np.load(Path(path)) as z:
            meta = json.loads(str(z["meta"]))
            st = {k: z[k] for k in z.files if k != "meta"}
        self.grid = tuple(float(v) for v in meta["grid"])
        self.head_name = meta["head"]
        self.p_desat = float(meta["p_desat"])
        self.info = meta.get("info", {})
        self.embedder = Embedder(meta["backbone"], int(meta["img_size"]))
        self.head = head_from_state(self.head_name, self.grid, st)


# ---------------------------------------------------------------- medición del atajo


def paint_report(
    y: np.ndarray,
    p: np.ndarray,
    p_desat: np.ndarray,
    painted: np.ndarray,
    grid: Sequence[float],
) -> dict[str, Any]:
    """G1 punto 2, sobre las imágenes de GS en val: MAE, sesgo y accuracy ±1 paso por clase
    para pintadas vs. no pintadas, y Δ esperanza = E[BCS | desaturada] - E[BCS | original]."""
    g = np.asarray(grid, dtype=float)
    step = float(g[1] - g[0]) if len(g) > 1 else 1.0
    pred = g[p.argmax(axis=1)]
    err = pred - y
    exp, exp_d = p @ g, p_desat @ g

    def block(m: np.ndarray) -> dict[str, Any]:
        if not m.any():
            return {"n": 0}
        e = err[m]
        return {
            "n": int(m.sum()),
            "mae": float(np.abs(e).mean()),
            "bias": float(e.mean()),
            "acc_within_1": float((np.abs(e) <= step + 1e-6).mean()),
        }

    by_class = {}
    for c in g:
        mc = np.isclose(y, c)
        by_class[f"{c:.2f}"] = {"painted": block(mc & painted), "unpainted": block(mc & ~painted)}
    delta = exp_d - exp
    return {
        "n": len(y),
        "painted": block(painted),
        "unpainted": block(~painted),
        "by_class": by_class,
        "delta_expected_painted": float(delta[painted].mean()) if painted.any() else None,
        "delta_expected_painted_abs": float(np.abs(delta[painted]).mean())
        if painted.any()
        else None,
        "delta_expected_all": float(delta.mean()),
        "pred_changed_painted": float((p_desat[painted].argmax(1) != p[painted].argmax(1)).mean())
        if painted.any()
        else None,
    }


# ---------------------------------------------------------------- experimento


def _farm(image_ids: pd.Series) -> np.ndarray:
    return image_ids.str.removeprefix("sdb_bcs_").str.split("_").str[0].to_numpy(dtype=str)


def run_experiment(task: str = "T09", config: Path | None = None) -> dict[str, Any]:
    from bcs.eval.report import build_report, synthetic_predictions, write_run
    from bcs.paths import data_dir, runs_dir
    from bcs.scale import load_scale
    from bcs.train import check_no_leakage, load_config, load_dataset, resolve_config_path

    t0 = time.monotonic()
    cfg = load_config(config or resolve_config_path("train_m2"))
    scale = load_scale(cfg.get("scale"))
    grid = scale.grid
    p_cfg = float((cfg.get("augment") or {}).get("red_paint_desaturate", 0.5))
    m = cfg.get("model") or {}
    embedder = Embedder(str(m.get("backbone") or "vit_small_patch14_dinov2.lvd142m"),
                        int(m.get("img_size") or 224))  # fmt: skip
    cache = EmbeddingCache(task_dir() / "emb" / embedder.name)

    df, _ = load_dataset(data_dir(), allow_synthetic=False)
    df["bcs"] = [scale.snap(float(b)) for b in df["bcs"]]
    tr = pd.DataFrame(df.loc[df["split"] == "train"]).reset_index(drop=True)
    va = pd.DataFrame(df.loc[df["split"] == "val"]).reset_index(drop=True)
    check_no_leakage(tr, va)

    t_emb = time.monotonic()
    emb = {
        (s, v): cache.get(list(d["image_id"]), list(d["path"]), v, embedder.embed, log=True)
        for s, d in (("train", tr), ("val", va))
        for v in ("orig", "desat")
    }
    emb_seconds = time.monotonic() - t_emb
    y_tr, y_va = np.asarray(tr["bcs"], dtype=float), np.asarray(va["bcs"], dtype=float)
    groups = list(tr["group_id"])

    paint_csv = runs_dir() / "T06" / "paint_frac.csv"
    paint = pd.read_csv(paint_csv, dtype={"image_id": str})
    red_of = dict(zip(paint["image_id"], paint["red_frac"].astype(float), strict=True))
    val_ids = pd.Series(va["image_id"], dtype=str)
    red_va = np.array([red_of.get(i, np.nan) for i in val_ids], dtype=float)
    gs = _farm(val_ids) == "GS"
    painted = red_va > PAINT_THRESHOLD

    specs = [("dummy", "prior", 0.0)] + [
        (f"{h}{'' if p > 0 else '_nomit'}", h, p)
        for h in ("ridge", "ordinal")
        for p in (p_cfg, 0.0)
    ]
    out_dir = task_dir()
    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_dir = runs_dir() / task / ts
    results: dict[str, Any] = {}
    preds_by_model: dict[str, pd.DataFrame] = {}
    t_heads = time.monotonic()
    for name, head_name, p in specs:
        print(f"[T09] {name}: cabeza {head_name}, p_desat={p}", file=sys.stderr)
        head, info = select_and_fit(
            head_name,
            grid,
            emb[("train", "orig")],
            emb[("train", "desat")],
            y_tr,
            groups,
            p,
        )
        probs = head.predict_proba(emb[("val", "orig")])
        probs_d = head.predict_proba(emb[("val", "desat")])
        preds = synthetic_predictions(
            y_va,
            probs,
            scale,
            image_id=list(va["image_id"]),
            group_id=list(va["group_id"]),
            source=list(va["source"]),
            fold=["val"] * len(va),
        )
        preds.to_csv(out_dir / f"predictions_val_{name}.csv", index=False)
        preds_by_model[name] = preds
        rep = build_report(preds, scale, model=name, task=task, y_train=y_tr)
        results[name] = {
            "info": info,
            "mean": {
                k: rep["mean"][k]
                for k in ("mae", "mae_expected", "bias", "accuracy", "acc_within_1", "nll", "ece")
            },
            "mae_ci95": rep["pooled"]["mae_ci95"],
            "mae_group_mean": rep["pooled"]["mae_group_mean"],
            "mae_improvement_vs_dummy": rep["mae_improvement_vs_dummy"],
            "dummy_mae": rep["dummy"]["mae"],
            "paint_gs_val": paint_report(y_va[gs], probs[gs], probs_d[gs], painted[gs], grid),
        }
        print(
            f"[T09] {name}: mae={rep['mean']['mae']:.4f} "
            f"(mejora vs. tonto {rep['mae_improvement_vs_dummy']:.3f}) "
            f"fit {info['fit_seconds']} s",
            file=sys.stderr,
        )
    heads_seconds = time.monotonic() - t_heads

    # elegido: el mejor MAE de val entre los modelos con la mitigación de G1
    mitigated = [n for n, _, p in specs if p > 0]
    best = min(mitigated, key=lambda n: results[n]["mean"]["mae"])
    preds_by_model[best].to_csv(out_dir / "predictions_val.csv", index=False)
    report = build_report(preds_by_model[best], scale, model=f"baseline:{best}", task=task,
                          stage="m2", y_train=y_tr)  # fmt: skip
    report["baseline"] = {
        "best": best,
        "selection": "menor MAE en val entre los modelos con mitigación de pintura (G1)",
        "models": results,
        "table": [
            {
                "model": n,
                "mae": r["mean"]["mae"],
                "acc_within_1": r["mean"]["acc_within_1"],
                "accuracy": r["mean"]["accuracy"],
                "mae_improvement_vs_dummy": r["mae_improvement_vs_dummy"],
                "paint_gs_bias_painted": r["paint_gs_val"]["painted"].get("bias"),
                "paint_gs_bias_unpainted": r["paint_gs_val"]["unpainted"].get("bias"),
                "delta_expected_painted": r["paint_gs_val"]["delta_expected_painted"],
            }
            for n, r in results.items()
        ],
        "embeddings": {
            "backbone": embedder.backbone,
            "img_size": embedder.img_size,
            "cache": str(cache.root),
            "seconds_this_run": round(emb_seconds, 1),
        },
        "heads_seconds": round(heads_seconds, 1),
        "paint_csv": str(paint_csv),
        "n_train": len(tr),
        "n_val": len(va),
        "n_val_gs": int(gs.sum()),
        "n_val_gs_painted": int((gs & painted).sum()),
        "seconds": round(time.monotonic() - t0, 1),
    }
    paths = write_run(report, run_dir, cfg)
    (run_dir / "summary.md").write_text(summary_markdown(report))
    print(f"[T09] mejor {best} → {paths['report']}", file=sys.stderr)
    report["report_path"] = str(paths["report"])
    return report


def summary_markdown(report: dict[str, Any]) -> str:
    b = report["baseline"]
    lines = [
        f"# T09 · baseline sobre val (n={b['n_val']}, train n={b['n_train']})",
        "",
        f"Elegido: **{b['best']}** ({b['selection']}).",
        "",
        "| modelo | MAE | acc | ±1 paso | mejora vs. tonto | sesgo GS pint. | sesgo GS no pint. "
        "| Δ esperanza pint. |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in b["table"]:

        def f(v: Any) -> str:
            return "—" if v is None else f"{v:+.3f}" if isinstance(v, float) else str(v)

        lines.append(
            f"| {r['model']} | {r['mae']:.4f} | {r['accuracy']:.3f} | {r['acc_within_1']:.3f} "
            f"| {r['mae_improvement_vs_dummy']:.3f} | {f(r['paint_gs_bias_painted'])} "
            f"| {f(r['paint_gs_bias_unpainted'])} | {f(r['delta_expected_painted'])} |"
        )
    lines += ["", f"Cabezas: {b['heads_seconds']} s en total (todas, con búsqueda)."]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m bcs.models.baseline", description="T09")
    ap.add_argument("--task", default="T09")
    ap.add_argument("--config", type=Path, default=None, help="default: config/train_m2.yaml")
    args = ap.parse_args(argv)
    run_experiment(args.task, args.config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
