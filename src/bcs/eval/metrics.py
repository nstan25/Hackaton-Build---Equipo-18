"""Métricas de evaluación (SPEC §2.0, §5.6.1 T02).

Todo se mide sobre la grilla que se recibe (sale de `config/scale.yaml`); nada asume una escala.
Las definiciones siguen al contrato (§2.2):
- `bcs` = valor de la grilla con mayor probabilidad (argmax).
- `confidence` = P(BCS real a un paso o menos de `bcs`) = suma de la probabilidad en
  argmax-1, argmax y argmax+1. "Acierto" de la confianza = |y - bcs| ≤ 1 paso.
- ECE de `confidence`: 10 bins de igual ancho en [0, 1], promedio de |acierto - confianza|
  pesado por la cantidad de casos de cada bin.
- Cobertura: fracción de casos con `y` dentro de `[min, max]` del intervalo conformal.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

N_BINS = 10
BOOTSTRAP_RESAMPLES = 1000
BOOTSTRAP_SEED = 0
_TOL = 1e-6


def evaluate(
    probs: np.ndarray,
    y: np.ndarray,
    grid: Sequence[float],
    groups: Sequence[str],
    sources: Sequence[str],
) -> dict[str, Any]:
    """`probs` [n, K] sobre `grid`; `y` [n] BCS reales (valores de la grilla).

    Sin intervalos, `coverage_90` e `interval_width_steps` quedan en `None`
    (ver `evaluate_with_intervals`).
    """
    return evaluate_with_intervals(probs, y, grid, groups, sources, intervals=None)


def evaluate_with_intervals(
    probs: np.ndarray,
    y: np.ndarray,
    grid: Sequence[float],
    groups: Sequence[str],
    sources: Sequence[str],
    intervals: Sequence[tuple[float, float]] | np.ndarray | None,
) -> dict[str, Any]:
    """Como `evaluate`, más la cobertura del intervalo; agrega las métricas por fuente."""
    g = np.asarray(grid, dtype=float)
    p = _check_probs(probs, len(g))
    y_idx = snap_to_grid(y, g)
    grp = np.asarray(groups, dtype=str)
    src = np.asarray(sources, dtype=str)
    iv = None if intervals is None else np.asarray(intervals, dtype=float).reshape(-1, 2)
    for name, arr in (("groups", grp), ("sources", src), ("intervals", iv)):
        if arr is not None and len(arr) != len(p):
            raise ValueError(f"{name} tiene {len(arr)} filas y probs {len(p)}")

    out = _core(p, y_idx, g, grp, src, iv)
    out["by_source"] = {
        s: _core(p[src == s], y_idx[src == s], g, grp[src == s], src[src == s], _rows(iv, src == s))
        for s in sorted(set(src.tolist()))
    }
    return out


def snap_to_grid(y: np.ndarray | Sequence[float], grid: np.ndarray) -> np.ndarray:
    """Índice de cada `y` en la grilla. Fuera del rango se agrupa en la punta (collapse_*);
    dentro del rango, un valor que no está en la grilla es un error."""
    yv = np.asarray(y, dtype=float)
    idx = np.abs(yv[:, None] - grid[None, :]).argmin(axis=1)
    inside = (yv >= grid[0] - _TOL) & (yv <= grid[-1] + _TOL)
    off = inside & (np.abs(grid[idx] - yv) > _TOL)
    if off.any():
        raise ValueError(f"etiquetas fuera de la grilla: {sorted(set(yv[off].tolist()))[:5]}")
    return idx


def confidence_from_probs(probs: np.ndarray) -> np.ndarray:
    """`confidence` de §2.2 por fila: masa en argmax ± 1 paso (igual que `run_pipeline`)."""
    top = probs.argmax(axis=1)
    k = probs.shape[1]
    conf = probs[np.arange(len(probs)), top].copy()
    left = top - 1 >= 0
    conf[left] += probs[np.flatnonzero(left), top[left] - 1]
    right = top + 1 < k
    conf[right] += probs[np.flatnonzero(right), top[right] + 1]
    return conf


def reliability_bins(conf: np.ndarray, correct: np.ndarray, n_bins: int = N_BINS) -> list[dict]:
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    which = np.clip(np.digitize(conf, edges[1:-1], right=True), 0, n_bins - 1)
    bins = []
    for b in range(n_bins):
        m = which == b
        n = int(m.sum())
        bins.append(
            {
                "lo": float(edges[b]),
                "hi": float(edges[b + 1]),
                "count": n,
                "mean_confidence": float(conf[m].mean()) if n else None,
                "accuracy": float(correct[m].mean()) if n else None,
            }
        )
    return bins


def expected_calibration_error(bins: list[dict]) -> float:
    total = sum(b["count"] for b in bins)
    if total == 0:
        return float("nan")
    return float(
        sum(b["count"] * abs(b["accuracy"] - b["mean_confidence"]) for b in bins if b["count"])
        / total
    )


def confusion_matrix(y_idx: np.ndarray, pred_idx: np.ndarray, k: int) -> list[list[int]]:
    """Filas = BCS real, columnas = BCS predicho, en el orden de la grilla."""
    m = np.zeros((k, k), dtype=int)
    np.add.at(m, (y_idx, pred_idx), 1)
    return m.tolist()


def _check_probs(probs: np.ndarray, k: int) -> np.ndarray:
    p = np.asarray(probs, dtype=float)
    if p.ndim != 2 or p.shape[1] != k:
        raise ValueError(f"probs debe ser [n, {k}] y es {p.shape}")
    if len(p) == 0:
        raise ValueError("no hay predicciones")
    if not np.all(np.isfinite(p)) or (p < -_TOL).any():
        raise ValueError("probs con valores negativos o no finitos")
    if not np.allclose(p.sum(axis=1), 1.0, atol=1e-3):
        raise ValueError("cada fila de probs tiene que sumar 1")
    return p


def _rows(iv: np.ndarray | None, mask: np.ndarray) -> np.ndarray | None:
    return None if iv is None else iv[mask]


def _core(
    p: np.ndarray,
    y_idx: np.ndarray,
    g: np.ndarray,
    grp: np.ndarray,
    src: np.ndarray,
    iv: np.ndarray | None,
) -> dict[str, Any]:
    k = len(g)
    step = float(g[1] - g[0]) if k > 1 else 1.0
    yv = g[y_idx]
    pred_idx = p.argmax(axis=1)
    pred = g[pred_idx]
    err = np.abs(pred - yv)
    err_steps = np.abs(pred_idx - y_idx)
    expected = p @ g
    conf = confidence_from_probs(p)
    correct = (err_steps <= 1).astype(float)
    bins = reliability_bins(conf, correct)
    true_p = p[np.arange(len(p)), y_idx]

    covered = width = None
    if iv is not None:
        lo, hi = iv[:, 0], iv[:, 1]
        covered = float(np.mean((yv >= lo - _TOL) & (yv <= hi + _TOL)))
        width = float(np.mean((hi - lo) / step))

    return {
        "n": len(p),
        "n_groups": len(set(grp.tolist())),
        "n_sources": len(set(src.tolist())),
        "mae": float(err.mean()),
        "mae_ci95": _group_bootstrap_ci(err, grp),
        "mae_group_mean": float(np.mean([err[grp == u].mean() for u in np.unique(grp)])),
        "mae_expected": float(np.abs(expected - yv).mean()),
        "bias": float((pred - yv).mean()),
        "accuracy": float((err_steps == 0).mean()),
        "acc_within_1": float((err_steps <= 1).mean()),
        "acc_within_2": float((err_steps <= 2).mean()),
        "nll": float(-np.log(np.clip(true_p, 1e-12, None)).mean()),
        "mean_confidence": float(conf.mean()),
        "ece": expected_calibration_error(bins),
        "reliability": bins,
        "coverage_90": covered,
        "interval_width_steps": width,
        "confusion": {
            "labels": [f"{v:.2f}" for v in g],
            "matrix": confusion_matrix(y_idx, pred_idx, k),
        },
    }


def _group_bootstrap_ci(err: np.ndarray, grp: np.ndarray) -> list[float]:
    """IC 95% del MAE remuestreando vacas (`group_id`), no imágenes: los frames de una vaca
    están correlacionados. Semilla fija."""
    uniq, inv = np.unique(grp, return_inverse=True)
    sums = np.bincount(inv, weights=err, minlength=len(uniq))
    counts = np.bincount(inv, minlength=len(uniq)).astype(float)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = rng.integers(0, len(uniq), size=(BOOTSTRAP_RESAMPLES, len(uniq)))
    maes = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
    return [float(np.quantile(maes, 0.025)), float(np.quantile(maes, 0.975))]
