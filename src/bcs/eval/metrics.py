"""Métricas de evaluación (SPEC §2.0). Stub de T00: MAE y accuracy; T02 lo completa."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np


def evaluate(
    probs: np.ndarray,
    y: np.ndarray,
    grid: Sequence[float],
    groups: Sequence[str],
    sources: Sequence[str],
) -> dict[str, Any]:
    """`probs` [n, K] sobre `grid`; `y` [n] BCS reales (valores de la grilla)."""
    g = np.asarray(grid, dtype=float)
    pred = g[np.argmax(probs, axis=1)]
    y = np.asarray(y, dtype=float)
    return {
        "n": int(len(y)),
        "n_groups": len(set(groups)),
        "n_sources": len(set(sources)),
        "mae": float(np.mean(np.abs(pred - y))),
        "accuracy": float(np.mean(np.isclose(pred, y))),
    }
