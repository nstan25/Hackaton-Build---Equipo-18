"""Interfaces de calibración e intervalo (SPEC §2.0, §2.3)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class Calibrator(Protocol):
    def fit(self, probs: np.ndarray, y: np.ndarray) -> None:
        """`probs` [n, K]; `y` [n] índices de clase en la grilla."""
        ...

    def transform(self, probs: np.ndarray) -> np.ndarray:
        """Probabilidades calibradas [n, K]."""
        ...


@runtime_checkable
class IntervalPredictor(Protocol):
    grid: tuple[float, ...]

    def fit(self, probs: np.ndarray, y: np.ndarray, alpha: float = 0.1) -> None:
        """Ajusta el conjunto conformal al nivel 1 - alpha (90% por defecto)."""
        ...

    def interval(self, probs: np.ndarray) -> list[tuple[float, float]]:
        """Un (min, max) sobre la grilla por fila de `probs`."""
        ...
