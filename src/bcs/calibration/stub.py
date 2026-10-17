"""Stubs de T00: calibrador identidad e intervalo = grilla completa."""

from __future__ import annotations

import numpy as np


class IdentityCalibrator:
    def fit(self, probs: np.ndarray, y: np.ndarray) -> None:
        return None

    def transform(self, probs: np.ndarray) -> np.ndarray:
        return probs


class FullGridInterval:
    def __init__(self, grid: tuple[float, ...]) -> None:
        self.grid = grid

    def fit(self, probs: np.ndarray, y: np.ndarray, alpha: float = 0.1) -> None:
        return None

    def interval(self, probs: np.ndarray) -> list[tuple[float, float]]:
        return [(self.grid[0], self.grid[-1]) for _ in range(len(probs))]
