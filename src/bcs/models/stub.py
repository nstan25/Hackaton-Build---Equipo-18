"""Scorer stub de T00: probabilidades uniformes sobre la grilla."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


class UniformScorer:
    model_version = "stub-uniform-0"

    def __init__(self, grid: tuple[float, ...]) -> None:
        self.grid = grid

    def fit(self, df: pd.DataFrame) -> None:
        return None

    def predict_proba(self, images: Sequence[Image.Image]) -> np.ndarray:
        k = len(self.grid)
        return np.full((len(images), k), 1.0 / k)

    def save(self, path: Path) -> None:
        path.write_text(json.dumps({"model_version": self.model_version, "grid": self.grid}))

    def load(self, path: Path) -> None:
        self.grid = tuple(json.loads(path.read_text())["grid"])
