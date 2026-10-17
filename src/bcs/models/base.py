"""Interfaz del modelo (SPEC §2.0). El `Scorer` nunca recibe el `cow_id`."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np
import pandas as pd
from PIL import Image


@runtime_checkable
class Scorer(Protocol):
    grid: tuple[float, ...]
    model_version: str

    def fit(self, df: pd.DataFrame) -> None:
        """Entrena sobre filas de `labels.csv` (columnas de `LabelRow`)."""
        ...

    def predict_proba(self, images: Sequence[Image.Image]) -> np.ndarray:
        """Devuelve probabilidades `[n, K]` sobre `grid` (K = len(grid)); cada fila suma 1."""
        ...

    def save(self, path: Path) -> None: ...

    def load(self, path: Path) -> None: ...
