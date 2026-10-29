"""Embeddings de la EfficientNet-B0 para el puntaje OOD (T27; T28: cualquier `.pt`).

El vector es la salida del pooling global (1.280-d), justo antes de la cabeza de regresión, con la
transformación que pide el checkpoint (`efficientnet.eval_transform`: la de `train_cnn.EVAL_TF`
para el modelo de producción, letterbox para el de T25) aplicada al recorte que recibe el modelo.
Sin TTA: el flip duplicaría el costo y el puntaje OOD no lo necesita. De la misma pasada sale el
valor continuo del modelo (sin TTA) para medir error. Es exactamente lo que calcula
`EfficientNetScorer.predict_views_features` en producción (vista sin flip).
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
from PIL import Image

from bcs.models.efficientnet import WEIGHTS, EfficientNetScorer, _resolve

BATCH = 32
DIM = 1280


class FeatureExtractor:
    def __init__(self, weights: str | Path = WEIGHTS) -> None:
        self.weights = _resolve(weights)
        self._scorer = EfficientNetScorer((3.75,), {"model": {"weights": str(self.weights)}})

    def __call__(self, images: Sequence[Image.Image]) -> tuple[np.ndarray, np.ndarray]:
        """(features [n, 1280] float32, valor BCS continuo [n] sin TTA)."""
        if not images:
            return np.zeros((0, DIM), dtype=np.float32), np.zeros(0)
        views, feats = self._scorer.predict_views_features(images, flip=False)
        return feats, views[:, 0]
