"""Embeddings de la EfficientNet-B0 de producción (`models/bcs_efficientnet_b0.pt`, solo lectura).

El vector es la salida del pooling global (1.280-d), justo antes de la cabeza de regresión, con la
transformación de `inference.py` (`train_cnn.EVAL_TF`: 224×320, normalización ImageNet) aplicada
al recorte que recibe el modelo. Sin TTA: el flip duplicaría el costo y el puntaje OOD no lo
necesita. De la misma pasada sale el valor continuo del modelo (sin TTA) para medir error.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from bcs.models.efficientnet import WEIGHTS, _resolve, team_module

BATCH = 32
DIM = 1280


class FeatureExtractor:
    def __init__(self, weights: str | Path = WEIGHTS) -> None:
        self.weights = _resolve(weights)
        self._model: Any = None
        self._loss = "huber"

    def _load(self) -> Any:
        if self._model is None:
            import torch

            train_cnn = team_module("train_cnn")
            ck = torch.load(self.weights, map_location="cpu")
            self._loss = ck.get("loss", "huber")
            model = train_cnn.build_model(ck["arch"], pretrained=False, loss=self._loss)
            model.load_state_dict(ck["state_dict"])
            self._model = model.eval()
        return self._model

    def __call__(self, images: Sequence[Image.Image]) -> tuple[np.ndarray, np.ndarray]:
        """(features [n, 1280] float32, valor BCS continuo [n] sin TTA)."""
        import torch

        model = self._load()
        train_cnn = team_module("train_cnn")
        feats, vals = [], []
        with torch.inference_mode():
            for i in range(0, len(images), BATCH):
                x = torch.stack(
                    [train_cnn.EVAL_TF(im.convert("RGB")) for im in images[i : i + BATCH]]
                )
                f = model.forward_head(model.forward_features(x), pre_logits=True)
                out = model.get_classifier()(f)
                feats.append(f.numpy().astype(np.float32))
                vals.append(train_cnn.decode(out, self._loss).numpy().astype(float))
        if not feats:
            return np.zeros((0, DIM), dtype=np.float32), np.zeros(0)
        return np.concatenate(feats), np.concatenate(vals).reshape(-1)
