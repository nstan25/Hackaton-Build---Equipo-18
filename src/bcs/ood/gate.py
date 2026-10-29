"""Detección OOD en producción (T28, G6 opción 1): Mahalanobis con una gaussiana en dos niveles.

El puntaje es la distancia de Mahalanobis al cuadrado del pooling (1.280-d, vista sin flip) a una
gaussiana ajustada con covarianza Ledoit-Wolf sobre el banco de entrenamiento del modelo (T27 §7).
Se compara con dos umbrales:

- `puntaje ≥ warn`   → nivel 1: `low_confidence` con `ISSUE_OOD_WARN` (la foto no se parece a las
  del entrenamiento).
- `puntaje ≥ reject` → nivel 2: `rejected` con `ISSUE_NOT_COW_REAR`.

El artefacto (`models/ood_mahalanobis_zen.npz`) guarda media y precisión en float32, los dos
umbrales y el sha256 del `.pt` cuyo embedding usa: `load(..., model_sha256=...)` se niega a cargar
un banco de otro modelo. Los umbrales y el banco salen de `python -m bcs.ood.fit_zen`.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np
from PIL import Image

LEVEL_OK, LEVEL_WARN, LEVEL_REJECT = 0, 1, 2


class OODModelMismatch(ValueError):
    """El banco OOD se armó con el embedding de otro modelo."""


@runtime_checkable
class FeatureScorer(Protocol):
    """Un `Scorer` que además devuelve el pooling de la misma pasada (para el puntaje OOD)."""

    def predict_proba_features(
        self, images: Sequence[Image.Image]
    ) -> tuple[np.ndarray, np.ndarray]: ...


@dataclass
class MahalanobisGate:
    mean: np.ndarray
    precision: np.ndarray
    warn: float
    reject: float
    model_sha256: str
    info: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def fit(
        cls, feats: np.ndarray, model_sha256: str, info: dict[str, Any] | None = None
    ) -> MahalanobisGate:
        """Media y precisión (Ledoit-Wolf, una gaussiana) del banco; umbrales en +inf hasta que
        se fijen con `with_thresholds`."""
        from bcs.ood.methods import MahalanobisScore

        m = MahalanobisScore().fit(np.asarray(feats, dtype=np.float64))
        return cls(
            mean=m.mean0.astype(np.float32),
            precision=m.precision.astype(np.float32),
            warn=float("inf"),
            reject=float("inf"),
            model_sha256=model_sha256,
            info=dict(info or {}),
        )

    def with_thresholds(self, warn: float, reject: float) -> MahalanobisGate:
        if not warn <= reject:
            raise ValueError(
                f"el umbral de aviso ({warn}) tiene que ser ≤ el de rechazo ({reject})"
            )
        return MahalanobisGate(
            self.mean, self.precision, float(warn), float(reject), self.model_sha256, self.info
        )

    def score(self, feats: np.ndarray) -> np.ndarray:
        """Distancia de Mahalanobis al cuadrado de cada fila [n, d] (más alto = más raro)."""
        x = np.asarray(feats, dtype=np.float32).reshape(-1, self.mean.shape[0]) - self.mean
        return np.einsum("ij,ij->i", x @ self.precision, x).astype(np.float64)

    def levels(self, scores: np.ndarray) -> np.ndarray:
        s = np.asarray(scores, dtype=float)
        return np.where(s >= self.reject, LEVEL_REJECT, np.where(s >= self.warn, LEVEL_WARN, 0))

    def level(self, feat: np.ndarray) -> int:
        return int(self.levels(self.score(feat))[0])

    def save(self, path: Path) -> None:
        np.savez(
            path,
            mean=self.mean,
            precision=self.precision,
            thresholds=np.array([self.warn, self.reject], dtype=np.float64),
            model_sha256=np.array(self.model_sha256),
            info=np.array(json.dumps(self.info, default=str)),
        )

    @classmethod
    def load(cls, path: Path, model_sha256: str | None = None) -> MahalanobisGate:
        with np.load(path, allow_pickle=False) as z:
            gate = cls(
                mean=z["mean"].astype(np.float32),
                precision=z["precision"].astype(np.float32),
                warn=float(z["thresholds"][0]),
                reject=float(z["thresholds"][1]),
                model_sha256=str(z["model_sha256"]),
                info=json.loads(str(z["info"])),
            )
        if model_sha256 is not None and gate.model_sha256 != model_sha256:
            raise OODModelMismatch(
                f"{Path(path).name} es del modelo {gate.model_sha256[:8]}, no del "
                f"{model_sha256[:8]}: rehacé el banco con `python -m bcs.ood.fit_zen`"
            )
        return gate
