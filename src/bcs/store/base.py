"""Interfaz del registro de evaluaciones (SPEC §2.0, §2.2)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

from bcs.contract import Prediction, Status


@dataclass(frozen=True)
class CowFilters:
    status: Status | None = None  # status de la última evaluación
    bcs_min: float | None = None  # sobre el último BCS válido
    bcs_max: float | None = None
    since: str | None = None  # ISO 8601: última evaluación en o después de esta fecha


@dataclass
class CowSummary:
    cow_id: str
    bcs: float | None  # último BCS válido (status ok o low_confidence)
    confidence: float | None
    interval: tuple[float, float] | None
    bcs_at: str | None  # created_at de esa evaluación válida
    n_evaluations: int  # todas, incluidas las rejected
    last_prediction_id: str  # la última foto, sea cual sea su status
    last_status: Status
    last_at: str
    last_rejected: bool  # la última evaluación fue rejected (se indica aparte)


@runtime_checkable
class Store(Protocol):
    def save(
        self,
        prediction: Prediction,
        image_bytes: bytes,
        *,
        taken_at: str | None = None,
        notes: str | None = None,
    ) -> None:
        """Guarda la predicción (incluso rejected) y la miniatura (lado mayor 512 px).

        `taken_at` y `notes` vienen del form de `/predict`; no son parte de `Prediction`.
        """
        ...

    def latest_by_cow(self, filters: CowFilters | None = None) -> list[CowSummary]: ...

    def history(self, cow_id: str) -> list[Prediction]:
        """Historial de la vaca, de la más reciente a la más vieja."""
        ...

    def get(self, prediction_id: str) -> Prediction | None: ...

    def thumbnail_path(self, prediction_id: str) -> Path | None: ...
