"""Interfaz de una fuente de datos (SPEC §1.2, §2.0)."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Protocol, runtime_checkable

from bcs.data.schema import LabelRow


@runtime_checkable
class Source(Protocol):
    source_id: str

    def fetch(self, sample: int | None = None) -> None:
        """Baja a $BCS_DATA_DIR/raw/<source_id>/. Idempotente; `sample` = N imágenes por clase."""
        ...

    def adapt(self) -> Iterable[LabelRow]:
        """Normaliza a $BCS_DATA_DIR/images/<source_id>/ y devuelve las filas de labels."""
        ...
