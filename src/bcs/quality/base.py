"""Interfaz del filtro de calidad: ¿es una vaca de atrás y la foto sirve? (SPEC §2.0)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from PIL import Image

from bcs.contract import QualityReport


@runtime_checkable
class QualityChecker(Protocol):
    def check(self, image: Image.Image) -> QualityReport: ...
