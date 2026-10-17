"""Stub de T00: acepta todo."""

from __future__ import annotations

from PIL import Image

from bcs.contract import QualityReport


class AcceptAllChecker:
    def check(self, image: Image.Image) -> QualityReport:
        return QualityReport(is_cow_rear=True, score=1.0, issues=[])
