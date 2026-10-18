"""Filtro de calidad sin CLIP (decisión del humano 18/10: la app no entraba en la RAM de Railway).

Solo brillo, sobreexposición y nitidez (los mismos chequeos y umbrales de T08). Que la foto sea
una vaca lo decide el detector del pipeline ("no se encontró la vaca en la foto"); que sea de
atrás no lo verifica nadie: `is_cow_rear` queda en True porque no hay evidencia en contra.
"""

from __future__ import annotations

from PIL import Image

from bcs.contract import QualityReport
from bcs.quality.clip_checker import Thresholds, image_stats, quality_issues


class BasicQualityChecker:
    def __init__(self, thresholds: Thresholds | None = None) -> None:
        self.thresholds = thresholds or Thresholds()

    def check(self, image: Image.Image) -> QualityReport:
        issues = quality_issues(image_stats(image), self.thresholds)
        return QualityReport(is_cow_rear=True, score=1.0, issues=issues)
