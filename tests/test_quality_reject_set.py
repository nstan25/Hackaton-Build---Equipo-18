"""Set de rechazo (tests/fixtures/reject/): generado, documentado y rechazado por el filtro real."""

from __future__ import annotations

import io

import pytest
from PIL import Image

from bcs.contract import QualityReport
from bcs.quality.clip_checker import ClipQualityChecker
from bcs.quality.reject_set import GENERATORS, reject_dir

REJECT_MIN = 0.85  # eval/thresholds.yaml m4.quality_filter.reject_negatives_min


def _paths():
    return sorted(reject_dir().glob("*.png"))


def test_reject_set_matches_generator() -> None:
    """Todas las imágenes salen de `bcs.quality.reject_set`: nada de terceros en el repo."""
    assert {p.stem for p in _paths()} == set(GENERATORS)
    for p in _paths():
        buf = io.BytesIO()
        GENERATORS[p.stem]().save(buf, format="PNG")
        assert buf.getvalue() == p.read_bytes(), f"{p.name} no coincide con su generador"


def test_reject_set_documented() -> None:
    readme = (reject_dir() / "README.md").read_text()
    missing = [p.name for p in _paths() if f"`{p.name}`" not in readme]
    assert not missing, f"sin documentar en README.md: {missing}"


@pytest.fixture(scope="module")
def reports() -> list[QualityReport]:
    images = [Image.open(p).convert("RGB") for p in _paths()]
    return ClipQualityChecker().check_batch(images)


def test_reject_set_rejection_rate(reports: list[QualityReport]) -> None:
    rate = sum(not r.passed for r in reports) / len(reports)
    accepted = [p.stem for p, r in zip(_paths(), reports, strict=True) if r.passed]
    assert rate >= REJECT_MIN, f"rechazo {rate:.2%} < {REJECT_MIN:.0%}; aceptadas: {accepted}"


def test_rejected_reports_explain_why(reports: list[QualityReport]) -> None:
    for p, r in zip(_paths(), reports, strict=True):
        assert r.passed or r.issues, f"{p.name} rechazada sin issues"
