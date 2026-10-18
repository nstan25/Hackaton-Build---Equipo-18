"""Filtro de calidad sin CLIP: brillo, sobreexposición y nitidez."""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageFilter

from bcs.quality.basic import BasicQualityChecker


def _noise(level: int) -> Image.Image:
    rng = np.random.default_rng(0)
    arr = np.clip(rng.normal(level, 40, (240, 320, 3)), 0, 255).astype("uint8")
    return Image.fromarray(arr)


def test_good_image_passes() -> None:
    assert BasicQualityChecker().check(_noise(128)).passed


def test_dark_overexposed_and_blurry_are_rejected() -> None:
    q = BasicQualityChecker()
    assert not q.check(Image.new("RGB", (320, 240), (12, 12, 12))).passed
    assert not q.check(Image.new("RGB", (320, 240), (250, 250, 250))).passed
    assert not q.check(_noise(128).filter(ImageFilter.GaussianBlur(10))).passed


def test_does_not_claim_rear_view_check() -> None:
    report = BasicQualityChecker().check(_noise(128))
    assert report.is_cow_rear and report.score == 1.0
