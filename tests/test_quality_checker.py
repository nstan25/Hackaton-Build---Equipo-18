"""Filtro de calidad con puntaje semántico falso: brillo, quemado, nitidez y reporte."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pytest
from PIL import Image, ImageFilter

from bcs.quality.base import QualityChecker
from bcs.quality.clip_checker import (
    ISSUE_BLURRY,
    ISSUE_DARK,
    ISSUE_NOT_COW_REAR,
    ISSUE_OVEREXPOSED,
    ClipQualityChecker,
    Thresholds,
    image_stats,
)


def _fixed(score: float):
    def scorer(images: Sequence[Image.Image]) -> np.ndarray:
        return np.full(len(images), score)

    return scorer


def _textured(seed: int = 0, size: tuple[int, int] = (320, 256)) -> Image.Image:
    """Textura de brillo medio y con bordes: pasa los chequeos de brillo y nitidez."""
    rng = np.random.default_rng(seed)
    small = rng.integers(40, 200, size=(size[1] // 4, size[0] // 4, 3), dtype=np.uint8)
    return Image.fromarray(small, mode="RGB").resize(size, Image.Resampling.NEAREST)


def test_implements_protocol() -> None:
    assert isinstance(ClipQualityChecker(scorer=_fixed(1.0)), QualityChecker)


def test_good_image_passes() -> None:
    report = ClipQualityChecker(scorer=_fixed(0.9)).check(_textured())
    assert report.is_cow_rear and report.issues == [] and report.passed
    assert report.score == pytest.approx(0.9)


def test_low_clip_score_is_not_cow_rear() -> None:
    report = ClipQualityChecker(scorer=_fixed(0.01)).check(_textured())
    assert not report.is_cow_rear and not report.passed
    assert report.issues == [ISSUE_NOT_COW_REAR]


def test_dark_image_rejected() -> None:
    dark = Image.fromarray((np.asarray(_textured()) * 0.1).astype(np.uint8), mode="RGB")
    report = ClipQualityChecker(scorer=_fixed(0.9)).check(dark)
    assert ISSUE_DARK in report.issues and not report.passed


def test_black_padding_does_not_count_as_dark() -> None:
    """El preprocesamiento rellena con negro: una vaca bien iluminada con bordes negros pasa."""
    arr = np.asarray(_textured()).copy()
    arr[:, 200:] = 0
    arr[180:, :] = 0
    stats = image_stats(Image.fromarray(arr, mode="RGB"))
    assert stats.brightness > Thresholds().brightness_min


def test_overexposed_image_rejected() -> None:
    bright = np.asarray(_textured(), dtype=np.float32) * 0.1 + 240
    img = Image.fromarray(bright.clip(0, 255).astype(np.uint8), mode="RGB")
    report = ClipQualityChecker(scorer=_fixed(0.9)).check(img)
    assert ISSUE_OVEREXPOSED in report.issues and not report.passed


def test_blurry_image_rejected() -> None:
    img = _textured().filter(ImageFilter.GaussianBlur(12))
    report = ClipQualityChecker(scorer=_fixed(0.9)).check(img)
    assert ISSUE_BLURRY in report.issues and not report.passed


def test_sharpness_independent_of_resolution() -> None:
    """La nitidez se mide a tamaño fijo: la misma foto en 4x no cambia de veredicto."""
    small = _textured(seed=3)
    big = small.resize((1280, 1024), Image.Resampling.NEAREST)
    a, b = image_stats(small).sharpness, image_stats(big).sharpness
    assert a == pytest.approx(b, rel=0.5)


def test_check_batch_matches_check() -> None:
    checker = ClipQualityChecker(scorer=_fixed(0.5))
    images = [_textured(seed=s) for s in range(3)]
    assert checker.check_batch(images) == [checker.check(im) for im in images]


def test_custom_thresholds() -> None:
    strict = ClipQualityChecker(scorer=_fixed(0.5), thresholds=Thresholds(clip_min=0.6))
    assert not strict.check(_textured()).is_cow_rear
