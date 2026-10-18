"""T12: el pipeline de producción de G3 (`bcs.inference.model_pipeline`).

La EfficientNet y el detector están en el repo (models/) y corren en CPU en segundos. CLIP (578 MB)
no se baja en los tests: el filtro de calidad usa el `ClipQualityChecker` real con un puntaje fijo.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

from bcs import inference
from bcs.calibration.regression import CalibratedRegression
from bcs.contract import Prediction, validate_prediction_dict
from bcs.quality.clip_checker import ClipQualityChecker
from bcs.scale import load_scale

FIXTURES = Path(__file__).parent / "fixtures"
SCALE = load_scale("dairy_sdb")
VOLATILE = ("prediction_id", "created_at", "cow_id")


def _clip(score: float) -> ClipQualityChecker:
    return ClipQualityChecker(scorer=lambda images: np.full(len(images), score))


def _whole(image: Image.Image) -> Image.Image:
    """Locator que deja la foto entera: las fixtures son dibujos y el detector no ve vaca."""
    return image


@pytest.fixture(scope="module")
def pipe() -> inference.Pipeline:
    return replace(inference.model_pipeline(SCALE), quality=_clip(0.9), locate=_whole)


class FixedValue:
    """Sustituye a la EfficientNet: siempre el mismo valor continuo."""

    def __init__(self, value: float) -> None:
        self.value = value
        self.calls = 0

    def predict_value(self, images: Sequence[Image.Image]) -> np.ndarray:
        self.calls += 1
        return np.full(len(images), self.value)


def _with_value(pipe: inference.Pipeline, value: float) -> tuple[inference.Pipeline, FixedValue]:
    fake = FixedValue(value)
    scorer = inference.CalibratedEfficientNet(fake, pipe.scorer.dist, "test")  # type: ignore[attr-defined]
    return replace(pipe, scorer=scorer), fake


def _stable(pred: Prediction) -> dict[str, Any]:
    d = pred.to_dict()
    return {k: v for k, v in d.items() if k not in VOLATILE}


def test_same_photo_two_cow_ids_identical(
    pipe: inference.Pipeline, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Restricción de T12: misma foto, dos `cow_id` → misma predicción salvo ids y fecha."""
    monkeypatch.setenv(inference.PREDICTOR_ENV, "model")
    monkeypatch.setattr(inference, "_model_pipeline", lambda: pipe)
    raw = (FIXTURES / "rear_cow.jpg").read_bytes()
    a, b = inference.predict(raw, "UY-001"), inference.predict(raw, " vaca-77 ")
    assert (a.cow_id, b.cow_id) == ("UY-001", "VACA-77")
    assert a.prediction_id != b.prediction_id
    assert _stable(a) == _stable(b)
    assert a.status != "rejected" and a.bcs is not None
    for p in (a, b):
        assert validate_prediction_dict(p.to_dict(), SCALE.keys, SCALE.step) == []


def test_probabilities_are_t11_calibrated_regression(pipe: inference.Pipeline) -> None:
    """Scorer (distribución) + Calibrator (temperatura) del pipeline = `predict_proba` de T11."""
    cal = CalibratedRegression.load(inference.CALIBRATION)
    img = Image.open(FIXTURES / "rear_cow.jpg").convert("RGB")
    value = pipe.scorer.effnet.predict_value([img])  # type: ignore[attr-defined]
    got = pipe.calibrator.transform(pipe.scorer.predict_proba([img]))
    assert np.allclose(got, cal.predict_proba(value))
    pred = inference.run_pipeline(pipe, Image.open(FIXTURES / "rear_cow.jpg"), "A1")
    probs = np.array([pred.distribution[k] for k in SCALE.keys])
    assert np.allclose(probs, got[0])
    top = int(probs.argmax())
    assert pred.bcs == SCALE.grid[top]
    assert pred.confidence == pytest.approx(probs[max(top - 1, 0) : top + 2].sum())
    assert pred.bcs_expected == pytest.approx(float(np.dot(SCALE.grid, probs)))
    assert pred.interval == tuple(cal.interval(got)[0])


@pytest.mark.parametrize(
    ("value", "status", "note"),
    [
        (3.2, "low_confidence", inference.EDGE_LOW),
        (3.76, "ok", None),
        (4.3, "low_confidence", inference.EDGE_HIGH),
    ],
)
def test_edge_rule(pipe: inference.Pipeline, value: float, status: str, note: str | None) -> None:
    """G3: BCS en 3.25 o 4.25 → `low_confidence` con su texto, aunque la confianza sea alta."""
    p, _ = _with_value(pipe, value)
    pred = inference.run_pipeline(p, (FIXTURES / "rear_cow.jpg").read_bytes(), "A1")
    assert pred.status == status
    assert pred.confidence is not None and pred.confidence >= 0.5
    assert inference.edge_note(pred, SCALE.step) == note
    assert pred.quality.issues == []  # el texto va en la UI: un issue la volvería `rejected`
    assert validate_prediction_dict(pred.to_dict(), SCALE.keys, SCALE.step) == []


def test_no_cow_detected_is_rejected_without_scoring(pipe: inference.Pipeline) -> None:
    p, fake = _with_value(pipe, 3.75)
    p = replace(p, locate=inference.CowLocator())  # detector real: el dibujo no es una vaca
    pred = inference.run_pipeline(p, (FIXTURES / "rear_cow.jpg").read_bytes(), "A1")
    assert pred.status == "rejected" and pred.bcs is None
    assert inference.ISSUE_NO_COW in pred.quality.issues
    assert fake.calls == 0
    assert validate_prediction_dict(pred.to_dict(), SCALE.keys, SCALE.step) == []


def test_quality_rejection_skips_detector_and_model(pipe: inference.Pipeline) -> None:
    located: list[Image.Image] = []

    def locate(image: Image.Image) -> Image.Image:
        located.append(image)
        return image

    p, fake = _with_value(pipe, 3.75)
    p = replace(p, quality=_clip(0.01), locate=locate)
    pred = inference.run_pipeline(p, (FIXTURES / "dog.jpg").read_bytes(), "A1")
    assert pred.status == "rejected"
    assert pred.quality.issues[0] == "no parece una vaca de atrás"
    assert located == [] and fake.calls == 0


def test_locator_crop_is_what_the_model_sees(pipe: inference.Pipeline) -> None:
    seen: list[tuple[int, int]] = []

    class Spy(FixedValue):
        def predict_value(self, images: Sequence[Image.Image]) -> np.ndarray:
            seen.extend(im.size for im in images)
            return super().predict_value(images)

    scorer = inference.CalibratedEfficientNet(Spy(3.75), pipe.scorer.dist, "test")  # type: ignore[attr-defined]
    p = replace(pipe, scorer=scorer, locate=lambda im: im.crop((0, 0, 50, 40)))
    inference.run_pipeline(p, (FIXTURES / "rear_cow.jpg").read_bytes(), "A1")
    assert seen == [(50, 40)]


def test_calibration_in_repo_matches_model_grid() -> None:
    cal = CalibratedRegression.load(inference.CALIBRATION)
    assert cal.grid == SCALE.grid
    assert cal.info["variant"] == "ml_deg0_class" and cal.alpha == pytest.approx(0.1)
    assert len(cal.conformal.thresholds) == len(SCALE.grid)


def test_model_version_and_scale(pipe: inference.Pipeline) -> None:
    version = inference.model_version()
    assert version.startswith("efficientnet_b0-") and "+cal-" in version and "+det-" in version
    assert pipe.scorer.model_version == version
    assert pipe.scale.name == "dairy_sdb"
    with pytest.raises(ValueError, match="dairy_sdb"):
        inference.model_pipeline(load_scale("dairy_1_5"))


def test_predictor_switch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(inference.PREDICTOR_ENV, raising=False)
    assert not inference.use_model()
    assert inference.start_warmup() is None
    assert (
        inference.predict_model_version()
        == inference.predict_dummy((FIXTURES / "rear_cow.jpg").read_bytes(), "A1").model_version
    )
    monkeypatch.setenv(inference.PREDICTOR_ENV, "model")
    assert inference.use_model()
    assert inference.predict_model_version() == inference.model_version()
    monkeypatch.setenv(inference.PREDICTOR_ENV, "real")
    with pytest.raises(ValueError, match="BCS_PREDICTOR"):
        inference.use_model()
