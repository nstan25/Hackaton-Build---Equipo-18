"""Contrato de inferencia (SPEC §2.2). Lo congela G0: ningún agente lo edita (SPEC §4.1)."""

from __future__ import annotations

import hashlib
import inspect
import json
from dataclasses import fields
from pathlib import Path
from typing import Any, cast

import pytest

from bcs.calibration.base import Calibrator, IntervalPredictor
from bcs.contract import (
    PREDICTION_FIELDS,
    InvalidCowId,
    Prediction,
    QualityReport,
    decide_status,
    normalize_cow_id,
    validate_prediction_dict,
)
from bcs.inference import Pipeline, predict, run_pipeline, stub_pipeline
from bcs.models.base import Scorer
from bcs.quality.base import QualityChecker
from bcs.scale import load_scale

FIXTURES = Path(__file__).parent / "fixtures"
SCALE = load_scale("dairy_1_5")


@pytest.fixture
def example() -> dict:
    return json.loads((FIXTURES / "prediction_example.json").read_text())


# --- Prediction y fixture -------------------------------------------------------------------


def test_prediction_fields_match_spec() -> None:
    assert PREDICTION_FIELDS == (
        "prediction_id",
        "cow_id",
        "created_at",
        "image_sha256",
        "status",
        "bcs",
        "bcs_expected",
        "confidence",
        "interval",
        "distribution",
        "quality",
        "scale",
        "model_version",
    )
    assert [f.name for f in fields(QualityReport)] == ["is_cow_rear", "score", "issues"]


def test_fixture_is_valid_contract(example: dict) -> None:
    assert validate_prediction_dict(example, SCALE.keys, SCALE.step) == []
    for key in ("cow_id", "prediction_id", "created_at", "image_sha256"):
        assert example[key]
    assert example["scale"] == SCALE.name


def test_fixture_hash_matches_rear_cow(example: dict) -> None:
    digest = hashlib.sha256((FIXTURES / "rear_cow.jpg").read_bytes()).hexdigest()
    assert example["image_sha256"] == digest


def test_fixture_roundtrip(example: dict) -> None:
    pred = Prediction.from_dict(example)
    assert pred.interval == (3.25, 3.75)
    assert json.loads(json.dumps(pred.to_dict())) == example


@pytest.mark.parametrize(
    "mutate, msg",
    [
        (lambda d: d.pop("cow_id"), "faltan"),
        (lambda d: d.update(extra=1), "de más"),
        (lambda d: d.update(status="maybe"), "status"),
        (lambda d: d.update(confidence=1.5), "confidence"),
        (lambda d: d.update(interval=[3.75, 3.25]), "interval"),
        (lambda d: d.update(cow_id="uy-0042"), "cow_id"),
        (lambda d: d.update(bcs=3.6), "grilla"),
        (lambda d: d.update(bcs_expected=4.2), "bcs_expected"),
        (lambda d: d.update(confidence=0.4), "reglas"),
        (lambda d: d.update(image_sha256="abc"), "sha256"),
    ],
)
def test_validator_catches_broken_predictions(example: dict, mutate, msg: str) -> None:
    mutate(example)
    errors = validate_prediction_dict(example, SCALE.keys, SCALE.step)
    assert any(msg in e for e in errors), errors


def test_rejected_returns_no_bcs(example: dict) -> None:
    example.update(
        status="rejected",
        bcs=None,
        bcs_expected=None,
        confidence=None,
        interval=None,
        quality={"is_cow_rear": False, "score": 0.1, "issues": ["no parece una vaca de atrás"]},
    )
    assert validate_prediction_dict(example, SCALE.keys, SCALE.step) == []
    example["bcs"] = 3.5
    assert validate_prediction_dict(example, SCALE.keys, SCALE.step)


# --- Reglas de status -----------------------------------------------------------------------

GOOD = QualityReport(is_cow_rear=True, score=0.9, issues=[])
STEP = 0.25


@pytest.mark.parametrize(
    "quality, confidence, interval, expected",
    [
        (QualityReport(False, 0.2, ["no parece una vaca de atrás"]), 0.9, (3.25, 3.75), "rejected"),
        (QualityReport(True, 0.9, ["imagen muy oscura"]), 0.9, (3.25, 3.75), "rejected"),
        (GOOD, 0.49, (3.25, 3.75), "low_confidence"),
        (GOOD, 0.5, (3.25, 3.75), "ok"),
        (GOOD, 0.9, (3.0, 4.0), "ok"),  # 4 pasos: todavía ok
        (GOOD, 0.9, (3.0, 4.25), "low_confidence"),  # 5 pasos
        (GOOD, None, (3.25, 3.75), "low_confidence"),
    ],
)
def test_status_rules(quality, confidence, interval, expected) -> None:
    assert decide_status(quality, confidence, interval, STEP) == expected


# --- cow_id ---------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("  uy-0042 ", "UY-0042"),
        ("1234", "1234"),
        ("a.b_c-d", "A.B_C-D"),
        ("X" * 64, "X" * 64),
    ],
)
def test_cow_id_normalization(raw: str, expected: str) -> None:
    assert normalize_cow_id(raw) == expected


@pytest.mark.parametrize("raw", [None, "", "   ", "-123", ".A", "UY 42", "vaca/1", "Ñ1", "X" * 65])
def test_cow_id_invalid(raw: str | None) -> None:
    with pytest.raises(InvalidCowId):
        normalize_cow_id(raw)


# --- predict --------------------------------------------------------------------------------


def test_predict_dummy_returns_fixture_with_cow_id(example: dict) -> None:
    raw = (FIXTURES / "side_cow.jpg").read_bytes()
    pred = predict(raw, "  vaca-7 ")
    d = pred.to_dict()
    assert d["cow_id"] == "VACA-7"
    assert d["image_sha256"] == hashlib.sha256(raw).hexdigest()
    assert d["prediction_id"] != example["prediction_id"]
    for key in ("status", "bcs", "confidence", "interval", "distribution", "quality"):
        assert d[key] == example[key]
    assert validate_prediction_dict(d, SCALE.keys, SCALE.step) == []


def test_predict_rejects_invalid_cow_id() -> None:
    with pytest.raises(InvalidCowId):
        predict((FIXTURES / "rear_cow.jpg").read_bytes(), "no válido")


def test_no_component_receives_cow_id() -> None:
    """Ni el Scorer ni calibración, intervalo o filtro de calidad ven el cow_id."""
    for proto in (Scorer, Calibrator, IntervalPredictor, QualityChecker):
        for name, member in inspect.getmembers(proto, inspect.isfunction):
            assert "cow_id" not in inspect.signature(member).parameters, (proto, name)

    seen: list[tuple] = []

    class Spy:
        """Proxy que registra cada llamada a un método del componente envuelto."""

        def __init__(self, inner: object) -> None:
            self._inner = inner

        def __getattr__(self, name: str) -> Any:
            attr = getattr(self._inner, name)
            if not callable(attr):
                return attr

            def call(*args: Any, **kwargs: Any) -> Any:
                seen.append((type(self._inner).__name__, name, args, kwargs))
                return attr(*args, **kwargs)

            return call

    base = stub_pipeline(SCALE)
    pipe = Pipeline(
        SCALE,
        cast(Any, Spy(base.quality)),
        cast(Any, Spy(base.scorer)),
        cast(Any, Spy(base.calibrator)),
        cast(Any, Spy(base.interval)),
    )
    run_pipeline(pipe, (FIXTURES / "rear_cow.jpg").read_bytes(), "secret-cow-99")
    called = {(owner, method) for owner, method, _, _ in seen}
    assert {
        ("AcceptAllChecker", "check"),
        ("UniformScorer", "predict_proba"),
        ("IdentityCalibrator", "transform"),
        ("FullGridInterval", "interval"),
    } <= called
    assert "SECRET-COW-99" not in repr(seen) and "secret-cow-99" not in repr(seen)
