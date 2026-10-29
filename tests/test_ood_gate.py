"""T28: la detección OOD de producción (`bcs.ood.gate`) y su lugar en `_run_decoded`."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from bcs import inference
from bcs.calibration.stub import FullGridInterval, IdentityCalibrator
from bcs.contract import (
    ISSUE_NOT_COW_REAR,
    ISSUE_OOD_WARN,
    QualityReport,
    decide_status,
    validate_prediction_dict,
)
from bcs.ood.fit_zen import rates, reject_threshold, split_groups, warn_threshold
from bcs.ood.gate import (
    LEVEL_OK,
    LEVEL_REJECT,
    LEVEL_WARN,
    FeatureScorer,
    MahalanobisGate,
    OODModelMismatch,
)
from bcs.ood.methods import MahalanobisScore
from bcs.quality.stub import AcceptAllChecker
from bcs.scale import load_scale

FIXTURES = Path(__file__).parent / "fixtures"
ZEN = load_scale("dairy_zen")
GOOD = QualityReport(is_cow_rear=True, score=0.9, issues=[])
DIM = 8


def _bank(n: int = 400, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).normal(size=(n, DIM)).astype(np.float32)


def _gate(warn: float = 20.0, reject: float = 60.0) -> MahalanobisGate:
    return MahalanobisGate.fit(_bank(), "a" * 64).with_thresholds(warn, reject)


def test_score_matches_t27_mahalanobis() -> None:
    bank = _bank()
    q = np.random.default_rng(1).normal(size=(50, DIM)) * 2
    ref = MahalanobisScore().fit(bank.astype(np.float64)).score(q)
    got = MahalanobisGate.fit(bank, "x").score(q)
    assert np.allclose(got, ref, rtol=1e-4)


def test_levels_two_thresholds() -> None:
    g = _gate(10.0, 30.0)
    assert g.levels(np.array([0.0, 9.99, 10.0, 29.9, 30.0, 1e6])).tolist() == [
        LEVEL_OK,
        LEVEL_OK,
        LEVEL_WARN,
        LEVEL_WARN,
        LEVEL_REJECT,
        LEVEL_REJECT,
    ]
    assert g.level(np.zeros(DIM)) == LEVEL_OK
    assert g.level(np.full(DIM, 50.0)) == LEVEL_REJECT
    with pytest.raises(ValueError):
        g.with_thresholds(5.0, 4.0)


def test_save_load_roundtrip_and_refuses_other_model(tmp_path: Path) -> None:
    g = _gate()
    g.info["bank"] = {"sdb_bcs_train": 3}
    path = tmp_path / "ood.npz"
    g.save(path)
    back = MahalanobisGate.load(path, model_sha256="a" * 64)
    assert (back.warn, back.reject, back.model_sha256) == (g.warn, g.reject, g.model_sha256)
    assert back.info == g.info
    assert np.array_equal(back.precision, g.precision) and back.precision.dtype == np.float32
    with pytest.raises(OODModelMismatch, match="aaaaaaaa"):
        MahalanobisGate.load(path, model_sha256="b" * 64)


def test_threshold_rules() -> None:
    a, b = np.arange(1000.0), np.arange(1000.0) * 2
    assert warn_threshold({"a": a, "b": b}, 0.99) == pytest.approx(
        np.quantile(b, 0.99, method="higher")
    )
    assert reject_threshold(np.array([1.0, 7.0, 3.0]), 1.25) == pytest.approx(8.75)
    groups = np.array(["g1", "g1", "g2", "g3", "g4", "g4"])
    m = split_groups(pd.Series(groups), 0.5, 0)
    # grupos enteros de un lado
    for gid in set(groups):
        assert len(set(m[groups == gid])) == 1
    r = rates(np.array([0, 1, 2, 0]), np.array([True, True, True, False]))
    assert r["reach"] == 3 and r["rejected_total"] == pytest.approx(0.5)
    assert r["warned_total"] == pytest.approx(0.25) and r["clean_total"] == pytest.approx(0.25)


# ---------------------------------------------------------------- contrato


def test_warning_issue_is_low_confidence_not_rejected() -> None:
    q = QualityReport(True, 0.9, [ISSUE_OOD_WARN])
    assert q.passed and q.warned
    assert decide_status(q, 0.95, (3.5, 3.75), 0.25) == "low_confidence"
    # un motivo de rechazo sigue rechazando aunque haya aviso
    both = QualityReport(True, 0.9, [ISSUE_OOD_WARN, "imagen muy oscura"])
    assert not both.passed and decide_status(both, 0.95, (3.5, 3.75), 0.25) == "rejected"
    assert decide_status(GOOD, 0.95, (3.5, 3.75), 0.25) == "ok"


# ---------------------------------------------------------------- en el pipeline


class FeatScorer:
    """Distribución fija en 3.75 y un pooling que se elige por test."""

    model_version = "fake-zen"

    def __init__(self, feat: np.ndarray) -> None:
        self.feat = feat
        self.calls = 0

    def predict_proba(self, images: Sequence[Image.Image]) -> np.ndarray:
        raise AssertionError("con OOD el pipeline usa predict_proba_features (una sola pasada)")

    def predict_proba_features(
        self, images: Sequence[Image.Image]
    ) -> tuple[np.ndarray, np.ndarray]:
        self.calls += 1
        p = np.full((len(images), len(ZEN.grid)), 0.01)
        p[:, ZEN.grid.index(3.75)] = 1 - 0.01 * (len(ZEN.grid) - 1)
        return p, np.tile(self.feat, (len(images), 1))


class TightInterval(FullGridInterval):
    def interval(self, probs: np.ndarray) -> list[tuple[float, float]]:
        return [(3.5, 4.0)] * len(probs)


def _pipe(feat: np.ndarray) -> tuple[inference.Pipeline, FeatScorer]:
    scorer = FeatScorer(feat)
    pipe = inference.Pipeline(
        scale=ZEN,
        quality=AcceptAllChecker(),
        scorer=scorer,  # type: ignore[arg-type]
        calibrator=IdentityCalibrator(),
        interval=TightInterval(ZEN.grid),
        ood=_gate(),
    )
    return pipe, scorer


@pytest.mark.parametrize(
    ("feat", "status", "issues"),
    [
        (np.zeros(DIM), "ok", []),
        (np.full(DIM, 1.9), "low_confidence", [ISSUE_OOD_WARN]),
        (np.full(DIM, 20.0), "rejected", [ISSUE_NOT_COW_REAR]),
    ],
)
def test_two_levels_in_run_pipeline(feat: np.ndarray, status: str, issues: list[str]) -> None:
    pipe, scorer = _pipe(feat)
    level = pipe.ood.level(feat)  # type: ignore[union-attr]
    assert level == {"ok": 0, "low_confidence": 1, "rejected": 2}[status]
    pred = inference.run_pipeline(pipe, (FIXTURES / "rear_cow.jpg").read_bytes(), "A1")
    assert scorer.calls == 1
    assert pred.status == status
    assert pred.quality.issues == issues
    if status == "rejected":
        assert pred.bcs is None and pred.quality.is_cow_rear is False
    else:
        assert pred.bcs == 3.75
    assert validate_prediction_dict(pred.to_dict(), ZEN.keys, ZEN.step) == []


def test_ood_requires_feature_scorer() -> None:
    pipe, _ = _pipe(np.zeros(DIM))
    stub = inference.stub_pipeline(ZEN)
    assert not isinstance(stub.scorer, FeatureScorer)
    with pytest.raises(TypeError, match="predict_proba_features"):
        inference.run_pipeline(
            replace(pipe, scorer=stub.scorer), (FIXTURES / "rear_cow.jpg").read_bytes(), "A1"
        )


def test_no_ood_when_detector_rejects() -> None:
    pipe, scorer = _pipe(np.full(DIM, 20.0))
    pred = inference.run_pipeline(
        replace(pipe, locate=lambda im: None), (FIXTURES / "rear_cow.jpg").read_bytes(), "A1"
    )
    assert pred.status == "rejected" and inference.ISSUE_NO_COW in pred.quality.issues
    assert scorer.calls == 0
