"""T30: el modelo de grupa integrado (default `BCS_MODEL=zen_grupa`).

Pesos `models/bcs_efficientnet_b0_zen_grupa.pt` (el de T25 ajustado con los recortes de zen que
da el detector de producción sobre la vaca sola), letterbox, grilla `dairy_zen` (3.0–5.0),
calibración `models/calibration_efficientnet_b0_zen_grupa.json` y banco OOD
`models/ood_mahalanobis_zen_grupa.npz` (sdb train + zen train con recorte de grupa).
`zen` (T25) y `sdb` (main) siguen disponibles.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from bcs import inference
from bcs.calibration.regression import CalibratedRegression
from bcs.contract import ISSUE_OOD_WARN, validate_prediction_dict
from bcs.models.efficientnet import EfficientNetScorer, sha256_file
from bcs.models.train_zen import load_config, make_transforms
from bcs.ood.gate import MahalanobisGate, OODModelMismatch
from bcs.quality.stub import AcceptAllChecker
from bcs.scale import load_scale

FIXTURES = Path(__file__).parent / "fixtures"
ZEN = load_scale("dairy_zen")


@pytest.fixture
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (inference.MODEL_ENV, "BCS_SCALE", inference.OOD_ENV):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture(scope="module")
def pipe() -> inference.Pipeline:
    with pytest.MonkeyPatch.context() as mp:
        for name in (inference.MODEL_ENV, "BCS_SCALE", inference.OOD_ENV):
            mp.delenv(name, raising=False)
        return inference.model_pipeline()


def test_default_model_is_zen_grupa(clean_env: None) -> None:
    spec = inference.model_spec()
    assert spec.name == "efficientnet_b0_zen_grupa" and spec.scale == "dairy_zen"
    assert spec.weights == inference.WEIGHTS_ZEN_GRUPA and spec.ood == inference.OOD_ZEN_GRUPA
    assert spec.calibration == inference.CALIBRATION_ZEN_GRUPA
    assert inference.model_scale() == "dairy_zen"
    assert inference.model_spec("zen").weights == inference.WEIGHTS_ZEN
    assert inference.model_spec("sdb").weights == inference.WEIGHTS


def test_artifacts_in_repo_are_consistent() -> None:
    sha = sha256_file(inference.WEIGHTS_ZEN_GRUPA)
    assert sha.startswith("2e457c78")  # variante G de T30, época 3
    meta = json.loads(inference.WEIGHTS_ZEN_GRUPA.with_suffix(".json").read_text())
    assert meta["weights_sha256"] == sha and meta["letterbox"] is True
    assert meta["variant"] == "G" and meta["init_weights"] == "models/bcs_efficientnet_b0_zen.pt"
    assert meta["init_sha256"] == sha256_file(inference.WEIGHTS_ZEN)
    cal = CalibratedRegression.load(inference.CALIBRATION_ZEN_GRUPA)
    assert cal.grid == ZEN.grid and cal.alpha == pytest.approx(0.1)
    assert cal.info["variant"] == "ml_deg0_class_min20" and cal.conformal.min_class_n == 20
    gate = MahalanobisGate.load(inference.OOD_ZEN_GRUPA, model_sha256=sha)
    assert gate.mean.shape == (1280,) and gate.precision.shape == (1280, 1280)
    assert 0 < gate.warn < gate.reject < np.inf
    assert gate.info["bank"]["sdb_bcs_train"] > 0 and gate.info["bank"]["zen_rgbd_train_grupa"] > 0
    # el banco de T25 no sirve para este modelo, y los anteriores siguen en su lugar
    with pytest.raises(OODModelMismatch):
        MahalanobisGate.load(inference.OOD_ZEN, model_sha256=sha)
    assert sha256_file(inference.WEIGHTS_ZEN).startswith("bd938c68")
    assert sha256_file(inference.WEIGHTS).startswith("5a5b28d6")


def test_pipeline_and_version(pipe: inference.Pipeline, clean_env: None) -> None:
    assert pipe.scale.name == "dairy_zen" and pipe.ood is not None
    version = inference.model_version()
    assert version.startswith("efficientnet_b0_zen_grupa-2e457c78+cal-") and "+ood-" in version
    assert pipe.scorer.model_version == version
    assert inference.model_version("zen").startswith("efficientnet_b0_zen-bd938c68+cal-")


def test_letterbox_is_the_training_transform() -> None:
    scorer = EfficientNetScorer(ZEN.grid, {"model": {"weights": str(inference.WEIGHTS_ZEN_GRUPA)}})
    assert scorer.checkpoint["letterbox"] is True
    _, eval_tf = make_transforms(load_config())
    img = Image.open(FIXTURES / "rear_cow.jpg").convert("RGB").crop((0, 0, 300, 200))  # horizontal
    assert torch.equal(scorer._tf(img), eval_tf(img))  # noqa: SLF001


def test_end_to_end_contract(pipe: inference.Pipeline) -> None:
    p = replace(pipe, quality=AcceptAllChecker(), locate=lambda im: im)
    pred = inference.run_pipeline(p, (FIXTURES / "rear_cow.jpg").read_bytes(), "UY-1")
    assert pred.scale == "dairy_zen" and set(pred.distribution) == set(ZEN.keys)
    assert validate_prediction_dict(pred.to_dict(), ZEN.keys, ZEN.step) == []


class FixedValueFeatures:
    def __init__(self, value: float, feat: np.ndarray) -> None:
        self.value, self.feat = value, feat

    def predict_value(self, images: Sequence[Image.Image]) -> np.ndarray:
        return np.full(len(images), self.value)

    def predict_value_features(
        self, images: Sequence[Image.Image]
    ) -> tuple[np.ndarray, np.ndarray]:
        return self.predict_value(images), np.tile(self.feat, (len(images), 1))


@pytest.mark.parametrize(
    ("value", "status", "note"),
    [
        (3.0, "low_confidence", inference.EDGE_LOW),
        (3.6, "ok", None),
        (5.3, "low_confidence", inference.EDGE_HIGH),
    ],
)
def test_edges_are_3_and_5(
    pipe: inference.Pipeline, value: float, status: str, note: str | None
) -> None:
    assert pipe.ood is not None
    fake = FixedValueFeatures(value, pipe.ood.mean)
    scorer = inference.CalibratedEfficientNet(fake, pipe.scorer.dist, "test")  # type: ignore[attr-defined]
    p = replace(pipe, scorer=scorer, quality=AcceptAllChecker(), locate=lambda im: im)
    pred = inference.run_pipeline(p, (FIXTURES / "rear_cow.jpg").read_bytes(), "A1")
    assert pred.status == status
    assert inference.edge_note(pred, ZEN.step) == note


def test_ood_levels_with_real_bank(pipe: inference.Pipeline) -> None:
    """Pooling apenas debajo / encima del aviso y encima del rechazo."""
    gate = pipe.ood
    assert gate is not None
    direction = np.random.default_rng(0).normal(size=1280).astype(np.float32)
    unit = gate.score(gate.mean + direction)[0]
    cases = (
        (0.5 * gate.warn, "ok", []),
        (1.05 * gate.warn, "low_confidence", [ISSUE_OOD_WARN]),
        (1.1 * gate.reject, "rejected", None),
    )
    for target, status, issues in cases:
        feat = gate.mean + direction * np.float32(np.sqrt(target / unit))
        dist = pipe.scorer.dist  # type: ignore[attr-defined]
        scorer = inference.CalibratedEfficientNet(FixedValueFeatures(3.76, feat), dist, "test")
        p = replace(pipe, scorer=scorer, quality=AcceptAllChecker(), locate=lambda im: im)
        pred = inference.run_pipeline(p, (FIXTURES / "rear_cow.jpg").read_bytes(), "A1")
        assert pred.status == status
        if issues is not None:
            assert pred.quality.issues == issues
