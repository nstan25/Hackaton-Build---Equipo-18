"""El pipeline completo corre de punta a punta con los stubs de T00 (SPEC §2.0)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from bcs.calibration.base import Calibrator, IntervalPredictor
from bcs.calibration.stub import FullGridInterval, IdentityCalibrator
from bcs.contract import QualityReport, validate_prediction_dict
from bcs.data.schema import LABEL_COLUMNS, load_labels
from bcs.data.sources.base import Source
from bcs.data.sources.synthetic import SyntheticSource
from bcs.eval.metrics import evaluate
from bcs.inference import Pipeline, run_pipeline, stub_pipeline
from bcs.models.base import Scorer
from bcs.models.stub import UniformScorer
from bcs.quality.base import QualityChecker
from bcs.quality.stub import AcceptAllChecker
from bcs.scale import load_scale
from bcs.store.base import CowFilters, Store
from bcs.store.memory import InMemoryStore

FIXTURES = Path(__file__).parent / "fixtures"
SCALE = load_scale("dairy_1_5")


def test_stubs_implement_protocols() -> None:
    assert isinstance(SyntheticSource(), Source)
    assert isinstance(UniformScorer(SCALE.grid), Scorer)
    assert isinstance(IdentityCalibrator(), Calibrator)
    assert isinstance(FullGridInterval(SCALE.grid), IntervalPredictor)
    assert isinstance(AcceptAllChecker(), QualityChecker)
    assert isinstance(InMemoryStore(Path("/nonexistent")), Store)


def test_scale_presets() -> None:
    assert len(SCALE.grid) == 17 and SCALE.grid[0] == 1.0 and SCALE.grid[-1] == 5.0
    assert load_scale("beef_1_8").grid == tuple(float(i) for i in range(1, 9))
    assert len(load_scale("beef_1_9").grid) == 9
    assert load_scale().name == "dairy_1_5"


def test_labels_fixture_schema() -> None:
    df = load_labels(FIXTURES / "labels_fixture.csv")
    assert tuple(df.columns) == LABEL_COLUMNS
    rows = list(SyntheticSource().adapt())
    assert len(rows) == len(df)


def test_end_to_end_with_stubs(tmp_path: Path) -> None:
    df = load_labels(FIXTURES / "labels_fixture.csv")
    pipe = stub_pipeline(SCALE)
    pipe.scorer.fit(df)
    pipe.scorer.save(tmp_path / "model.json")
    pipe.scorer.load(tmp_path / "model.json")

    store = InMemoryStore(tmp_path / "media")
    for name, cow in [("rear_cow", "A1"), ("side_cow", "A1"), ("dog", "B2")]:
        raw = (FIXTURES / f"{name}.jpg").read_bytes()
        pred = run_pipeline(pipe, raw, cow)
        errors = validate_prediction_dict(pred.to_dict(), SCALE.keys, SCALE.step)
        assert errors == []
        # probabilidades uniformes + intervalo = grilla completa → low_confidence
        assert pred.status == "low_confidence"
        store.save(pred, raw)

    cows = store.latest_by_cow()
    assert [c.cow_id for c in cows] == ["A1", "B2"]
    assert cows[0].n_evaluations == 2
    assert len(store.history("A1")) == 2
    thumb = store.thumbnail_path(cows[0].last_prediction_id)
    assert thumb is not None and thumb.exists()
    assert store.latest_by_cow(CowFilters(status="ok")) == []

    images = [Image.new("RGB", (8, 8))] * len(df)
    probs = pipe.calibrator.transform(pipe.scorer.predict_proba(images))
    metrics = evaluate(
        probs, df["bcs"].to_numpy(), SCALE.grid, list(df["group_id"]), list(df["source"])
    )
    assert {"mae", "accuracy"} <= metrics.keys()


def test_rejected_by_quality(tmp_path: Path) -> None:
    class RejectAll:
        def check(self, image) -> QualityReport:
            return QualityReport(is_cow_rear=False, score=0.0, issues=[])

    base = stub_pipeline(SCALE)
    pipe = Pipeline(SCALE, RejectAll(), base.scorer, base.calibrator, base.interval)
    pred = run_pipeline(pipe, (FIXTURES / "dog.jpg").read_bytes(), "perro")
    assert pred.status == "rejected" and pred.bcs is None and pred.quality.issues
    assert validate_prediction_dict(pred.to_dict(), SCALE.keys, SCALE.step) == []
    assert np.isclose(sum(pred.distribution.values()), 0.0)
