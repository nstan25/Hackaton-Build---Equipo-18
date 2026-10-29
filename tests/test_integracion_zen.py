"""T28: el modelo de T25 integrado en la rama `dev` (G5 + G6, opción 1).

Default `BCS_MODEL=zen`: pesos `models/bcs_efficientnet_b0_zen.pt` con letterbox, grilla
`dairy_zen` (3.0–5.0), calibración `models/calibration_efficientnet_b0_zen.json` y detección OOD
`models/ood_mahalanobis_zen.npz`. El modelo de `main` sigue disponible con `BCS_MODEL=sdb`.
"""

from __future__ import annotations

import importlib.util
import json
from collections.abc import Iterator, Sequence
from dataclasses import replace
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd
import pytest
import torch
from fastapi.testclient import TestClient
from PIL import Image

from bcs import inference
from bcs.calibration.regression import CalibratedRegression, ConformalInterval
from bcs.contract import ISSUE_OOD_WARN, QualityReport, validate_prediction_dict
from bcs.eval.report import synthetic_predictions
from bcs.models.efficientnet import EfficientNetScorer, sha256_file
from bcs.models.train_zen import load_config, make_transforms
from bcs.ood.gate import MahalanobisGate, OODModelMismatch
from bcs.paths import REPO_ROOT
from bcs.quality.stub import AcceptAllChecker
from bcs.scale import load_scale

FIXTURES = Path(__file__).parent / "fixtures"
ZEN = load_scale("dairy_zen")
WEB = REPO_ROOT / "web"


@pytest.fixture
def zen_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (inference.MODEL_ENV, "BCS_SCALE", inference.OOD_ENV):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture(scope="module")
def pipe() -> inference.Pipeline:
    with pytest.MonkeyPatch.context() as mp:
        for name in (inference.MODEL_ENV, "BCS_SCALE", inference.OOD_ENV):
            mp.delenv(name, raising=False)
        return inference.model_pipeline()


def test_default_model_is_zen(zen_env: None) -> None:
    spec = inference.model_spec()
    assert spec.name == "efficientnet_b0_zen" and spec.scale == "dairy_zen"
    assert spec.weights == inference.WEIGHTS_ZEN and spec.ood == inference.OOD_ZEN
    assert inference.model_scale() == "dairy_zen"
    assert inference.model_spec("sdb").weights == inference.WEIGHTS
    with pytest.raises(ValueError, match="BCS_MODEL"):
        inference.model_spec("otro")


def test_artifacts_in_repo_are_consistent() -> None:
    sha = sha256_file(inference.WEIGHTS_ZEN)
    assert sha.startswith("bd938c68")  # el candidato A de T25
    meta = json.loads(inference.WEIGHTS_ZEN.with_suffix(".json").read_text())
    assert meta["weights_sha256"] == sha and meta["letterbox"] is True
    cal = CalibratedRegression.load(inference.CALIBRATION_ZEN)
    assert cal.grid == ZEN.grid and cal.alpha == pytest.approx(0.1)
    assert cal.info["variant"] == "ml_deg0_class_min20" and cal.conformal.min_class_n == 20
    assert len(cal.conformal.thresholds) == len(ZEN.grid)
    gate = MahalanobisGate.load(inference.OOD_ZEN, model_sha256=sha)
    assert gate.mean.shape == (1280,) and gate.precision.shape == (1280, 1280)
    assert 0 < gate.warn < gate.reject < np.inf
    assert gate.info["bank"]["sdb_bcs_train"] > 0 and gate.info["bank"]["zen_rgbd_train"] > 0
    # el modelo viejo y su calibración siguen en su lugar
    assert sha256_file(inference.WEIGHTS).startswith("5a5b28d6")
    assert CalibratedRegression.load(inference.CALIBRATION).grid == load_scale("dairy_sdb").grid


def test_zen_pipeline(pipe: inference.Pipeline, zen_env: None) -> None:
    assert pipe.scale.name == "dairy_zen" and pipe.ood is not None
    version = inference.model_version()
    assert version.startswith("efficientnet_b0_zen-bd938c68+cal-") and "+ood-" in version
    assert pipe.scorer.model_version == version


def test_ood_off_and_bank_of_other_model(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv("BCS_SCALE", raising=False)
    monkeypatch.setenv(inference.OOD_ENV, "off")
    assert inference.model_pipeline(model="zen").ood is None
    assert "+ood-" not in inference.model_version("zen")
    monkeypatch.delenv(inference.OOD_ENV)
    other = tmp_path / "ood.npz"
    MahalanobisGate.fit(np.random.default_rng(0).normal(size=(50, 4)), "f" * 64).save(other)
    monkeypatch.setitem(inference.MODELS, "zen", replace(inference.MODELS["zen"], ood=other))
    with pytest.raises(OODModelMismatch):
        inference.model_pipeline(model="zen")


def test_scale_must_match_model(zen_env: None) -> None:
    with pytest.raises(ValueError, match="dairy_zen"):
        inference.model_pipeline(load_scale("dairy_sdb"))


def test_letterbox_is_the_training_transform() -> None:
    scorer = EfficientNetScorer(ZEN.grid, {"model": {"weights": str(inference.WEIGHTS_ZEN)}})
    assert scorer.checkpoint["letterbox"] is True
    _, eval_tf = make_transforms(load_config())
    img = Image.open(FIXTURES / "rear_cow.jpg").convert("RGB").crop((0, 0, 150, 300))  # vertical
    assert torch.equal(scorer._tf(img), eval_tf(img))  # noqa: SLF001
    # el viejo sigue estirando (train_cnn.EVAL_TF)
    old = EfficientNetScorer(load_scale("dairy_sdb").grid)
    assert old.checkpoint.get("letterbox") is None
    from bcs.models.efficientnet import team_module

    assert old._tf is team_module("train_cnn").EVAL_TF  # noqa: SLF001


def test_one_pass_gives_value_and_pooling(pipe: inference.Pipeline) -> None:
    img = Image.open(FIXTURES / "rear_cow.jpg").convert("RGB")
    effnet = pipe.scorer.effnet  # type: ignore[attr-defined]
    views, feats = effnet.predict_views_features([img, img.transpose(Image.Transpose.ROTATE_90)])
    assert feats.shape == (2, 1280) and feats.dtype == np.float32
    assert np.allclose(views, effnet.predict_views([img, img.transpose(Image.Transpose.ROTATE_90)]))
    probs, f2 = pipe.scorer.predict_proba_features([img])  # type: ignore[attr-defined]
    assert np.allclose(probs, pipe.scorer.predict_proba([img]))
    assert np.allclose(f2[0], feats[0], atol=1e-5)


def test_end_to_end_contract(pipe: inference.Pipeline) -> None:
    """Foto por el pipeline (sin detector: el fixture es un dibujo) → contrato en 3.0–5.0."""
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
        (2.6, "low_confidence", inference.EDGE_LOW),
        (3.0, "low_confidence", inference.EDGE_LOW),
        (3.26, "ok", None),
        (4.5, "ok", None),
        (4.76, "ok", None),
        (5.3, "low_confidence", inference.EDGE_HIGH),
    ],
)
def test_edges_are_3_and_5(
    pipe: inference.Pipeline, value: float, status: str, note: str | None
) -> None:
    """Regla de bordes de G3 en la grilla nueva: 3.0 y 5.0 → `low_confidence` (no 3.25/4.25)."""
    assert pipe.ood is not None
    fake = FixedValueFeatures(value, pipe.ood.mean)  # pooling en la media del banco: ID
    scorer = inference.CalibratedEfficientNet(fake, pipe.scorer.dist, "test")  # type: ignore[attr-defined]
    p = replace(pipe, scorer=scorer, quality=AcceptAllChecker(), locate=lambda im: im)
    pred = inference.run_pipeline(p, (FIXTURES / "rear_cow.jpg").read_bytes(), "A1")
    assert pred.quality.issues == []
    if status == "ok":
        assert pred.confidence is not None and pred.confidence >= 0.5
        assert pred.interval is not None and (pred.interval[1] - pred.interval[0]) / 0.25 <= 4
    assert pred.status == status
    assert inference.edge_note(pred, ZEN.step) == note
    assert validate_prediction_dict(pred.to_dict(), ZEN.keys, ZEN.step) == []


def test_ood_warning_with_real_bank(pipe: inference.Pipeline) -> None:
    """Un pooling apenas por encima del umbral de aviso → `low_confidence` con el motivo."""
    gate = pipe.ood
    assert gate is not None
    direction = np.random.default_rng(0).normal(size=1280).astype(np.float32)
    unit = gate.score(gate.mean + direction)[0]
    for target, status in ((0.5 * gate.warn, "ok"), (1.05 * gate.warn, "low_confidence")):
        feat = gate.mean + direction * np.float32(np.sqrt(target / unit))
        dist = pipe.scorer.dist  # type: ignore[attr-defined]
        scorer = inference.CalibratedEfficientNet(FixedValueFeatures(3.76, feat), dist, "test")
        p = replace(pipe, scorer=scorer, quality=AcceptAllChecker(), locate=lambda im: im)
        pred = inference.run_pipeline(p, (FIXTURES / "rear_cow.jpg").read_bytes(), "A1")
        assert pred.status == status
        assert pred.quality.issues == ([ISSUE_OOD_WARN] if status != "ok" else [])


# ---------------------------------------------------------------- calibración


def test_conformal_min_class_n_falls_back_to_marginal() -> None:
    grid = (3.0, 3.25, 3.5)
    rng = np.random.default_rng(0)
    y = np.r_[np.zeros(200, int), np.ones(200, int), np.full(5, 2)]
    p = rng.dirichlet([2, 2, 2], size=len(y))
    strict = ConformalInterval(grid)
    strict.fit(p, y)
    assert strict.thresholds[2] == 1.0  # 5 casos: la clase entra siempre
    fb = ConformalInterval(grid, min_class_n=20)
    fb.fit(p, y)
    marginal = ConformalInterval(grid, by_class=False)
    marginal.fit(p, y)
    assert fb.thresholds[2] == pytest.approx(marginal.thresholds[0])
    assert fb.thresholds[:2] == pytest.approx(strict.thresholds[:2])
    cal = CalibratedRegression(grid, min_class_n=20)
    back = CalibratedRegression.from_dict(json.loads(json.dumps(cal.to_dict())))
    assert back.conformal.min_class_n == 20


# ---------------------------------------------------------------- API y web


@pytest.fixture
def api(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    from bcs.api import create_app
    from bcs.store.sqlite import SqliteStore

    monkeypatch.setenv("BCS_APP_TOKEN", "t")
    for name in (inference.MODEL_ENV, "BCS_SCALE"):
        monkeypatch.delenv(name, raising=False)
    store = SqliteStore(tmp_path / "bcs.sqlite", tmp_path / "media")
    with TestClient(create_app(store=store), headers={"X-App-Token": "t"}) as c:
        yield c


def test_distribution_uses_zen_grid(api: TestClient) -> None:
    d = api.get("/cows/distribution").json()
    assert d["scale"] == "dairy_zen" and d["grid"] == list(ZEN.grid)
    assert list(d["counts"]) == list(ZEN.keys)


def test_web_range_and_ood_text() -> None:
    index, rodeo = (WEB / "index.html").read_text(), (WEB / "rodeo.html").read_text()
    for html in (index, rodeo):
        assert "const VALID_MIN = 3.0, VALID_MAX = 5.0;" in html
        assert "3.25 a 4.25" not in html and "3.25–4.25" not in html
    assert f'"{ISSUE_OOD_WARN}"' in index  # = contract.WARNING_ISSUES
    assert 'id="res-warn"' in index and "≤ " in index
    assert "«≤ 3.00»" in rodeo and "const label = (k)" in rodeo


# ---------------------------------------------------------------- eval y eval-final


def _load(path: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(Path(path).stem, REPO_ROOT / path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    for name in ("data", "artifacts", "runs", "holdout"):
        (tmp_path / name).mkdir()
    for var, name in (("BCS_DATA_DIR", "data"), ("BCS_ARTIFACTS_DIR", "artifacts"),
                      ("BCS_RUNS_DIR", "runs"), ("BCS_HOLDOUT_DIR", "holdout")):  # fmt: skip
        monkeypatch.setenv(var, str(tmp_path / name))
    monkeypatch.delenv("BCS_TASK_ID", raising=False)
    return tmp_path


def test_make_eval_reports_by_source(env: Path) -> None:
    ev = _load("eval/evaluate.py")
    rng = np.random.default_rng(0)
    n = 60
    y = np.array(ZEN.grid)[rng.integers(0, len(ZEN.grid), n)]
    probs = np.full((n, len(ZEN.grid)), 0.02)
    probs[np.arange(n), [ZEN.grid.index(v) for v in y]] = 1 - 0.02 * (len(ZEN.grid) - 1)
    src = ["sdb_bcs"] * 40 + ["zen_rgbd"] * 20
    df = synthetic_predictions(
        y, probs, ZEN, source=src, group_id=[f"{s}:{i // 2}" for i, s in enumerate(src)]
    )
    df["fold"] = [f"cf{i % 5}" for i in range(n)]
    path = env / "preds.csv"
    df.to_csv(path, index=False)
    code = ev.main(["--model", "z", "--stage", "m3", "--predictions", str(path), "--task", "T28t",
                    "--scale", "dairy_zen"])  # fmt: skip
    assert code == 0
    (run,) = (env / "runs" / "T28t").iterdir()
    rep = json.loads((run / "report.json").read_text())
    assert set(rep["by_source"]) == {"sdb_bcs", "zen_rgbd"}
    assert rep["by_source"]["zen_rgbd"]["n"] == 20
    assert rep["by_source"]["sdb_bcs"]["thresholds"]["passed"] is True


def test_eval_final_reads_zen_labels_and_raw_frames(tmp_path: Path) -> None:
    ef = _load("eval/evaluate_final.py")
    data = tmp_path / "data"
    rgb = data / "raw" / "zen_rgbd" / "rgb" / "27.03.2025" / "565" / "rgb"
    rgb.mkdir(parents=True)
    Image.new("RGB", (32, 24)).save(rgb / "11.png")
    pd.DataFrame(
        {
            "image_id": ["zen_rgbd_a", "zen_rgbd_b"],
            "rgb_path": ["27.03.2025/565/rgb/11.png", "x.png"],
        }
    ).to_csv(data / "raw" / "zen_rgbd" / "plan.csv", index=False)
    assert ef.raw_path(data, "zen_rgbd_a") == rgb / "11.png"
    assert ef.raw_path(data, "zen_rgbd_b") is None  # en el plan pero sin bajar
    assert ef.raw_path(data, "zen_rgbd_c") is None

    cols = {"path": "p.jpg", "group_id": "g", "bcs": 3.0}
    sdb = pd.DataFrame([{"image_id": "sdb_bcs_GS_1_1", "source": "sdb_bcs", **cols}])
    zen = pd.DataFrame([{"image_id": "zen_rgbd_a", "source": "zen_rgbd", **cols}])
    sdb.to_csv(tmp_path / "s.csv", index=False)
    zen.to_csv(tmp_path / "z.csv", index=False)
    df = ef.load_all_labels([tmp_path / "s.csv", tmp_path / "z.csv"], data=data)
    assert df["source"].tolist() == ["sdb_bcs", "zen_rgbd"]
    assert ef._origin(df["image_id"], df["source"]).tolist() == ["GS", "zen_rgbd"]  # noqa: SLF001
    with pytest.raises(ef.FinalEvalError, match="duplicado"):
        ef.load_all_labels([tmp_path / "z.csv", tmp_path / "z.csv"], data=data)
    # una imagen de train de zen no puede entrar
    pd.DataFrame({"image_id": ["zen_rgbd_a"], "split": ["train"]}).to_csv(
        data / "splits_zen_rgbd.csv", index=False
    )
    with pytest.raises(ef.FinalEvalError, match="train"):
        ef.load_all_labels([tmp_path / "z.csv"], data=data)


def test_quality_report_passed_semantics_unchanged() -> None:
    """Sin avisos, `passed` es lo de siempre: cualquier motivo rechaza."""
    assert QualityReport(True, 1.0, []).passed
    assert not QualityReport(True, 1.0, ["imagen borrosa"]).passed
    assert not QualityReport(False, 1.0, []).passed
    assert not QualityReport(False, 1.0, [ISSUE_OOD_WARN]).passed
