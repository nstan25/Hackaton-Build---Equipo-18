"""T09: la EfficientNet-B0 del equipo como Scorer. El .pt está en el repo (models/); los tests que
lo cargan usan una o dos imágenes de fixture y corren en CPU en segundos."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from bcs.models.base import Scorer
from bcs.models.efficientnet import (
    WEIGHTS,
    EfficientNetScorer,
    discretized_normal,
    origin,
    team_module,
)
from bcs.paths import REPO_ROOT, config_dir, fixtures_dir
from bcs.scale import load_scale
from bcs.train import build_scorer, load_config

GRID = load_scale("dairy_sdb").grid


def test_discretized_normal_rows_sum_one_and_peak_at_value() -> None:
    p = discretized_normal(np.array([3.5, 3.76, 4.24]), 0.1, GRID)
    assert p.shape == (3, len(GRID)) and np.allclose(p.sum(1), 1)
    assert list(np.asarray(GRID)[p.argmax(1)]) == [3.5, 3.75, 4.25]
    tails = discretized_normal(np.array([2.0, 5.0]), 0.1, GRID)  # fuera de rango → puntas
    assert tails[0, 0] > 0.99 and tails[1, -1] > 0.99


def test_origin_from_image_id() -> None:
    ids = ["sdb_bcs_GS_1000_1", "sdb_bcs_YM_12_3", "sdb_bcs_L-i10454", "sdb_bcs_R-i7"]
    assert list(origin(ids)) == ["GS", "YM", "L-R", "L-R"]


def test_config_builds_efficientnet_scorer() -> None:
    cfg = load_config(config_dir() / "efficientnet_b0.yaml")
    s = build_scorer(cfg, GRID)
    assert isinstance(s, EfficientNetScorer) and isinstance(s, Scorer)
    assert s.tta and s.weights == REPO_ROOT / WEIGHTS


@pytest.fixture(scope="module")
def scorer() -> EfficientNetScorer:
    weights = REPO_ROOT / WEIGHTS
    assert weights.exists(), f"falta {weights} (viene en el repo, commit 7cac8ae)"
    return EfficientNetScorer(GRID, {"model": {"weights": WEIGHTS, "tta": True}})


def _rear() -> Image.Image:
    with Image.open(fixtures_dir() / "rear_cow.jpg") as im:
        return im.convert("RGB")


def test_real_model_matches_team_pipeline(scorer: EfficientNetScorer) -> None:
    """Mismo número que `bcs_pipeline.BCSScorer.score_crop` (TTA con flip, EVAL_TF, decode)."""
    import torch

    im = _rear()
    views = scorer.predict_views([im, im.transpose(Image.Transpose.FLIP_LEFT_RIGHT)])
    assert views.shape == (2, 2)
    assert views[0, 0] == pytest.approx(views[1, 1], abs=1e-4)  # flip de la imagen = vista flip
    train_cnn = team_module("train_cnn")
    x = train_cnn.EVAL_TF(im).unsqueeze(0)
    with torch.inference_mode():
        model = scorer._load_model()
        ref = float(train_cnn.decode((model(x) + model(torch.flip(x, dims=[3]))) / 2)[0])
    value = scorer.predict_value([im])
    assert value[0] == pytest.approx(ref, abs=1e-4)
    assert 2.5 < value[0] < 5.0


def test_real_model_proba_fit_save_load(scorer: EfficientNetScorer, tmp_path: Path) -> None:
    path = tmp_path / "rear.png"  # sin pérdida: misma imagen que `_rear()`
    _rear().save(path)
    df = pd.DataFrame({"path": [str(path)] * 2, "bcs": [3.5, 3.5], "split": ["train", "val"]})
    scorer.fit(df)
    value = scorer.predict_value([_rear()])[0]
    assert scorer.sigma_info["n"] == 1  # solo la fila de val
    assert scorer.sigma == pytest.approx(max(abs(value - 3.5), 1e-3), abs=1e-6)
    p = scorer.predict_proba([_rear()])
    assert p.shape == (1, len(GRID)) and np.isclose(p.sum(), 1)
    scorer.save(tmp_path / "s.json")
    other = EfficientNetScorer(GRID)
    other.load(tmp_path / "s.json")
    assert other.sigma == scorer.sigma and other.tta
    assert np.allclose(other.predict_proba([_rear()]), p)
