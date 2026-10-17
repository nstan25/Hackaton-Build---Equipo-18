"""T09: baseline tonto, desaturación de la pintura roja y su medición (G1 punto 2)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from bcs.data.merge import paint_fractions
from bcs.models.base import Scorer
from bcs.models.baseline import PriorScorer, desaturate_red, paint_report, red_mask
from bcs.scale import load_scale

GRID = load_scale("dairy_sdb").grid


def _painted_image() -> Image.Image:
    arr = np.full((20, 20, 3), 120, dtype=np.uint8)
    arr[5:10, 5:10] = (220, 30, 20)  # rojo-naranja saturado
    arr[12:15, 12:15] = (30, 30, 200)  # azul: no se toca
    return Image.fromarray(arr)


def test_prior_scorer_predicts_train_mode(tmp_path: Path) -> None:
    s = PriorScorer(GRID)
    assert isinstance(s, Scorer)
    s.fit(pd.DataFrame({"bcs": [3.5, 3.5, 3.5, 4.0, 3.25]}))
    p = s.predict_proba([_painted_image()] * 3)
    assert p.shape == (3, len(GRID)) and np.allclose(p.sum(1), 1)
    assert (np.asarray(GRID)[p.argmax(1)] == 3.5).all()
    s.save(tmp_path / "prior.json")
    t = PriorScorer(GRID)
    t.load(tmp_path / "prior.json")
    assert np.allclose(t.prior, s.prior)


def test_desaturate_red_only_touches_red_pixels(tmp_path: Path) -> None:
    im = _painted_image()
    mask = red_mask(im)
    assert mask.sum() == 25
    path = tmp_path / "p.png"
    im.save(path)
    assert mask.mean() == pytest.approx(paint_fractions(path)[0])  # misma definición que T06
    out = np.asarray(desaturate_red(im)).astype(int)
    before = np.asarray(im).astype(int)
    assert not red_mask(Image.fromarray(out.astype(np.uint8))).any()
    red_px = out[mask]
    assert (red_px.max(1) - red_px.min(1) <= 2).all()  # gris
    assert np.abs(out[~mask] - before[~mask]).max() <= 2  # el resto queda igual


def test_paint_report_measures_bias_and_delta() -> None:
    y = np.array([3.25, 3.25, 3.25, 4.25, 4.25, 4.25])
    painted = np.array([True, True, False, True, False, False])
    pred = np.array([3.5, 3.5, 3.25, 4.0, 4.25, 4.25])  # las pintadas, sesgadas
    value = pred.copy()
    value_d = value + np.where(painted, 0.2, 0.0)
    r = paint_report(y, pred, value, value_d, painted, GRID)
    assert r["painted"]["n"] == 3 and r["unpainted"]["n"] == 3
    assert r["painted"]["mae"] == pytest.approx(0.25) and r["unpainted"]["mae"] == 0
    assert r["by_class"]["3.25"]["painted"]["bias"] == pytest.approx(0.25)
    assert r["by_class"]["4.25"]["painted"]["bias"] == pytest.approx(-0.25)
    assert r["delta_painted_mean"] == pytest.approx(0.2)
    assert r["pred_changed_painted"] == pytest.approx(1.0)  # 3.5→3.7 y 4.0→4.2 cambian de clase
    assert r["by_class"]["3.50"]["painted"] == {"n": 0}
