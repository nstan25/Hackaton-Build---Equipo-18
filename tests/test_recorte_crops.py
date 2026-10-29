"""T30: recortes de zen como en el uso real (`scripts/zen_crops/`) y su evaluación (sintético)."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from bcs.models.efficientnet import team_module

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts" / "zen_crops"


@pytest.fixture(scope="module")
def zc() -> Any:
    sys.path.insert(0, str(SCRIPTS))
    try:
        return importlib.import_module("crops")
    finally:
        sys.path.remove(str(SCRIPTS))


@pytest.fixture(scope="module")
def ev() -> Any:
    sys.path.insert(0, str(SCRIPTS))
    try:
        return importlib.import_module("evaluate")
    finally:
        sys.path.remove(str(SCRIPTS))


def test_expand_without_extra_is_production_crop(zc: Any) -> None:
    prep_crops = team_module("prep_crops")
    for box in ([100, 200, 400, 380], [0, 0, 50, 60], [900, 500, 1020, 575]):
        assert zc.expand(box, 1024, 576) == prep_crops.expand_bbox(box, 1024, 576)


def test_expand_up_grows_only_upwards(zc: Any) -> None:
    base = zc.expand([100, 300, 300, 400], 1000, 1000)
    up = zc.expand([100, 300, 300, 400], 1000, 1000, up=0.5)
    assert up[0] == base[0] and up[2] == base[2] and up[3] == base[3]
    assert up[1] == base[1] - 50
    assert zc.expand([100, 30, 300, 400], 1000, 1000, up=5.0)[1] == 0  # recortado a la imagen


def test_solo_box_adds_side_margin_and_clips(zc: Any) -> None:
    assert zc.solo_box((700, 0, 1200, 1000), 1920, 1080) == (650, 0, 1250, 1000)
    assert zc.solo_box((10, 5, 510, 1080), 1920, 1080) == (0, 5, 560, 1080)


def test_crop_from_solo(zc: Any) -> None:
    solo = Image.new("RGB", (600, 1000), (120, 100, 90))
    assert zc.crop_from_solo(solo, "") is None
    c = zc.crop_from_solo(solo, "100.0,500.0,500.0,850.0")
    assert c is not None and max(c.size) <= zc.MAX_SIDE
    tall = zc.crop_from_solo(solo, "100.0,500.0,500.0,850.0", up=1.0)
    assert tall is not None and tall.height / tall.width > c.height / c.width
    assert zc.half_box(600, 1000) == (0, 450, 600, 1000)


def _preds(pred: list[float], kind: list[str] | None = None) -> pd.DataFrame:
    n = len(pred)
    return pd.DataFrame(
        {
            "image_id": [f"i{i}" for i in range(n)],
            "group_id": [f"g{i // 2}" for i in range(n)],
            "y": [3.5] * n,
            "pred": pred,
            "kind": kind or ["det"] * n,
        }
    )


def test_metrics_exclude_rejected(ev: Any) -> None:
    m = ev.metrics(_preds([3.5, 3.75, np.nan, 4.5], ["det", "det", "rejected", "det"]))
    assert m["n"] == 3 and m["n_all"] == 4
    assert m["rejected"] == pytest.approx(0.25)
    assert m["mae"] == pytest.approx((0 + 0.25 + 1.0) / 3)
    assert m["exact"] == pytest.approx(1 / 3) and m["pm1"] == pytest.approx(2 / 3)


def test_paired_delta_same_rows_and_sign(ev: Any) -> None:
    base = _preds([4.0, 4.0, 4.0, 4.0])
    better = _preds([3.5, 3.6, 3.5, np.nan], ["det", "det", "det", "rejected"])
    d = ev.paired_delta(base, better)
    assert d["delta"] == pytest.approx(-(0.5 + 0.4 + 0.5) / 3)  # la fila rechazada no cuenta
    assert d["hi"] < 0 and d["animals"] == 2
