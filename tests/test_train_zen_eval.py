"""T25: letterbox, heurística de pelaje y métricas de `train_zen`."""

from __future__ import annotations

import numpy as np
import pandas as pd
from PIL import Image

from bcs.models.train_zen import Letterbox, _contrast, metrics, red_coat_fraction

GRID = (3.0, 3.25, 3.5, 3.75, 4.0, 4.25, 4.5, 4.75, 5.0)


def test_letterbox_keeps_aspect() -> None:
    lb = Letterbox((224, 320))
    tall = lb(Image.new("RGB", (160, 320), (255, 255, 255)))  # vertical, como zen
    assert tall.size == (320, 224)
    a = np.asarray(tall)
    cols = np.where((a == 255).all(axis=2).any(axis=0))[0]
    assert len(cols) == 112  # 160 * 224 / 320: no se estira a lo ancho
    assert tuple(a[0, 0]) == lb.fill  # relleno con el gris de la media
    wide = lb(Image.new("RGB", (640, 224), (255, 255, 255)))
    rows = np.where((np.asarray(wide) == 255).all(axis=2).any(axis=1))[0]
    assert len(rows) == 112  # 640×224 → 320×112


def test_red_coat_fraction() -> None:
    red = Image.new("RGB", (100, 200), (150, 70, 40))  # colorado/marrón
    holstein = Image.new("RGB", (100, 200), (20, 20, 20))
    holstein.paste((235, 235, 235), (0, 0, 50, 200))
    assert red_coat_fraction(red) > 0.9
    assert red_coat_fraction(holstein) == 0.0
    gray_red = red.convert("L").convert("RGB")  # la misma vaca en gris ya no es colorada
    assert red_coat_fraction(gray_red) == 0.0


def test_metrics_values() -> None:
    y = np.array([3.0, 4.5, 5.0, 3.25])
    pred = np.array([3.1, 4.2, 4.6, 3.3])
    m = metrics(y, pred, np.array(["a", "b", "b", "c"]), GRID)
    assert m["n"] == 4 and m["animals"] == 3
    assert np.isclose(m["mae"], np.mean([0.1, 0.3, 0.4, 0.05]))
    # clases: 3.0, 4.25, 4.5, 3.25 → exactas 2/4; ±1 paso: 3/4 (5.0 → 4.5 falla)
    assert m["exact"] == 0.5 and m["pm1"] == 0.75
    assert np.isclose(m["bias"], np.mean([0.1, -0.3, -0.4, 0.05]))
    assert metrics(np.array([]), np.array([]), np.array([]), GRID)["n"] == 0


def test_contrast_auc() -> None:
    hi = pd.DataFrame({"y": [4.5, 5.0], "pred": [4.4, 4.6], "group_id": ["a", "b"]})
    mid = pd.DataFrame(
        {"y": [4.0, 4.25, 4.0], "pred": [4.0, 4.5, 3.9], "group_id": ["c", "d", "c"]}
    )
    c = _contrast(hi, mid)
    assert c["n_hi"] == 2 and c["animals_mid"] == 2
    assert np.isclose(c["true_delta"], 4.75 - np.mean([4.0, 4.25, 4.0]))
    assert np.isclose(c["auc"], 5 / 6)  # 4.4 pierde contra 4.5
    assert _contrast(hi.iloc[:0], mid) == {}
