"""T25: letterbox, heurística de pelaje y métricas de `train_zen`."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from bcs.models.train_zen import (
    Letterbox,
    _contrast,
    build_report,
    load_config,
    metrics,
    paired_delta,
    red_coat_fraction,
    render_tables,
)

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


def test_build_report_and_tables(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BCS_ARTIFACTS_DIR", str(tmp_path))
    d = tmp_path / "T25"
    d.mkdir()
    recs = [  # (source, animal, fecha, y)
        ("sdb_bcs", "s1", "", 3.25),
        ("sdb_bcs", "s1", "", 3.25),
        ("sdb_bcs", "s2", "", 4.0),
        ("zen_rgbd", "z1", "2025-03-27", 4.5),
        ("zen_rgbd", "z2", "2025-03-27", 4.0),
        ("zen_rgbd", "z3", "2025-02-20", 3.0),
    ]
    val = pd.DataFrame(
        [
            {"image_id": f"i{k}", "path": f"/no/{k}.jpg", "source": s, "group_id": g,
             "date": dt, "bcs": y, "y": y}
            for k, (s, g, dt, y) in enumerate(recs)
        ]
    )  # fmt: skip
    val.to_csv(d / "manifest_val.csv", index=False)
    pd.DataFrame({"image_id": val["image_id"], "red_frac": [0, 0, 0, 0.9, 0.1, 0]}).to_csv(
        d / "coat_val.csv", index=False
    )
    for name, shift in (("prod", 0.0), ("A", 0.25)):
        p = val.drop(columns="path").assign(pred=val["y"] + shift)
        p.to_csv(d / f"preds_val_{name}.csv", index=False)
    rep = build_report(load_config(), ["prod", "A"])
    assert rep["by_source"]["prod"]["sdb_bcs"]["mae"] == 0.0
    assert np.isclose(rep["by_source"]["A"]["zen_rgbd"]["bias"], 0.25)
    assert rep["by_class"]["prod"]["sdb_bcs"]["3.25"]["animals"] == 1
    assert rep["sdb_animal"]["prod"]["n"] == 2
    assert set(rep["by_date"]["A"]) == {"2025-03-27", "2025-02-20"}
    assert rep["contrast_2703"]["prod"]["auc"] == 1.0
    coat = rep["coat_4.0-4.25"]["prod"]
    assert set(coat) == {"zen_rgbd"} and coat["zen_rgbd"]["no colorada"]["n"] == 1
    md = render_tables(rep)
    assert "### zen_rgbd val, por fecha" in md and "| 2025-03-27 |" in md
    assert "Contraste dentro del 2025-03-27" in md
    d = rep["delta_vs_prod"]["A"]["sdb_bcs"]
    assert np.isclose(d["delta"], 0.25) and d["animals"] == 2
    assert "| sdb_bcs 3.25 | A | +0.250 |" in md


def test_paired_delta_bootstrap_by_animal() -> None:
    y = np.repeat([3.5, 4.0, 3.25, 3.75], 5)
    groups = np.repeat(["a", "b", "c", "d"], 5)
    base = pd.DataFrame({"y": y, "pred": y + 0.2, "group_id": groups})
    same = paired_delta(base, base)
    assert same == {"delta": 0.0, "lo": 0.0, "hi": 0.0, "animals": 4}
    better = base.assign(pred=y + np.repeat([0.0, 0.1, 0.2, 0.0], 5))
    d = paired_delta(base, better)
    assert np.isclose(d["delta"], np.mean([-0.2, -0.1, 0.0, -0.2]))
    assert d["lo"] <= d["delta"] <= d["hi"] <= 0.0
