"""T25: manifiestos de `train_zen` sobre datos sintéticos (sin imágenes reales)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from bcs.models.train_zen import build_manifests, load_config, selection_rows, variant_rows


def _write_source(root: Path, src: str, rows: list[dict[str, object]]) -> None:
    labels = pd.DataFrame(rows)
    labels["path"] = [f"images/{src}/{i}.jpg" for i in labels["image_id"]]
    labels["source"] = src
    labels["animal_id"] = labels["group_id"]
    for c in ("scale", "view", "scorer", "bbox"):
        labels[c] = ""
    split = labels[["image_id", "split"]]
    labels[labels["split"] != "test"].drop(columns="split").to_csv(
        root / f"labels_{src}.csv", index=False
    )
    split.to_csv(root / f"splits_{src}.csv", index=False)


@pytest.fixture
def data_root(tmp_path: Path) -> Path:
    sdb = []
    for g in range(6):
        split = "val" if g == 5 else "train"
        for f in range(1, 13):
            sdb.append(
                {"image_id": f"sdb_bcs_GS_{g}_{f}", "group_id": f"sdb_bcs:cow_{g}",
                 "bcs": [3.25, 3.5, 3.75, 4.0, 4.25][g % 5], "date": "", "split": split}
            )  # fmt: skip
    zen = []
    spec = [  # (animal, fecha, bcs, split)
        (1, "2024-12-06", 2.75, "train"),
        (2, "2024-12-06", 3.0, "train"),
        (3, "2025-02-20", 3.25, "train"),
        (4, "2025-03-27", 4.5, "train"),
        (5, "2025-03-27", 5.0, "train"),
        (6, "2025-03-27", 4.0, "val"),
        (7, "2025-03-27", 4.75, "test"),
    ]
    for a, d, b, s in spec:
        for f in range(3):
            zen.append(
                {"image_id": f"zen_rgbd_{a}_{f}", "group_id": f"zen_rgbd:{a}", "bcs": b,
                 "date": d, "split": s}
            )  # fmt: skip
    _write_source(tmp_path, "sdb_bcs", sdb)
    _write_source(tmp_path, "zen_rgbd", zen)
    return tmp_path


def test_manifest_train_val_only_and_subsample(data_root: Path) -> None:
    cfg = load_config()
    train, val = build_manifests(cfg, data_root)
    assert set(train["split"]) == {"train"} and set(val["split"]) == {"val"}
    assert not train["image_id"].str.startswith("zen_rgbd_7").any()  # test nunca entra
    assert not val["image_id"].str.startswith("zen_rgbd_7").any()
    assert not set(train["group_id"]) & set(val["group_id"])
    sdb = train[train["source"] == "sdb_bcs"]
    k = cfg["data"]["sdb_max_per_group"]
    assert sdb.groupby("group_id").size().max() == k
    # frames repartidos a lo largo de la secuencia: el primero y el último de cada grupo
    ids = sdb.loc[sdb["group_id"] == "sdb_bcs:cow_0", "image_id"]
    frames = sorted(int(i.rsplit("_", 1)[-1]) for i in ids)
    assert frames[0] == 1 and frames[-1] == 12
    # val de sdb no se submuestrea (el reporte usa todo val)
    assert len(val[val["source"] == "sdb_bcs"]) == 12
    assert all(Path(p).is_absolute() for p in train["path"])


def test_tail_collapsed_to_grid(data_root: Path) -> None:
    train, _ = build_manifests(load_config(), data_root)
    zen1 = train[train["group_id"] == "zen_rgbd:1"]
    assert set(zen1["bcs"]) == {2.75} and set(zen1["y"]) == {3.0}
    assert train["y"].min() == 3.0 and train["y"].max() == 5.0


def test_leakage_raises(data_root: Path) -> None:
    sp = pd.read_csv(data_root / "splits_zen_rgbd.csv")
    sp.loc[sp["image_id"] == "zen_rgbd_6_0", "split"] = "train"  # mismo animal en train y val
    sp.to_csv(data_root / "splits_zen_rgbd.csv", index=False)
    with pytest.raises(ValueError, match="group_id"):
        build_manifests(load_config(), data_root)


def test_variants_dates_and_source_mix(data_root: Path) -> None:
    cfg = load_config()
    train, _ = build_manifests(cfg, data_root)
    a, b = variant_rows(train, cfg, "A"), variant_rows(train, cfg, "B")
    assert "2024-12-06" not in set(a["date"]) and "2024-12-06" in set(b["date"])
    for rows in (a, b):
        w = rows["w"].to_numpy(dtype=float)
        assert np.isclose(w.sum(), 1.0)
        for src in ("sdb_bcs", "zen_rgbd"):
            assert np.isclose(w[(rows["source"] == src).to_numpy()].sum(), 0.5)
    # B: descontado el balanceo por clase, una imagen de 06.12 pesa `date_weight` veces otra
    zen = b.loc[b["source"] == "zen_rgbd"]
    n = zen["y"].map(zen["y"].value_counts()).to_numpy(dtype=float)
    p = cfg["data"]["class_balance_power"]["zen_rgbd"]
    raw = zen["w"].to_numpy(dtype=float) * n**p
    is612 = (zen["date"] == "2024-12-06").to_numpy()
    dw = cfg["variants"]["B"]["date_weight"]["2024-12-06"]
    assert np.allclose(raw[is612] / raw[~is612].mean(), dw)


def test_selection_rows_subsamples_sdb_only(data_root: Path) -> None:
    cfg = load_config()
    _, val = build_manifests(cfg, data_root)
    sel = selection_rows(val, cfg)
    assert len(sel[sel["source"] == "sdb_bcs"]) == cfg["data"]["select_sdb_max_per_group"]
    assert len(sel[sel["source"] == "zen_rgbd"]) == len(val[val["source"] == "zen_rgbd"])
