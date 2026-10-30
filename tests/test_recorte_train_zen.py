"""T30: `train_zen` con recortes de grupa en zen (`train_zen_grupa.yaml`), sin datos reales."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from bcs.models import train_zen
from bcs.models.train_zen import build_manifests, load_config, variant_rows
from bcs.paths import config_dir


def _write_source(root: Path, src: str, rows: list[dict[str, object]]) -> None:
    labels = pd.DataFrame(rows)
    labels["path"] = [f"images/{src}/{i}.jpg" for i in labels["image_id"]]
    labels["source"] = src
    labels["animal_id"] = labels["group_id"]
    for c in ("scale", "view", "scorer", "bbox"):
        labels[c] = ""
    labels[labels["split"] != "test"].drop(columns="split").to_csv(
        root / f"labels_{src}.csv", index=False
    )
    labels[["image_id", "split"]].to_csv(root / f"splits_{src}.csv", index=False)


@pytest.fixture
def setup(tmp_path: Path) -> tuple[Path, dict[str, Any]]:
    sdb = [
        {"image_id": f"sdb_bcs_GS_{g}_{f}", "group_id": f"sdb_bcs:cow_{g}",
         "bcs": [3.25, 3.5, 3.75, 4.0, 4.25][g % 5], "date": "",
         "split": "val" if g == 5 else "train"}
        for g in range(6) for f in range(1, 4)
    ]  # fmt: skip
    spec = [(1, "2024-12-06", 3.0, "train"), (2, "2025-02-20", 3.25, "train"),
            (3, "2025-03-27", 4.5, "train"), (4, "2025-03-27", 4.0, "val"),
            (5, "2025-03-27", 4.75, "test")]  # fmt: skip
    zen = [
        {"image_id": f"zen_rgbd_{a}_{f}", "group_id": f"zen_rgbd:{a}", "bcs": b, "date": d,
         "split": s}
        for a, d, b, s in spec for f in range(3)
    ]  # fmt: skip
    _write_source(tmp_path, "sdb_bcs", sdb)
    _write_source(tmp_path, "zen_rgbd", zen)
    # recortes: todos los de train/val; zen_rgbd_2_0 y zen_rgbd_4_2 sin detección (respaldo)
    ids = [r["image_id"] for r in zen if r["split"] != "test"]
    man = pd.DataFrame(
        {
            "image_id": ids,
            "path": [f"/crops/grupa/{i}.jpg" for i in ids],
            "kind": ["half" if i in ("zen_rgbd_2_0", "zen_rgbd_4_2") else "det" for i in ids],
        }
    )
    man.to_csv(tmp_path / "manifest.csv", index=False)
    cfg = load_config(config_dir() / "train_zen_grupa.yaml")
    cfg["data"]["zen_crops"] = str(tmp_path / "manifest.csv")
    return tmp_path, cfg


def test_grupa_config_is_t30_and_starts_from_dev_model() -> None:
    cfg = load_config(config_dir() / "train_zen_grupa.yaml")
    assert cfg["task"] == "T30"
    assert cfg["init_weights"] == "models/bcs_efficientnet_b0_zen.pt"
    assert cfg["export_name"].startswith("bcs_efficientnet_b0_zen_")
    assert cfg["export_name"] != "bcs_efficientnet_b0_zen.pt"  # nunca pisa el de dev


def test_set_task_switches_output_dir() -> None:
    try:
        train_zen.set_task({"task": "T30"})
        assert train_zen.out_dir().name == "T30"
        train_zen.set_task({})
        assert train_zen.out_dir().name == "T25"  # default: lo de T25 sigue igual
    finally:
        train_zen.set_task({})


def test_t25_config_has_no_frames(setup: tuple[Path, dict[str, Any]]) -> None:
    root, _ = setup
    train, val = build_manifests(load_config(), root)
    assert "frame" not in train.columns
    assert all("/crops/" not in p for p in train["path"])


def test_grupa_swaps_zen_paths_and_val_keeps_only_detections(
    setup: tuple[Path, dict[str, Any]],
) -> None:
    root, cfg = setup
    train, val = build_manifests(cfg, root)
    g = variant_rows(train, cfg, "G")
    zen = g[g["source"] == "zen_rgbd"]
    assert set(zen["frame"]) == {"grupa"}
    assert all(p.startswith("/crops/grupa/") for p in zen["path"])
    assert "zen_rgbd_2_0" in set(zen["image_id"])  # el respaldo entra en train
    assert "2024-12-06" not in set(g["date"])
    assert all("/crops/" not in p for p in g.loc[g["source"] == "sdb_bcs", "path"])
    zv = val[val["source"] == "zen_rgbd"]
    assert set(zv["image_id"]) == {"zen_rgbd_4_0", "zen_rgbd_4_1"}  # sin la que no detectó
    assert not val["image_id"].str.startswith("zen_rgbd_5").any()  # test nunca
    assert all(p.startswith("/crops/grupa/") for p in zv["path"])


def test_mix_has_both_frames_half_weight_each(setup: tuple[Path, dict[str, Any]]) -> None:
    root, cfg = setup
    train, _ = build_manifests(cfg, root)
    m = variant_rows(train, cfg, "M")
    zen = m[m["source"] == "zen_rgbd"]
    assert set(zen["frame"]) == {"grupa", "entera"}
    assert (zen.groupby("image_id").size() == 2).all()
    w = m["w"].to_numpy(dtype=float)
    assert np.isclose(w[(m["source"] == "zen_rgbd").to_numpy()].sum(), 0.5)
    wz = np.asarray(zen["w"], dtype=float)
    is_grupa = np.asarray(zen["frame"]) == "grupa"
    assert np.isclose(wz[is_grupa].sum(), wz[~is_grupa].sum())
    assert all("/crops/" not in p for p in zen.loc[zen["frame"] == "entera", "path"])


def test_bad_frame_raises(setup: tuple[Path, dict[str, Any]]) -> None:
    root, cfg = setup
    train, _ = build_manifests(cfg, root)
    cfg["variants"]["X"] = {"zen_frame": "otro"}
    with pytest.raises(ValueError, match="zen_frame"):
        variant_rows(train, cfg, "X")


def test_export_name_and_dest_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """T31: exportar otra variante sin pisar la exportada ni escribir en la carpeta de la tarea."""
    import torch

    monkeypatch.setenv("BCS_ARTIFACTS_DIR", str(tmp_path))
    cfg = load_config(config_dir() / "train_zen_grupa.yaml")
    try:
        train_zen.set_task(cfg)
        (tmp_path / "T30" / "M").mkdir(parents=True)
        torch.save({"state_dict": {}, "variant": "M"}, tmp_path / "T30" / "M" / "best.pt")
        dest = train_zen.export(cfg, "M", "bcs_x_mix.pt", tmp_path / "T31")
    finally:
        train_zen.set_task({})
    assert dest == tmp_path / "T31" / "bcs_x_mix.pt" and dest.exists()
    meta = json.loads(dest.with_suffix(".json").read_text())
    assert meta["variant"] == "M" and meta["weights"] == str(dest)
    assert not (tmp_path / "T30" / cfg["export_name"]).exists()
