"""T25: entrenamiento reanudable de `train_zen` con imágenes sintéticas y pocos pasos (CPU)."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

from bcs.models.train_zen import epoch_indices, export, load_candidate, load_config, train


def _tiny_cfg() -> dict[str, Any]:
    cfg = copy.deepcopy(load_config())
    cfg["init_weights"] = None  # sin pesos de producción: arquitectura sola, sin bajar nada
    cfg["size"] = [32, 48]
    cfg["train"].update(
        epochs=2, patience=5, epoch_samples=8, batch_size=4, warmup_steps=1,
        num_workers=0, torch_threads=1, ckpt_every_steps=1, val_batch_size=8,
    )  # fmt: skip
    return cfg


@pytest.fixture
def rows(tmp_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(0)
    recs = []
    for i in range(12):
        src = "sdb_bcs" if i % 2 else "zen_rgbd"
        size = (60, 40) if src == "sdb_bcs" else (30, 60)
        p = tmp_path / f"{i}.jpg"
        Image.fromarray(rng.integers(0, 255, (size[1], size[0], 3), dtype=np.uint8)).save(p)
        recs.append(
            {"image_id": f"img{i}", "path": str(p), "source": src, "group_id": f"{src}:{i}",
             "y": [3.0, 3.5, 4.0, 5.0][i % 4], "date": "", "w": 1 / 12}
        )  # fmt: skip
    df = pd.DataFrame(recs)
    return df.iloc[:8].reset_index(drop=True), df.iloc[8:].reset_index(drop=True)


def test_train_writes_checkpoints_and_log(
    tmp_path: Path, rows: tuple[pd.DataFrame, pd.DataFrame]
) -> None:
    tr, va = rows
    dest = tmp_path / "run"
    st = train(_tiny_cfg(), "A", train_rows=tr, val_rows=va, dest=dest, log=lambda s: None)
    assert st["done"] and st["epoch"] == 2
    log = pd.read_csv(dest / "log.csv")
    assert list(log["epoch"]) == [1, 2]
    assert {"mae_sdb_bcs", "mae_zen_rgbd", "score"} <= set(log.columns)
    model, meta = load_candidate(dest / "best.pt")
    assert meta["letterbox"] and meta["center"] == 3.75 and meta["grid"][0] == 3.0
    assert meta["grid"][-1] == 5.0
    with torch.inference_mode():
        assert model(torch.zeros(1, 3, 32, 48)).shape == (1, 1)
    # ya terminada: relanzar no entrena de nuevo
    again = train(_tiny_cfg(), "A", train_rows=tr, val_rows=va, dest=dest, log=lambda s: None)
    assert again["log"] == st["log"]


def test_resume_mid_epoch(tmp_path: Path, rows: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    tr, va = rows
    dest = tmp_path / "run"
    msgs: list[str] = []
    st = train(
        _tiny_cfg(), "B", train_rows=tr, val_rows=va, dest=dest, max_steps=1, log=msgs.append
    )
    assert not st["done"] and st["epoch"] == 0 and st["batch"] == 1
    assert (dest / "last.pt").exists() and not (dest / "best.pt").exists()
    st = train(_tiny_cfg(), "B", train_rows=tr, val_rows=va, dest=dest, log=msgs.append)
    assert any("reanudo B: época 1, lote 1/2" in m for m in msgs)
    assert st["done"] and [r["epoch"] for r in st["log"]] == [1, 2]


def test_epoch_indices_deterministic_and_weighted() -> None:
    w = np.array([0.0, 1.0, 0.0, 3.0])
    a, b = epoch_indices(w, 400, 42, 1), epoch_indices(w, 400, 42, 1)
    assert a == b and a != epoch_indices(w, 400, 42, 2)
    assert set(a) <= {1, 3} and a.count(3) > 2 * a.count(1)


def test_export_new_name_outside_models(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, rows: tuple[pd.DataFrame, pd.DataFrame]
) -> None:
    monkeypatch.setenv("BCS_ARTIFACTS_DIR", str(tmp_path / "art"))
    tr, va = rows
    cfg = _tiny_cfg()
    train(
        cfg,
        "A",
        train_rows=tr,
        val_rows=va,
        dest=tmp_path / "art" / "T25" / "A",
        log=lambda s: None,
    )
    cfg["init_weights"] = "models/bcs_efficientnet_b0.pt"  # solo se lee para el sha256
    dest = export(cfg, "A")
    assert dest == tmp_path / "art" / "T25" / "bcs_efficientnet_b0_zen.pt"
    meta = json.loads(dest.with_suffix(".json").read_text())
    assert meta["variant"] == "A" and meta["weights_sha256"] and meta["init_sha256"]
    load_candidate(dest)
