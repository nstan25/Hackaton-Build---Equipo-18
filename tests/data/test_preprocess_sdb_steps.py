"""Preprocesamiento de sdb_bcs (SPEC §2.5) sobre el mini .rar sintético de fixture."""

from __future__ import annotations

import json
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from bcs.data.preprocess_sdb import MANIFEST, Config, PreprocessError, run, sha256_file

FIX = Path(__file__).parents[1] / "fixtures" / "sdb_mini"
SCRIPTS = Path(__file__).parents[2] / "preprocesamiento"
CLASSES = {"3.25": 13, "3.75": 13, "4.25": 13}


def make_cfg(root: Path, sample: int | None = None, reference: Path | None = None) -> Config:
    raw = root / "raw" / "sdb_bcs"
    raw.mkdir(parents=True, exist_ok=True)
    if not (raw / "dataset.rar").exists():
        shutil.copy(FIX / "dataset.rar", raw / "dataset.rar")
    return Config(
        rar=raw / "dataset.rar",
        raw_dir=raw,
        out_dir=root / "raw" / "sdb_bcs_pre",
        scripts_dir=SCRIPTS,
        reference_split=reference or FIX / "split.csv",
        sample=sample,
    )


@pytest.fixture(scope="module")
def done(tmp_path_factory: pytest.TempPathFactory) -> tuple[Config, dict]:
    cfg = make_cfg(tmp_path_factory.mktemp("pre"))
    return cfg, run(cfg)


def _step(manifest: dict, n: int) -> dict:
    return next(s for s in manifest["steps"] if s["step"] == n)


def test_preprocess_steps(done: tuple[Config, dict]) -> None:
    cfg, m = done
    assert [s["function"] for s in m["steps"]] == [
        "step1_make_split",
        "step2_apply_split",
        "step3_prep_crops",
        "step4_prep_yolo",
    ]
    assert m["images_in_rar"] == CLASSES

    # paso 1: make_split.py reproduce el split versionado byte a byte
    s1 = _step(m, 1)
    assert s1["split_reproduced"] and s1["before"] == s1["after"] == CLASSES
    assert (cfg.out_dir / "split.csv").read_bytes() == (FIX / "split.csv").read_bytes()
    split = pd.read_csv(cfg.out_dir / "split.csv")
    assert set(split["split"]) == {"train", "val", "test"}

    # paso 2: cada imagen y su xml en dataset_split/<split>/<clase>/
    assert _step(m, 2)["after"] == CLASSES
    for s, p in zip(split["split"], split["path"], strict=True):
        img = cfg.out_dir / "dataset_split" / s / p
        assert img.exists() and img.with_suffix(".xml").exists()

    # paso 3: recorte = bbox del xml +10%, lado máx 320
    assert _step(m, 3)["after"] == CLASSES
    crops = pd.read_csv(cfg.out_dir / "crops" / "manifest.csv")
    assert len(crops) == len(split)
    for s, p, c in zip(crops["split"], crops["path"], crops["crop"], strict=True):
        src = cfg.out_dir / "dataset_split" / s / p
        box = ET.parse(src.with_suffix(".xml")).find("object/bndbox")
        assert box is not None
        x0, y0, x1, y1 = (int(box.findtext(k) or 0) for k in ("xmin", "ymin", "xmax", "ymax"))
        dx, dy = (x1 - x0) * 0.1, (y1 - y0) * 0.1
        with Image.open(src) as o, Image.open(cfg.out_dir / c) as cr:
            w, h = o.size
            exp = (
                max(0, int(x0 - dx)),
                max(0, int(y0 - dy)),
                min(w, int(x1 + dx)),
                min(h, int(y1 + dy)),
            )
            ref = o.convert("RGB").crop(exp)
            ref.thumbnail((320, 320))
            assert cr.size == ref.size and max(cr.size) <= 320
            diff = np.abs(np.asarray(cr, np.int16) - np.asarray(ref, np.int16)).mean()
            assert diff < 4, f"{p}: el recorte no es el bbox +10% ({diff:.1f})"

    # paso 4: yolo_data de 1 clase, solo train y val
    s4 = _step(m, 4)
    assert set(s4["after_by_split"]) == {"train", "val"}
    labels = sorted((cfg.out_dir / "yolo_data" / "labels").glob("*/*.txt"))
    images = sorted((cfg.out_dir / "yolo_data" / "images").glob("*/*.jpg"))
    assert len(labels) == len(images) == sum(s4["after"].values()) > 0
    test_names = set(split.loc[split["split"] == "test", "path"].str.split("/").str[1])
    assert not test_names & {p.name for p in images}
    for lab in labels:
        cls, *xywh = lab.read_text().split()
        assert cls == "0" and all(0 < float(v) <= 1 for v in xywh)
    assert "no ejecutado" in s4["eval"]


def test_manifest_hashes(done: tuple[Config, dict]) -> None:
    cfg, m = done
    on_disk = json.loads((cfg.out_dir / MANIFEST).read_text())
    assert on_disk == m
    assert m["rar_sha256"] == sha256_file(cfg.rar)
    assert m["readme_sha256"] == sha256_file(SCRIPTS / "README.md")
    assert set(m["scripts_sha256"]) >= {"make_split.py", "apply_split.py", "prep_crops.py"}
    assert m["seed"] == 42 and m["sample"] is None
    for s in m["steps"]:
        assert {"before", "after", "discarded"} <= set(s)
        assert {"count", "reason", "files"} <= set(s["discarded"])


def test_preprocess_idempotent(done: tuple[Config, dict]) -> None:
    cfg, m = done
    files = sorted(p for p in cfg.out_dir.rglob("*") if p.is_file())
    before = {p: (p.stat().st_mtime_ns, p.read_bytes()) for p in files}
    m2 = run(cfg)
    assert m2 == m
    after = sorted(p for p in cfg.out_dir.rglob("*") if p.is_file())
    assert after == files
    assert {p: (p.stat().st_mtime_ns, p.read_bytes()) for p in after} == before


def test_preprocess_reproducible(tmp_path: Path, done: tuple[Config, dict]) -> None:
    """Mismo .rar + README + semilla en otra carpeta: mismos bytes en split y recortes."""
    cfg, m = done
    cfg2 = make_cfg(tmp_path)
    m2 = run(cfg2)
    assert m2["steps"] == m["steps"]
    for rel in ("split.csv", "crops/manifest.csv"):
        assert (cfg2.out_dir / rel).read_bytes() == (cfg.out_dir / rel).read_bytes()
    for p in (cfg.out_dir / "crops").rglob("*.jpg"):
        assert (cfg2.out_dir / p.relative_to(cfg.out_dir)).read_bytes() == p.read_bytes()


def test_split_mismatch_stops(tmp_path: Path) -> None:
    """Si make_split.py no reproduce el split versionado, se para (no se elige otro)."""
    bad = tmp_path / "split.csv"
    lines = (FIX / "split.csv").read_text().splitlines(keepends=True)
    bad.write_text("".join(lines[:1] + lines[2:]))
    with pytest.raises(PreprocessError, match="byte a byte"):
        run(make_cfg(tmp_path / "d", reference=bad))


def test_preprocess_sample(tmp_path: Path) -> None:
    cfg = make_cfg(tmp_path, sample=3)
    m = run(cfg)
    assert m["sample"] == 3
    s1 = _step(m, 1)
    assert s1["before"] == CLASSES and s1["after"] == {c: 3 for c in CLASSES}
    assert s1["not_sampled"] == sum(CLASSES.values()) - 9
    assert _step(m, 3)["after"] == {c: 3 for c in CLASSES}
    # el split del sample es el versionado, filtrado
    ref = pd.read_csv(FIX / "split.csv").set_index("path")
    got = pd.read_csv(cfg.out_dir / "split.csv").set_index("path")
    assert (ref.loc[got.index, "split"] == got["split"]).all()


def test_preprocess_sample_after_full(tmp_path: Path) -> None:
    """Un --sample sobre la carpeta de una corrida completa no cuenta restos de esa corrida."""
    run(make_cfg(tmp_path))
    m = run(make_cfg(tmp_path, sample=3))
    for n in (1, 2, 3):
        assert _step(m, n)["after"] == {c: 3 for c in CLASSES}
