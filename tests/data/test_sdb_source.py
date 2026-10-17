"""Fetch y adaptador de sdb_bcs (SPEC §1.1, §1.2, §5.6.1) sobre el mini .rar sintético."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from bcs.data.preprocess_sdb import Config, run, sha256_file
from bcs.data.schema import load_labels
from bcs.data.sources.base import Source
from bcs.data.sources.sdb_bcs import (
    SdbBcsSource,
    _dhash,
    _load_prep_crops,
    fetch_rar,
    near_duplicates_across_splits,
)

FIX = Path(__file__).parents[1] / "fixtures" / "sdb_mini"
SCRIPTS = Path(__file__).parents[2] / "preprocesamiento"


@pytest.fixture(scope="module")
def adapted(tmp_path_factory: pytest.TempPathFactory) -> SdbBcsSource:
    root = tmp_path_factory.mktemp("data")
    raw = root / "raw" / "sdb_bcs"
    raw.mkdir(parents=True)
    shutil.copy(FIX / "dataset.rar", raw / "dataset.rar")
    run(
        Config(
            rar=raw / "dataset.rar",
            raw_dir=raw,
            out_dir=root / "raw" / "sdb_bcs_pre",
            scripts_dir=SCRIPTS,
            reference_split=FIX / "split.csv",
        )
    )
    src = SdbBcsSource(root=root, scripts_dir=SCRIPTS, workers=2)
    src.adapt()
    return src


def _split(src: SdbBcsSource) -> pd.DataFrame:
    return pd.read_csv(src.pre_dir / "split.csv", dtype={"cow": str})


def test_is_source() -> None:
    assert isinstance(SdbBcsSource(), Source)


def test_labels_schema(adapted: SdbBcsSource) -> None:
    labels = load_labels(adapted.root / "labels_sdb_bcs.csv")
    holdout = load_labels(adapted.root / "holdout_labels_sdb_bcs.csv")
    splits = pd.read_csv(adapted.root / "splits_sdb_bcs.csv")
    assert list(splits.columns) == ["image_id", "split"]
    by_id = dict(zip(splits["image_id"], splits["split"], strict=True))
    # labels = train + val; holdout = test; splits cubre todo, sin solaparse
    assert {by_id[i] for i in labels["image_id"]} == {"train", "val"}
    assert {by_id[i] for i in holdout["image_id"]} == {"test"}
    assert len(labels) + len(holdout) == len(splits) == len(_split(adapted))
    assert set(labels["source"]) == {"sdb_bcs"} and set(labels["scale"]) == {"dairy_1_5"}
    assert set(labels["bcs"]) <= {3.25, 3.5, 3.75, 4.0, 4.25}
    for p in pd.concat([labels, holdout])["path"]:
        assert (adapted.root / p).exists()
    # adapt() devuelve exactamente las filas de labels_sdb_bcs.csv
    rows = adapted.adapt()
    assert [r.image_id for r in rows] == list(labels["image_id"])


def test_group_id_recoverable(adapted: SdbBcsSource) -> None:
    """group_id = 'sdb_bcs:' + group de split.csv, para el 100% de las imágenes."""
    split = _split(adapted)
    every = pd.concat(
        [
            load_labels(adapted.root / "labels_sdb_bcs.csv"),
            load_labels(adapted.root / "holdout_labels_sdb_bcs.csv"),
        ]
    )
    group_of = dict(zip("sdb_bcs_" + split["path"].map(lambda p: Path(p).stem), split["group"],
                        strict=True))  # fmt: skip
    assert len(every) == len(split)
    for i, g in zip(every["image_id"], every["group_id"], strict=True):
        assert g == f"sdb_bcs:{group_of[i]}"
    # se reconstruye del nombre: GS_/YM_<vaca> → cow_<vaca>; L/R → su tramo
    for i, g in zip(every["image_id"], every["group_id"], strict=True):
        name = i.removeprefix("sdb_bcs_")
        if name[:3] in ("GS_", "YM_"):
            assert g == f"sdb_bcs:cow_{name.split('_')[1]}"
        else:
            assert g.startswith("sdb_bcs:LR_seg")
    # un grupo nunca queda en dos splits
    assert split.groupby("group")["split"].nunique().max() == 1


def test_sdb_overlay_removed(adapted: SdbBcsSource) -> None:
    """Ningún píxel de texto sobreimpreso del original sobrevive en la imagen procesada.

    En la fixture el texto es blanco puro (lo único > 248 en la imagen); se ubica en el original,
    se proyecta al recorte con la geometría de prep_crops.py y se exige negro ahí.
    """
    prep = _load_prep_crops(SCRIPTS)
    split = _split(adapted)
    checked = 0
    for s, p in zip(split["split"], split["path"], strict=True):
        src = adapted.pre_dir / "dataset_split" / s / p
        out = adapted.images_dir / f"sdb_bcs_{Path(p).stem}.jpg"
        orig = np.asarray(Image.open(src).convert("RGB"), np.int16)
        h, w = orig.shape[:2]
        x0, y0, x1, y1 = prep.expand_bbox(prep.read_bbox(src.with_suffix(".xml")), w, h)
        proc = np.asarray(Image.open(out).convert("L"), np.int16)
        sx, sy = proc.shape[1] / (x1 - x0), proc.shape[0] / (y1 - y0)
        ys, xs = np.nonzero((orig > 248).all(axis=2))
        inside = (xs >= x0) & (xs < x1) & (ys >= y0) & (ys < y1)
        if Path(p).name[:3] in ("GS_", "YM_"):
            assert len(xs) > 0, f"{p}: la fixture debería traer texto"
        for x, y in zip(xs[inside], ys[inside], strict=True):
            px, py = int((x - x0) * sx), int((y - y0) * sy)
            assert proc[py, px] < 40, f"{p}: queda texto en ({px},{py})"
            checked += 1
    assert checked > 50  # el +10% de margen mete parte de la franja en los recortes


def test_masked_regions_black(adapted: SdbBcsSource) -> None:
    qa = json.loads((adapted.images_dir / "qa_sdb_bcs.json").read_text())
    assert qa["masked_images"] > 0
    split = _split(adapted)
    src = SdbBcsSource(root=adapted.root, scripts_dir=SCRIPTS)
    prep = _load_prep_crops(SCRIPTS)
    for s, p in zip(split["split"], split["path"], strict=True):
        row = {
            "split": s,
            "path": p,
            "image_id": f"sdb_bcs_{Path(p).stem}",
            "crop": f"crops/{s}/{p}",
        }
        info = src._process(row, prep, force=False)
        img = np.asarray(Image.open(src.images_dir / f"sdb_bcs_{Path(p).stem}.jpg").convert("L"))
        for bx0, by0, bx1, by1 in info["masked"]:
            inner = img[by0 + 2 : by1 - 2, bx0 + 2 : bx1 - 2]
            if inner.size:
                assert inner.max() < 40, f"{p}: franja tapada no negra"
        if Path(p).name[0] in "LR":
            assert info["masked"] == []  # L/R no traen texto: no se tapa nada


def test_no_near_duplicates_across_splits(adapted: SdbBcsSource) -> None:
    splits = pd.read_csv(adapted.root / "splits_sdb_bcs.csv")
    by_id = dict(zip(splits["image_id"], splits["split"], strict=True))
    hashes = {}
    for i in by_id:
        with Image.open(adapted.images_dir / f"{i}.jpg") as im:
            hashes[i] = _dhash(im)
    assert near_duplicates_across_splits(hashes, by_id) == []
    qa = json.loads((adapted.images_dir / "qa_sdb_bcs.json").read_text())
    assert qa["near_duplicates_across_splits"]["n"] == 0
    # control positivo: una copia de una foto puesta en otro split se detecta
    a = sorted(by_id)[0]
    other = "test" if by_id[a] != "test" else "train"
    pairs = near_duplicates_across_splits({**hashes, "copy": hashes[a]}, {**by_id, "copy": other})
    assert (a, "copy", 0) in pairs or ("copy", a, 0) in pairs


def test_adapt_idempotent(adapted: SdbBcsSource) -> None:
    files = sorted(p for p in adapted.images_dir.iterdir()) + sorted(
        adapted.root.glob("*_sdb_bcs.csv")
    )
    before = {p: p.stat().st_mtime_ns for p in files}
    adapted.adapt()
    assert {p: p.stat().st_mtime_ns for p in files} == before


def test_fetch_idempotent(tmp_path: Path) -> None:
    rar = tmp_path / "raw" / "sdb_bcs" / "dataset.rar"
    rar.parent.mkdir(parents=True)
    shutil.copy(FIX / "dataset.rar", rar)
    lock = tmp_path / "sources.lock.sdb_bcs.json"
    size, sha = rar.stat().st_size, sha256_file(rar)

    def no_download(url: str, dest: Path) -> None:
        raise AssertionError("no debería bajar nada")

    e1 = fetch_rar(rar, lock, size=size, sha256=sha, download=no_download)
    stamp = (rar.stat().st_mtime_ns, lock.stat().st_mtime_ns, lock.read_text())
    e2 = fetch_rar(rar, lock, size=size, sha256=sha, download=no_download)
    assert e1 == e2 and e1["sha256"] == sha and e1["size"] == size
    assert (rar.stat().st_mtime_ns, lock.stat().st_mtime_ns, lock.read_text()) == stamp
    assert json.loads(lock.read_text())["sdb_bcs"]["doi"] == "10.57760/sciencedb.16704"


def test_fetch_resumes_partial(tmp_path: Path) -> None:
    rar = tmp_path / "dataset.rar"
    full = (FIX / "dataset.rar").read_bytes()
    rar.write_bytes(full[:1000])
    calls: list[str] = []

    def resume(url: str, dest: Path) -> None:
        calls.append(url)
        with dest.open("ab") as f:
            f.write(full[dest.stat().st_size :])

    fetch_rar(rar, tmp_path / "lock.json", size=len(full), sha256=sha256_file(FIX / "dataset.rar"),
              download=resume)  # fmt: skip
    assert len(calls) == 1 and rar.read_bytes() == full


def test_fetch_rejects_bad_sha(tmp_path: Path) -> None:
    rar = tmp_path / "dataset.rar"
    shutil.copy(FIX / "dataset.rar", rar)
    with pytest.raises(RuntimeError, match="sha256"):
        fetch_rar(rar, tmp_path / "lock.json", size=rar.stat().st_size, sha256="0" * 64)
