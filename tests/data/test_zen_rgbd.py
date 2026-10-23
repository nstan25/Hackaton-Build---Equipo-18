"""Fetch y adaptador de zen_rgbd (SPEC §1.1, §1.2 trampas 2 y 3) sobre zips y xlsx sintéticos."""

from __future__ import annotations

import io
import json
import zlib
from pathlib import Path

import pandas as pd
import pytest
from PIL import Image

from bcs.data.schema import load_labels
from bcs.data.sources.base import Source
from bcs.data.sources.zen_rgbd import (
    Box,
    ZenRgbdSource,
    animal_stratum,
    build_plan,
    central_box,
    image_id,
    md5_file,
    split_animals,
)

W, H = 192, 108  # frames de 1920x1080 a escala 1/10
CENTRAL: Box = (75, 0, 118, 86)  # vaca del puesto central
SIDE: tuple[Box, ...] = ((0, 10, 50, 90), (145, 5, 192, 85))
# BCS por animal; 102 aparece en dos fechas con distinto BCS (trampa 3)
ANIMALS = {
    100 + k: b for k, b in enumerate([2.75, 3.0, 3.25, 3.5, 3.75, 4.0, 4.25, 4.5, 4.75, 5.0] * 4)
}


def _png(seed: int) -> bytes:
    img = Image.new("RGB", (W, H), (seed % 256, 80, 120))
    img.paste((250, 250, 250), CENTRAL)  # la vaca central, blanca
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _xlsx_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    visit = 0
    for date in ("06.12.2024", "27.03.2025"):
        for a, b in ANIMALS.items():
            if date == "27.03.2025" and a % 3:  # solo un tercio vuelve en la segunda fecha
                continue
            visit += 1
            bcs = 4.0 if (date == "27.03.2025" and a == 102) else b
            for k in range(6):
                rgb = f"{date}\\{visit}\\rgb\\10.00.{visit:02d}.{k:03d}.png"
                rows.append({"N": len(rows) + 1, "animal_id": a, "BCS": bcs, "data": date,
                             "frame": visit, "RGB_path": rgb,
                             "depth_path": rgb.replace("rgb", "depth")})  # fmt: skip
    # una visita listada para dos animales: se descarta
    for k in range(6):
        for a in (900, 901):
            rgb = f"06.12.2024\\999\\rgb\\11.00.00.{k:03d}.png"
            rows.append({"N": len(rows) + 1, "animal_id": a, "BCS": 3.25, "data": "06.12.2024",
                         "frame": 999, "RGB_path": rgb, "depth_path": ""})  # fmt: skip
    return rows


class FakeReader:
    """Dos zips: uno con la fecha como carpeta raíz y otro sin ella (como 27.03.2025.*)."""

    def __init__(self, rows: list[dict[str, object]]) -> None:
        self.members: dict[str, dict[str, bytes]] = {"06.12.2024": {}, "27.03.2025.1": {}}
        for i, r in enumerate(rows):
            p = str(r["RGB_path"]).replace("\\", "/")
            if p.startswith("27.03.2025/"):
                self.members["27.03.2025.1"][p.split("/", 1)[1]] = _png(i)
            else:
                self.members["06.12.2024"][p] = _png(i)
        self.reads = 0

    def list(self, zip_name: str) -> list[tuple[str, int, int]]:
        m = self.members.get(zip_name, {})
        return [(n, zlib.crc32(d), len(d)) for n, d in m.items()]

    def read(self, zip_name: str, member: str) -> bytes:
        self.reads += 1
        return self.members[zip_name][member]


def _detector(img: Image.Image) -> list[Box]:
    assert img.size == (W, H)
    return [SIDE[0], CENTRAL, SIDE[1], (60, 0, 190, 30)]  # la última: cabeza baja, se descarta


@pytest.fixture(scope="module")
def xlsx(tmp_path_factory: pytest.TempPathFactory) -> Path:
    p = tmp_path_factory.mktemp("xlsx") / "Dataset.xlsx"
    pd.DataFrame(_xlsx_rows()).to_excel(p, index=False)
    return p


def _source(root: Path, xlsx: Path, reader: FakeReader) -> ZenRgbdSource:
    def download(url: str, dest: Path) -> None:
        assert url.endswith("Dataset.xlsx/content")
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(xlsx.read_bytes())

    return ZenRgbdSource(
        root=root,
        reader=reader,
        download=download,
        detector=_detector,
        workers=2,
        xlsx_md5=md5_file(xlsx),
    )


@pytest.fixture(scope="module")
def adapted(tmp_path_factory: pytest.TempPathFactory, xlsx: Path) -> ZenRgbdSource:
    src = _source(tmp_path_factory.mktemp("data"), xlsx, FakeReader(_xlsx_rows()))
    src.fetch()
    src.adapt()
    return src


def test_is_source() -> None:
    assert isinstance(ZenRgbdSource(), Source)


def test_image_id() -> None:
    assert image_id("27.03.2025/1/rgb/08.15.20.700.png") == "zen_rgbd_20250327_1_081520700"


def test_plan_three_frames_per_visit(adapted: ZenRgbdSource) -> None:
    plan = pd.read_csv(adapted.raw_dir / "plan.csv")
    per = plan.groupby(["animal_id", "date"]).size()
    assert set(per) == {3}
    # primero, medio y último de cada visita (6 frames: 0, 2 o 3, 5)
    first = plan[(plan["animal_id"] == 100) & (plan["date"] == "2024-12-06")]
    assert [p.rsplit(".", 2)[-2] for p in first["rgb_path"]] == ["000", "002", "005"]
    assert not ({900, 901} & set(plan["animal_id"]))  # visita ambigua descartada
    assert set(plan["zip"]) == {"06.12.2024", "27.03.2025.1"}


def test_labels_schema(adapted: ZenRgbdSource) -> None:
    root = adapted.root
    labels = load_labels(root / "labels_zen_rgbd.csv")
    holdout = load_labels(root / "holdout_labels_zen_rgbd.csv")
    splits = pd.read_csv(root / "splits_zen_rgbd.csv")
    assert list(splits.columns) == ["image_id", "split"]
    by_id = dict(zip(splits["image_id"], splits["split"], strict=True))
    assert {by_id[i] for i in labels["image_id"]} == {"train", "val"}
    assert {by_id[i] for i in holdout["image_id"]} == {"test"}
    every = pd.concat([labels, holdout])
    assert len(every) == len(splits) == len(pd.read_csv(adapted.raw_dir / "plan.csv"))
    assert set(every["source"]) == {"zen_rgbd"} and set(every["scale"]) == {"dairy_1_5"}
    assert set(every["view"]) == {"rear_dorsal"}
    assert set(every["bcs"]) <= {2.75, 3.0, 3.25, 3.5, 3.75, 4.0, 4.25, 4.5, 4.75, 5.0}
    assert all(a.startswith("zen_rgbd:") for a in every["animal_id"])
    assert (every["group_id"] == every["animal_id"]).all()  # group_id por animal
    assert set(every["date"]) == {"2024-12-06", "2025-03-27"}
    # 102 tiene dos BCS en dos fechas: las dos filas comparten group_id
    a102 = every[every["animal_id"] == "zen_rgbd:102"]
    assert set(a102["bcs"]) == {3.25, 4.0} and len(set(a102["group_id"])) == 1
    for p in every["path"]:
        with Image.open(root / p) as im:
            assert max(im.size) <= 320
    rows = adapted.adapt()
    assert [r.image_id for r in rows] == list(labels["image_id"])


def test_crop_is_central_cow(adapted: ZenRgbdSource) -> None:
    """El recorte toma la caja central (+10%): la vaca blanca llena el centro del recorte."""
    labels = load_labels(adapted.root / "labels_zen_rgbd.csv")
    assert set(labels["bbox"]) == {",".join(map(str, CENTRAL))}
    with Image.open(adapted.root / labels["path"].iloc[0]) as im:
        w, h = im.size
        assert im.convert("RGB").getpixel((w // 2, h // 2)) == pytest.approx((250, 250, 250), abs=6)


def test_central_box_rules() -> None:
    w, h = 1920, 1080
    assert central_box([(0, 300, 500, 900), (760, 0, 1200, 860), (1450, 100, 1920, 700)], w, h) == (
        760, 0, 1200, 860,
    )  # fmt: skip
    # una caja que junta dos vacas (muy ancha) o una cabeza (muy baja) no cuenta
    assert central_box([(600, 0, 1800, 900), (700, 0, 1170, 330)], w, h) is None
    assert central_box([], w, h) is None


def test_fetch_idempotent(tmp_path: Path, xlsx: Path) -> None:
    reader = FakeReader(_xlsx_rows())
    src = _source(tmp_path, xlsx, reader)
    first = src.fetch(sample=2)
    assert first["downloaded"] == 20  # 2 por clase × 10 clases
    lock = (tmp_path / "sources.lock.zen_rgbd.json").read_text()
    mtimes = {p: p.stat().st_mtime_ns for p in (tmp_path / "raw" / "zen_rgbd").rglob("*.png")}
    again = src.fetch(sample=2)
    assert again["downloaded"] == 0 and reader.reads == 20
    assert (tmp_path / "sources.lock.zen_rgbd.json").read_text() == lock
    assert {p: p.stat().st_mtime_ns for p in mtimes} == mtimes
    # reanudable: un frame truncado se vuelve a bajar; el resto no
    victim = next(iter(mtimes))
    victim.write_bytes(victim.read_bytes()[:10])
    assert src.fetch(sample=2)["downloaded"] == 1
    full = src.fetch()
    assert full["present"] == full["plan"] and full["downloaded"] == full["plan"] - 20
    entry = json.loads((tmp_path / "sources.lock.zen_rgbd.json").read_text())["zen_rgbd"]
    assert entry["doi"] == "10.5281/zenodo.20290988" and entry["license"].startswith("CC BY 4.0")
    assert entry["frames_present"] == full["plan"]


def test_no_group_leakage_zen(adapted: ZenRgbdSource) -> None:
    splits = pd.read_csv(adapted.root / "splits_zen_rgbd.csv")
    every = pd.concat(
        [
            load_labels(adapted.root / "labels_zen_rgbd.csv"),
            load_labels(adapted.root / "holdout_labels_zen_rgbd.csv"),
        ]
    ).merge(splits, on="image_id")
    assert every.groupby("group_id")["split"].nunique().max() == 1
    assert every.groupby("animal_id")["split"].nunique().max() == 1


def test_split_animals_stratified_and_fixed() -> None:
    animals = pd.DataFrame(
        {"animal_id": list(range(200)), "bcs": [3.0] * 40 + [3.5] * 140 + [5.0] * 20}
    )
    a, b = split_animals(animals), split_animals(animals)
    assert a == b  # semilla fija
    for lo, hi, n in ((0, 40, 40), (40, 180, 140), (180, 200, 20)):
        got = pd.Series([a[i] for i in range(lo, hi)]).value_counts().to_dict()
        assert got == {"train": round(0.7 * n), "val": round(0.15 * n), "test": round(0.15 * n)}
    assert animal_stratum([3.25, 4.75, 3.5]) == 4.75
    assert animal_stratum([3.0, 3.75]) == 3.0


def test_build_plan_member_names() -> None:
    rows = _xlsx_rows()
    df = pd.DataFrame(rows)
    df["rgb_path"] = df["RGB_path"].str.replace("\\", "/", regex=False)
    reader = FakeReader(rows)
    index = {n: [z, c, s] for z in reader.members for n, c, s in reader.list(z)}
    plan = build_plan(df, index)
    late = plan[plan["date"] == "2025-03-27"]
    assert all(not m.startswith("27.03.2025/") for m in late["member"])
