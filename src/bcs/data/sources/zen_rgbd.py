"""Fuente `zen_rgbd` (Zenodo 20290988, CC BY 4.0): fetch por frame y adaptador al formato canónico.

Sala rotativa, cámara fija desde atrás y arriba, **tres vacas por frame** (trampa 2 de §1.2). La
etiquetada es la del puesto central: se confirmó mirando el mismo `animal_id` en dos fechas (la
central coincide en pelaje y las vecinas cambian; ver el reporte de T04). El recorte usa YOLO11m
COCO (clase `cow`) y elige la caja plausible más cercana al centro. `models/cow_det.pt`
(entrenado en recortes de `sdb_bcs`) no sirve en este dominio: en 16/25 frames no da caja central
y las que da toman solo la grupa; YOLO11m encontró la vaca central entera en 25/25.

Fetch (nunca los zips enteros, 83,5 GB):

    raw/zen_rgbd/Dataset.xlsx         metadata: N, animal_id, BCS, data, frame, RGB_path, ...
    raw/zen_rgbd/zip_index.json       miembro rgb → (zip, crc32, tamaño), del directorio central
    raw/zen_rgbd/plan.csv             3 frames por (animal_id, fecha): primero, medio y último
    raw/zen_rgbd/rgb/<RGB_path>       los frames del plan, bajados con remotezip (crc verificado)

Adaptador:

    images/zen_rgbd/<image_id>.jpg    vaca central, caja +10% (prep_crops.expand_bbox), lado máx 320
    labels_zen_rgbd.csv               train + val (esquema de bcs.data.schema)
    splits_zen_rgbd.csv               image_id,split; por animal, 70/15/15 estratificado, semilla 42
    holdout_labels_zen_rgbd.csv       test, mismo esquema (solo lo lee `make eval-final`)

`animal_id` = `group_id` = "zen_rgbd:<id>" (trampa 3: el mismo animal en varias fechas y BCS).

Uso: python -m bcs.data.sources.zen_rgbd [--sample N]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
import threading
import zlib
from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

from bcs.data.preprocess_sdb import sha256_file
from bcs.data.schema import LABEL_COLUMNS, LabelRow
from bcs.data.sources.sdb_bcs import _load_prep_crops
from bcs.paths import REPO_ROOT, artifacts_dir, data_dir, runs_dir

SOURCE_ID = "zen_rgbd"
SCALE = "dairy_1_5"
VIEW = "rear_dorsal"
RECORD = "https://zenodo.org/api/records/20290988"
DOI = "10.5281/zenodo.20290988"
XLSX_URL = f"{RECORD}/files/Dataset.xlsx/content"
XLSX_MD5 = "30868f345998cac63a48158b11c47983"  # checksum publicado por Zenodo
ZIPS = ("06.12.2024", "20.02.2025", "20.03.2025", "27.03.2025.1", "27.03.2025.2")
FRAMES_PER_VISIT = 3
SEED = 42
SPLIT_FRACS = (("train", 0.70), ("val", 0.15), ("test", 0.15))

# Detector: YOLO11m COCO, clase 19 = cow (AGPL-3.0, pesos de ultralytics; viven en artifacts/T04,
# nunca en el repo).
DET_URL = "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11m.pt"
DET_SHA256 = "d5ffc1a674953a08e11a8d21e022781b1b23a19b730afc309290bd9fb5305b95"
COCO_COW = 19
DET_CONF = 0.2
# Caja de una vaca entera: alta (≥ 40% del frame) y no más ancha que ~2 puestos.
MIN_BOX_H, MAX_BOX_W = 0.40, 0.45
MAX_SIDE = 320

Box = tuple[int, int, int, int]


def zip_url(name: str) -> str:
    return f"{RECORD}/files/{name}.zip/content"


# --- metadata y plan ---------------------------------------------------------------------


def load_xlsx(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path, dtype={"data": str})
    df["rgb_path"] = df["RGB_path"].str.replace("\\", "/", regex=False)
    return df


def ambiguous_paths(df: pd.DataFrame) -> set[str]:
    """Frames listados para más de un animal (una visita de 20.02.2025 trae dos ids y dos BCS)."""
    animals: dict[str, set[int]] = {}
    for path, a in zip(df["rgb_path"], df["animal_id"], strict=True):
        animals.setdefault(str(path), set()).add(int(a))
    return {path for path, ids in animals.items() if len(ids) > 1}


def member_name(rgb_path: str, index: dict[str, list[Any]]) -> str:
    """Los zips de 27.03.2025 no llevan la fecha como carpeta raíz; los otros sí."""
    if rgb_path in index:
        return rgb_path
    tail = rgb_path.split("/", 1)[1]
    if tail in index:
        return tail
    raise KeyError(f"{rgb_path} no está en ningún zip")


def build_plan(df: pd.DataFrame, index: dict[str, list[Any]]) -> pd.DataFrame:
    """3 frames por (animal_id, fecha), temporalmente separados: primero, medio y último."""
    bad = ambiguous_paths(df)
    ok = _select(df, ~df["rgb_path"].isin(sorted(bad))).sort_values(
        by=["animal_id", "data", "rgb_path"]
    )
    picks = []
    for _, g in ok.groupby(["animal_id", "data"], sort=True):
        n = len(g)
        idx = sorted({round(i * (n - 1) / (FRAMES_PER_VISIT - 1)) for i in range(FRAMES_PER_VISIT)})
        picks.append(g.iloc[idx])
    plan = pd.concat(picks, ignore_index=True)
    members = [member_name(p, index) for p in plan["rgb_path"]]
    plan = pd.DataFrame(
        {
            "image_id": [image_id(p) for p in plan["rgb_path"]],
            "animal_id": plan["animal_id"].astype(int),
            "bcs": plan["BCS"].astype(float),
            "date": [iso_date(d) for d in plan["data"]],
            "visit": plan["frame"].astype(int),
            "rgb_path": plan["rgb_path"],
            "zip": [index[m][0] for m in members],
            "member": members,
            "crc": [int(index[m][1]) for m in members],
            "size": [int(index[m][2]) for m in members],
        }
    )
    if bool(plan["image_id"].duplicated().any()):
        raise RuntimeError("image_id duplicado en el plan")
    return plan


def iso_date(d: str) -> str:
    day, month, year = d.split(".")[:3]
    return f"{year}-{month}-{day}"


def image_id(rgb_path: str) -> str:
    """'27.03.2025/1/rgb/08.15.20.700.png' → 'zen_rgbd_20250327_1_081520700'."""
    date, visit, _, name = rgb_path.split("/")
    return (
        f"{SOURCE_ID}_{iso_date(date).replace('-', '')}_{visit}_{Path(name).stem.replace('.', '')}"
    )


def sample_plan(plan: pd.DataFrame, sample: int | None) -> pd.DataFrame:
    """`--sample N`: N frames por clase de BCS (al azar con semilla fija); None = todo el plan."""
    if sample is None:
        return plan
    keep: list[int] = []
    for _, g in plan.groupby("bcs", sort=True):
        idx = list(g.index)
        random.Random(SEED).shuffle(idx)
        keep += idx[:sample]
    return plan.loc[sorted(keep)]


# --- fetch -------------------------------------------------------------------------------


class ZipReader(Protocol):
    def list(self, zip_name: str) -> list[tuple[str, int, int]]: ...
    def read(self, zip_name: str, member: str) -> bytes: ...


class RemoteZipReader:
    """Un RemoteZip abierto por hilo y por zip: el directorio central se lee una sola vez."""

    def __init__(self) -> None:
        self._local = threading.local()

    def _zip(self, zip_name: str) -> Any:
        from remotezip import RemoteZip

        cache = self._local.__dict__.setdefault("zips", {})
        if zip_name not in cache:
            cache[zip_name] = RemoteZip(zip_url(zip_name))
        return cache[zip_name]

    def list(self, zip_name: str) -> list[tuple[str, int, int]]:
        return [(i.filename, i.CRC, i.file_size) for i in self._zip(zip_name).infolist()]

    def read(self, zip_name: str, member: str) -> bytes:
        return self._zip(zip_name).read(member)


def _curl(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["curl", "-sSL", "--retry", "3", "-o", str(dest), url], check=True)


def md5_file(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_text(path: Path, content: str) -> None:
    if not path.exists() or path.read_text() != content:  # no reescribir si no cambió
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".part")
        tmp.write_text(content)
        tmp.replace(path)


def _select(df: pd.DataFrame, rows: Any, cols: list[str] | None = None) -> pd.DataFrame:
    """`df.loc[rows, cols]` como DataFrame nuevo (filas por máscara booleana o slice)."""
    if not isinstance(rows, slice):
        rows = np.asarray(rows, dtype=bool)
    return pd.DataFrame(df.loc[rows, cols if cols is not None else list(df.columns)]).copy()


def _write_csv(df: pd.DataFrame, path: Path) -> None:
    _write_text(path, df.to_csv(index=False))


def fetch_xlsx(
    dest: Path, *, md5: str = XLSX_MD5, download: Callable[[str, Path], None] = _curl
) -> None:
    if dest.exists() and md5_file(dest) == md5:
        return
    download(XLSX_URL, dest)
    got = md5_file(dest)
    if got != md5:
        raise RuntimeError(f"{dest}: md5 {got} != {md5}")


def fetch_index(dest: Path, reader: ZipReader, zips: Sequence[str] = ZIPS) -> dict[str, list[Any]]:
    """Miembros rgb de cada zip → [zip, crc32, tamaño]; se lee de la red una sola vez."""
    if dest.exists():
        return json.loads(dest.read_text())
    index: dict[str, list[Any]] = {}
    for z in zips:
        for name, crc, size in reader.list(z):
            if "/rgb/" in f"/{name}" and name.endswith(".png"):
                index[name] = [z, crc, size]
    _write_text(dest, json.dumps(index, sort_keys=True) + "\n")
    return index


def frame_ok(path: Path, size: int) -> bool:
    return path.exists() and path.stat().st_size == size


def fetch_frames(
    plan: pd.DataFrame, rgb_dir: Path, reader: ZipReader, workers: int = 8
) -> dict[str, int]:
    """Baja los frames que falten (o estén truncados); el crc32 se verifica antes de renombrar."""

    def one(r: dict[str, Any]) -> bool:
        dest = rgb_dir / r["rgb_path"]
        if frame_ok(dest, int(r["size"])):
            return False
        data = reader.read(r["zip"], r["member"])
        if zlib.crc32(data) != int(r["crc"]) or len(data) != int(r["size"]):
            raise RuntimeError(f"{r['zip']}:{r['member']}: crc o tamaño no coinciden")
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(".part")
        tmp.write_bytes(data)
        tmp.replace(dest)
        return True

    rows = plan.to_dict(orient="records")
    with ThreadPoolExecutor(workers) as ex:
        got = list(ex.map(one, rows))
    return {"planned": len(rows), "downloaded": sum(got), "present": len(rows)}


def write_lock(lock: Path, raw: Path, plan: pd.DataFrame, present: int) -> dict[str, Any]:
    xlsx = raw / "Dataset.xlsx"
    entry = {
        "url": RECORD,
        "xlsx_url": XLSX_URL,
        "zips": {z: zip_url(z) for z in ZIPS},
        "doi": DOI,
        "title": (
            "RGB-D Image Dataset for Automated Body Condition Scoring of Dairy Cows in Rotary "
            "Milking Parlors (Ruchay, Kolpakov, Riazanov)"
        ),
        "license": "CC BY 4.0 (citar el DOI)",
        "xlsx_sha256": sha256_file(xlsx),
        "plan_sha256": sha256_file(raw / "plan.csv"),
        "frames_in_plan": len(plan),
        "frames_present": present,
        "bytes_present": int(np.asarray(plan["size"]).sum()) if present == len(plan) else None,
        # fecha del archivo, no de la corrida: la misma copia da el mismo lock
        "downloaded": datetime.fromtimestamp(xlsx.stat().st_mtime, UTC).date().isoformat(),
    }
    _write_text(lock, json.dumps({SOURCE_ID: entry}, indent=2, ensure_ascii=False) + "\n")
    return entry


# --- recorte -----------------------------------------------------------------------------


Detector = Callable[[Image.Image], list[Box]]


def yolo_detector(weights: Path) -> Detector:
    from ultralytics import YOLO  # import diferido: pesa y no hace falta en los tests

    model = YOLO(str(weights))
    lock = threading.Lock()

    def detect(img: Image.Image) -> list[Box]:
        with lock:
            r: Any = list(model.predict(img, verbose=False, conf=DET_CONF, classes=[COCO_COW]))[0]
        out: list[Box] = []
        for b in r.boxes:
            x0, y0, x1, y1 = (int(round(v)) for v in b.xyxy[0].tolist())
            out.append((x0, y0, x1, y1))
        return out

    return detect


def ensure_weights(dest: Path, download: Callable[[str, Path], None] = _curl) -> Path:
    if not (dest.exists() and sha256_file(dest) == DET_SHA256):
        download(DET_URL, dest)
        if sha256_file(dest) != DET_SHA256:
            raise RuntimeError(f"{dest}: sha256 no coincide con {DET_SHA256}")
    return dest


def central_box(boxes: Iterable[Box], w: int, h: int) -> Box | None:
    """La caja de vaca entera cuyo centro queda más cerca del centro del frame.

    Plausible = alta (≥ 40% del alto) y no más ancha que ~2 puestos (≤ 45% del ancho), así se
    descartan las cajas que solo toman la cabeza de una vecina o que juntan dos vacas.
    """
    cands = [
        b
        for b in boxes
        if (b[3] - b[1]) >= MIN_BOX_H * h and (b[2] - b[0]) <= MAX_BOX_W * w and b[0] < w / 2 < b[2]
    ]
    if not cands:
        return None
    return min(cands, key=lambda b: ((b[0] + b[2] - w) / 2) ** 2 + ((b[1] + b[3] - h) / 2) ** 2)


# --- split por animal --------------------------------------------------------------------


def animal_stratum(bcs: Iterable[float]) -> float:
    """El BCS más extremo del animal (el más lejos de 3.5): así las colas quedan en los 3 splits."""
    return max(bcs, key=lambda b: (abs(b - 3.5), b))


def split_animals(animal_bcs: pd.DataFrame, seed: int = SEED) -> dict[int, str]:
    """train/val/test por animal, 70/15/15 dentro de cada estrato, semilla fija.

    Cuotas por resto mayor (empates al azar con la semilla). Un estrato de 3 o más animales deja
    al menos uno en val y uno en test, para que las colas chicas (2.75: 4 animales) se puedan
    evaluar.
    """
    strata = animal_bcs.groupby("animal_id")["bcs"].agg(lambda s: animal_stratum(s.tolist()))
    rng = random.Random(seed)
    out: dict[int, str] = {}
    for _, animals in strata.groupby(strata, sort=True):
        ids = sorted(int(a) for a in animals.index)
        rng.shuffle(ids)
        n = len(ids)
        quota = [n * f for _, f in SPLIT_FRACS]
        counts = [int(q) for q in quota]
        if n >= len(SPLIT_FRACS):
            counts = [max(c, 1) for c in counts]
            counts[0] = n - sum(counts[1:]) if sum(counts) > n else counts[0]
        ties = [rng.random() for _ in quota]
        order = sorted(range(len(quota)), key=lambda k: (counts[k] - quota[k], ties[k]))
        for k in order[: n - sum(counts)]:
            counts[k] += 1
        start = 0
        for (name, _), c in zip(SPLIT_FRACS, counts, strict=True):
            for a in ids[start : start + c]:
                out[a] = name
            start += c
    return out


# --- fuente ------------------------------------------------------------------------------


class ZenRgbdSource:
    source_id = SOURCE_ID

    def __init__(
        self,
        root: Path | None = None,
        *,
        reader: ZipReader | None = None,
        download: Callable[[str, Path], None] = _curl,
        detector: Detector | None = None,
        weights: Path | None = None,
        workers: int = 8,
        xlsx_md5: str = XLSX_MD5,
    ) -> None:
        self.root = root or data_dir()
        self.reader = reader or RemoteZipReader()
        self.download = download
        self._detector = detector
        self.weights = weights or artifacts_dir() / "T04" / "yolo11m.pt"
        self.workers = workers
        self.xlsx_md5 = xlsx_md5

    @property
    def raw_dir(self) -> Path:
        return self.root / "raw" / SOURCE_ID

    @property
    def images_dir(self) -> Path:
        return self.root / "images" / SOURCE_ID

    def table(self) -> pd.DataFrame:
        return load_xlsx(self.raw_dir / "Dataset.xlsx")

    def fetch(self, sample: int | None = None) -> dict[str, Any]:
        """Idempotente y reanudable: solo baja lo que falta. `sample` = N frames por clase."""
        fetch_xlsx(self.raw_dir / "Dataset.xlsx", md5=self.xlsx_md5, download=self.download)
        index = fetch_index(self.raw_dir / "zip_index.json", self.reader)
        plan = build_plan(self.table(), index)
        _write_csv(plan, self.raw_dir / "plan.csv")
        todo = sample_plan(plan, sample)
        stats = fetch_frames(todo, self.raw_dir / "rgb", self.reader, self.workers)
        present = sum(
            frame_ok(self.raw_dir / "rgb" / p, s)
            for p, s in zip(plan["rgb_path"], plan["size"], strict=True)
        )
        write_lock(self.root / f"sources.lock.{SOURCE_ID}.json", self.raw_dir, plan, present)
        return {**stats, "present": present, "plan": len(plan)}

    def detector(self) -> Detector:
        if self._detector is None:
            self._detector = yolo_detector(ensure_weights(self.weights, self.download))
        return self._detector

    def _detections(self, plan: pd.DataFrame) -> dict[str, dict[str, Any]]:
        """Cajas por image_id, cacheadas en images/zen_rgbd/.detections.json."""
        cache_path = self.images_dir / ".detections.json"
        cache: dict[str, dict[str, Any]] = (
            json.loads(cache_path.read_text()) if cache_path.exists() else {}
        )
        todo = [r for r in plan.to_dict(orient="records") if r["image_id"] not in cache]
        if todo:
            detect = self.detector()

            def one(r: dict[str, Any]) -> tuple[str, dict[str, Any]]:
                with Image.open(self.raw_dir / "rgb" / r["rgb_path"]) as im:
                    im = im.convert("RGB")
                    return r["image_id"], {"size": im.size, "boxes": detect(im)}

            with ThreadPoolExecutor(self.workers) as ex:
                for i, d in ex.map(one, todo):
                    cache[i] = d
            _write_text(cache_path, json.dumps(cache, sort_keys=True) + "\n")
        return cache

    def adapt(self) -> list[LabelRow]:
        """Recorta los frames del plan que estén bajados; devuelve las filas de train + val."""
        plan = pd.read_csv(self.raw_dir / "plan.csv", dtype={"rgb_path": str})
        all_animals = _select(plan, slice(None), ["animal_id", "bcs"])
        present = [
            frame_ok(self.raw_dir / "rgb" / p, s)
            for p, s in zip(plan["rgb_path"], plan["size"], strict=True)
        ]
        here = _select(plan, present).reset_index(drop=True)
        self.images_dir.mkdir(parents=True, exist_ok=True)
        dets = self._detections(here)
        prep = _load_prep_crops(REPO_ROOT)

        stamp = self.images_dir / ".adapt.json"
        code = sha256_file(Path(__file__))
        force = not (stamp.exists() and json.loads(stamp.read_text()).get("code") == code)

        def one(r: dict[str, Any]) -> str | None:
            d = dets[r["image_id"]]
            w, h = d["size"]
            box = central_box([tuple(b) for b in d["boxes"]], w, h)
            if box is None:
                return None
            dst = self.images_dir / f"{r['image_id']}.jpg"
            if force or not dst.exists():
                with Image.open(self.raw_dir / "rgb" / r["rgb_path"]) as im:
                    crop = im.convert("RGB").crop(prep.expand_bbox(list(box), w, h))
                crop.thumbnail((MAX_SIDE, MAX_SIDE))
                tmp = dst.with_suffix(".part.jpg")
                crop.save(tmp, quality=95)
                tmp.replace(dst)
            return ",".join(map(str, box))

        with ThreadPoolExecutor(self.workers) as ex:
            bboxes = list(ex.map(one, here.to_dict(orient="records")))
        _write_text(stamp, json.dumps({"code": code}) + "\n")
        here["bbox"] = bboxes
        dropped = _select(here, [b is None for b in bboxes])
        df = _select(here, [b is not None for b in bboxes])
        keep = {f"{i}.jpg" for i in df["image_id"]}
        for p in self.images_dir.glob("*.jpg"):  # restos de otra corrida
            if p.name not in keep:
                p.unlink()

        split_of = split_animals(all_animals)
        df["split"] = [split_of[int(a)] for a in df["animal_id"]]
        df = df.sort_values(by="image_id", kind="stable").reset_index(drop=True)
        labels = pd.DataFrame(
            {
                "image_id": df["image_id"],
                "path": [f"images/{SOURCE_ID}/{i}.jpg" for i in df["image_id"]],
                "source": SOURCE_ID,
                "animal_id": [f"{SOURCE_ID}:{a}" for a in df["animal_id"]],
                "group_id": [f"{SOURCE_ID}:{a}" for a in df["animal_id"]],
                "bcs": df["bcs"].astype(float),
                "scale": SCALE,
                "view": VIEW,
                "scorer": "",
                "date": df["date"],
                "bbox": df["bbox"],
            }
        ).loc[:, list(LABEL_COLUMNS)]
        is_test = (df["split"] == "test").to_numpy()
        train_val = _select(labels, ~is_test)
        _write_csv(train_val, self.root / f"labels_{SOURCE_ID}.csv")
        _write_csv(_select(labels, is_test), self.root / f"holdout_labels_{SOURCE_ID}.csv")
        _write_csv(
            _select(df, slice(None), ["image_id", "split"]), self.root / f"splits_{SOURCE_ID}.csv"
        )

        qa = qa_report(df, dropped, self.table(), split_of)
        _write_text(self.images_dir / "qa_zen_rgbd.json", json.dumps(qa, indent=2) + "\n")
        rows = train_val.to_dict(orient="records")
        return [LabelRow(**{str(k): v for k, v in r.items()}) for r in rows]


def n_animals(df: pd.DataFrame) -> int:
    return len(set(df["animal_id"].tolist()))


def qa_report(
    df: pd.DataFrame, dropped: pd.DataFrame, xlsx: pd.DataFrame, split_of: dict[int, str]
) -> dict[str, Any]:
    """Conteos por clase y split (imágenes y animales), lo que cae fuera de 3.25–4.25, descartes."""

    def counts(g: pd.DataFrame) -> dict[str, dict[str, int]]:
        return {
            f"{b:.2f}": {"images": len(x), "animals": n_animals(x)}
            for b, x in g.groupby("bcs", sort=True)
        }

    bcs = np.asarray(df["bcs"], dtype=float)
    outside = _select(df, (bcs < 3.25) | (bcs > 4.25))
    by_split = {s: counts(g) for s, g in df.groupby("split", sort=True)}
    return {
        "images": len(df),
        "animals": n_animals(df),
        "by_class": counts(df),
        "by_split": by_split,
        "split_totals": {
            s: {"images": len(g), "animals": n_animals(g)}
            for s, g in df.groupby("split", sort=True)
        },
        "animals_by_split_all": {
            s: sum(1 for v in split_of.values() if v == s) for s in ("train", "val", "test")
        },
        "outside_3.25_4.25": {
            "images": len(outside),
            "animals": n_animals(outside),
            "by_split": {
                s: {"images": len(g), "animals": n_animals(g)}
                for s, g in outside.groupby("split", sort=True)
            },
        },
        "dropped_no_central_box": sorted(dropped["image_id"].tolist()),
        "dropped_ambiguous_frames": sorted(ambiguous_paths(xlsx)),
    }


def contact_sheet(
    root: Path, out: Path, *, per_class: int = 8, tile: int = 160, seed: int = SEED
) -> Path:
    """Hoja de recortes de train + val (nunca test): una fila por clase de BCS."""
    labels = pd.read_csv(root / f"labels_{SOURCE_ID}.csv", keep_default_na=False)
    classes = sorted(labels["bcs"].unique())
    sheet = Image.new("RGB", (60 + per_class * tile, len(classes) * tile), "white")
    draw = ImageDraw.Draw(sheet)
    rng = random.Random(seed)
    for k, b in enumerate(classes):
        g = labels[labels["bcs"] == b]
        # un recorte por animal cuando se puede, para ver variedad y no la misma vaca 8 veces
        firsts = g.groupby("animal_id").head(1)
        pick = rng.sample(list(firsts.index), min(per_class, len(firsts)))
        draw.text((4, k * tile + tile // 2), f"{b:.2f}", fill="black")
        for j, i in enumerate(pick):
            with Image.open(root / labels.at[i, "path"]) as im:
                im = im.convert("RGB")
                im.thumbnail((tile - 4, tile - 4))
                sheet.paste(im, (60 + j * tile, k * tile))
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out, quality=90)
    return out


def raw_sheet(src: ZenRgbdSource, out: Path, n: int = 25, seed: int = SEED) -> Path:
    """25 frames crudos al azar (train + val) con todas las cajas y la central elegida en verde."""
    plan = pd.read_csv(src.raw_dir / "plan.csv")
    split_of = split_animals(_select(plan, slice(None), ["animal_id", "bcs"]))
    dets = json.loads((src.images_dir / ".detections.json").read_text())
    keep = [
        split_of[int(a)] != "test" and i in dets
        for a, i in zip(plan["animal_id"], plan["image_id"], strict=True)
    ]
    plan = _select(plan, keep)
    pick = plan.iloc[random.Random(seed).sample(range(len(plan)), min(n, len(plan)))]
    tw, th, cols = 384, 216, 5
    sheet = Image.new("RGB", (cols * tw, -(-len(pick) // cols) * (th + 14)), "white")
    draw = ImageDraw.Draw(sheet)
    for k, r in enumerate(pick.to_dict(orient="records")):
        d = dets[r["image_id"]]
        w, h = d["size"]
        with Image.open(src.raw_dir / "rgb" / r["rgb_path"]) as im:
            im = im.convert("RGB")
        dr = ImageDraw.Draw(im)
        dr.line([(w / 2, 0), (w / 2, h)], fill="yellow", width=3)
        for b in d["boxes"]:
            dr.rectangle(b, outline="red", width=4)
        c = central_box([tuple(b) for b in d["boxes"]], w, h)
        if c:
            dr.rectangle(c, outline="lime", width=10)
        im.thumbnail((tw, th))
        x, y = (k % cols) * tw, (k // cols) * (th + 14)
        sheet.paste(im, (x, y))
        draw.text(
            (x + 2, y + th + 1), f"{r['animal_id']} {r['bcs']} {r['image_id'][9:]}", fill="black"
        )
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out, quality=88)
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Adaptador zen_rgbd → formato canónico")
    ap.add_argument("--sample", type=int, default=None, help="informativo: lo aplica el fetch")
    args = ap.parse_args(argv)
    src = ZenRgbdSource()
    rows = src.adapt()
    qa = json.loads((src.images_dir / "qa_zen_rgbd.json").read_text())
    out = runs_dir() / "T04" / "contact_sheets"
    sheets = [
        contact_sheet(src.root, out / "crops_by_class.jpg"),
        raw_sheet(src, out / "raw_25.jpg"),
    ]
    print(f"zen_rgbd (sample={args.sample}): {len(rows)} filas train+val; {qa['split_totals']}")
    print(f"fuera de 3.25–4.25: {qa['outside_3.25_4.25']}")
    print(
        f"sin caja central: {len(qa['dropped_no_central_box'])}; hojas: {[str(s) for s in sheets]}"
    )


if __name__ == "__main__":
    main()
