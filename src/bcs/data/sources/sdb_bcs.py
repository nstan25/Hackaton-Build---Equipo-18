"""Fuente `sdb_bcs` (ScienceDB 16704 V3): fetch del .rar y adaptador al formato canónico (§1.1).

El adaptador parte de los recortes del README (`raw/sdb_bcs_pre/crops/`, bbox del xml +10%,
lado máx 320 px), tapa con negro las franjas de texto sobreimpreso (fecha y "BCS", trampa 1 de
§1.2) y escribe:

    images/sdb_bcs/<image_id>.jpg
    labels_sdb_bcs.csv           train + val (esquema de bcs.data.schema)
    splits_sdb_bcs.csv           image_id,split para train/val/test
    holdout_labels_sdb_bcs.csv   test, mismo esquema (solo lo lee `make eval-final`)

`group_id` = "sdb_bcs:" + columna `group` de split.csv (G0).

Uso: python -m bcs.data.sources.sdb_bcs [--sample N]
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import random
import subprocess
import sys
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

from bcs.data.preprocess_sdb import MANIFEST, RAR_SHA256, SOURCE_ID, sha256_file
from bcs.data.schema import LABEL_COLUMNS, LabelRow
from bcs.paths import REPO_ROOT, data_dir, runs_dir

RAR_SIZE = 4_407_548_599
URL_API = (
    "https://www.scidb.cn/api/sdb-filetree-service/getAllUrl?dataSetId="
    "16b8bdaf31ee4c8b9891fc7e9df6e41c&type=journalDataset&version=V3&global=en"
)
URL = (
    "https://download.scidb.cn/download?fileId=4516a6866954512e08446bc3504b32c9"
    "&path=/V3/dataset.rar&fileName=dataset.rar"
)
SCALE = "dairy_1_5"
VIEW = "rear_dorsal"

# Franjas de texto sobreimpreso, en coordenadas de un frame de 1024x576 (se escalan para los
# frames de 1364x768). Medidas sobre 400 frames por familia (frecuencia de píxeles de texto) y
# mirando tiras de 20 frames; con margen. L-i / R-i no traen texto.
OVERLAYS: dict[str, tuple[tuple[int, int, int, int], ...]] = {
    "GS": ((0, 0, 400, 34), (918, 538, 982, 572)),  # fecha arriba-izq, "BCS" abajo-der
    "YM": ((0, 0, 410, 64), (0, 478, 112, 528), (700, 486, 836, 548)),  # fecha, "BCS2", "BCS1"
}
REF_W, REF_H = 1024, 576


# --- fetch -------------------------------------------------------------------------------


def _curl(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["curl", "-L", "-C", "-", "-A", "Mozilla/5.0", "-o", str(dest), url], check=True)


def fetch_rar(
    rar: Path,
    lock: Path,
    *,
    size: int = RAR_SIZE,
    sha256: str = RAR_SHA256,
    download: Callable[[str, Path], None] = _curl,
) -> dict[str, Any]:
    """Idempotente: si el .rar está con el tamaño y el sha256 esperados, no baja nada."""
    if not rar.exists() or rar.stat().st_size < size:
        download(URL, rar)  # curl -C - reanuda una descarga parcial
    got_size = rar.stat().st_size
    if got_size != size:
        raise RuntimeError(f"{rar}: tamaño {got_size} != {size}")
    got_sha = sha256_file(rar)
    if got_sha != sha256:
        raise RuntimeError(f"{rar}: sha256 {got_sha} != {sha256}")
    entry = {
        "url": URL,
        "url_api": URL_API,
        "doi": "10.57760/sciencedb.16704",
        "version": "V3",
        "title": "Dairy cow BCS target detection data set (Huang et al.)",
        "size": got_size,
        "sha256": got_sha,
        # fecha del archivo, no de la corrida: la misma copia da el mismo lock
        "downloaded": datetime.fromtimestamp(rar.stat().st_mtime, UTC).date().isoformat(),
        "license": "pública en ScienceDB; verificar términos en la página del DOI y citar",
    }
    content = json.dumps({SOURCE_ID: entry}, indent=2, ensure_ascii=False) + "\n"
    if not lock.exists() or lock.read_text() != content:
        lock.write_text(content)
    return entry


# --- tapado del texto ----------------------------------------------------------------------


def family(name: str) -> str:
    return name[:2] if name[:3] in ("GS_", "YM_") else "LR"


def overlay_boxes(fam: str, w: int, h: int) -> list[tuple[int, int, int, int]]:
    """Franjas de texto en píxeles de una imagen original de w x h."""
    sx, sy = w / REF_W, h / REF_H
    return [
        (int(x0 * sx), int(y0 * sy), int(np.ceil(x1 * sx)), int(np.ceil(y1 * sy)))
        for x0, y0, x1, y1 in OVERLAYS.get(fam, ())
    ]


def boxes_in_crop(
    boxes: list[tuple[int, int, int, int]],
    crop_box: tuple[int, int, int, int],
    crop_size: tuple[int, int],
) -> list[tuple[int, int, int, int]]:
    """Proyecta las franjas al recorte (ya achicado a `crop_size`), redondeando hacia afuera."""
    cx0, cy0, cx1, cy1 = crop_box
    sx, sy = crop_size[0] / (cx1 - cx0), crop_size[1] / (cy1 - cy0)
    out = []
    for x0, y0, x1, y1 in boxes:
        ix0, iy0, ix1, iy1 = max(x0, cx0), max(y0, cy0), min(x1, cx1), min(y1, cy1)
        if ix0 >= ix1 or iy0 >= iy1:
            continue
        out.append(
            (
                max(0, int((ix0 - cx0) * sx) - 1),
                max(0, int((iy0 - cy0) * sy) - 1),
                min(crop_size[0], int(np.ceil((ix1 - cx0) * sx)) + 1),
                min(crop_size[1], int(np.ceil((iy1 - cy0) * sy)) + 1),
            )
        )
    return out


def mask_overlay(img: Image.Image, boxes: Iterable[tuple[int, int, int, int]]) -> Image.Image:
    img = img.convert("RGB")
    draw = ImageDraw.Draw(img)
    for x0, y0, x1, y1 in boxes:
        draw.rectangle([x0, y0, x1 - 1, y1 - 1], fill=(0, 0, 0))
    return img


# --- adaptador ---------------------------------------------------------------------------


def _load_prep_crops(scripts_dir: Path) -> ModuleType:
    """`prep_crops.py` del equipo, importado tal cual (misma geometría de recorte)."""
    spec = importlib.util.spec_from_file_location("prep_crops", scripts_dir / "prep_crops.py")
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    # Los scripts del equipo son de solo lectura: que el import no deje __pycache__ al lado.
    prev, sys.dont_write_bytecode = sys.dont_write_bytecode, True
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.dont_write_bytecode = prev
    return mod


def image_id(path: str) -> str:
    """'3.25/GS_1003_1.jpg' → 'sdb_bcs_GS_1003_1' (los nombres no se repiten entre clases)."""
    return f"{SOURCE_ID}_{Path(path).stem}"


def animal_id(row: pd.Series) -> str:
    if family(str(row["path"]).split("/")[1]) == "LR":  # L/R no traen id de vaca: el tramo
        return f"{SOURCE_ID}:{row['group']}"
    return f"{SOURCE_ID}:{row['source']}_{row['cow']}"


def _dhash(img: Image.Image) -> int:
    a = np.asarray(img.convert("L").resize((9, 8), Image.Resampling.BILINEAR), dtype=np.int16)
    bits = (a[:, 1:] > a[:, :-1]).flatten()
    return int(sum(1 << i for i, b in enumerate(bits) if b))


def near_duplicates_across_splits(
    hashes: dict[str, int], splits: dict[str, str], max_dist: int = 4
) -> list[tuple[str, str, int]]:
    """Pares de imágenes de splits distintos con dHash a distancia <= max_dist.

    Pigeonhole: si dos hashes de 64 bits difieren en <= max_dist bits, coinciden en al menos uno
    de max_dist + 1 tramos; solo se comparan los que comparten un tramo.
    """
    ids = sorted(hashes)
    chunks = max_dist + 1
    width = -(-64 // chunks)
    found: dict[tuple[str, str], int] = {}
    for c in range(chunks):
        buckets: dict[int, list[str]] = {}
        for i in ids:
            buckets.setdefault((hashes[i] >> (c * width)) & ((1 << width) - 1), []).append(i)
        for members in buckets.values():
            for a_i, a in enumerate(members):
                for b in members[a_i + 1 :]:
                    if splits[a] == splits[b]:
                        continue
                    d = (hashes[a] ^ hashes[b]).bit_count()
                    if d <= max_dist:
                        found[(a, b)] = d
    return sorted((a, b, d) for (a, b), d in found.items())


def compare_documented(
    found: list[tuple[str, str, int]], documented: Iterable[tuple[str, str]], present: set[str]
) -> dict[str, list[tuple[str, str]]]:
    """Casi-duplicados encontrados vs. documentados (restringidos a las imágenes presentes)."""
    got = {(a, b) for a, b, _ in found}
    doc = {(min(a, b), max(a, b)) for a, b in documented if a in present and b in present}
    return {"undocumented": sorted(got - doc), "missing": sorted(doc - got)}


class SdbBcsSource:
    source_id = SOURCE_ID

    def __init__(
        self,
        root: Path | None = None,
        scripts_dir: Path | None = None,
        workers: int = 8,
    ) -> None:
        self.root = root or data_dir()
        self.scripts_dir = scripts_dir or REPO_ROOT
        self.workers = workers

    @property
    def pre_dir(self) -> Path:
        return self.root / "raw" / f"{SOURCE_ID}_pre"

    @property
    def images_dir(self) -> Path:
        return self.root / "images" / SOURCE_ID

    def fetch(self, sample: int | None = None) -> None:
        """El .rar es un único archivo: `sample` no cambia la descarga (lo aplica preprocess)."""
        fetch_rar(
            self.root / "raw" / SOURCE_ID / "dataset.rar",
            self.root / f"sources.lock.{SOURCE_ID}.json",
        )

    def _table(self) -> pd.DataFrame:
        split = pd.read_csv(self.pre_dir / "split.csv", dtype={"cow": str})
        crops = pd.read_csv(self.pre_dir / "crops" / "manifest.csv")
        df = split.merge(crops[["path", "crop"]], on="path", how="left", validate="one_to_one")
        if bool(df["crop"].isna().any()):
            raise RuntimeError("hay filas de split.csv sin recorte: correr `make preprocess`")
        df["image_id"] = df["path"].map(image_id)
        df["animal_id"] = df.apply(animal_id, axis=1)
        df["group_id"] = SOURCE_ID + ":" + df["group"].astype(str)
        return df

    def documented_pairs(self) -> tuple[str | None, list[tuple[str, str]]]:
        """Pares de `known_leakage.pairs_file` del manifiesto de preprocess (ninguno si no hay)."""
        leak = json.loads((self.pre_dir / MANIFEST).read_text()).get("known_leakage")
        if not leak:
            return None, []
        path = Path(leak["pairs_file"])
        if sha256_file(path) != leak["pairs_sha256"]:
            raise RuntimeError(f"{path} no coincide con el sha256 de known_leakage")
        return str(path), [(str(p[0]), str(p[1])) for p in json.loads(path.read_text())]

    def _process(self, row: dict[str, Any], prep: ModuleType, force: bool) -> dict[str, Any]:
        src = self.pre_dir / "dataset_split" / row["split"] / row["path"]
        dst = self.images_dir / f"{row['image_id']}.jpg"
        with Image.open(src) as orig:
            w, h = orig.size
        bbox = prep.read_bbox(src.with_suffix(".xml"))
        crop_box = prep.expand_bbox(bbox, w, h)
        with Image.open(self.pre_dir / row["crop"]) as crop:
            size = crop.size
            boxes = boxes_in_crop(overlay_boxes(family(src.name), w, h), crop_box, size)
            if force or not dst.exists():
                mask_overlay(crop, boxes).save(dst, quality=95)
        with Image.open(dst) as out:
            dh = _dhash(out)
        return {"bbox": ",".join(map(str, bbox)), "dhash": dh, "masked": boxes}

    def adapt(self) -> list[LabelRow]:
        """Escribe imágenes y CSVs; devuelve las filas de train + val (`labels_sdb_bcs.csv`)."""
        df = self._table()
        self.images_dir.mkdir(parents=True, exist_ok=True)
        stamp = self.images_dir / ".adapt.json"
        code = sha256_file(Path(__file__))
        force = not (stamp.exists() and json.loads(stamp.read_text()).get("code") == code)
        prep = _load_prep_crops(self.scripts_dir)
        rows = df.to_dict(orient="records")
        with ThreadPoolExecutor(self.workers) as ex:
            done = list(ex.map(lambda r: self._process(r, prep, force), rows))
        _write_text(stamp, json.dumps({"code": code}) + "\n")
        keep = {f"{i}.jpg" for i in df["image_id"]}
        for p in self.images_dir.glob("*.jpg"):  # restos de una corrida con otro --sample
            if p.name not in keep:
                p.unlink()

        df["bbox"] = [d["bbox"] for d in done]
        labels = pd.DataFrame(
            {
                "image_id": df["image_id"],
                "path": [f"images/{SOURCE_ID}/{i}.jpg" for i in df["image_id"]],
                "source": SOURCE_ID,
                "animal_id": df["animal_id"],
                "group_id": df["group_id"],
                "bcs": df["label"].astype(float),
                "scale": SCALE,
                "view": VIEW,
                "scorer": "",
                "date": "",
                "bbox": df["bbox"],
            }
        ).loc[:, list(LABEL_COLUMNS)]
        is_test = (df["split"] == "test").to_numpy()
        order = np.argsort(df["image_id"].to_numpy(), kind="stable")
        labels, df = labels.iloc[order], df.iloc[order]
        is_test = is_test[order]
        _write_csv(labels[~is_test], self.root / f"labels_{SOURCE_ID}.csv")
        _write_csv(labels[is_test], self.root / f"holdout_labels_{SOURCE_ID}.csv")
        _write_csv(df[["image_id", "split"]], self.root / f"splits_{SOURCE_ID}.csv")

        hashes = dict(zip(df["image_id"], [done[i]["dhash"] for i in order], strict=True))
        splits = dict(zip(df["image_id"], df["split"], strict=True))
        dups = near_duplicates_across_splits(hashes, splits)
        doc_file, documented = self.documented_pairs()
        diff = compare_documented(dups, documented, set(splits))
        qa = {
            "images": len(df),
            "by_split": {k: int(v) for k, v in df["split"].value_counts().sort_index().items()},
            "masked_images": sum(bool(d["masked"]) for d in done),
            "near_duplicates_across_splits": {
                "max_dist": 4,
                "n": len(dups),
                "documented_in": doc_file,
                **diff,
                "pairs": dups,
            },
        }
        _write_text(self.images_dir / "qa_sdb_bcs.json", json.dumps(qa, indent=2) + "\n")
        if diff["undocumented"] or diff["missing"]:
            raise RuntimeError(
                f"casi-duplicados entre splits distintos de los documentados ({doc_file}): "
                f"{len(diff['undocumented'])} nuevos, {len(diff['missing'])} faltantes; "
                f"ver {self.images_dir / 'qa_sdb_bcs.json'}"
            )
        train_val = labels[~is_test].to_dict(orient="records")
        return [LabelRow(**{str(k): v for k, v in r.items()}) for r in train_val]


def _write_text(path: Path, content: str) -> None:
    if not path.exists() or path.read_text() != content:  # no reescribir si no cambió
        path.write_text(content)


def _write_csv(df: pd.DataFrame, path: Path) -> None:
    _write_text(path, df.to_csv(index=False))


def contact_sheet(root: Path, out: Path, n: int = 25, tile: int = 200, seed: int = 42) -> Path:
    """Hoja de contactos de imágenes procesadas de train + val (nunca test)."""
    labels = pd.read_csv(root / f"labels_{SOURCE_ID}.csv", keep_default_na=False)
    pick = random.Random(seed).sample(range(len(labels)), min(n, len(labels)))
    cols = 5
    rows = -(-len(pick) // cols)
    sheet = Image.new("RGB", (cols * tile, rows * (tile + 16)), "white")
    draw = ImageDraw.Draw(sheet)
    for k, i in enumerate(pick):
        r = labels.iloc[i]
        with Image.open(root / r["path"]) as im:
            im = im.convert("RGB")
            im.thumbnail((tile, tile))
            x, y = (k % cols) * tile, (k // cols) * (tile + 16)
            sheet.paste(im, (x, y))
        draw.text((x + 2, y + tile + 2), f"{r['bcs']} {r['image_id'][8:]}", fill="black")
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out, quality=90)
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Adaptador sdb_bcs → formato canónico")
    ap.add_argument("--sample", type=int, default=None, help="informativo: lo aplica preprocess")
    args = ap.parse_args(argv)
    src = SdbBcsSource()
    rows = src.adapt()
    qa = json.loads((src.images_dir / "qa_sdb_bcs.json").read_text())
    sheet = contact_sheet(src.root, runs_dir() / "T03" / "contact_sheets" / "sdb_bcs_pre.jpg")
    print(f"sdb_bcs (sample={args.sample}): {len(rows)} filas train+val; qa={qa['by_split']}")
    nd = qa["near_duplicates_across_splits"]
    print(f"tapadas: {qa['masked_images']}; casi-duplicados entre splits: {nd['n']}, "
          f"todos documentados en {nd['documented_in']}")  # fmt: skip
    print(f"hoja de contactos: {sheet}")


if __name__ == "__main__":
    main()
