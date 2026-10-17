"""Recorta cada imagen por su bbox (+ margen) y la guarda en crops/{split}/{clase}/.

Uso: python prep_crops.py [dataset_split] [split.csv] [crops]
"""
import sys
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
from PIL import Image

MARGIN = 0.10
MAX_SIDE = 320


def read_bbox(xml_path):
    box = ET.parse(xml_path).find("object/bndbox")
    return [int(float(box.find(k).text)) for k in ("xmin", "ymin", "xmax", "ymax")]


def expand_bbox(bbox, w, h, margin=MARGIN):
    x0, y0, x1, y1 = bbox
    dx, dy = (x1 - x0) * margin, (y1 - y0) * margin
    return max(0, int(x0 - dx)), max(0, int(y0 - dy)), min(w, int(x1 + dx)), min(h, int(y1 + dy))


def crop_one(src, dst):
    if dst.exists():
        return
    img = Image.open(src).convert("RGB")
    crop = img.crop(expand_bbox(read_bbox(src.with_suffix(".xml")), *img.size))
    crop.thumbnail((MAX_SIDE, MAX_SIDE))
    dst.parent.mkdir(parents=True, exist_ok=True)
    crop.save(dst, quality=95)


def main():
    src_root = Path(sys.argv[1] if len(sys.argv) > 1 else "dataset_split")
    df = pd.read_csv(sys.argv[2] if len(sys.argv) > 2 else "split.csv", dtype={"cow": str})
    out = Path(sys.argv[3] if len(sys.argv) > 3 else "crops")

    df["crop"] = [f"{out.as_posix()}/{s}/{p}" for s, p in zip(df["split"], df["path"])]
    jobs = [(src_root / s / p, Path(c)) for s, p, c in zip(df["split"], df["path"], df["crop"])]
    with ThreadPoolExecutor(8) as ex:
        for i, _ in enumerate(ex.map(lambda j: crop_one(*j), jobs), 1):
            if i % 10000 == 0:
                print(f"{i}/{len(jobs)}", flush=True)

    df[["crop", "path", "label", "split", "source", "video"]].to_csv(out / "manifest.csv", index=False)
    print(f"Listo: {len(df)} recortes en {out}")


if __name__ == "__main__":
    main()
