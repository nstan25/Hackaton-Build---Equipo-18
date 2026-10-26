"""Muestra chica de COCO val2017 (licencias abiertas) como OOD de fotos reales (T27).

Baja solo `instances_{val,train}2017.json` del zip de anotaciones (remotezip, sin bajar el zip
entero) y las imágenes elegidas una por una desde images.cocodataset.org. Todo queda en
`$BCS_ARTIFACTS_DIR/T27/ood/coco/`, fuera del repo, con `manifest.csv` (licencia, autor/URL de
Flickr, categorías). Idempotente: no vuelve a bajar lo que ya está.

    uv run python scripts/ood/fetch_coco.py
"""

from __future__ import annotations

import json
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from bcs.paths import REPO_ROOT, artifacts_dir

ANN_ZIP = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
IMG_URL = "http://images.cocodataset.org/{split}/{file}"
COW = 21
ANIMALS = {16, 17, 18, 19, 20, 22, 23, 24, 25}  # bird, cat, dog, horse, sheep, elephant, ...


def out_dir() -> Path:
    return artifacts_dir() / "T27" / "ood" / "coco"


def load_annotations(dest: Path, split: str) -> dict:
    path = dest / f"instances_{split}.json"
    if not path.exists():
        from remotezip import RemoteZip

        with RemoteZip(ANN_ZIP) as z:
            path.write_bytes(z.read(f"annotations/instances_{split}.json"))
    return json.loads(path.read_text())


def candidates(ann: dict, cfg: dict, split: str) -> pd.DataFrame:
    cats_by_img: dict[int, set[int]] = {}
    for a in ann["annotations"]:
        cats_by_img.setdefault(a["image_id"], set()).add(a["category_id"])
    names = {c["id"]: c["name"] for c in ann["categories"]}
    lic = {x["id"]: x for x in ann["licenses"]}
    rows = []
    for im in ann["images"]:
        if im["license"] not in cfg["licenses"]:
            continue
        cats = cats_by_img.get(im["id"], set())
        kind = "cow" if COW in cats else "animal" if cats & ANIMALS else "other"
        rows.append(
            {
                "coco_id": im["id"],
                "coco_split": split,
                "file": im["file_name"],
                "kind": kind,
                "categories": ";".join(sorted(names[c] for c in cats)),
                "license_id": im["license"],
                "license": lic[im["license"]]["name"],
                "license_url": lic[im["license"]]["url"],
                "flickr_url": im.get("flickr_url", ""),
                "coco_url": im["coco_url"],
            }
        )
    return pd.DataFrame(rows).sort_values("coco_id")


def choose(df: pd.DataFrame, cfg: dict, seed: int) -> pd.DataFrame:
    """Por tipo, primero val2017; si no alcanza (vacas con licencia abierta: ~20), completa con
    train2017. El modelo de producción nunca vio COCO, así que el split de COCO no importa."""
    rng = np.random.default_rng(seed)
    want = {"cow": cfg["cows"], "animal": cfg["animals"], "other": cfg["other"]}
    parts = []
    for kind, n in want.items():
        for split in ("val2017", "train2017"):
            sub = df[(df["kind"] == kind) & (df["coco_split"] == split)]
            k = min(n, len(sub))
            parts.append(sub.iloc[np.sort(rng.choice(len(sub), k, replace=False))])
            n -= k
            if n <= 0:
                break
    return pd.concat(parts, ignore_index=True)


def download(row: pd.Series, dest: Path) -> bool:
    path = dest / "images" / row["file"]
    if path.exists() and path.stat().st_size > 0:
        return True
    url = IMG_URL.format(split=row["coco_split"], file=row["file"])
    req = urllib.request.Request(url, headers={"User-Agent": "bcs"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            data = r.read()
    except OSError as e:
        print(f"falló {row['file']}: {e}", file=sys.stderr)
        return False
    path.write_bytes(data)
    return True


def main() -> int:
    cfg = yaml.safe_load((REPO_ROOT / "config" / "ood.yaml").read_text())
    dest = out_dir()
    (dest / "images").mkdir(parents=True, exist_ok=True)
    manifest = dest / "manifest.csv"
    if manifest.exists():
        done = pd.read_csv(manifest)
        if all((dest / "images" / f).exists() for f in done["file"]):
            print(f"ya está: {manifest} ({len(done)} imágenes)")
            return 0
    cands = []
    for split in ("val2017", "train2017"):
        cands.append(candidates(load_annotations(dest, split), cfg["coco"], split))
    df = choose(pd.concat(cands, ignore_index=True), cfg["coco"], cfg["seed"])
    with ThreadPoolExecutor(8) as ex:
        ok = list(ex.map(lambda r: download(r, dest), [r for _, r in df.iterrows()]))
    df = df[np.asarray(ok)]
    df.to_csv(manifest, index=False)
    (dest / "SOURCE.md").write_text(
        "# COCO val2017 (muestra T27)\n\n"
        "Fuente: https://cocodataset.org (Lin et al. 2014), val2017 (+ train2017 para\n"
        "completar vacas), imágenes de Flickr.\n"
        "Anotaciones: CC BY 4.0 (COCO Consortium). Imágenes: licencia por imagen en manifest.csv\n"
        f"(solo ids {cfg['coco']['licenses']}: CC BY / BY-SA / BY-ND 2.0).\n"
        "Uso: solo evaluación interna de OOD; nunca se commitean al repo.\n"
    )
    print(df.groupby(["kind", "license"]).size().to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
