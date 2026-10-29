"""T30: recortes de `zen_rgbd` como los daría el uso real (una vaca sola, de atrás → detector de
producción → recorte), para medir y para entrenar. Solo train y val de `splits_zen_rgbd.csv`;
nunca test ni `holdout_labels_*`. Todo sale a `$BCS_ARTIFACTS_DIR/T30/` (nada al repo).

- **solo**: el frame crudo (1920×1080) recortado a la caja etiquetada de la vaca central
  (`labels_zen_rgbd.bbox`, la de T04) con `SOLO_MARGIN` a cada lado en x. Simula la foto de
  celular de una sola vaca de atrás: es lo que el usuario le manda a la app.
- **grupa**: `solo` → `models/cow_det.pt` (conf `DETECTOR_CONF`, como `inference.CowLocator`) →
  caja +10% (`prep_crops.expand_bbox`), lado mayor 320. Es lo que ve el modelo en producción.
- **alto**: la misma caja del detector, ampliada hacia arriba (`--up`) para acercarse al encuadre
  vertical de los recortes de entrenamiento de T25 (variante B, recorte en inferencia).

    uv run python scripts/zen_crops/crops.py detect --set zen_solo|zen_raw|sdb_raw   # dets_<set>
    uv run python scripts/zen_crops/crops.py crops --name grupa --fallback half     # entrenar
    uv run python scripts/zen_crops/crops.py sheet                                  # contactos
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

from bcs.paths import artifacts_dir, data_dir, runs_dir

TASK = "T30"
SOLO_MARGIN = 0.10  # margen horizontal de la caja etiquetada (proporción del ancho)
MAX_SIDE = 320  # prep_crops.MAX_SIDE


def out_dir() -> Path:
    return artifacts_dir() / TASK


def parse_bbox(s: str) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = (int(float(v)) for v in str(s).split(","))
    return x0, y0, x1, y1


def solo_box(
    bbox: Sequence[int], w: int, h: int, margin: float = SOLO_MARGIN
) -> tuple[int, int, int, int]:
    """Caja de la "vaca sola" dentro del frame: la etiquetada con `margin` en x (en y ya ocupa
    casi todo el alto del frame), recortada a la imagen."""
    x0, y0, x1, y1 = bbox
    dx = (x1 - x0) * margin
    return max(0, int(x0 - dx)), max(0, int(y0)), min(w, int(x1 + dx)), min(h, int(y1))


def expand(
    box: Sequence[float], w: int, h: int, margin: float = 0.10, up: float = 0.0, down: float = 0.0
) -> tuple[int, int, int, int]:
    """`prep_crops.expand_bbox` (margen `margin` a cada lado) + `up`/`down` extra (proporción del
    alto de la caja) arriba/abajo. Con `up = down = 0` es exactamente el recorte de producción."""
    x0, y0, x1, y1 = (int(v) for v in box)
    dx, dy = (x1 - x0) * margin, (y1 - y0) * margin
    return (
        max(0, int(x0 - dx)),
        max(0, int(y0 - dy - (y1 - y0) * up)),
        min(w, int(x1 + dx)),
        min(h, int(y1 + dy + (y1 - y0) * down)),
    )


def thumb(im: Image.Image, side: int = MAX_SIDE) -> Image.Image:
    im = im.copy()
    im.thumbnail((side, side))
    return im


def zen_rows(root: Path | None = None) -> pd.DataFrame:
    """Filas train + val de zen con la ruta al frame crudo (`plan.csv`). Nunca test."""
    root = root or data_dir()
    labels = pd.read_csv(root / "labels_zen_rgbd.csv", dtype={"date": str})
    splits = pd.read_csv(root / "splits_zen_rgbd.csv")
    df = labels.merge(splits, on="image_id", how="inner")
    df = df.loc[df["split"].isin(["train", "val"])].copy()
    plan = pd.read_csv(root / "raw" / "zen_rgbd" / "plan.csv")[["image_id", "rgb_path"]]
    df = df.merge(plan, on="image_id", how="left")
    df["raw"] = [str(root / "raw" / "zen_rgbd" / "rgb" / p) for p in df["rgb_path"]]
    return pd.DataFrame(df).reset_index(drop=True)


def load_solo(row: Any) -> Image.Image:
    with Image.open(row["raw"]) as im:
        frame = im.convert("RGB")
    return frame.crop(solo_box(parse_bbox(row["bbox"]), *frame.size))


class Detector:
    """`models/cow_det.pt` con la misma regla que `inference.CowLocator` (devuelve la caja)."""

    def __init__(self) -> None:
        from ultralytics import YOLO

        from bcs.inference import DETECTOR, DETECTOR_CONF
        from bcs.models.efficientnet import team_module

        self.model = YOLO(str(DETECTOR))
        self.conf = DETECTOR_CONF
        self.prep_yolo = team_module("prep_yolo")

    def __call__(self, im: Image.Image) -> list[float] | None:
        return self.prep_yolo.detect_cow(self.model, im, conf=self.conf)


SETS = ("zen_solo", "zen_raw", "sdb_raw")


def sdb_val_rows(root: Path | None = None) -> pd.DataFrame:
    """sdb val, ≤ 2 frames por grupo (los de `pipe_val` de T28: `T28/val_labels_sdb.csv`), con la
    foto cruda (`raw/sdb_bcs/dataset/<clase>/<archivo>.jpg`)."""
    root = root or data_dir()
    df = pd.read_csv(artifacts_dir() / "T28" / "val_labels_sdb.csv")
    base = root / "raw" / "sdb_bcs" / "dataset"
    dirs = sorted(d for d in base.iterdir() if d.is_dir())
    names = [f"{i.removeprefix('sdb_bcs_')}.jpg" for i in df["image_id"]]
    df["raw"] = [str(next((d / n for d in dirs if (d / n).exists()), "")) for n in names]
    df["split"] = "val"
    return pd.DataFrame(df.loc[df["raw"] != ""]).reset_index(drop=True)


def set_rows(name: str) -> pd.DataFrame:
    """Filas del conjunto: `zen_solo` (train + val, la vaca sola), `zen_raw` (val, frame entero de
    la sala) o `sdb_raw` (val, foto cruda de sdb)."""
    if name == "zen_solo":
        return zen_rows()
    if name == "zen_raw":
        df = zen_rows()
        return pd.DataFrame(df.loc[df["split"] == "val"]).reset_index(drop=True)
    if name == "sdb_raw":
        return sdb_val_rows()
    raise ValueError(name)


def load_src(name: str, row: Any) -> Image.Image:
    """La foto que recibe el detector en el conjunto `name`."""
    if name == "zen_solo":
        return load_solo(row)
    with Image.open(row["raw"]) as im:
        return im.convert("RGB")


def dets_path(name: str) -> Path:
    return out_dir() / f"dets_{name}.csv"


def cmd_detect(name: str = "zen_solo", limit: int | None = None) -> Path:
    """Para cada fila: tamaño de la foto y caja del detector (en coordenadas de esa foto)."""
    df = set_rows(name)
    if limit:
        df = df.groupby("split", group_keys=False).head(limit)
    dest = dets_path(name)
    dest.parent.mkdir(parents=True, exist_ok=True)
    done = pd.read_csv(dest) if dest.exists() else pd.DataFrame()
    seen = set(done["image_id"]) if len(done) else set()
    det = Detector()
    rows: list[dict[str, Any]] = []
    todo = df.loc[~df["image_id"].isin(seen)]
    for i, (_, r) in enumerate(todo.iterrows(), 1):
        src = load_src(name, r)
        box = det(src)
        rows.append(
            {
                "image_id": r["image_id"],
                "solo_w": src.width,
                "solo_h": src.height,
                "det": "" if box is None else ",".join(f"{v:.1f}" for v in box),
            }
        )
        if i % 200 == 0:
            done = pd.concat([done, pd.DataFrame(rows)], ignore_index=True)
            done.to_csv(dest, index=False)
            rows = []
            print(f"  {i}/{len(todo)}", flush=True)
    done = pd.concat([done, pd.DataFrame(rows)], ignore_index=True)
    done.to_csv(dest, index=False)
    found = done["det"].fillna("").astype(str).str.len() > 0
    print(f"{name}: {found.mean():.1%} con caja de {len(done)} → {dest}")
    return dest


def det_table(name: str = "zen_solo") -> pd.DataFrame:
    df = set_rows(name).merge(pd.read_csv(dets_path(name)), on="image_id", how="inner")
    df["det"] = df["det"].fillna("").astype(str)
    return pd.DataFrame(df)


def crop_from_solo(
    solo: Image.Image, det: str, up: float = 0.0, down: float = 0.0
) -> Image.Image | None:
    if not det:
        return None
    box = [float(v) for v in det.split(",")]
    return thumb(solo.crop(expand(box, *solo.size, up=up, down=down)))


def cmd_crops(name: str, up: float, down: float, fallback: str) -> Path:
    """Recortes `name` de zen (train + val) desde `solo` con la caja del detector. Sin caja,
    `fallback`: `skip` (no hay recorte) o `half` (parte inferior de la vaca sola, `half_box`)."""
    df = det_table("zen_solo")
    d = out_dir() / "crops" / name
    d.mkdir(parents=True, exist_ok=True)
    rows = []
    for _, r in df.iterrows():
        dest = d / f"{r['image_id']}.jpg"
        kind = "det" if r["det"] else fallback
        if kind == "skip":
            continue
        if not dest.exists():
            solo = load_solo(r)
            crop = (
                crop_from_solo(solo, r["det"], up, down)
                if r["det"]
                else thumb(solo.crop(half_box(*solo.size)))
            )
            assert crop is not None
            crop.save(dest, quality=95)
        rows.append({"image_id": r["image_id"], "path": str(dest), "kind": kind})
    man = pd.DataFrame(rows)
    man.to_csv(d / "manifest.csv", index=False)
    print(f"{name}: {len(man)} recortes ({(man['kind'] == 'det').mean():.1%} del detector) → {d}")
    return d / "manifest.csv"


def half_box(w: int, h: int, top: float = 0.45) -> tuple[int, int, int, int]:
    """Sin detección: la parte inferior de la vaca sola desde `top` del alto (la grupa en la toma
    de la rotativa, que mira desde arriba y atrás)."""
    return 0, int(h * top), w, h


# ---------------------------------------------------------------- hojas de contactos


def sheet(
    images: list[Image.Image], captions: list[str], cols: int = 8, tile: int = 180
) -> Image.Image:
    rows = (len(images) + cols - 1) // cols
    out = Image.new("RGB", (cols * tile, rows * (tile + 16)), "white")
    d = ImageDraw.Draw(out)
    for i, (im, cap) in enumerate(zip(images, captions, strict=True)):
        x, y = (i % cols) * tile, (i // cols) * (tile + 16)
        t = thumb(im, tile)
        out.paste(t, (x + (tile - t.width) // 2, y + 16))
        d.text((x + 2, y + 2), cap, fill="black")
    return out


def cmd_sheet(n: int = 24, seed: int = 0) -> list[Path]:
    """Hoja de `solo` con la caja del detector en verde, y las de los recortes de cada variante,
    para las mismas `n` imágenes de val al azar; más una de recortes de sdb para comparar."""
    df = det_table()
    val = df.loc[df["split"] == "val"].sample(n, random_state=seed)
    out = runs_dir() / TASK
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    ims, caps = [], []
    for _, r in val.iterrows():
        solo = load_solo(r)
        dr = ImageDraw.Draw(solo)
        if r["det"]:
            box = expand([float(v) for v in r["det"].split(",")], *solo.size)
            dr.rectangle(box, outline=(0, 255, 0), width=6)
        ims.append(solo)
        caps.append(f"{r['bcs']:.2f} {'det' if r['det'] else 'SIN det'}")
    p = out / "sheet_solo_det.jpg"
    sheet(ims, caps).save(p, quality=85)
    paths.append(p)
    for name in sorted(x.name for x in (out_dir() / "crops").glob("*") if x.is_dir()):
        man = pd.read_csv(out_dir() / "crops" / name / "manifest.csv").set_index("image_id")
        ims, caps = [], []
        for _, r in val.iterrows():
            if r["image_id"] in man.index:
                with Image.open(man.loc[r["image_id"], "path"]) as im:
                    ims.append(im.convert("RGB"))
                caps.append(f"{r['bcs']:.2f} {man.loc[r['image_id'], 'kind']}")
        p = out / f"sheet_{name}.jpg"
        sheet(ims, caps).save(p, quality=85)
        paths.append(p)
    ims, caps = [], []
    for _, r in val.iterrows():
        with Image.open(data_dir() / r["path"]) as im:
            ims.append(im.convert("RGB"))
        caps.append(f"{r['bcs']:.2f} T04")
    p = out / "sheet_t04_entera.jpg"
    sheet(ims, caps).save(p, quality=85)
    paths.append(p)
    sdb = pd.read_csv(data_dir() / "labels_sdb_bcs.csv").merge(
        pd.read_csv(data_dir() / "splits_sdb_bcs.csv"), on="image_id"
    )
    sdb = sdb.loc[sdb["split"] == "val"].sample(n, random_state=seed)
    ims = []
    for rel in sdb["path"]:
        with Image.open(data_dir() / rel) as im:
            ims.append(im.convert("RGB"))
    p = out / "sheet_sdb_val.jpg"
    sheet(ims, [f"{b:.2f} sdb" for b in sdb["bcs"]]).save(p, quality=85)
    paths.append(p)
    for p in paths:
        print(f"→ {p}")
    return paths


def box_stats() -> pd.DataFrame:
    """Proporciones de la caja del detector dentro de la vaca sola (ancho/alto, alto relativo,
    posición vertical), por split."""
    df = det_table()
    df = df.loc[df["det"] != ""].copy()
    b = np.array([[float(v) for v in s.split(",")] for s in df["det"]])
    df["aspect"] = (b[:, 2] - b[:, 0]) / (b[:, 3] - b[:, 1])
    df["rel_h"] = (b[:, 3] - b[:, 1]) / df["solo_h"].to_numpy()
    df["rel_top"] = b[:, 1] / df["solo_h"].to_numpy()
    df["rel_w"] = (b[:, 2] - b[:, 0]) / df["solo_w"].to_numpy()
    return pd.DataFrame(df)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="scripts/zen_crops/crops.py")
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("detect")
    d.add_argument("--set", choices=SETS, default="zen_solo")
    d.add_argument("--limit", type=int, default=None, help="n por split (prueba)")
    c = sub.add_parser("crops")
    c.add_argument("--name", required=True)
    c.add_argument("--up", type=float, default=0.0)
    c.add_argument("--down", type=float, default=0.0)
    c.add_argument("--fallback", choices=["skip", "half"], default="skip")
    s = sub.add_parser("sheet")
    s.add_argument("--n", type=int, default=24)
    sub.add_parser("stats")
    a = ap.parse_args(argv)
    if a.cmd == "detect":
        cmd_detect(a.set, a.limit)
    elif a.cmd == "crops":
        cmd_crops(a.name, a.up, a.down, a.fallback)
    elif a.cmd == "sheet":
        cmd_sheet(a.n)
    elif a.cmd == "stats":
        st = box_stats()
        cols = ["aspect", "rel_h", "rel_top", "rel_w"]
        print(st.groupby("split")[cols].describe().T.round(3).to_string())
        n = det_table().groupby("split")["det"].apply(lambda s: (s != "").mean())
        print("tasa de detección por split:\n" + n.round(3).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
