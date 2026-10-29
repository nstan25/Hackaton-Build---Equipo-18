"""Hojas de contactos de T27 para mirar a mano qué puntúa alto y bajo (en `$BCS_RUNS_DIR/T27/`).

Por conjunto, los recortes (lo que ve el modelo) ordenados por el puntaje Mahalanobis, con el
puntaje encima. Nada de esto va al repo (incluye imágenes de COCO y de los datasets).

    uv run python scripts/ood/contact_sheet.py
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

from bcs.paths import artifacts_dir, runs_dir

TILE = (200, 140)


def sheet(images: list[Image.Image], captions: list[str], cols: int = 6) -> Image.Image:
    rows = (len(images) + cols - 1) // cols
    out = Image.new("RGB", (cols * TILE[0], rows * (TILE[1] + 16)), "white")
    d = ImageDraw.Draw(out)
    for i, (im, cap) in enumerate(zip(images, captions, strict=True)):
        x, y = (i % cols) * TILE[0], (i // cols) * (TILE[1] + 16)
        t = im.copy()
        t.thumbnail(TILE)
        out.paste(t, (x, y + 16))
        d.text((x + 2, y + 2), cap, fill="black")
    return out


def crops(paths: list[str]) -> list[Image.Image]:
    from bcs.inference import CowLocator

    loc = CowLocator()
    out = []
    for p in paths:
        with Image.open(p) as im:
            img = im.convert("RGB")
        c = loc(img)
        if c is None:
            c = img
        out.append(c)
    return out


def run() -> None:
    out = runs_dir() / "T27"
    sc = pd.read_csv(out / "scores_pipeline.csv")
    res = json.loads((out / "results.json").read_text())
    t99 = res["thresholds"]["val_p99"]["maha"]["threshold"]
    emb = artifacts_dir() / "T27" / "emb"
    paths = {}
    for name in ("coco", "zen_raw", "reject"):
        m = pd.read_csv(emb / f"{name}.csv")
        paths.update(dict(zip(m["image_id"].astype(str), m["path"], strict=True)))
    for name in ("coco", "zen_raw"):
        s = sc[(sc["set"] == name) & sc["quality_ok"].astype(bool) & sc["detected"].astype(bool)]
        s = s.sort_values("maha")
        pick = pd.concat([s.head(12), s.tail(6)])
        ims = crops([paths[str(i)] for i in pick["image_id"]])
        caps = []
        for _, r in pick.iterrows():
            tag = (
                r["kind"]
                if isinstance(r.get("kind"), str)
                else f"err {abs(r['value'] - r['bcs']):.2f}"
            )
            caps.append(f"{'ACEPTA' if r['maha'] <= t99 else 'rechaza'} {r['maha']:.0f} {tag}")
        sheet(ims, caps).save(out / f"sheet_{name}_maha.jpg", quality=85)
    # recortes de zen val (los que ve el modelo en T25) más y menos raros
    zen = pd.read_csv(emb / "val_zen.csv")
    z = np.load(emb / "val_zen.npz")["feats"]
    from bcs.ood.methods import MahalanobisScore

    bank = np.load(emb / "bank.npz")["feats"]
    s = MahalanobisScore().fit(bank).score(z)
    order = np.argsort(s)
    pick = np.concatenate([order[:9], order[-9:]])
    ims = []
    for i in pick:
        with Image.open(zen["path"][i]) as im:
            ims.append(im.convert("RGB"))
    caps = [f"{s[i]:.0f} err {abs(zen['value'][i] - zen['bcs'][i]):.2f}" for i in pick]
    sheet(ims, caps).save(out / "sheet_val_zen_maha.jpg", quality=85)


if __name__ == "__main__":
    run()
