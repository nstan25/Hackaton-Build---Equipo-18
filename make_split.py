"""Split train/val/test del dataset de BCS, agrupado por vaca/video para evitar fuga.

Esquemas de nombre:
  GS_<vaca>_<frame>.jpg, YM_<vaca>_<frame>.jpg  -> video = (fuente, vaca, clase)
  L-i<n>.jpg, R-i<n>.jpg (sin id de vaca)        -> video = tramo de n consecutivos

Grupos (lo que nunca se separa entre splits):
  - GS y YM con el mismo numero de vaca van juntos (podria ser la misma vaca).
  - L-i<n> y R-i<n> comparten numero y van juntos; un tramo de n consecutivos es un grupo.

Uso: python make_split.py <carpeta_dataset> [salida.csv]
"""
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}
SEED = 42


def load(root):
    rows = []
    for cls_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for img in cls_dir.glob("*.jpg"):
            name = img.stem
            if name[:3] in ("GS_", "YM_"):
                src, cow, frame = name.split("_")
                rows.append(dict(source=src, cow=cow, frame=int(frame), num=None))
            elif name[1:3] == "-i":
                rows.append(dict(source=name[0] + "-i", cow=None, frame=None, num=int(name[3:])))
            else:
                raise ValueError(f"nombre inesperado: {img}")
            rows[-1].update(path=f"{cls_dir.name}/{img.name}", label=float(cls_dir.name))
    return pd.DataFrame(rows)


def assign_groups(df):
    gs = df["cow"].notna()
    df.loc[gs, "video"] = df.loc[gs, "source"] + "_" + df.loc[gs, "cow"] + "_" + df.loc[gs, "label"].astype(str)
    df.loc[gs, "group"] = "cow_" + df.loc[gs, "cow"]

    # L/R: tramos de numeros consecutivos (union de ambas camaras)
    nums = sorted(df.loc[~gs, "num"].unique())
    seg, seg_of = 0, {}
    for i, n in enumerate(nums):
        if i and n != nums[i - 1] + 1:
            seg += 1
        seg_of[n] = seg
    lr_seg = df.loc[~gs, "num"].map(seg_of)
    df.loc[~gs, "video"] = "LR_seg" + lr_seg.astype(str)
    df.loc[~gs, "group"] = "LR_seg" + lr_seg.astype(str)
    return df


def split(df):
    # estrato de cada grupo = (familia de fuente, clase mayoritaria)
    fam = df["source"].map(lambda s: "LR" if s.endswith("-i") else "GSYM")
    g = df.assign(fam=fam).groupby("group").agg(
        n=("path", "size"), fam=("fam", "first"), label=("label", lambda x: x.mode()[0]))
    rng = random.Random(SEED)
    assign = {}
    for _, sub in g.groupby(["fam", "label"]):
        groups = list(sub.index)
        rng.shuffle(groups)
        total = sub["n"].sum()
        filled = Counter()
        for grp in groups:
            # va al split que esta mas lejos de su cuota
            s = max(RATIOS, key=lambda k: RATIOS[k] * total - filled[k])
            assign[grp] = s
            filled[s] += sub.at[grp, "n"]
    df["split"] = df["group"].map(assign)
    return df


def main():
    root = Path(sys.argv[1])
    out = Path(sys.argv[2] if len(sys.argv) > 2 else "split.csv")
    df = split(assign_groups(load(root)))

    assert df.groupby("group")["split"].nunique().max() == 1, "un grupo quedo en dos splits"

    print(f"Imagenes: {len(df)}  |  videos: {df['video'].nunique()}  |  grupos: {df['group'].nunique()}")
    print("\nVideos por fuente:")
    print(df.groupby("source")["video"].nunique().to_string())
    print("\nImagenes por split y clase:")
    print(pd.crosstab(df["label"], df["split"], margins=True).to_string())
    print("\nProporcion por split:")
    print(df["split"].value_counts(normalize=True).round(3).to_string())
    print("\nVideos por split:")
    print(df.groupby("split")["video"].nunique().to_string())

    cols = ["path", "label", "split", "source", "cow", "frame", "num", "video", "group"]
    df[cols].sort_values(["split", "path"]).to_csv(out, index=False)
    print(f"\nGuardado en {out}")


if __name__ == "__main__":
    main()
