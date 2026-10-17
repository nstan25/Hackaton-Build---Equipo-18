"""Copia el dataset a <salida>/{train,val,test}/<clase>/ segun split.csv (imagen + xml).

Uso: python apply_split.py <carpeta_dataset> [split.csv] [salida]
"""
import shutil
import sys
from pathlib import Path

import pandas as pd


def main():
    src = Path(sys.argv[1])
    csv = Path(sys.argv[2] if len(sys.argv) > 2 else "split.csv")
    out = Path(sys.argv[3] if len(sys.argv) > 3 else "dataset_split")

    df = pd.read_csv(csv)
    for i, (path, split) in enumerate(zip(df["path"], df["split"]), 1):
        img = src / path
        dst = out / split / path  # path ya es <clase>/<archivo>
        dst.parent.mkdir(parents=True, exist_ok=True)
        for f, d in ((img, dst), (img.with_suffix(".xml"), dst.with_suffix(".xml"))):
            if not d.exists():
                shutil.copy2(f, d)
        if i % 5000 == 0:
            print(f"{i}/{len(df)}", flush=True)
    print(f"Listo: {len(df)} imagenes en {out}")


if __name__ == "__main__":
    main()
