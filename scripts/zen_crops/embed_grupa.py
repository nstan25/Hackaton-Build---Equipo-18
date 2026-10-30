"""T30: embeddings OOD del modelo de grupa (`bcs.ood.fit_zen --section integration_grupa`).

Los conjuntos de T27/T28 salen de `scripts/ood/build_sets.py --task T30 --weights <pt>`; este
script agrega, en la misma carpeta (`$BCS_ARTIFACTS_DIR/T30/emb/`), los de zen con el recorte de
uso real (`crops.py crops --name grupa`):

- `bank_zen_grupa`: zen train, recorte de grupa (lo que vio el modelo en entrenamiento).
- `val_zen_grupa`: zen val, recorte del detector sobre la vaca sola (solo las que detecta: las
  demás la app las rechaza antes del modelo). Es la vista `uso` de `evaluate.py`.

    uv run python scripts/zen_crops/embed_grupa.py --weights models/bcs_efficientnet_b0_zen_grupa.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ood"))
import build_sets as bs  # noqa: E402  # pyright: ignore[reportMissingImports]
import crops as zc  # noqa: E402

from bcs.ood.embed import FeatureExtractor  # noqa: E402


def grupa_rows(split: str, only_det: bool) -> pd.DataFrame:
    man = pd.read_csv(zc.out_dir() / "crops" / "grupa" / "manifest.csv", dtype={"image_id": str})
    if only_det:
        man = man.loc[man["kind"] == "det"]
    zen = bs.zen_rows(split)[["image_id", "group_id", "bcs"]]
    return zen.merge(man[["image_id", "path"]], on="image_id", how="inner")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--task", default="T30")
    args = ap.parse_args(argv)
    bs.TASK = args.task
    bs.emb_dir().mkdir(parents=True, exist_ok=True)
    fe = FeatureExtractor(args.weights)

    def by_path(r: pd.Series) -> Image.Image:
        return bs._open(r["path"])

    bs.embed_set("bank_zen_grupa", grupa_rows("train", only_det=False), by_path, fe)
    bs.embed_set("val_zen_grupa", grupa_rows("val", only_det=True), by_path, fe)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
