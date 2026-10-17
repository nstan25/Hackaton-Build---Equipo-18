"""Split train / val / test de `sdb_bcs` (G0 punto 3): `preprocesamiento/split.csv` tal cual.

No hay K-Fold ni `make holdout`: test es el holdout. De acá salen

    holdout_groups.txt   group_id con split=test (uno por línea, ordenados)
    splits.csv           image_id,split solo para las filas de labels.csv (train / val)

y los chequeos que `make labels` exige antes de escribir: ningún grupo en dos splits y ningún
grupo del holdout en `labels.csv`. Las etiquetas de test nunca se leen acá.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from bcs.data.sources.sdb_bcs import SOURCE_ID, image_id
from bcs.paths import REPO_ROOT

README_SPLIT = REPO_ROOT / "preprocesamiento" / "split.csv"
TRAIN_SPLITS = ("train", "val")
HOLDOUT_SPLIT = "test"


def read_readme_split(path: Path = README_SPLIT) -> pd.DataFrame:
    """`split.csv` del README → image_id, group_id, split (mismo id y prefijo que el adaptador)."""
    df = pd.read_csv(path, usecols=["path", "split", "group"], dtype=str)
    return pd.DataFrame(
        {
            "image_id": df["path"].map(image_id),
            "group_id": SOURCE_ID + ":" + df["group"],
            "split": df["split"],
        }
    )


def check_source_splits(readme: pd.DataFrame, source_splits: pd.DataFrame) -> list[str]:
    """`splits_<id>.csv` del adaptador tiene que coincidir imagen por imagen con el README."""
    a = readme.set_index("image_id")["split"].sort_index()
    b = source_splits.set_index("image_id")["split"].sort_index()
    if not a.index.equals(b.index):
        only_a, only_b = a.index.difference(b.index), b.index.difference(a.index)
        return [
            f"imágenes distintas: {len(only_a)} solo en el README, {len(only_b)} solo en splits"
        ]
    diff = a[a != b]
    return [f"{len(diff)} imágenes con otro split que el README"] if len(diff) else []


def holdout_groups(readme: pd.DataFrame) -> list[str]:
    return sorted(readme.loc[readme["split"] == HOLDOUT_SPLIT, "group_id"].unique())


def group_leakage(table: pd.DataFrame) -> list[str]:
    """group_id que aparecen en más de un split (`test_no_group_leakage`)."""
    n = table.groupby("group_id")["split"].nunique()
    return sorted(str(g) for g, k in n.items() if k > 1)


def holdout_violations(labels: pd.DataFrame, groups: list[str]) -> list[str]:
    """group_id del holdout presentes en `labels.csv` (`test_holdout_excluded`)."""
    return sorted(set(labels["group_id"]) & set(groups))


def build_splits(labels: pd.DataFrame, readme: pd.DataFrame) -> pd.DataFrame:
    """image_id,split para cada fila de `labels.csv`, en el mismo orden; solo train / val."""
    split = readme.set_index("image_id")["split"]
    missing = labels.loc[~labels["image_id"].isin(split.index.tolist()), "image_id"]
    if len(missing):
        raise ValueError(
            f"{len(missing)} imágenes de labels.csv sin split (p. ej. {missing.iat[0]})"
        )
    out = pd.DataFrame({"image_id": labels["image_id"].to_numpy()})
    out["split"] = split.loc[out["image_id"]].to_numpy()
    bad = out.loc[~out["split"].isin(TRAIN_SPLITS)]
    if len(bad):
        raise ValueError(f"{len(bad)} imágenes de labels.csv fuera de train/val")
    return out
