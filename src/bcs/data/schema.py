"""Esquema de `labels.csv` (SPEC §1.1)."""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path

import pandas as pd

VIEWS = ("rear", "rear_dorsal", "dorsal", "side", "other")


@dataclass(frozen=True)
class LabelRow:
    image_id: str
    path: str  # relativo a $BCS_DATA_DIR
    source: str
    animal_id: str  # siempre prefijado con la fuente: "zen_rgbd:230211"
    group_id: str  # lo que no puede separarse entre splits
    bcs: float
    scale: str
    view: str
    scorer: str
    date: str  # ISO 8601 (fecha) o vacío
    bbox: str  # "x0,y0,x1,y1" en píxeles de la imagen original, o vacío


LABEL_COLUMNS: tuple[str, ...] = tuple(f.name for f in fields(LabelRow))


class LabelsSchemaError(ValueError):
    pass


def validate_labels(df: pd.DataFrame) -> list[str]:
    """Devuelve la lista de errores de esquema (vacía si está bien)."""
    errors: list[str] = []
    missing = [c for c in LABEL_COLUMNS if c not in df.columns]
    if missing:
        return [f"faltan columnas: {missing}"]
    if bool(df["image_id"].duplicated().any()):
        errors.append("image_id duplicado")
    for col in ("image_id", "path", "source", "animal_id", "group_id", "scale"):
        if bool(df[col].isna().any()) or bool((df[col].astype(str).str.len() == 0).any()):
            errors.append(f"{col} tiene valores vacíos")
    prefixed = [
        str(a).startswith(f"{s}:") for a, s in zip(df["animal_id"], df["source"], strict=True)
    ]
    if not all(prefixed):
        errors.append("animal_id debe estar prefijado con la fuente ('<source>:<id>')")
    if not pd.api.types.is_numeric_dtype(df["bcs"]) or bool(df["bcs"].isna().any()):
        errors.append("bcs debe ser numérico y no vacío")
    bad_view = ~df["view"].isin(VIEWS)
    if bool(bad_view.any()):
        errors.append(f"view fuera de {VIEWS}: {sorted(df.loc[bad_view, 'view'].unique())}")
    return errors


def load_labels(path: Path) -> pd.DataFrame:
    df = pd.read_csv(
        path, dtype={c: str for c in LABEL_COLUMNS if c != "bcs"}, keep_default_na=False
    )
    df["bcs"] = pd.to_numeric(df["bcs"], errors="coerce")
    errors = validate_labels(df)
    if errors:
        raise LabelsSchemaError(f"{path}: " + "; ".join(errors))
    return df.loc[:, list(LABEL_COLUMNS)]
