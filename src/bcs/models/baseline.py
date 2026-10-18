"""Baseline tonto y medición del atajo de pintura roja (T09).

- `PriorScorer`: el baseline tonto de SPEC §2.3 paso 1 como `Scorer`: la distribución de clases
  de train para toda imagen (argmax = moda). `make eval` calcula además su MAE con moda y media
  (`report.json` → `dummy`).
- `red_mask` / `desaturate_red`: los píxeles rojo-naranja con la misma definición que T06
  (`bcs.data.merge.paint_fractions`: HSV de PIL, tono < 18 o > 241, saturación > 128,
  valor > 77) y la misma imagen con esos píxeles en saturación 0.
- `paint_report`: la medición obligatoria de G1 punto 2 sobre las imágenes de GS en val.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image

PAINT_THRESHOLD = 0.005  # G1: "pintada" = > 0,5% de píxeles rojo-naranja (T06)


def grid_index(y: np.ndarray | Sequence[float], grid: Sequence[float] | np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=float)
    return np.abs(np.asarray(y, dtype=float)[:, None] - g[None, :]).argmin(axis=1)


class PriorScorer:
    """Baseline tonto: la frecuencia de cada clase en train, igual para toda imagen."""

    model_version = "t09-prior"

    def __init__(self, grid: tuple[float, ...], config: dict[str, Any] | None = None) -> None:
        self.grid = grid
        self.prior = np.full(len(grid), 1.0 / len(grid))

    def fit(self, df: pd.DataFrame) -> None:
        counts = np.bincount(
            grid_index(df["bcs"].to_numpy(dtype=float), self.grid), minlength=len(self.grid)
        )
        self.prior = counts / counts.sum()

    def predict_proba(self, images: Sequence[Image.Image]) -> np.ndarray:
        return np.tile(self.prior, (len(images), 1))

    def save(self, path: Path) -> None:
        Path(path).write_text(json.dumps({"grid": list(self.grid), "prior": self.prior.tolist()}))

    def load(self, path: Path) -> None:
        st = json.loads(Path(path).read_text())
        self.grid = tuple(float(v) for v in st["grid"])
        self.prior = np.asarray(st["prior"], dtype=float)


def red_mask(im: Image.Image) -> np.ndarray:
    """Píxeles rojo-naranja; misma definición que `bcs.data.merge.paint_fractions` (T06)."""
    hsv = np.asarray(im.convert("HSV"), dtype=np.uint8)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    return ((h < 18) | (h > 241)) & (s > 128) & (v > 77)


def desaturate_red(im: Image.Image) -> Image.Image:
    """La misma imagen con los píxeles rojo-naranja en saturación 0 (gris del mismo valor)."""
    mask = red_mask(im)
    if not mask.any():
        return im.convert("RGB")
    hsv = np.array(im.convert("HSV"), dtype=np.uint8)
    hsv[..., 1][mask] = 0
    return Image.fromarray(hsv, mode="HSV").convert("RGB")


def paint_report(
    y: np.ndarray,
    pred: np.ndarray,
    value: np.ndarray,
    value_desat: np.ndarray,
    painted: np.ndarray,
    grid: Sequence[float],
) -> dict[str, Any]:
    """G1 punto 2 (imágenes de GS en val): MAE, sesgo y accuracy ±1 paso, por clase, para
    pintadas vs. no pintadas (`pred` = BCS discreto), y Δ = valor con la franja roja desaturada
    - valor original en la misma imagen (`value` = predicción continua o esperanza)."""
    g = np.asarray(grid, dtype=float)
    step = float(g[1] - g[0]) if len(g) > 1 else 1.0
    err = pred - y

    def block(m: np.ndarray) -> dict[str, Any]:
        if not m.any():
            return {"n": 0}
        e = err[m]
        return {
            "n": int(m.sum()),
            "mae": float(np.abs(e).mean()),
            "bias": float(e.mean()),
            "acc_within_1": float((np.abs(e) <= step + 1e-6).mean()),
        }

    by_class = {
        f"{c:.2f}": {
            "painted": block(np.isclose(y, c) & painted),
            "unpainted": block(np.isclose(y, c) & ~painted),
        }
        for c in g
    }
    delta = value_desat - value
    any_p = bool(painted.any())
    return {
        "n": len(y),
        "painted": block(painted),
        "unpainted": block(~painted),
        "by_class": by_class,
        "delta_painted_mean": float(delta[painted].mean()) if any_p else None,
        "delta_painted_abs_mean": float(np.abs(delta[painted]).mean()) if any_p else None,
        "delta_painted_abs_p95": float(np.quantile(np.abs(delta[painted]), 0.95))
        if any_p
        else None,
        "pred_changed_painted": float(
            (grid_index(value_desat[painted], g) != grid_index(value[painted], g)).mean()
        )
        if any_p
        else None,
    }
