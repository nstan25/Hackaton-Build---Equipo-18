"""Escala de BCS configurable (SPEC §0.1). Ningún código asume una escala fija."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from bcs.paths import config_dir


@dataclass(frozen=True)
class Scale:
    name: str
    min: float
    max: float
    step: float
    collapse_below: float | None = None
    collapse_above: float | None = None

    @property
    def grid(self) -> tuple[float, ...]:
        """Valores de la grilla después de agrupar las colas."""
        n = round((self.max - self.min) / self.step)
        values = [round(self.min + i * self.step, 4) for i in range(n + 1)]
        if self.collapse_below is not None:
            values = [v for v in values if v >= self.collapse_below]
        if self.collapse_above is not None:
            values = [v for v in values if v <= self.collapse_above]
        return tuple(values)

    @property
    def keys(self) -> tuple[str, ...]:
        return tuple(grid_key(v) for v in self.grid)

    def snap(self, value: float) -> float:
        """Lleva un valor (p. ej. una etiqueta en la cola) al punto más cercano de la grilla."""
        return min(self.grid, key=lambda g: abs(g - value))


def grid_key(value: float) -> str:
    """Clave de `Prediction.distribution` para un valor de la grilla (p. ej. 3.5 → "3.50")."""
    return f"{value:.2f}"


def load_scale(name: str | None = None, path: Path | None = None) -> Scale:
    path = path or config_dir() / "scale.yaml"
    raw: dict[str, Any] = yaml.safe_load(path.read_text())
    chosen: str = name or os.environ.get("BCS_SCALE") or raw["active"]
    preset = raw["presets"][chosen]
    return Scale(
        name=chosen,
        min=float(preset["min"]),
        max=float(preset["max"]),
        step=float(preset["step"]),
        collapse_below=preset.get("collapse_below"),
        collapse_above=preset.get("collapse_above"),
    )
