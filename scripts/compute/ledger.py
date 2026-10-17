"""Bitácora de gasto (SPEC §2.4): una línea por corrida en `runs/compute_ledger.csv`."""

from __future__ import annotations

import csv
import os
from datetime import UTC, datetime
from pathlib import Path

from bcs.paths import runs_dir

COLUMNS = (
    "ts",
    "backend",
    "task",
    "model",
    "fold",
    "smoke",
    "machine",
    "seconds",
    "est_cost_usd",
    "status",
    "report",
)

# USD por hora, estimados (verificar contra la factura). Se pisan con BCS_COST_<BACKEND>_USD_H.
USD_PER_HOUR = {
    "local": 0.0,
    "kaggle": 0.0,  # cuota semanal gratuita
    "modal": 0.80,  # L4 (~$0.000222/s) + CPU/RAM del contenedor
    "tenki": 0.06,  # runner CPU de GitHub Actions
}


def ledger_path() -> Path:
    return runs_dir() / "compute_ledger.csv"


def estimate_cost(backend: str, seconds: float) -> float:
    rate = float(os.environ.get(f"BCS_COST_{backend.upper()}_USD_H", USD_PER_HOUR.get(backend, 0)))
    return round(rate * seconds / 3600, 4)


def append(
    *,
    backend: str,
    task: str,
    model: str,
    fold: str,
    smoke: bool,
    machine: str,
    seconds: float,
    status: str,
    report: str = "",
    path: Path | None = None,
) -> dict[str, str]:
    path = path or ledger_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "ts": datetime.now(UTC).isoformat(timespec="seconds"),
        "backend": backend,
        "task": task,
        "model": model,
        "fold": fold,
        "smoke": str(int(smoke)),
        "machine": machine,
        "seconds": f"{seconds:.1f}",
        "est_cost_usd": f"{estimate_cost(backend, seconds):.4f}",
        "status": status,
        "report": report,
    }
    new = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        if new:
            w.writeheader()
        w.writerow(row)
    return row
