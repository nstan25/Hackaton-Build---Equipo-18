"""Rutas compartidas fuera del repo (SPEC §3.0): env vars con defaults en ~/bcs-shared."""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SHARED_ROOT = Path.home() / "bcs-shared"


def _env_path(name: str, default: Path) -> Path:
    value = os.environ.get(name)
    return Path(value).expanduser() if value else default


def data_dir() -> Path:
    return _env_path("BCS_DATA_DIR", SHARED_ROOT / "data")


def artifacts_dir() -> Path:
    return _env_path("BCS_ARTIFACTS_DIR", SHARED_ROOT / "artifacts")


def runs_dir() -> Path:
    return _env_path("BCS_RUNS_DIR", SHARED_ROOT / "runs")


def holdout_dir() -> Path:
    """Solo el humano lo usa (SPEC §4.1). El código de los agentes nunca lo lee."""
    return _env_path("BCS_HOLDOUT_DIR", SHARED_ROOT / "holdout")


def db_path() -> Path:
    return _env_path("BCS_DB_PATH", SHARED_ROOT / "app" / "bcs.sqlite")


def media_dir() -> Path:
    return _env_path("BCS_MEDIA_DIR", SHARED_ROOT / "app" / "media")


def config_dir() -> Path:
    return REPO_ROOT / "config"


def fixtures_dir() -> Path:
    return REPO_ROOT / "tests" / "fixtures"
