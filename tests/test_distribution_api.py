"""`GET /cows/distribution` (T20): cada vaca una vez, con el promedio de sus BCS válidos."""

from __future__ import annotations

import json
import uuid
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from bcs.api_cows import bcs_distribution, grid_bin
from bcs.contract import Prediction, QualityReport
from bcs.scale import load_scale
from bcs.store.memory import CowMean, InMemoryStore
from bcs.store.sqlite import SqliteStore

FIXTURES = Path(__file__).parent / "fixtures"
EXAMPLE = Prediction.from_dict(json.loads((FIXTURES / "prediction_example.json").read_text()))
REAR = (FIXTURES / "rear_cow.jpg").read_bytes()
TOKEN = "test-token"
SDB = load_scale("dairy_sdb")
KEYS = ["3.25", "3.50", "3.75", "4.00", "4.25"]


def _pred(cow_id: str, created_at: str, bcs: float | None, status: str = "ok") -> Prediction:
    p = replace(
        EXAMPLE,
        prediction_id=str(uuid.uuid4()),
        cow_id=cow_id,
        created_at=created_at,
        bcs=bcs,
        status=status,  # type: ignore[arg-type]
    )
    if status == "rejected":
        p = replace(
            p,
            bcs=None,
            bcs_expected=None,
            confidence=None,
            interval=None,
            quality=QualityReport(is_cow_rear=False, score=0.1, issues=["no parece una vaca"]),
        )
    return p


@pytest.fixture
def store(tmp_path: Path) -> SqliteStore:
    return SqliteStore(tmp_path / "bcs.sqlite", tmp_path / "media")


@pytest.fixture
def api(store: SqliteStore, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    from bcs.api import create_app

    monkeypatch.setenv("BCS_APP_TOKEN", TOKEN)
    monkeypatch.delenv("BCS_SCALE", raising=False)
    # T28: el default de la rama dev es el modelo zen (dairy_zen); estos casos son sobre la grilla
    # del modelo sdb. El default zen se prueba en tests/test_integracion_zen.py.
    monkeypatch.setenv("BCS_MODEL", "sdb")
    with TestClient(create_app(store=store), headers={"X-App-Token": TOKEN}) as c:
        yield c


def _save(store: SqliteStore | InMemoryStore, *preds: Prediction) -> None:
    for p in preds:
        store.save(p, REAR)


def test_empty_store_all_categories_zero(api: TestClient) -> None:
    r = api.get("/cows/distribution")
    assert r.status_code == 200
    assert r.json() == {
        "scale": "dairy_sdb",
        "grid": [3.25, 3.5, 3.75, 4.0, 4.25],
        "counts": dict.fromkeys(KEYS, 0),
        "animals": 0,
        "mean_bcs": None,
        "without_bcs": 0,
    }


def test_requires_token(api: TestClient) -> None:
    assert api.get("/cows/distribution", headers={"X-App-Token": "otro"}).status_code == 401


def test_each_cow_counted_once_with_its_mean(api: TestClient, store: SqliteStore) -> None:
    _save(
        store,
        # UY-1: 3.25, 3.5, 3.75 → 3.5 → barra 3.5 (una vaca, no tres)
        _pred("UY-1", "2026-10-01T10:00:00Z", 3.25),
        _pred("UY-1", "2026-10-02T10:00:00Z", 3.5),
        _pred("UY-1", "2026-10-03T10:00:00Z", 3.75, "low_confidence"),
        # UY-2: 3.25 y 3.5 → 3.375, mitad de camino → redondeo estándar hacia arriba: 3.5
        _pred("UY-2", "2026-10-01T10:00:00Z", 3.25),
        _pred("UY-2", "2026-10-02T10:00:00Z", 3.5),
        # UY-3: la rejected (última) no entra en el promedio
        _pred("UY-3", "2026-10-01T10:00:00Z", 4.25),
        _pred("UY-3", "2026-10-05T10:00:00Z", None, "rejected"),
        # UY-4: solo rejected → no es un animal evaluado
        _pred("UY-4", "2026-10-01T10:00:00Z", None, "rejected"),
    )
    d = api.get("/cows/distribution").json()
    assert d["counts"] == {"3.25": 0, "3.50": 2, "3.75": 0, "4.00": 0, "4.25": 1}
    assert d["animals"] == 3
    assert d["without_bcs"] == 1
    # promedio de los promedios por vaca: (3.5 + 3.375 + 4.25) / 3
    assert d["mean_bcs"] == pytest.approx((3.5 + 3.375 + 4.25) / 3, abs=1e-4)
    assert sum(d["counts"].values()) == d["animals"]


def test_same_timestamp_measurements_all_averaged(api: TestClient, store: SqliteStore) -> None:
    # Empate de fecha: con el promedio no hay que elegir una; entran las dos.
    _save(
        store,
        _pred("UY-1", "2026-10-01T10:00:00Z", 3.75),
        _pred("UY-1", "2026-10-01T07:00:00-03:00", 4.25),  # mismo instante UTC
    )
    d = api.get("/cows/distribution").json()
    assert d["counts"]["4.00"] == 1 and d["animals"] == 1
    assert d["mean_bcs"] == pytest.approx(4.0)


def test_since_averages_only_inside_window(api: TestClient, store: SqliteStore) -> None:
    _save(
        store,
        _pred("UY-1", "2026-09-01T10:00:00Z", 3.25),
        _pred("UY-1", "2026-10-10T10:00:00Z", 4.0),
        _pred("UY-2", "2026-09-01T10:00:00Z", 3.5),
    )
    d = api.get("/cows/distribution", params={"since": "2026-10-01T00:00:00Z"}).json()
    assert d["counts"] == {"3.25": 0, "3.50": 0, "3.75": 0, "4.00": 1, "4.25": 0}
    assert d["animals"] == 1 and d["mean_bcs"] == pytest.approx(4.0)
    assert api.get("/cows/distribution", params={"since": "ayer"}).status_code == 422


def test_route_does_not_shadow_cow_history(api: TestClient, store: SqliteStore) -> None:
    _save(store, _pred("DISTRIBUTION", "2026-10-01T10:00:00Z", 3.5))
    assert len(api.get("/cows/DISTRIBUTION").json()) == 1
    assert api.get("/cows/distribution").json()["animals"] == 1


def test_grid_bin_rounding() -> None:
    assert grid_bin(3.375, SDB) == 3.5  # mitad de camino → arriba
    assert grid_bin(3.374, SDB) == 3.25
    assert grid_bin(3.6249, SDB) == 3.5
    assert grid_bin(3.625, SDB) == 3.75
    assert grid_bin(4.25, SDB) == 4.25
    assert grid_bin(2.0, SDB) == 3.25  # fuera de la grilla: al extremo, como las colas
    assert grid_bin(4.9, SDB) == 4.25


def test_memory_store_matches_sqlite(store: SqliteStore, tmp_path: Path) -> None:
    mem = InMemoryStore(tmp_path / "mem")
    preds = [
        _pred("UY-1", "2026-10-01T10:00:00Z", 3.25),
        _pred("UY-1", "2026-10-02T10:00:00Z", 3.5),
        _pred("UY-2", "2026-10-03T10:00:00Z", None, "rejected"),
        _pred("UY-3", "2026-10-04T10:00:00Z", 4.0),
    ]
    _save(store, *preds)
    _save(mem, *preds)
    for since in (None, "2026-10-02T00:00:00Z"):
        assert mem.mean_bcs_by_cow(since) == store.mean_bcs_by_cow(since)
    assert store.mean_bcs_by_cow() == [
        CowMean("UY-1", 3.375, 2),
        CowMean("UY-2", None, 0),
        CowMean("UY-3", 4.0, 1),
    ]


def test_bcs_distribution_pure() -> None:
    d = bcs_distribution([CowMean("A", 3.3, 1), CowMean("B", 4.1, 2)], SDB)
    assert d["counts"] == {"3.25": 1, "3.50": 0, "3.75": 0, "4.00": 1, "4.25": 0}
    assert d["mean_bcs"] == pytest.approx(3.7)
