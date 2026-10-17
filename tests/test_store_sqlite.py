"""`SqliteStore` (T15): guardar/leer, último válido, historial, filtros, duplicados, reinicio."""

from __future__ import annotations

import io
import json
import sqlite3
import uuid
from dataclasses import replace
from pathlib import Path

import pytest
from PIL import Image

from bcs.contract import Prediction, QualityReport
from bcs.store.base import CowFilters, Store
from bcs.store.sqlite import SqliteStore

FIXTURES = Path(__file__).parent / "fixtures"
EXAMPLE = Prediction.from_dict(json.loads((FIXTURES / "prediction_example.json").read_text()))
REAR = (FIXTURES / "rear_cow.jpg").read_bytes()


def _pred(cow_id: str, created_at: str, *, bcs: float = 3.5, **kw) -> Prediction:
    return replace(
        EXAMPLE,
        prediction_id=str(uuid.uuid4()),
        cow_id=cow_id,
        created_at=created_at,
        bcs=bcs,
        **kw,
    )


def _rejected(cow_id: str, created_at: str) -> Prediction:
    return replace(
        _pred(cow_id, created_at),
        status="rejected",
        bcs=None,
        bcs_expected=None,
        confidence=None,
        interval=None,
        quality=QualityReport(is_cow_rear=False, score=0.1, issues=["no parece una vaca de atrás"]),
    )


@pytest.fixture
def store(tmp_path: Path) -> SqliteStore:
    return SqliteStore(tmp_path / "db" / "bcs.sqlite", tmp_path / "media")


def test_is_a_store(store: SqliteStore) -> None:
    assert isinstance(store, Store)


def test_save_and_get_roundtrip(store: SqliteStore) -> None:
    p = _pred("UY-1", "2026-10-17T10:00:00Z")
    store.save(p, REAR, taken_at="2026-10-16T08:00:00-03:00", notes="renguea")
    got = store.get(p.prediction_id)
    assert got == p
    assert got is not None and got.to_dict() == p.to_dict()
    assert store.extras(p.prediction_id) == {
        "taken_at": "2026-10-16T08:00:00-03:00",
        "notes": "renguea",
    }
    assert store.get("nope") is None and store.extras("nope") is None


def test_rejected_is_saved(store: SqliteStore) -> None:
    r = _rejected("UY-1", "2026-10-17T10:00:00Z")
    store.save(r, REAR)
    assert store.get(r.prediction_id) == r
    assert store.history("UY-1") == [r]


def test_thumbnail_is_jpg_512(store: SqliteStore, tmp_path: Path) -> None:
    buf = io.BytesIO()
    Image.new("RGB", (2000, 1000), (120, 80, 40)).save(buf, format="PNG")
    p = _pred("UY-1", "2026-10-17T10:00:00Z")
    store.save(p, buf.getvalue())
    path = store.thumbnail_path(p.prediction_id)
    assert path is not None and path.parent == tmp_path / "media"
    with Image.open(path) as img:
        assert img.format == "JPEG" and img.size == (512, 256)
    assert store.thumbnail_path("nope") is None


def test_history_sorted_newest_first(store: SqliteStore) -> None:
    times = ["2026-10-15T10:00:00Z", "2026-10-17T10:00:00Z", "2026-10-16T10:00:00Z"]
    preds = [_pred("UY-1", t) for t in times]
    for p in preds:
        store.save(p, REAR)
    store.save(_pred("UY-2", "2026-10-18T10:00:00Z"), REAR)
    hist = store.history("UY-1")
    assert [p.created_at for p in hist] == sorted(times, reverse=True)
    assert store.history("NADIE") == []


def test_history_orders_by_instant_not_string(store: SqliteStore) -> None:
    # 12:00-03:00 es 15:00Z: más nueva que 14:00Z aunque el string compare menor
    a = _pred("UY-1", "2026-10-17T14:00:00Z")
    b = _pred("UY-1", "2026-10-17T12:00:00-03:00")
    store.save(b, REAR)
    store.save(a, REAR)
    assert [p.prediction_id for p in store.history("UY-1")] == [b.prediction_id, a.prediction_id]


def test_same_timestamp_last_saved_wins(store: SqliteStore) -> None:
    a, b = _pred("UY-1", "2026-10-17T10:00:00Z"), _pred("UY-1", "2026-10-17T10:00:00Z")
    store.save(a, REAR)
    store.save(b, REAR)
    assert store.history("UY-1")[0].prediction_id == b.prediction_id
    assert store.latest_by_cow()[0].last_prediction_id == b.prediction_id


def test_latest_valid_ignores_rejected(store: SqliteStore) -> None:
    valid = _pred("UY-1", "2026-10-16T10:00:00Z", bcs=3.25, confidence=0.7, interval=(3.0, 3.5))
    rej = _rejected("UY-1", "2026-10-17T10:00:00Z")
    store.save(valid, REAR)
    store.save(rej, REAR)
    (row,) = store.latest_by_cow()
    assert row.cow_id == "UY-1" and row.n_evaluations == 2
    assert (row.bcs, row.confidence, row.interval) == (3.25, 0.7, (3.0, 3.5))
    assert row.bcs_at == valid.created_at
    assert row.last_prediction_id == rej.prediction_id and row.last_at == rej.created_at
    assert row.last_status == "rejected" and row.last_rejected


def test_low_confidence_counts_as_valid(store: SqliteStore) -> None:
    store.save(_pred("UY-1", "2026-10-16T10:00:00Z", bcs=3.0), REAR)
    store.save(_pred("UY-1", "2026-10-17T10:00:00Z", bcs=4.0, status="low_confidence"), REAR)
    (row,) = store.latest_by_cow()
    assert row.bcs == 4.0 and row.last_status == "low_confidence" and not row.last_rejected


def test_only_rejected_has_no_bcs(store: SqliteStore) -> None:
    store.save(_rejected("UY-1", "2026-10-17T10:00:00Z"), REAR)
    (row,) = store.latest_by_cow()
    assert row.bcs is None and row.confidence is None and row.interval is None
    assert row.bcs_at is None and row.last_rejected and row.n_evaluations == 1


@pytest.fixture
def herd(store: SqliteStore) -> SqliteStore:
    store.save(_pred("A", "2026-10-10T10:00:00Z", bcs=2.5), REAR)
    store.save(_pred("A", "2026-10-12T10:00:00Z", bcs=3.0), REAR)
    store.save(_pred("B", "2026-10-15T10:00:00Z", bcs=3.75, status="low_confidence"), REAR)
    store.save(_pred("C", "2026-10-01T10:00:00Z", bcs=4.25), REAR)
    store.save(_rejected("C", "2026-10-16T10:00:00Z"), REAR)
    store.save(_rejected("D", "2026-10-17T10:00:00Z"), REAR)
    return store


def _ids(store: SqliteStore, **kw) -> list[str]:
    return [r.cow_id for r in store.latest_by_cow(CowFilters(**kw))]


def test_filters(herd: SqliteStore) -> None:
    assert _ids(herd) == ["A", "B", "C", "D"]
    assert _ids(herd, status="ok") == ["A"]
    assert _ids(herd, status="low_confidence") == ["B"]
    assert _ids(herd, status="rejected") == ["C", "D"]
    assert _ids(herd, bcs_max=3.0) == ["A"]  # el último válido de A es 3.0, no 2.5
    # C: su último válido es 4.25 aunque la última evaluación fue rejected
    assert _ids(herd, bcs_min=3.75) == ["B", "C"]
    assert _ids(herd, bcs_min=3.0, bcs_max=3.75) == ["A", "B"]
    assert _ids(herd, since="2026-10-15T10:00:00Z") == ["B", "C", "D"]  # inclusivo
    assert _ids(herd, since="2026-10-16") == ["C", "D"]
    assert _ids(herd, since="2026-10-16T07:00:01-03:00") == ["D"]  # = 10:00:01Z
    assert _ids(herd, status="rejected", bcs_min=4.0) == ["C"]
    assert _ids(herd, since="2027-01-01") == []


def test_invalid_since_raises(herd: SqliteStore) -> None:
    with pytest.raises(ValueError):
        herd.latest_by_cow(CowFilters(since="ayer"))


def test_duplicate_image_sha256(store: SqliteStore) -> None:
    a = _pred("UY-1", "2026-10-16T10:00:00Z")
    b = _pred("UY-2", "2026-10-17T10:00:00Z")  # misma foto, otra vaca
    other = _pred("UY-1", "2026-10-17T11:00:00Z", image_sha256="0" * 64)
    for p in (a, b, other):
        store.save(p, REAR)
    # se guardan todas; la foto repetida se detecta por hash
    dups = store.find_by_image_sha256(EXAMPLE.image_sha256)
    assert [p.prediction_id for p in dups] == [b.prediction_id, a.prediction_id]
    assert store.find_by_image_sha256("f" * 64) == []
    assert len(store.history("UY-1")) == 2


def test_duplicate_prediction_id_rejected(store: SqliteStore) -> None:
    p = _pred("UY-1", "2026-10-17T10:00:00Z")
    store.save(p, REAR)
    with pytest.raises(ValueError, match="repetido"):
        store.save(replace(p, cow_id="UY-2"), REAR)
    assert store.get(p.prediction_id) == p


def test_bad_image_leaves_no_row(store: SqliteStore) -> None:
    p = _pred("UY-1", "2026-10-17T10:00:00Z")
    with pytest.raises(OSError):
        store.save(p, b"no es una imagen")
    assert store.get(p.prediction_id) is None


def test_persists_across_restart(tmp_path: Path) -> None:
    db, media = tmp_path / "bcs.sqlite", tmp_path / "media"
    first = SqliteStore(db, media)
    p = _pred("UY-1", "2026-10-17T10:00:00Z")
    r = _rejected("UY-1", "2026-10-17T11:00:00Z")
    first.save(p, REAR, notes="n")
    first.save(r, REAR)
    del first

    again = SqliteStore(db, media)
    assert again.get(p.prediction_id) == p
    assert again.history("UY-1") == [r, p]
    (row,) = again.latest_by_cow()
    assert row.bcs == p.bcs and row.last_rejected and row.n_evaluations == 2
    assert again.thumbnail_path(p.prediction_id) is not None
    assert again.extras(p.prediction_id) == {"taken_at": None, "notes": "n"}


def test_schema_has_cow_index(store: SqliteStore) -> None:
    with sqlite3.connect(store.db_path) as conn:
        cols = [r[2] for r in conn.execute("PRAGMA index_info(idx_predictions_cow_created)")]
    assert cols[:2] == ["cow_id", "created_ts"]
