"""Registro de evaluaciones en SQLite (`$BCS_DB_PATH`) + miniaturas JPG en `$BCS_MEDIA_DIR`.

Una fila por evaluación, incluidas las `rejected` (SPEC §2.2). Cada operación abre su propia
conexión: los endpoints sync de FastAPI corren en un threadpool y `sqlite3` no comparte
conexiones entre hilos.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from bcs.contract import Prediction, QualityReport
from bcs.store.base import CowFilters, CowSummary
from bcs.store.memory import CowMean, make_thumbnail

SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
    seq INTEGER PRIMARY KEY AUTOINCREMENT,  -- orden de guardado: desempata created_at iguales
    prediction_id TEXT NOT NULL UNIQUE,
    cow_id TEXT NOT NULL,
    created_at TEXT NOT NULL,               -- tal cual vino en la Prediction
    created_ts REAL NOT NULL,               -- created_at en epoch UTC, para ordenar y filtrar
    image_sha256 TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('ok', 'low_confidence', 'rejected')),
    bcs REAL,
    bcs_expected REAL,
    confidence REAL,
    interval_min REAL,
    interval_max REAL,
    distribution TEXT NOT NULL,             -- JSON clase → prob
    quality TEXT NOT NULL,                  -- JSON QualityReport
    scale TEXT NOT NULL,
    model_version TEXT NOT NULL,
    notes TEXT,
    taken_at TEXT,
    thumbnail TEXT NOT NULL                 -- nombre del JPG dentro de media_dir
);
CREATE INDEX IF NOT EXISTS idx_predictions_cow_created ON predictions (cow_id, created_ts, seq);
CREATE INDEX IF NOT EXISTS idx_predictions_sha256 ON predictions (image_sha256);
-- cache de explicaciones de OpenAI (T17): una por predicción, para no pagarla dos veces
CREATE TABLE IF NOT EXISTS explanations (
    prediction_id TEXT PRIMARY KEY REFERENCES predictions (prediction_id),
    data TEXT NOT NULL                      -- JSON de `/predictions/{id}/explain`
);
"""

_PRED_COLUMNS = (
    "prediction_id, cow_id, created_at, image_sha256, status, bcs, bcs_expected, confidence, "
    "interval_min, interval_max, distribution, quality, scale, model_version"
)


def iso_to_ts(value: str) -> float:
    """ISO 8601 → epoch. Sin zona horaria se asume UTC. `ValueError` si no es ISO."""
    dt = datetime.fromisoformat(value.strip())
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.timestamp()


def _row_to_prediction(row: sqlite3.Row) -> Prediction:
    lo, hi = row["interval_min"], row["interval_max"]
    return Prediction(
        prediction_id=row["prediction_id"],
        cow_id=row["cow_id"],
        created_at=row["created_at"],
        image_sha256=row["image_sha256"],
        status=row["status"],
        bcs=row["bcs"],
        bcs_expected=row["bcs_expected"],
        confidence=row["confidence"],
        interval=(lo, hi) if lo is not None and hi is not None else None,
        distribution=json.loads(row["distribution"]),
        quality=QualityReport(**json.loads(row["quality"])),
        scale=row["scale"],
        model_version=row["model_version"],
    )


class SqliteStore:
    def __init__(self, db_path: Path, media_dir: Path) -> None:
        self.db_path = db_path
        self.media_dir = media_dir
        self._write_lock = threading.Lock()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.executescript(SCHEMA)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.row_factory = sqlite3.Row
        try:
            with conn:  # commit al salir bien, rollback si hay excepción
                yield conn
        finally:
            conn.close()

    def save(
        self,
        prediction: Prediction,
        image_bytes: bytes,
        *,
        taken_at: str | None = None,
        notes: str | None = None,
    ) -> None:
        p = prediction
        thumb_name = f"{p.prediction_id}.jpg"
        thumb = make_thumbnail(
            image_bytes
        )  # antes de tocar la base: si la foto no abre, no queda fila
        values: dict[str, Any] = {
            "prediction_id": p.prediction_id,
            "cow_id": p.cow_id,
            "created_at": p.created_at,
            "created_ts": iso_to_ts(p.created_at),
            "image_sha256": p.image_sha256,
            "status": p.status,
            "bcs": p.bcs,
            "bcs_expected": p.bcs_expected,
            "confidence": p.confidence,
            "interval_min": p.interval[0] if p.interval is not None else None,
            "interval_max": p.interval[1] if p.interval is not None else None,
            "distribution": json.dumps(p.distribution),
            "quality": json.dumps(
                {
                    "is_cow_rear": p.quality.is_cow_rear,
                    "score": p.quality.score,
                    "issues": list(p.quality.issues),
                }
            ),
            "scale": p.scale,
            "model_version": p.model_version,
            "notes": notes,
            "taken_at": taken_at,
            "thumbnail": thumb_name,
        }
        cols = ", ".join(values)
        params = ", ".join(f":{k}" for k in values)
        self.media_dir.mkdir(parents=True, exist_ok=True)
        with self._write_lock, self._connect() as conn:
            try:
                conn.execute(f"INSERT INTO predictions ({cols}) VALUES ({params})", values)
            except sqlite3.IntegrityError as e:
                raise ValueError(f"prediction_id repetido: {p.prediction_id}") from e
            # la miniatura se escribe dentro de la transacción: si falla, no queda la fila
            tmp = self.media_dir / f".{thumb_name}.tmp"
            tmp.write_bytes(thumb)
            tmp.replace(self.media_dir / thumb_name)

    def history(self, cow_id: str) -> list[Prediction]:
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT {_PRED_COLUMNS} FROM predictions WHERE cow_id = ? "
                "ORDER BY created_ts DESC, seq DESC",
                (cow_id,),
            ).fetchall()
        return [_row_to_prediction(r) for r in rows]

    def get(self, prediction_id: str) -> Prediction | None:
        with self._connect() as conn:
            row = conn.execute(
                f"SELECT {_PRED_COLUMNS} FROM predictions WHERE prediction_id = ?",
                (prediction_id,),
            ).fetchone()
        return _row_to_prediction(row) if row is not None else None

    def find_by_image_sha256(self, image_sha256: str) -> list[Prediction]:
        """Evaluaciones de la misma foto (duplicados), de la más reciente a la más vieja."""
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT {_PRED_COLUMNS} FROM predictions WHERE image_sha256 = ? "
                "ORDER BY created_ts DESC, seq DESC",
                (image_sha256,),
            ).fetchall()
        return [_row_to_prediction(r) for r in rows]

    def extras(self, prediction_id: str) -> dict[str, str | None] | None:
        """`taken_at` y `notes` del form de `/predict` (no son parte de `Prediction`)."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT taken_at, notes FROM predictions WHERE prediction_id = ?",
                (prediction_id,),
            ).fetchone()
        return {"taken_at": row["taken_at"], "notes": row["notes"]} if row is not None else None

    def get_explanation(self, prediction_id: str) -> dict[str, Any] | None:
        """Explicación de OpenAI ya generada (T17), o None. No es parte de `Store`."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT data FROM explanations WHERE prediction_id = ?", (prediction_id,)
            ).fetchone()
        return json.loads(row["data"]) if row is not None else None

    def save_explanation(self, prediction_id: str, explanation: dict[str, Any]) -> None:
        with self._write_lock, self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO explanations (prediction_id, data) VALUES (?, ?)",
                (prediction_id, json.dumps(explanation, ensure_ascii=False)),
            )

    def thumbnail_path(self, prediction_id: str) -> Path | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT thumbnail FROM predictions WHERE prediction_id = ?", (prediction_id,)
            ).fetchone()
        if row is None:
            return None
        path = self.media_dir / row["thumbnail"]
        return path if path.is_file() else None

    def latest_by_cow(self, filters: CowFilters | None = None) -> list[CowSummary]:
        f = filters or CowFilters()
        where: list[str] = ["l.rn = 1"]
        params: list[Any] = []
        if f.status is not None:
            where.append("l.status = ?")
            params.append(f.status)
        if f.bcs_min is not None:
            where.append("v.bcs >= ?")
            params.append(f.bcs_min)
        if f.bcs_max is not None:
            where.append("v.bcs <= ?")
            params.append(f.bcs_max)
        if f.since is not None:
            where.append("l.created_ts >= ?")
            params.append(iso_to_ts(f.since))
        sql = f"""
            WITH last AS (
                SELECT cow_id, prediction_id, status, created_at, created_ts,
                       ROW_NUMBER() OVER w AS rn, COUNT(*) OVER (PARTITION BY cow_id) AS n
                FROM predictions
                WINDOW w AS (PARTITION BY cow_id ORDER BY created_ts DESC, seq DESC)
            ),
            valid AS (
                SELECT cow_id, bcs, confidence, interval_min, interval_max, created_at,
                       ROW_NUMBER() OVER w AS rn
                FROM predictions
                WHERE status != 'rejected'
                WINDOW w AS (PARTITION BY cow_id ORDER BY created_ts DESC, seq DESC)
            )
            SELECT l.cow_id, l.n, l.prediction_id, l.status, l.created_at AS last_at,
                   v.bcs, v.confidence, v.interval_min, v.interval_max, v.created_at AS bcs_at
            FROM last l LEFT JOIN valid v ON v.cow_id = l.cow_id AND v.rn = 1
            WHERE {" AND ".join(where)}
            ORDER BY l.cow_id
        """
        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [
            CowSummary(
                cow_id=r["cow_id"],
                bcs=r["bcs"],
                confidence=r["confidence"],
                interval=(r["interval_min"], r["interval_max"])
                if r["interval_min"] is not None and r["interval_max"] is not None
                else None,
                bcs_at=r["bcs_at"],
                n_evaluations=r["n"],
                last_prediction_id=r["prediction_id"],
                last_status=r["status"],
                last_at=r["last_at"],
                last_rejected=r["status"] == "rejected",
            )
            for r in rows
        ]

    def mean_bcs_by_cow(self, since: str | None = None) -> list[CowMean]:
        """Una fila por vaca con evaluaciones en o después de `since` (ISO 8601; todas si None),
        con el promedio de sus BCS válidos (ok o low_confidence; rejected no cuenta)."""
        where, params = ("WHERE created_ts >= ?", [iso_to_ts(since)]) if since else ("", [])
        sql = f"""
            SELECT cow_id,
                   AVG(CASE WHEN status != 'rejected' THEN bcs END) AS mean_bcs,
                   COUNT(CASE WHEN status != 'rejected' THEN bcs END) AS n_valid
            FROM predictions {where}
            GROUP BY cow_id
            ORDER BY cow_id
        """
        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [CowMean(r["cow_id"], r["mean_bcs"], r["n_valid"]) for r in rows]
