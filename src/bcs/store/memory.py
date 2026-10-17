"""Store en memoria (stub de T00). Las miniaturas sí van a disco, en `media_dir`."""

from __future__ import annotations

import io
from pathlib import Path

from PIL import Image, ImageOps

from bcs.contract import Prediction
from bcs.store.base import CowFilters, CowSummary

THUMBNAIL_SIZE = 512


def make_thumbnail(image_bytes: bytes) -> bytes:
    img = ImageOps.exif_transpose(Image.open(io.BytesIO(image_bytes))).convert("RGB")
    img.thumbnail((THUMBNAIL_SIZE, THUMBNAIL_SIZE))
    out = io.BytesIO()
    img.save(out, format="JPEG", quality=85)
    return out.getvalue()


class InMemoryStore:
    def __init__(self, media_dir: Path) -> None:
        self.media_dir = media_dir
        self._predictions: dict[str, Prediction] = {}

    def save(
        self,
        prediction: Prediction,
        image_bytes: bytes,
        *,
        taken_at: str | None = None,
        notes: str | None = None,
    ) -> None:
        self.media_dir.mkdir(parents=True, exist_ok=True)
        thumb = self.media_dir / f"{prediction.prediction_id}.jpg"
        thumb.write_bytes(make_thumbnail(image_bytes))
        self._predictions[prediction.prediction_id] = prediction

    def history(self, cow_id: str) -> list[Prediction]:
        # de la más nueva a la más vieja; ante empate de created_at, gana la última guardada
        preds = [p for p in reversed(self._predictions.values()) if p.cow_id == cow_id]
        return sorted(preds, key=lambda p: p.created_at, reverse=True)

    def get(self, prediction_id: str) -> Prediction | None:
        return self._predictions.get(prediction_id)

    def thumbnail_path(self, prediction_id: str) -> Path | None:
        if prediction_id not in self._predictions:
            return None
        path = self.media_dir / f"{prediction_id}.jpg"
        return path if path.exists() else None

    def latest_by_cow(self, filters: CowFilters | None = None) -> list[CowSummary]:
        f = filters or CowFilters()
        rows: list[CowSummary] = []
        for cow_id in sorted({p.cow_id for p in self._predictions.values()}):
            hist = self.history(cow_id)
            last = hist[0]
            valid = next((p for p in hist if p.status != "rejected"), None)
            row = CowSummary(
                cow_id=cow_id,
                bcs=valid.bcs if valid else None,
                confidence=valid.confidence if valid else None,
                interval=valid.interval if valid else None,
                bcs_at=valid.created_at if valid else None,
                n_evaluations=len(hist),
                last_prediction_id=last.prediction_id,
                last_status=last.status,
                last_at=last.created_at,
                last_rejected=last.status == "rejected",
            )
            if f.status is not None and row.last_status != f.status:
                continue
            if f.bcs_min is not None and (row.bcs is None or row.bcs < f.bcs_min):
                continue
            if f.bcs_max is not None and (row.bcs is None or row.bcs > f.bcs_max):
                continue
            if f.since is not None and row.last_at < f.since:
                continue
            rows.append(row)
        return rows
