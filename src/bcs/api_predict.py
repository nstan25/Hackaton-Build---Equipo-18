"""`GET /health` y `POST /predict` (SPEC §2.2)."""

from __future__ import annotations

import threading
from datetime import datetime
from typing import Annotated, Any

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool

from bcs import inference
from bcs.auth import require_token
from bcs.contract import InvalidCowId, normalize_cow_id
from bcs.store.base import Store
from bcs.store.memory import thumbnail_jpeg

router = APIRouter()

MAX_NOTES = 280
MAX_IMAGE_BYTES = 20 * 1024 * 1024  # fotos de celular de sobra; corta uploads absurdos
# A lo sumo dos fotos decodificadas a la vez (hasta 40 MP c/u): la RAM no crece con las requests
# en paralelo (R3). Con el modelo, `inference` ya serializa la predicción con su propio lock.
_decode_slots = threading.BoundedSemaphore(2)


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "model_version": inference.predict_model_version()}


def _check_taken_at(taken_at: str | None) -> str | None:
    taken_at = (taken_at or "").strip()
    if not taken_at:
        return None
    try:
        datetime.fromisoformat(taken_at)
    except ValueError as e:
        raise HTTPException(status_code=422, detail="taken_at debe ser ISO 8601") from e
    return taken_at


def _check_notes(notes: str | None) -> str | None:
    notes = (notes or "").strip()
    if len(notes) > MAX_NOTES:
        raise HTTPException(status_code=422, detail=f"notes admite hasta {MAX_NOTES} caracteres")
    return notes or None


def _predict_and_save(store: Store, raw: bytes, cow_id: str, **extra: str | None) -> dict[str, Any]:
    with _decode_slots:
        try:
            pred, img = inference.predict_raw(raw, cow_id)
        except inference.ImageTooLarge as e:
            raise HTTPException(status_code=413, detail=str(e)) from e
        except inference.InvalidImage as e:
            raise HTTPException(status_code=422, detail=str(e)) from e
        # La miniatura sale de la foto ya decodificada: la original no se vuelve a abrir.
        thumb = thumbnail_jpeg(img)
        del img
    store.save(pred, thumb, **extra)
    return pred.to_dict()


@router.post("/predict", dependencies=[Depends(require_token)])
async def predict(
    request: Request,
    image: Annotated[UploadFile | None, File()] = None,
    cow_id: Annotated[str | None, Form()] = None,
    taken_at: Annotated[str | None, Form()] = None,
    notes: Annotated[str | None, Form()] = None,
) -> dict[str, Any]:
    # El cow_id se valida antes de mirar la foto: si no cumple, no se evalúa nada.
    try:
        cow_id = normalize_cow_id(cow_id)
    except InvalidCowId as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    if image is None:
        raise HTTPException(status_code=422, detail="image es obligatoria")
    taken_at = _check_taken_at(taken_at)
    notes = _check_notes(notes)
    raw = await image.read(MAX_IMAGE_BYTES + 1)
    if not raw:
        raise HTTPException(status_code=422, detail="image está vacía")
    if len(raw) > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=413, detail=f"image supera {MAX_IMAGE_BYTES // 2**20} MB")
    # La inferencia es CPU: corre en un thread para no frenar el event loop.
    return await run_in_threadpool(
        _predict_and_save, request.app.state.store, raw, cow_id, taken_at=taken_at, notes=notes
    )
