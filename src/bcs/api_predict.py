"""`GET /health` y `POST /predict` (SPEC §2.2).

Stub de T00 sobre el `predict` dummy; T01 lo completa.
"""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Any

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile

from bcs import inference
from bcs.auth import require_token
from bcs.contract import InvalidCowId, normalize_cow_id

router = APIRouter()

MAX_NOTES = 280


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "model_version": inference.predict_model_version()}


@router.post("/predict", dependencies=[Depends(require_token)])
async def predict(
    request: Request,
    image: Annotated[UploadFile | None, File()] = None,
    cow_id: Annotated[str | None, Form()] = None,
    taken_at: Annotated[str | None, Form()] = None,
    notes: Annotated[str | None, Form()] = None,
) -> dict[str, Any]:
    try:
        cow_id = normalize_cow_id(cow_id)
    except InvalidCowId as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    if image is None:
        raise HTTPException(status_code=422, detail="image es obligatoria")
    if notes is not None and len(notes) > MAX_NOTES:
        raise HTTPException(status_code=422, detail=f"notes admite hasta {MAX_NOTES} caracteres")
    if taken_at:
        try:
            datetime.fromisoformat(taken_at)
        except ValueError as e:
            raise HTTPException(status_code=422, detail="taken_at debe ser ISO 8601") from e
    raw = await image.read()
    try:
        pred = inference.predict(raw, cow_id)
    except OSError as e:  # PIL no puede abrir la imagen
        raise HTTPException(status_code=422, detail="image no es una imagen válida") from e
    request.app.state.store.save(pred, raw, taken_at=taken_at or None, notes=notes)
    return pred.to_dict()
