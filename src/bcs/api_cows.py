"""Endpoints de rodeo (SPEC §2.2). Stub de T00 sobre el `Store` en memoria; T15 lo completa."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import FileResponse

from bcs.auth import require_token
from bcs.contract import InvalidCowId, Status, normalize_cow_id
from bcs.store.base import CowFilters, Store

router = APIRouter(dependencies=[Depends(require_token)])


def _store(request: Request) -> Store:
    return request.app.state.store


@router.get("/cows")
def list_cows(
    request: Request,
    status: Status | None = None,
    bcs_max: float | None = None,
    bcs_min: float | None = None,
    since: str | None = None,
) -> list[dict[str, Any]]:
    filters = CowFilters(status=status, bcs_min=bcs_min, bcs_max=bcs_max, since=since)
    return [asdict(row) for row in _store(request).latest_by_cow(filters)]


@router.get("/cows/{cow_id}")
def cow_history(request: Request, cow_id: str) -> list[dict[str, Any]]:
    try:
        cow_id = normalize_cow_id(cow_id)
    except InvalidCowId as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    history = _store(request).history(cow_id)
    if not history:
        raise HTTPException(status_code=404, detail="vaca sin evaluaciones")
    return [p.to_dict() for p in history]


@router.get("/predictions/{prediction_id}")
def get_prediction(request: Request, prediction_id: str) -> dict[str, Any]:
    pred = _store(request).get(prediction_id)
    if pred is None:
        raise HTTPException(status_code=404, detail="predicción inexistente")
    return pred.to_dict()


@router.get("/media/{prediction_id}.jpg")
def get_media(request: Request, prediction_id: str) -> FileResponse:
    path = _store(request).thumbnail_path(prediction_id)
    if path is None:
        raise HTTPException(status_code=404, detail="miniatura inexistente")
    return FileResponse(path, media_type="image/jpeg")
