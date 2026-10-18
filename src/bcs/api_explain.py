"""`GET /predictions/{prediction_id}/explain` (T17): por qué la foto dio ese BCS, según OpenAI.

Detrás del token, como el resto. La explicación se guarda por `prediction_id` (en el store si
sabe hacerlo, si no en memoria del proceso): la segunda llamada no vuelve a pagar a OpenAI.
Errores: sin clave → 503, timeout → 504, respuesta inválida → 502.
"""

from __future__ import annotations

import threading
from collections import defaultdict
from collections.abc import Iterator
from typing import Annotated, Any

import httpx
from fastapi import APIRouter, Depends, HTTPException, Request

from bcs import explain
from bcs.auth import require_token
from bcs.store.base import Store

router = APIRouter(dependencies=[Depends(require_token)])

# Un candado por predicción: dos clics seguidos no generan (ni pagan) dos explicaciones.
_locks: defaultdict[str, threading.Lock] = defaultdict(threading.Lock)
_locks_guard = threading.Lock()


def openai_client() -> Iterator[httpx.Client]:
    """Cliente HTTP hacia OpenAI. Los tests lo reemplazan con `dependency_overrides`."""
    with httpx.Client(timeout=explain.timeout_s()) as client:
        yield client


def _store(request: Request) -> Store:
    return request.app.state.store


def _cached(request: Request, prediction_id: str) -> dict[str, Any] | None:
    get = getattr(_store(request), "get_explanation", None)
    if get is not None:
        return get(prediction_id)
    return getattr(request.app.state, "explanations", {}).get(prediction_id)


def _remember(request: Request, prediction_id: str, data: dict[str, Any]) -> None:
    save = getattr(_store(request), "save_explanation", None)
    if save is not None:
        save(prediction_id, data)
    else:
        if not hasattr(request.app.state, "explanations"):
            request.app.state.explanations = {}
        request.app.state.explanations[prediction_id] = data


@router.get("/predictions/{prediction_id}/explain")
def explain_prediction(
    request: Request, prediction_id: str, client: Annotated[httpx.Client, Depends(openai_client)]
) -> dict[str, Any]:
    pred = _store(request).get(prediction_id)
    if pred is None:
        raise HTTPException(status_code=404, detail="predicción inexistente")
    if pred.status == "rejected" or pred.bcs is None:
        raise HTTPException(status_code=409, detail="foto rechazada: no hay puntaje que explicar")
    with _locks_guard:
        lock = _locks[prediction_id]
    with lock:
        cached = _cached(request, prediction_id)
        if cached is not None:
            return cached
        path = _store(request).thumbnail_path(prediction_id)
        if path is None:
            raise HTTPException(status_code=404, detail="miniatura inexistente")
        try:
            data = explain.explain(pred, path.read_bytes(), client)
        except explain.ExplainError as e:
            raise HTTPException(status_code=e.status_code, detail=e.detail) from e
        _remember(request, prediction_id, data)
        return data
