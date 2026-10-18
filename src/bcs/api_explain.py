"""`GET /predictions/{prediction_id}/explain` (T17): qué se observa en la foto, según OpenAI.

Detrás del token, como el resto. La explicación se guarda por `prediction_id` (en el store si
sabe hacerlo, si no en memoria del proceso): la segunda llamada no vuelve a pagar a OpenAI.
Errores: sin clave → 503, timeout → 504, respuesta inválida → 502.

En paralelo con el BCS (T19): `POST /predict` lanza la observación ciega con `start()` apenas tiene
la foto y la asocia a la predicción con `attach()`. Si la predicción sale `rejected` se descarta; si
no, el endpoint espera la que está en curso en vez de pedir otra.
"""

from __future__ import annotations

import inspect
import threading
from collections import defaultdict
from collections.abc import Callable, Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from typing import Annotated, Any, cast

import httpx
from fastapi import APIRouter, Depends, FastAPI, HTTPException, Request

from bcs import explain
from bcs.auth import require_token
from bcs.contract import Prediction
from bcs.store.base import Store

router = APIRouter(dependencies=[Depends(require_token)])

# Un candado por predicción: dos clics seguidos no generan (ni pagan) dos explicaciones.
_locks: defaultdict[str, threading.Lock] = defaultdict(threading.Lock)
_locks_guard = threading.Lock()

# Observaciones lanzadas por `/predict` que todavía no terminaron, por `prediction_id`. La
# concurrencia real hacia OpenAI la limita `explain.MAX_CONCURRENT_CALLS`.
_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="explain")
_pending: dict[str, Future[dict[str, Any]]] = {}
_pending_guard = threading.Lock()


def openai_client() -> Iterator[httpx.Client]:
    """Cliente HTTP hacia OpenAI. Los tests lo reemplazan con `dependency_overrides`."""
    with httpx.Client(timeout=explain.timeout_s()) as client:
        yield client


def _observe(factory: Callable[[], Any], thumbnail: bytes) -> dict[str, Any]:
    made: Any = factory()
    if not inspect.isgenerator(made):
        return explain.observe(thumbnail, made)
    try:
        client = cast(httpx.Client, next(made))
        return explain.observe(thumbnail, client)
    finally:
        made.close()


def start(app: FastAPI, thumbnail: bytes) -> Future[dict[str, Any]] | None:
    """Lanza la observación ciega en segundo plano. None si no hay clave de OpenAI."""
    if not explain.configured():
        return None
    # El mismo cliente que usaría el endpoint (los tests lo reemplazan por uno falso).
    factory = app.dependency_overrides.get(openai_client, openai_client)
    return _executor.submit(_observe, factory, thumbnail)


def attach(app: FastAPI, pred: Prediction, future: Future[dict[str, Any]]) -> None:
    """Asocia la observación en curso a su predicción; con `rejected` (o sin BCS) se descarta."""
    if pred.status == "rejected" or pred.bcs is None:
        future.cancel()  # si ya arrancó, el resultado no se guarda ni se muestra
        return
    pid = pred.prediction_id
    with _pending_guard:
        _pending[pid] = future

    def done(f: Future[dict[str, Any]]) -> None:
        try:
            if not f.cancelled() and f.exception() is None:
                with _lock(pid):
                    if _cached(app, pid) is None:
                        _remember(app, pid, explain.for_prediction(f.result(), pred))
        finally:
            with _pending_guard:
                if _pending.get(pid) is f:
                    del _pending[pid]

    future.add_done_callback(done)


def _lock(prediction_id: str) -> threading.Lock:
    with _locks_guard:
        return _locks[prediction_id]


def _store(app: FastAPI) -> Store:
    return app.state.store


def _cached(app: FastAPI, prediction_id: str) -> dict[str, Any] | None:
    get = getattr(_store(app), "get_explanation", None)
    if get is not None:
        data = get(prediction_id)
    else:
        data = getattr(app.state, "explanations", {}).get(prediction_id)
    # Una explicación de otra versión del prompt no sirve: se regenera y se pisa.
    if data is None or data.get("prompt_version") != explain.PROMPT_VERSION:
        return None
    return data


def _remember(app: FastAPI, prediction_id: str, data: dict[str, Any]) -> None:
    save = getattr(_store(app), "save_explanation", None)
    if save is not None:
        save(prediction_id, data)
    else:
        if not hasattr(app.state, "explanations"):
            app.state.explanations = {}
        app.state.explanations[prediction_id] = data


def _wait_pending(prediction_id: str) -> dict[str, Any] | None:
    """La observación lanzada por `/predict`, esperándola si sigue en curso; None si no hay."""
    with _pending_guard:
        future = _pending.get(prediction_id)
    if future is None:
        return None
    try:
        return future.result(timeout=explain.timeout_s())
    except FutureTimeout as e:
        raise explain.ExplainTimeout("OpenAI tardó demasiado en responder; probá de nuevo") from e
    except explain.ExplainError:
        raise
    except Exception as e:  # noqa: BLE001 — un fallo inesperado del hilo es un 502, nunca un 500
        raise explain.ExplainBadResponse("no se pudo generar la explicación") from e


@router.get("/predictions/{prediction_id}/explain")
def explain_prediction(
    request: Request, prediction_id: str, client: Annotated[httpx.Client, Depends(openai_client)]
) -> dict[str, Any]:
    app = request.app
    pred = _store(app).get(prediction_id)
    if pred is None:
        raise HTTPException(status_code=404, detail="predicción inexistente")
    if pred.status == "rejected" or pred.bcs is None:
        raise HTTPException(status_code=409, detail="foto rechazada: no hay puntaje que explicar")
    try:
        observation = _wait_pending(prediction_id)
    except explain.ExplainError as e:
        raise HTTPException(status_code=e.status_code, detail=e.detail) from e
    with _lock(prediction_id):
        cached = _cached(app, prediction_id)
        if cached is not None:
            return cached
        if observation is None:
            path = _store(app).thumbnail_path(prediction_id)
            if path is None:
                raise HTTPException(status_code=404, detail="miniatura inexistente")
            try:
                observation = explain.observe(path.read_bytes(), client)
            except explain.ExplainError as e:
                raise HTTPException(status_code=e.status_code, detail=e.detail) from e
        data = explain.for_prediction(observation, pred)
        _remember(app, prediction_id, data)
        return data
