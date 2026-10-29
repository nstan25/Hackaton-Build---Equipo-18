"""Endpoints de rodeo (SPEC §2.2).

`/cows`, `/cows/distribution`, `/cows/{cow_id}`, `/predictions/{id}` y `/media/{id}.jpg`,
todas detrás del token. Trabajan contra el `Store` montado en `app.state.store`.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import asdict
from typing import Any, Protocol, runtime_checkable

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import FileResponse

from bcs.auth import require_token
from bcs.contract import InvalidCowId, Status, normalize_cow_id
from bcs.inference import model_scale
from bcs.scale import Scale, grid_key, load_scale
from bcs.store.base import CowFilters, Store
from bcs.store.memory import CowMean
from bcs.store.sqlite import iso_to_ts

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
    if since is not None:
        try:
            iso_to_ts(since)
        except ValueError as e:
            raise HTTPException(status_code=422, detail="since debe ser ISO 8601") from e
    if bcs_min is not None and bcs_max is not None and bcs_min > bcs_max:
        raise HTTPException(status_code=422, detail="bcs_min no puede ser mayor que bcs_max")
    filters = CowFilters(status=status, bcs_min=bcs_min, bcs_max=bcs_max, since=since)
    return [asdict(row) for row in _store(request).latest_by_cow(filters)]


@runtime_checkable
class _CowMeans(Protocol):
    def mean_bcs_by_cow(self, since: str | None = None) -> list[CowMean]: ...


def grid_bin(value: float, scale: Scale) -> float:
    """Punto de la grilla más cercano; a mitad de camino entre dos (p. ej. 3.375), redondeo
    estándar hacia arriba (3.5). Fuera de la grilla, al extremo (como las colas de scale.yaml)."""
    grid = scale.grid
    i = math.floor((value - grid[0]) / scale.step + 0.5 + 1e-9)
    return grid[min(max(i, 0), len(grid) - 1)]


def bcs_distribution(rows: Sequence[CowMean], scale: Scale) -> dict[str, Any]:
    """Cuántas vacas hay en cada valor de la grilla (T20). Cada vaca se cuenta una sola vez, con
    el promedio de sus BCS válidos, asignado a la barra de `grid_bin`. Las vacas que solo tienen
    fotos rejected no entran en conteos ni promedio: se informan en `without_bcs`.
    `mean_bcs` es el promedio de los promedios por vaca."""
    counts = dict.fromkeys(scale.keys, 0)
    means = [r.mean_bcs for r in rows if r.mean_bcs is not None]
    for m in means:
        counts[grid_key(grid_bin(m, scale))] += 1
    return {
        "scale": scale.name,
        "grid": list(scale.grid),
        "counts": counts,
        "animals": len(means),
        "mean_bcs": round(sum(means) / len(means), 4) if means else None,
        "without_bcs": len(rows) - len(means),
    }


# Antes de `/cows/{cow_id}`: un cow_id siempre va en mayúsculas, no choca con "distribution".
@router.get("/cows/distribution")
def cows_distribution(request: Request, since: str | None = None) -> dict[str, Any]:
    if since is not None:
        try:
            iso_to_ts(since)
        except ValueError as e:
            raise HTTPException(status_code=422, detail="since debe ser ISO 8601") from e
    store = _store(request)
    if not isinstance(store, _CowMeans):
        raise HTTPException(status_code=501, detail="el registro no agrega promedios por vaca")
    # La grilla del modelo de producción (la misma que usa `model_pipeline`), no la `active`.
    scale = load_scale(model_scale())
    return bcs_distribution(store.mean_bcs_by_cow(since), scale)


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
