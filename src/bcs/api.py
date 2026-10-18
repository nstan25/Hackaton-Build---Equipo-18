"""App FastAPI: solo crea la app, monta los routers y sirve `web/` (SPEC §2)."""

from __future__ import annotations

from fastapi import FastAPI

from bcs import api_cows, api_explain, api_predict, api_web
from bcs.auth import PredictGuard, install_log_redaction
from bcs.paths import db_path, media_dir
from bcs.store.base import Store
from bcs.store.sqlite import SqliteStore


def create_app(store: Store | None = None) -> FastAPI:
    # Sin /docs, /redoc ni /openapi.json: serían rutas públicas fuera del token (SPEC §2.2).
    app = FastAPI(title="BCS", docs_url=None, redoc_url=None, openapi_url=None)
    app.state.store = store or SqliteStore(db_path(), media_dir())
    # Token y tamaño de `POST /predict` antes de leer el cuerpo (R6); token fuera de los logs (R5).
    app.add_middleware(PredictGuard)
    install_log_redaction()
    app.include_router(api_predict.router)
    app.include_router(api_cows.router)
    app.include_router(api_explain.router)
    app.include_router(api_web.router)  # último: su ruta comodín no tapa a las de la API
    return app


app = create_app()
