"""App FastAPI: solo crea la app, monta los routers y sirve `web/` (SPEC §2)."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from bcs import api_cows, api_predict
from bcs.paths import REPO_ROOT, media_dir
from bcs.store.base import Store
from bcs.store.memory import InMemoryStore


def create_app(store: Store | None = None) -> FastAPI:
    app = FastAPI(title="BCS")
    app.state.store = store or InMemoryStore(media_dir())
    app.include_router(api_predict.router)
    app.include_router(api_cows.router)
    app.mount("/", StaticFiles(directory=REPO_ROOT / "web", html=True), name="web")
    return app


app = create_app()
