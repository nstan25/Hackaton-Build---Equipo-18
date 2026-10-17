"""Sirve `web/` con el mismo token que la API (SPEC §2.2: todo salvo /health lo exige)."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse

from bcs.auth import require_token
from bcs.paths import REPO_ROOT

WEB_DIR = (REPO_ROOT / "web").resolve()

router = APIRouter(dependencies=[Depends(require_token)])


@router.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html")


@router.get("/{name:path}", include_in_schema=False)
def static_file(name: str) -> FileResponse:
    path = (WEB_DIR / name).resolve()
    if not path.is_relative_to(WEB_DIR) or not path.is_file():
        raise HTTPException(status_code=404, detail="no existe")
    return FileResponse(path)
