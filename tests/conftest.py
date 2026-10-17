from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

TOKEN = "test-token"


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    monkeypatch.setenv("BCS_APP_TOKEN", TOKEN)
    monkeypatch.setenv("BCS_MEDIA_DIR", str(tmp_path / "media"))
    monkeypatch.setenv("BCS_DB_PATH", str(tmp_path / "bcs.sqlite"))
    from bcs.api import create_app

    with TestClient(create_app(), headers={"X-App-Token": TOKEN}) as c:
        yield c
