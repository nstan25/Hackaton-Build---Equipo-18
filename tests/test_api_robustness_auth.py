"""Token y cuerpos grandes: el token fuera de los logs (R5) y auth antes del cuerpo (R6)."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from bcs import auth
from bcs.auth import MAX_PREDICT_BODY, RedactTokenFilter, redact
from bcs.paths import REPO_ROOT
from bcs.store.memory import InMemoryStore

FIXTURES = Path(__file__).parent / "fixtures"
TOKEN = "test-token"
REAR = (FIXTURES / "rear_cow.jpg").read_bytes()
BIG = 50 * 1024 * 1024


@pytest.fixture
def store(tmp_path: Path) -> InMemoryStore:
    return InMemoryStore(tmp_path / "media")


@pytest.fixture
def client(store: InMemoryStore, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    monkeypatch.setenv("BCS_APP_TOKEN", TOKEN)
    from bcs.api import create_app

    with TestClient(create_app(store)) as c:
        yield c


@pytest.fixture
def no_spool(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """Cuenta los temporales que abre el parser multipart de Starlette (van a disco)."""
    import starlette.formparsers as fp

    made: list[int] = []
    real = fp.SpooledTemporaryFile  # pyright: ignore[reportPrivateImportUsage]

    def spy(*a: Any, **kw: Any) -> Any:
        made.append(1)
        return real(*a, **kw)

    monkeypatch.setattr(fp, "SpooledTemporaryFile", spy)
    return made


def _post(client: TestClient, image: bytes, **kw: Any) -> Any:
    return client.post("/predict", data={"cow_id": "A1"}, files={"image": ("f.jpg", image)}, **kw)


# ---------------------------------------------------------------- R6


def test_big_body_without_token_is_401_without_touching_disk(
    client: TestClient, store: InMemoryStore, no_spool: list[int]
) -> None:
    r = _post(client, b"\0" * BIG)
    assert r.status_code == 401
    assert no_spool == []
    assert store.latest_by_cow() == []


def test_big_body_with_token_is_413_without_touching_disk(
    client: TestClient, no_spool: list[int]
) -> None:
    r = _post(client, b"\0" * BIG, headers={"X-App-Token": TOKEN})
    assert r.status_code == 413
    assert no_spool == []


def test_chunked_body_over_limit_is_413(
    client: TestClient, store: InMemoryStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    # sin Content-Length: se cuenta lo recibido y se corta al pasar el tope
    monkeypatch.setattr(auth, "MAX_PREDICT_BODY", 1024)
    from bcs.api import create_app

    app = create_app(store)

    def chunks() -> Iterator[bytes]:
        for _ in range(64):
            yield b"x" * 1024

    with TestClient(app) as c:
        r = c.post(
            "/predict",
            content=chunks(),
            headers={"X-App-Token": TOKEN, "Content-Type": "multipart/form-data; boundary=b"},
        )
    assert r.status_code == 413
    assert store.latest_by_cow() == []


def test_guard_keeps_normal_predict_working(client: TestClient) -> None:
    assert len(REAR) < MAX_PREDICT_BODY
    assert _post(client, REAR, headers={"X-App-Token": TOKEN}).status_code == 200
    assert _post(client, REAR, params={"token": TOKEN}).status_code == 200


@pytest.mark.parametrize(
    "auth_kw",
    [
        {},
        {"headers": {"X-App-Token": "wrong"}},
        {"headers": {"X-App-Token": "tökén".encode("latin-1")}},
        {"params": {"token": "wrong"}},
        {"cookies": {"bcs_token": "wrong"}},
    ],
)
def test_guard_rejects_bad_tokens(
    client: TestClient, store: InMemoryStore, auth_kw: dict[str, Any]
) -> None:
    for name, value in auth_kw.get("cookies", {}).items():
        client.cookies.set(name, value)
    kw = {k: v for k, v in auth_kw.items() if k != "cookies"}
    assert _post(client, REAR, **kw).status_code == 401
    assert store.latest_by_cow() == []


def test_cookie_token_works_for_pages_and_api(client: TestClient) -> None:
    # la web guarda el token del link en la cookie: las páginas cargan sin ?token= en la URL
    assert client.get("/rodeo.html").status_code == 401
    client.cookies.set("bcs_token", TOKEN)
    assert client.get("/rodeo.html").status_code == 200
    assert client.get("/cows").status_code == 200
    assert _post(client, REAR).status_code == 200


def test_health_stays_public(client: TestClient) -> None:
    assert client.get("/health").status_code == 200


# ---------------------------------------------------------------- R5


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("/cows?token=secreto&bcs_min=2", "/cows?token=***&bcs_min=2"),
        ("/media/x.jpg?token=secreto", "/media/x.jpg?token=***"),
        ("/?a=1&token=s%20c", "/?a=1&token=***"),
        ("/cows?mytoken=1", "/cows?mytoken=1"),
        ("/cows", "/cows"),
    ],
)
def test_redact(url: str, expected: str) -> None:
    assert redact(url) == expected


def test_access_log_has_no_token(client: TestClient) -> None:
    # el mismo formato que usa uvicorn.access: '%s - "%s %s HTTP/%s" %d'
    logger = logging.getLogger("uvicorn.access")
    assert any(isinstance(f, RedactTokenFilter) for f in logger.filters)
    record = logger.makeRecord(
        logger.name,
        logging.INFO,
        __file__,
        0,
        '%s - "%s %s HTTP/%s" %d',
        ("127.0.0.1:5000", "GET", f"/cows?token={TOKEN}", "1.1", 200),
        None,
    )
    for f in logger.filters:
        assert isinstance(f, logging.Filter)
        f.filter(record)
    message = record.getMessage()
    assert TOKEN not in message and "token=***" in message


def test_log_filter_is_installed_once(client: TestClient) -> None:
    from bcs.api import create_app

    create_app(InMemoryStore(Path("/nonexistent")))
    filters = logging.getLogger("uvicorn.access").filters
    assert sum(isinstance(f, RedactTokenFilter) for f in filters) == 1


def test_web_does_not_put_token_in_urls() -> None:
    for page in ("index.html", "rodeo.html"):
        html = (REPO_ROOT / "web" / page).read_text()
        assert "withToken" not in html and '"token="' not in html
        assert "history.replaceState" in html
        assert "createObjectURL" in html
