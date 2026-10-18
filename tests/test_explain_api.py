"""`GET /predictions/{id}/explain` (T17) con un OpenAI falso (`httpx.MockTransport`).

Ningún test llama a la API real de OpenAI.
"""

from __future__ import annotations

import base64
import json
import uuid
from collections.abc import Callable, Iterator
from dataclasses import replace
from pathlib import Path
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient

from bcs import explain
from bcs.api_explain import openai_client
from bcs.contract import Prediction
from bcs.store.base import Store
from bcs.store.memory import InMemoryStore
from bcs.store.sqlite import SqliteStore

FIXTURES = Path(__file__).parent / "fixtures"
TOKEN = "test-token"
FAKE_KEY = "sk-test-fake-key-not-real"
GOOD = {
    "explanation": "Los ilíacos y los isquiones se ven redondeados, con algo de grasa encima.",
    "traits": [
        {"region": "ilíacos", "observation": "redondeados, sin bordes filosos"},
        {"region": "base de la cola", "observation": "no se ve: la tapa la cola"},
    ],
}

Handler = Callable[[httpx.Request], httpx.Response]


def _responses_body(payload: Any, model: str = "gpt-test") -> dict[str, Any]:
    text = payload if isinstance(payload, str) else json.dumps(payload)
    return {
        "model": model,
        "status": "completed",
        "output": [
            {"type": "reasoning", "summary": []},
            {"type": "message", "content": [{"type": "output_text", "text": text}]},
        ],
    }


class FakeOpenAI:
    def __init__(self, handler: Handler | None = None) -> None:
        self.calls: list[httpx.Request] = []
        self.handler = handler or (lambda _req: httpx.Response(200, json=_responses_body(GOOD)))

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.calls.append(request)
        return self.handler(request)


def _prediction(status: str = "ok", bcs: float | None = 3.5) -> Prediction:
    base = Prediction.from_dict(json.loads((FIXTURES / "prediction_example.json").read_text()))
    return replace(base, prediction_id=str(uuid.uuid4()), status=status, bcs=bcs)  # type: ignore[arg-type]


@pytest.fixture(params=["memory", "sqlite"])
def store(request: pytest.FixtureRequest, tmp_path: Path) -> Store:
    if request.param == "memory":
        return InMemoryStore(tmp_path / "media")
    return SqliteStore(tmp_path / "bcs.sqlite", tmp_path / "media")


@pytest.fixture
def fake() -> FakeOpenAI:
    return FakeOpenAI()


@pytest.fixture
def api(store: Store, fake: FakeOpenAI, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    from bcs.api import create_app

    monkeypatch.setenv("BCS_APP_TOKEN", TOKEN)
    monkeypatch.setenv("OPENAI_API_KEY", FAKE_KEY)
    monkeypatch.delenv("OPENAI_MODEL", raising=False)
    monkeypatch.delenv("OPENAI_REASONING_EFFORT", raising=False)
    app = create_app(store=store)

    def client() -> Iterator[httpx.Client]:
        with httpx.Client(transport=httpx.MockTransport(fake)) as c:
            yield c

    app.dependency_overrides[openai_client] = client
    with TestClient(app, headers={"X-App-Token": TOKEN}) as c:
        yield c


def _saved(store: Store, **kw: Any) -> Prediction:
    pred = _prediction(**kw)
    store.save(pred, (FIXTURES / "rear_cow.jpg").read_bytes())
    return pred


def test_explain_ok_and_cached(api: TestClient, store: Store, fake: FakeOpenAI) -> None:
    pred = _saved(store)
    r = api.get(f"/predictions/{pred.prediction_id}/explain")
    assert r.status_code == 200, r.text
    data = r.json()
    assert set(data) == {
        "prediction_id",
        "bcs",
        "explanation",
        "traits",
        "model",
        "generated_at",
        "disclaimer",
    }
    assert data["prediction_id"] == pred.prediction_id
    assert data["bcs"] == 3.5
    assert data["explanation"] == GOOD["explanation"]
    assert data["traits"] == GOOD["traits"]
    assert data["model"] == "gpt-test"
    assert "IA" in data["disclaimer"] and "OpenAI" in data["disclaimer"]
    assert len(fake.calls) == 1

    again = api.get(f"/predictions/{pred.prediction_id}/explain")
    assert again.status_code == 200
    assert again.json() == data
    assert len(fake.calls) == 1  # de la cache: no se vuelve a pagar


def test_request_to_openai(api: TestClient, store: Store, fake: FakeOpenAI) -> None:
    pred = _saved(store, status="low_confidence", bcs=4.25)
    assert api.get(f"/predictions/{pred.prediction_id}/explain").status_code == 200
    req = fake.calls[0]
    assert str(req.url) == explain.OPENAI_URL
    assert req.headers["authorization"] == f"Bearer {FAKE_KEY}"
    body = json.loads(req.content)
    assert body["model"] == explain.DEFAULT_MODEL
    assert body["reasoning"] == {"effort": "low"}
    assert body["text"]["format"]["type"] == "json_schema"
    assert body["text"]["format"]["strict"] is True
    user = body["input"][1]["content"]
    text = next(p["text"] for p in user if p["type"] == "input_text")
    assert "4.25" in text and "confianza baja" in text
    assert pred.cow_id not in json.dumps(body)  # el cow_id no viaja a OpenAI
    image = next(p["image_url"] for p in user if p["type"] == "input_image")
    prefix = "data:image/jpeg;base64,"
    assert image.startswith(prefix)
    thumb = store.thumbnail_path(pred.prediction_id)
    assert thumb is not None
    assert base64.b64decode(image[len(prefix) :]) == thumb.read_bytes()


def test_model_from_env(
    api: TestClient, store: Store, fake: FakeOpenAI, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENAI_MODEL", "gpt-otro")
    monkeypatch.setenv("OPENAI_REASONING_EFFORT", "")
    pred = _saved(store)
    assert api.get(f"/predictions/{pred.prediction_id}/explain").status_code == 200
    body = json.loads(fake.calls[0].content)
    assert body["model"] == "gpt-otro"
    assert "reasoning" not in body


def test_without_key_is_503(
    api: TestClient, store: Store, fake: FakeOpenAI, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY")
    pred = _saved(store)
    r = api.get(f"/predictions/{pred.prediction_id}/explain")
    assert r.status_code == 503
    assert "explicación no disponible" in r.json()["detail"]
    assert fake.calls == []


def test_timeout_is_504(api: TestClient, store: Store, fake: FakeOpenAI) -> None:
    def slow(req: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("timeout", request=req)

    fake.handler = slow
    pred = _saved(store)
    r = api.get(f"/predictions/{pred.prediction_id}/explain")
    assert r.status_code == 504
    # tras un error no queda nada en la cache: el reintento vuelve a llamar
    fake.handler = lambda _req: httpx.Response(200, json=_responses_body(GOOD))
    assert api.get(f"/predictions/{pred.prediction_id}/explain").status_code == 200
    assert len(fake.calls) == 2


@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(200, json=_responses_body("esto no es json")),
        httpx.Response(200, json=_responses_body({"explanation": "x"})),
        httpx.Response(200, json=_responses_body({"explanation": "", "traits": []})),
        httpx.Response(200, json=_responses_body({"explanation": "x", "traits": ["y"]})),
        httpx.Response(200, json={"status": "incomplete", "output": []}),
        httpx.Response(200, json={"output": []}),
        httpx.Response(200, json=[1, 2]),
        httpx.Response(200, text="<html>"),
        httpx.Response(
            200,
            json={"output": [{"type": "message", "content": [{"type": "refusal"}]}]},
        ),
        httpx.Response(500, json={"error": {"message": "boom"}}),
    ],
)
def test_invalid_response_is_502(
    api: TestClient, store: Store, fake: FakeOpenAI, response: httpx.Response
) -> None:
    fake.handler = lambda _req: response
    pred = _saved(store)
    r = api.get(f"/predictions/{pred.prediction_id}/explain")
    assert r.status_code == 502, r.text


def test_openai_error_body_not_forwarded(api: TestClient, store: Store, fake: FakeOpenAI) -> None:
    leak = {"error": {"message": f"Incorrect API key provided: {FAKE_KEY[:10]}***"}}
    fake.handler = lambda _req: httpx.Response(401, json=leak)
    pred = _saved(store)
    r = api.get(f"/predictions/{pred.prediction_id}/explain")
    assert r.status_code == 503
    assert FAKE_KEY[:10] not in r.text


def test_connection_error_is_502(api: TestClient, store: Store, fake: FakeOpenAI) -> None:
    def down(req: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("sin red", request=req)

    fake.handler = down
    pred = _saved(store)
    assert api.get(f"/predictions/{pred.prediction_id}/explain").status_code == 502


def test_rejected_and_missing(api: TestClient, store: Store, fake: FakeOpenAI) -> None:
    pred = _saved(store, status="rejected", bcs=None)
    assert api.get(f"/predictions/{pred.prediction_id}/explain").status_code == 409
    assert api.get("/predictions/no-existe/explain").status_code == 404
    assert fake.calls == []


def test_requires_token(api: TestClient, store: Store, fake: FakeOpenAI) -> None:
    pred = _saved(store)
    r = api.get(f"/predictions/{pred.prediction_id}/explain", headers={"X-App-Token": ""})
    assert r.status_code == 401
    assert fake.calls == []
