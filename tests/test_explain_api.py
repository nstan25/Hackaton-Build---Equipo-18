"""`GET /predictions/{id}/explain` (T17) con un OpenAI falso (`httpx.MockTransport`).

Ningún test llama a la API real de OpenAI. Es una sola llamada ciega: la foto, sin el BCS.
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
    "traits": [
        {"region": "isquiones", "observation": "sobresalen con el borde a la vista"},
        {"region": "base de la cola y hueco", "observation": "hueco profundo a cada lado"},
    ],
    "summary": "Se ven los isquiones con el borde marcado y huecos profundos junto a la cola.",
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


def _ok(_req: httpx.Request) -> httpx.Response:
    return httpx.Response(200, json=_responses_body(GOOD))


class FakeOpenAI:
    def __init__(self) -> None:
        self.calls: list[httpx.Request] = []
        self.handler: Handler = _ok

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
    monkeypatch.delenv("OPENAI_TIMEOUT", raising=False)
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


def _get(api: TestClient, pred: Prediction):  # el Response de TestClient (httpx2), no el de httpx
    return api.get(f"/predictions/{pred.prediction_id}/explain")


def test_explain_ok_and_cached(api: TestClient, store: Store, fake: FakeOpenAI) -> None:
    pred = _saved(store)
    r = _get(api, pred)
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
        "prompt_version",
    }
    assert data["prediction_id"] == pred.prediction_id
    assert data["bcs"] == 3.5
    assert data["explanation"] == GOOD["summary"]
    assert data["traits"] == GOOD["traits"]
    assert data["model"] == "gpt-test"
    assert data["prompt_version"] == explain.PROMPT_VERSION
    assert "IA" in data["disclaimer"] and "OpenAI" in data["disclaimer"]
    assert len(fake.calls) == 1  # una sola llamada

    again = _get(api, pred)
    assert again.status_code == 200
    assert again.json() == data
    assert len(fake.calls) == 1  # de la cache: no se vuelve a pagar


def test_request_is_blind(api: TestClient, store: Store, fake: FakeOpenAI) -> None:
    pred = _saved(store, status="low_confidence", bcs=4.25)
    assert _get(api, pred).status_code == 200
    req = fake.calls[0]
    assert str(req.url) == explain.OPENAI_URL
    assert req.headers["authorization"] == f"Bearer {FAKE_KEY}"
    body = json.loads(req.content)
    assert body["model"] == explain.DEFAULT_MODEL
    assert body["reasoning"] == {"effort": "low"}
    assert body["text"]["format"]["type"] == "json_schema"
    assert body["text"]["format"]["strict"] is True
    assert list(body["text"]["format"]["schema"]["properties"]) == ["traits", "summary"]
    user = body["input"][1]
    dumped = json.dumps(user, ensure_ascii=False)
    assert pred.cow_id not in dumped  # el cow_id no viaja a OpenAI
    assert "4.25" not in dumped  # ni el BCS
    assert "confianza" not in dumped and "low_confidence" not in dumped  # ni el estado
    image = next(p["image_url"] for p in user["content"] if p["type"] == "input_image")
    prefix = "data:image/jpeg;base64,"
    thumb = store.thumbnail_path(pred.prediction_id)
    assert thumb is not None
    assert base64.b64decode(image[len(prefix) :]) == thumb.read_bytes()


def test_prompt_describes_and_does_not_score() -> None:
    source = Path(explain.__file__).read_text()
    assert "Ferguson" in source and "Elanco" in source  # la guía cita la escala de referencia
    for word in ("hueco", "Isquiones", "Ilíacos", "sacro", "apófisis", "en V", "en U"):
        assert word in explain.GUIDE
    assert explain.GUIDE in explain.SYSTEM_PROMPT
    assert "No des ningún puntaje" in explain.SYSTEM_PROMPT
    assert "BCS" not in explain.GUIDE  # la guía no asocia signos con puntajes


def test_stale_cache_is_regenerated(api: TestClient, store: Store, fake: FakeOpenAI) -> None:
    pred = _saved(store)
    old = {
        "prediction_id": pred.prediction_id,
        "bcs": 3.5,
        "explanation": "La foto sugiere una condición más baja",
        "traits": [],
        "agreement": "sugiere_mas_baja",
        "prompt_version": "2-ciego",
    }
    store.save_explanation(pred.prediction_id, old)  # type: ignore[attr-defined]
    data = _get(api, pred).json()
    assert data["explanation"] == GOOD["summary"]
    assert "agreement" not in data
    assert len(fake.calls) == 1
    assert store.get_explanation(pred.prediction_id) == data  # type: ignore[attr-defined]
    assert _get(api, pred).json() == data
    assert len(fake.calls) == 1


def test_model_from_env(
    api: TestClient, store: Store, fake: FakeOpenAI, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENAI_MODEL", "gpt-otro")
    monkeypatch.setenv("OPENAI_REASONING_EFFORT", "")
    assert _get(api, _saved(store)).status_code == 200
    body = json.loads(fake.calls[0].content)
    assert body["model"] == "gpt-otro"
    assert "reasoning" not in body


def test_without_key_is_503(
    api: TestClient, store: Store, fake: FakeOpenAI, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY")
    r = _get(api, _saved(store))
    assert r.status_code == 503
    assert "explicación no disponible" in r.json()["detail"]
    assert fake.calls == []


def test_timeout_is_504(api: TestClient, store: Store, fake: FakeOpenAI) -> None:
    def slow(req: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("timeout", request=req)

    fake.handler = slow
    pred = _saved(store)
    assert _get(api, pred).status_code == 504
    # tras un error no queda nada en la cache: el reintento vuelve a llamar
    fake.handler = _ok
    assert _get(api, pred).status_code == 200
    assert len(fake.calls) == 2


@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(200, json=_responses_body("esto no es json")),
        httpx.Response(200, json=_responses_body({"summary": "x"})),
        httpx.Response(200, json=_responses_body({"summary": "x", "traits": []})),
        httpx.Response(200, json=_responses_body({"summary": "", "traits": GOOD["traits"]})),
        httpx.Response(200, json=_responses_body({"summary": "x", "traits": ["y"]})),
        httpx.Response(200, json={"status": "incomplete", "output": []}),
        httpx.Response(200, json={"output": []}),
        httpx.Response(200, json=[1, 2]),
        httpx.Response(200, text="<html>"),
        httpx.Response(
            200, json={"output": [{"type": "message", "content": [{"type": "refusal"}]}]}
        ),
        httpx.Response(500, json={"error": {"message": "boom"}}),
    ],
)
def test_invalid_response_is_502(
    api: TestClient, store: Store, fake: FakeOpenAI, response: httpx.Response
) -> None:
    fake.handler = lambda _req: response
    r = _get(api, _saved(store))
    assert r.status_code == 502, r.text


def test_openai_error_body_not_forwarded(api: TestClient, store: Store, fake: FakeOpenAI) -> None:
    leak = {"error": {"message": f"Incorrect API key provided: {FAKE_KEY[:10]}***"}}
    fake.handler = lambda _req: httpx.Response(401, json=leak)
    r = _get(api, _saved(store))
    assert r.status_code == 503
    assert FAKE_KEY[:10] not in r.text


def test_connection_error_is_502(api: TestClient, store: Store, fake: FakeOpenAI) -> None:
    def down(req: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("sin red", request=req)

    fake.handler = down
    assert _get(api, _saved(store)).status_code == 502


def test_rejected_and_missing(api: TestClient, store: Store, fake: FakeOpenAI) -> None:
    assert _get(api, _saved(store, status="rejected", bcs=None)).status_code == 409
    assert api.get("/predictions/no-existe/explain").status_code == 404
    assert fake.calls == []


def test_requires_token(api: TestClient, store: Store, fake: FakeOpenAI) -> None:
    pred = _saved(store)
    r = api.get(f"/predictions/{pred.prediction_id}/explain", headers={"X-App-Token": ""})
    assert r.status_code == 401
    assert fake.calls == []
