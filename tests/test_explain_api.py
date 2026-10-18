"""`GET /predictions/{id}/explain` (T17) con un OpenAI falso (`httpx.MockTransport`).

Ningún test llama a la API real de OpenAI. El flujo tiene dos pasos: lectura ciega de la foto
(sin el BCS) y comparación (solo texto, con el BCS).
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
TRAITS = [
    {"region": "isquiones", "observation": "marcados, con el borde a la vista"},
    {"region": "base de la cola y hueco", "observation": "hueco profundo a cada lado"},
]
EXPLANATION = "La foto sugiere una condición más baja: hueco de la cola profundo."

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


def _step(req: httpx.Request) -> str:
    return json.loads(req.content)["text"]["format"]["name"]


def answers(estimate: str = "baja", describe: Any = None, compare: Any = None) -> Handler:
    """OpenAI falso que contesta cada paso según el nombre del schema pedido."""
    blind = {"traits": TRAITS, "estimate_reason": "hueco profundo", "photo_estimate": estimate}

    def handler(req: httpx.Request) -> httpx.Response:
        if _step(req) == "bcs_lectura_ciega":
            return describe or httpx.Response(200, json=_responses_body(blind))
        return compare or httpx.Response(200, json=_responses_body({"explanation": EXPLANATION}))

    return handler


class FakeOpenAI:
    def __init__(self) -> None:
        self.calls: list[httpx.Request] = []
        self.handler: Handler = answers()

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
        "photo_estimate",
        "agreement",
        "explanation",
        "traits",
        "model",
        "generated_at",
        "disclaimer",
        "prompt_version",
    }
    assert data["prediction_id"] == pred.prediction_id
    assert data["bcs"] == 3.5
    assert data["photo_estimate"] == "baja"
    assert data["agreement"] == "sugiere_mas_baja"
    assert data["explanation"] == EXPLANATION
    assert data["traits"] == TRAITS
    assert data["model"] == "gpt-test"
    assert data["prompt_version"] == explain.PROMPT_VERSION
    assert "IA" in data["disclaimer"] and "OpenAI" in data["disclaimer"]
    assert [_step(c) for c in fake.calls] == ["bcs_lectura_ciega", "bcs_explicacion"]

    again = _get(api, pred)
    assert again.status_code == 200
    assert again.json() == data
    assert len(fake.calls) == 2  # de la cache: no se vuelve a pagar


def test_blind_step_does_not_see_the_score(api: TestClient, store: Store, fake: FakeOpenAI) -> None:
    pred = _saved(store, status="low_confidence", bcs=4.25)
    assert _get(api, pred).status_code == 200
    describe, compare = (json.loads(c.content) for c in fake.calls)
    for c in fake.calls:
        assert str(c.url) == explain.OPENAI_URL
        assert c.headers["authorization"] == f"Bearer {FAKE_KEY}"
    for body in (describe, compare):
        assert body["model"] == explain.DEFAULT_MODEL
        assert body["reasoning"] == {"effort": "low"}
        assert body["text"]["format"]["type"] == "json_schema"
        assert body["text"]["format"]["strict"] is True
        user = json.dumps(body["input"][1], ensure_ascii=False)
        assert pred.cow_id not in user  # el cow_id no viaja a OpenAI
        assert "confianza" not in user and "low_confidence" not in user  # ni el estado

    # paso 1: la foto, sin el BCS
    user = describe["input"][1]["content"]
    assert "4.25" not in json.dumps(user)
    image = next(p["image_url"] for p in user if p["type"] == "input_image")
    prefix = "data:image/jpeg;base64,"
    thumb = store.thumbnail_path(pred.prediction_id)
    assert thumb is not None
    assert base64.b64decode(image[len(prefix) :]) == thumb.read_bytes()
    schema = describe["text"]["format"]["schema"]
    assert list(schema["properties"])[0] == "traits"  # describe antes de estimar
    assert schema["properties"]["photo_estimate"]["enum"] == list(explain.ESTIMATES)

    # paso 2: solo texto, con el BCS y el acuerdo calculado
    user = compare["input"][1]["content"]
    assert all(p["type"] == "input_text" for p in user)
    text = user[0]["text"]
    assert "4.25" in text and "más baja" in text and TRAITS[1]["observation"] in text


@pytest.mark.parametrize(
    ("estimate", "bcs", "expected"),
    [
        ("baja", 3.25, "coincide"),
        ("baja", 3.5, "sugiere_mas_baja"),
        ("media", 3.75, "coincide"),
        ("media", 3.25, "sugiere_mas_alta"),
        ("media", 4.0, "sugiere_mas_baja"),
        ("alta", 4.25, "coincide"),
        ("alta", 3.75, "sugiere_mas_alta"),
        ("no_se_puede", 3.5, "no_se_puede"),
    ],
)
def test_agreement_is_computed(estimate: str, bcs: float, expected: str) -> None:
    assert explain.agreement(estimate, bcs) == expected


def test_prompts_reference_criteria_and_forbid_system_talk() -> None:
    assert "Ferguson" in explain.CRITERIA and "Elanco" in explain.CRITERIA
    for word in ("hueco", "isquiones", "ilíacos", "sacro", "apófisis", "en V", "en U"):
        assert word in explain.CRITERIA
    assert explain.CRITERIA in explain.DESCRIBE_PROMPT
    assert "No sabés qué puntaje" in explain.DESCRIBE_PROMPT
    assert "verificar a mano" in explain.COMPARE_PROMPT  # como prohibición
    assert "No lo suavices ni avales" in explain.COMPARE_PROMPT


def test_stale_cache_is_regenerated(api: TestClient, store: Store, fake: FakeOpenAI) -> None:
    pred = _saved(store)
    old = {"prediction_id": pred.prediction_id, "bcs": 3.5, "explanation": "vieja", "traits": []}
    store.save_explanation(pred.prediction_id, old)  # type: ignore[attr-defined]
    data = _get(api, pred).json()
    assert data["explanation"] == EXPLANATION
    assert len(fake.calls) == 2
    assert store.get_explanation(pred.prediction_id) == data  # type: ignore[attr-defined]
    assert _get(api, pred).json() == data
    assert len(fake.calls) == 2


def test_model_from_env(
    api: TestClient, store: Store, fake: FakeOpenAI, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENAI_MODEL", "gpt-otro")
    monkeypatch.setenv("OPENAI_REASONING_EFFORT", "")
    pred = _saved(store)
    assert _get(api, pred).status_code == 200
    for c in fake.calls:
        body = json.loads(c.content)
        assert body["model"] == "gpt-otro"
        assert "reasoning" not in body


def test_unreadable_photo_cant_compare(api: TestClient, store: Store, fake: FakeOpenAI) -> None:
    fake.handler = answers("no_se_puede")
    data = _get(api, _saved(store)).json()
    assert data["photo_estimate"] == "no_se_puede"
    assert data["agreement"] == "no_se_puede"


def test_without_key_is_503(
    api: TestClient, store: Store, fake: FakeOpenAI, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY")
    r = _get(api, _saved(store))
    assert r.status_code == 503
    assert "explicación no disponible" in r.json()["detail"]
    assert fake.calls == []


@pytest.mark.parametrize("step", ["bcs_lectura_ciega", "bcs_explicacion"])
def test_timeout_is_504(api: TestClient, store: Store, fake: FakeOpenAI, step: str) -> None:
    ok = answers()

    def slow(req: httpx.Request) -> httpx.Response:
        if _step(req) == step:
            raise httpx.ReadTimeout("timeout", request=req)
        return ok(req)

    fake.handler = slow
    pred = _saved(store)
    assert _get(api, pred).status_code == 504
    # tras un error no queda nada en la cache: el reintento vuelve a llamar
    fake.handler = ok
    n = len(fake.calls)
    assert _get(api, pred).status_code == 200
    assert len(fake.calls) == n + 2


def test_both_steps_share_the_deadline(
    api: TestClient, store: Store, fake: FakeOpenAI, monkeypatch: pytest.MonkeyPatch
) -> None:
    clock = iter([0.0, 0.0, explain.MAX_TIMEOUT_S - 0.5])  # el paso 1 se come casi todo
    monkeypatch.setattr(explain, "_now", lambda: next(clock))
    assert _get(api, _saved(store)).status_code == 504
    assert [_step(c) for c in fake.calls] == ["bcs_lectura_ciega"]


_INVALID = [
    httpx.Response(200, json=_responses_body("esto no es json")),
    httpx.Response(200, json={"status": "incomplete", "output": []}),
    httpx.Response(200, json={"output": []}),
    httpx.Response(200, json=[1, 2]),
    httpx.Response(200, text="<html>"),
    httpx.Response(200, json={"output": [{"type": "message", "content": [{"type": "refusal"}]}]}),
    httpx.Response(500, json={"error": {"message": "boom"}}),
]
_BAD_DESCRIBE = [
    {"traits": TRAITS, "estimate_reason": "x"},
    {"traits": TRAITS, "estimate_reason": "x", "photo_estimate": "3.25"},
    {"traits": [], "estimate_reason": "x", "photo_estimate": "baja"},
    {"traits": ["y"], "estimate_reason": "x", "photo_estimate": "baja"},
    {"traits": TRAITS, "estimate_reason": "", "photo_estimate": "baja"},
]


@pytest.mark.parametrize(
    "handler",
    [answers(describe=r) for r in _INVALID]
    + [answers(compare=r) for r in _INVALID]
    + [answers(describe=httpx.Response(200, json=_responses_body(b))) for b in _BAD_DESCRIBE]
    + [answers(compare=httpx.Response(200, json=_responses_body({"explanation": ""})))],
)
def test_invalid_response_is_502(
    api: TestClient, store: Store, fake: FakeOpenAI, handler: Handler
) -> None:
    fake.handler = handler
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
