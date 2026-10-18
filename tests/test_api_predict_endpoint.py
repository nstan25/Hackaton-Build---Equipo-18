"""`POST /predict` y el token (T01, SPEC §2.2)."""

from __future__ import annotations

import hashlib
import io
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from bcs import api_predict, inference
from bcs.contract import Prediction, validate_prediction_dict
from bcs.scale import load_scale
from bcs.store.memory import THUMBNAIL_SIZE, InMemoryStore

FIXTURES = Path(__file__).parent / "fixtures"
SCALE = load_scale("dairy_1_5")
TOKEN = "test-token"
REAR = (FIXTURES / "rear_cow.jpg").read_bytes()


class SpyStore(InMemoryStore):
    def __init__(self, media_dir: Path) -> None:
        super().__init__(media_dir)
        self.calls: list[dict[str, Any]] = []

    def save(
        self,
        prediction: Prediction,
        image_bytes: bytes,
        *,
        taken_at: str | None = None,
        notes: str | None = None,
    ) -> None:
        self.calls.append(
            {"pred": prediction, "raw": image_bytes, "taken_at": taken_at, "notes": notes}
        )
        super().save(prediction, image_bytes, taken_at=taken_at, notes=notes)


@pytest.fixture
def spy(tmp_path: Path) -> SpyStore:
    return SpyStore(tmp_path / "media")


@pytest.fixture
def client(spy: SpyStore, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    monkeypatch.setenv("BCS_APP_TOKEN", TOKEN)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)  # sin explicación en segundo plano (T19)
    from bcs.api import create_app

    with TestClient(create_app(spy), headers={"X-App-Token": TOKEN}) as c:
        yield c


def _post(client: TestClient, data: dict[str, str], image: bytes | None = REAR, **kw: Any) -> Any:
    files = {"image": ("foto.jpg", image, "image/jpeg")} if image is not None else None
    return client.post("/predict", data=data, files=files, **kw)


def test_predict_saves_prediction_and_thumbnail(client: TestClient, spy: SpyStore) -> None:
    r = _post(client, {"cow_id": " uy-7 ", "taken_at": "2026-10-17T08:30:00Z", "notes": " flaca "})
    assert r.status_code == 200
    body = r.json()
    assert validate_prediction_dict(body, SCALE.keys, SCALE.step) == []
    assert body["cow_id"] == "UY-7"
    assert body["image_sha256"] == hashlib.sha256(REAR).hexdigest()
    [call] = spy.calls
    assert call["pred"].to_dict() == body
    # el store recibe la miniatura armada desde la foto ya decodificada (R3), no la original
    stored = Image.open(io.BytesIO(call["raw"]))
    assert stored.format == "JPEG" and max(stored.size) <= THUMBNAIL_SIZE
    assert call["taken_at"] == "2026-10-17T08:30:00Z" and call["notes"] == "flaca"
    thumb = spy.thumbnail_path(body["prediction_id"])
    assert thumb is not None and thumb.read_bytes()[:2] == b"\xff\xd8"
    assert client.get(f"/predictions/{body['prediction_id']}").json() == body


def test_optional_fields_default_to_none(client: TestClient, spy: SpyStore) -> None:
    assert _post(client, {"cow_id": "A1", "taken_at": "", "notes": "  "}).status_code == 200
    assert spy.calls[0]["taken_at"] is None and spy.calls[0]["notes"] is None


def test_each_call_is_a_new_prediction(client: TestClient) -> None:
    a, b = (_post(client, {"cow_id": "A1"}).json() for _ in range(2))
    assert a["prediction_id"] != b["prediction_id"]
    assert a["image_sha256"] == b["image_sha256"]


@pytest.mark.parametrize(
    "data",
    [
        {},
        {"cow_id": ""},
        {"cow_id": "   "},
        {"cow_id": "uy 7"},
        {"cow_id": "-UY7"},
        {"cow_id": "Ñandú"},
        {"cow_id": "A" * 65},
    ],
)
def test_bad_cow_id_is_422_and_not_evaluated(
    client: TestClient, spy: SpyStore, monkeypatch: pytest.MonkeyPatch, data: dict[str, str]
) -> None:
    def boom(*_: Any) -> None:
        raise AssertionError("no debería evaluar la foto")

    monkeypatch.setattr(inference, "predict", boom)
    monkeypatch.setattr(inference, "predict_raw", boom)
    r = _post(client, data)
    assert r.status_code == 422 and "cow_id" in r.json()["detail"]
    assert spy.calls == []


@pytest.mark.parametrize(
    ("data", "image", "field"),
    [
        ({"cow_id": "A1"}, None, "image"),
        ({"cow_id": "A1"}, b"", "image"),
        ({"cow_id": "A1"}, b"no soy un jpeg", "image"),
        ({"cow_id": "A1", "taken_at": "ayer"}, REAR, "taken_at"),
        ({"cow_id": "A1", "notes": "x" * 281}, REAR, "notes"),
    ],
)
def test_invalid_input_is_422(
    client: TestClient, spy: SpyStore, data: dict[str, str], image: bytes | None, field: str
) -> None:
    r = _post(client, data, image)
    assert r.status_code == 422 and field in r.json()["detail"]
    assert spy.calls == []


def test_notes_at_limit_is_ok(client: TestClient) -> None:
    assert _post(client, {"cow_id": "A1", "notes": "x" * 280}).status_code == 200


def test_oversized_image_is_413(
    client: TestClient, spy: SpyStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(api_predict, "MAX_IMAGE_BYTES", len(REAR) - 1)
    assert _post(client, {"cow_id": "A1"}).status_code == 413
    assert spy.calls == []


@pytest.mark.parametrize(
    "auth",
    [
        {},
        {"headers": {"X-App-Token": ""}},
        {"headers": {"X-App-Token": "wrong"}},
        {"headers": {"X-App-Token": "tökén".encode("latin-1")}},  # no ASCII: no debe ser 500
        {"headers": {"X-App-Token": ""}, "params": {"token": "wrong"}},
    ],
)
def test_predict_without_valid_token_is_401(
    client: TestClient, spy: SpyStore, auth: dict[str, Any]
) -> None:
    client.headers.pop("X-App-Token")
    r = _post(client, {"cow_id": "A1"}, **auth)
    assert r.status_code == 401
    assert spy.calls == []


def test_token_checked_before_cow_id(client: TestClient) -> None:
    client.headers.pop("X-App-Token")
    assert _post(client, {}).status_code == 401


def test_token_in_query_works(client: TestClient) -> None:
    client.headers.pop("X-App-Token")
    assert _post(client, {"cow_id": "A1"}, params={"token": TOKEN}).status_code == 200
