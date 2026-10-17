"""Rutas de §2.2 sobre los stubs de T00 (T01 y T15 agregan sus propios tests)."""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from bcs.contract import validate_prediction_dict
from bcs.scale import load_scale

FIXTURES = Path(__file__).parent / "fixtures"
SCALE = load_scale("dairy_1_5")


def _post(client: TestClient, cow_id: str | None, name: str = "rear_cow", **kw):
    data = {"cow_id": cow_id} if cow_id is not None else {}
    files = {"image": (f"{name}.jpg", (FIXTURES / f"{name}.jpg").read_bytes(), "image/jpeg")}
    return client.post("/predict", data=data, files=files, **kw)


def test_health_needs_no_token(client: TestClient) -> None:
    r = client.get("/health", headers={"X-App-Token": ""})
    assert r.status_code == 200 and r.json()["status"] == "ok"


def test_token_required(client: TestClient) -> None:
    for path in ("/cows", "/cows/X1", "/predictions/abc", "/media/abc.jpg"):
        assert client.get(path, headers={"X-App-Token": "wrong"}).status_code == 401
    assert _post(client, "X1", headers={"X-App-Token": ""}).status_code == 401
    assert client.get(
        "/cows", headers={"X-App-Token": ""}, params={"token": "test-token"}
    ).is_success


def test_cow_id_validation(client: TestClient) -> None:
    assert _post(client, None).status_code == 422
    r = _post(client, "vaca con espacios")
    assert r.status_code == 422 and "cow_id" in r.json()["detail"]


def test_predict_and_rodeo(client: TestClient) -> None:
    r = _post(client, " uy-7 ")
    assert r.status_code == 200
    pred = r.json()
    assert pred["cow_id"] == "UY-7"
    assert validate_prediction_dict(pred, SCALE.keys, SCALE.step) == []
    _post(client, "UY-7", "side_cow")

    cows = client.get("/cows").json()
    assert [c["cow_id"] for c in cows] == ["UY-7"] and cows[0]["n_evaluations"] == 2
    hist = client.get("/cows/uy-7").json()
    assert len(hist) == 2 and hist[0]["created_at"] >= hist[1]["created_at"]
    assert client.get(f"/predictions/{pred['prediction_id']}").json() == pred
    media = client.get(f"/media/{pred['prediction_id']}.jpg")
    assert media.status_code == 200 and media.headers["content-type"] == "image/jpeg"
    assert client.get("/cows/NADIE").status_code == 404
    assert client.get("/predictions/nope").status_code == 404


def test_web_is_served(client: TestClient) -> None:
    assert client.get("/").status_code == 200
    assert client.get("/rodeo.html").status_code == 200
