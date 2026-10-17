"""Endpoints de rodeo (T15) sobre `SqliteStore`, con token."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from bcs.contract import validate_prediction_dict
from bcs.scale import load_scale
from bcs.store.sqlite import SqliteStore

FIXTURES = Path(__file__).parent / "fixtures"
TOKEN = "test-token"
SCALE = load_scale("dairy_1_5")


def _app(tmp_path: Path) -> TestClient:
    from bcs.api import create_app

    store = SqliteStore(tmp_path / "bcs.sqlite", tmp_path / "media")
    return TestClient(create_app(store=store), headers={"X-App-Token": TOKEN})


@pytest.fixture
def api(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    monkeypatch.setenv("BCS_APP_TOKEN", TOKEN)
    with _app(tmp_path) as c:
        yield c


def _post(c: TestClient, cow_id: str, name: str = "rear_cow", **data: str) -> dict:
    files = {"image": (f"{name}.jpg", (FIXTURES / f"{name}.jpg").read_bytes(), "image/jpeg")}
    r = c.post("/predict", data={"cow_id": cow_id, **data}, files=files)
    assert r.status_code == 200, r.text
    return r.json()


def test_rodeo_endpoints(api: TestClient) -> None:
    p1 = _post(api, "uy-1", notes="primera")
    p2 = _post(api, "UY-1", "side_cow")
    p3 = _post(api, "UY-2")

    cows = {row["cow_id"]: row for row in api.get("/cows").json()}
    assert set(cows) == {"UY-1", "UY-2"}
    assert cows["UY-1"]["n_evaluations"] == 2
    assert cows["UY-1"]["last_prediction_id"] == p2["prediction_id"]
    assert cows["UY-2"]["last_prediction_id"] == p3["prediction_id"]

    hist = api.get("/cows/ uy-1 ").json()
    assert [h["prediction_id"] for h in hist] == [p2["prediction_id"], p1["prediction_id"]]
    for h in hist:
        assert validate_prediction_dict(h, SCALE.keys, SCALE.step) == []

    assert api.get(f"/predictions/{p1['prediction_id']}").json() == p1
    media = api.get(f"/media/{p1['prediction_id']}.jpg")
    assert media.status_code == 200 and media.headers["content-type"] == "image/jpeg"
    assert media.content[:2] == b"\xff\xd8"


def test_not_found_and_invalid(api: TestClient) -> None:
    assert api.get("/cows/NADIE").status_code == 404
    assert api.get("/cows/vaca con espacios").status_code == 422
    assert api.get("/predictions/nope").status_code == 404
    assert api.get("/media/nope.jpg").status_code == 404


def test_cows_filters(api: TestClient) -> None:
    _post(api, "UY-1")
    assert api.get("/cows", params={"status": "rejected"}).json() == []
    assert len(api.get("/cows", params={"since": "2000-01-01T00:00:00Z"}).json()) == 1
    assert api.get("/cows", params={"since": "2999-01-01"}).json() == []
    assert api.get("/cows", params={"since": "ayer"}).status_code == 422
    assert api.get("/cows", params={"status": "otro"}).status_code == 422
    assert api.get("/cows", params={"bcs_min": 4, "bcs_max": 3}).status_code == 422
    assert api.get("/cows", params={"bcs_max": "x"}).status_code == 422


def test_token_required(api: TestClient) -> None:
    pid = _post(api, "UY-1")["prediction_id"]
    for path in ("/cows", "/cows/UY-1", f"/predictions/{pid}", f"/media/{pid}.jpg"):
        assert api.get(path, headers={"X-App-Token": ""}).status_code == 401, path
        assert api.get(path, headers={"X-App-Token": "mal"}).status_code == 401, path
        assert api.get(path, headers={"X-App-Token": ""}, params={"token": TOKEN}).is_success


def test_persists_across_app_restart(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BCS_APP_TOKEN", TOKEN)
    with _app(tmp_path) as c:
        pred = _post(c, "UY-9")
    with _app(tmp_path) as c:
        assert c.get(f"/predictions/{pred['prediction_id']}").json() == pred
        assert [row["cow_id"] for row in c.get("/cows").json()] == ["UY-9"]
        assert c.get(f"/media/{pred['prediction_id']}.jpg").status_code == 200
