"""Fotos hostiles en `POST /predict`: bomba de píxeles (R3) y archivos que no son imagen (R4)."""

from __future__ import annotations

import hashlib
import io
import resource
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from bcs import inference
from bcs.store.memory import THUMBNAIL_SIZE, InMemoryStore, make_thumbnail

FIXTURES = Path(__file__).parent / "fixtures"
TOKEN = "test-token"
REAR = (FIXTURES / "rear_cow.jpg").read_bytes()


def _encode(img: Image.Image, fmt: str, **kw: Any) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format=fmt, **kw)
    return buf.getvalue()


@pytest.fixture
def store(tmp_path: Path) -> InMemoryStore:
    return InMemoryStore(tmp_path / "media")


@pytest.fixture
def client(store: InMemoryStore, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    monkeypatch.setenv("BCS_APP_TOKEN", TOKEN)
    from bcs.api import create_app

    with TestClient(create_app(store), headers={"X-App-Token": TOKEN}) as c:
        yield c


def _post(client: TestClient, image: bytes, name: str = "foto.jpg") -> Any:
    return client.post("/predict", data={"cow_id": "A1"}, files={"image": (name, image)})


def _max_rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024  # Linux: KB


# ---------------------------------------------------------------- R3


def test_pixel_bomb_png_is_rejected_without_decoding(
    client: TestClient, store: InMemoryStore
) -> None:
    bomb = _encode(Image.new("RGB", (10000, 8000), (120, 100, 80)), "PNG")
    assert len(bomb) < 400 * 1024  # chica en bytes: el tope de 20 MB no la frena
    before = _max_rss_mb()
    r = _post(client, bomb, "bomba.png")
    grew = _max_rss_mb() - before
    assert r.status_code == 413
    assert "MP" in r.json()["detail"]
    assert store.latest_by_cow() == []
    # decodificada serían 240 MB en RGB (antes el pico llegaba a +0,6 GB); acá no se decodifica
    assert grew < 100, f"el pico de RSS creció {grew:.0f} MB"


def test_pixel_limit_checked_before_decoding(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(inference, "MAX_IMAGE_PIXELS", 100 * 100)
    raw = _encode(Image.new("RGB", (101, 100)), "PNG")

    def no_load(*_: Any, **__: Any) -> None:
        raise AssertionError("no debería decodificar")

    monkeypatch.setattr(Image.Image, "load", no_load)
    with pytest.raises(inference.ImageTooLarge):
        inference._to_image(raw)  # pyright: ignore[reportPrivateUsage]


def test_large_jpeg_is_decoded_reduced_with_draft(monkeypatch: pytest.MonkeyPatch) -> None:
    # una JPEG por encima del tope (p. ej. 48 MP de celular) se decodifica achicada, no se rechaza
    monkeypatch.setattr(inference, "MAX_IMAGE_PIXELS", 450 * 300)
    raw = _encode(Image.new("RGB", (900, 600), (90, 80, 70)), "JPEG")
    img, same = inference._to_image(raw)  # pyright: ignore[reportPrivateUsage]
    assert same == raw
    assert img.size == (450, 300) and img.size[0] * img.size[1] <= inference.MAX_IMAGE_PIXELS


def test_jpeg_too_large_even_with_draft_is_413(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(inference, "MAX_IMAGE_PIXELS", 10 * 10)
    raw = _encode(Image.new("RGB", (900, 600)), "JPEG")
    assert _post(client, raw).status_code == 413


def test_thumbnail_is_small_and_sha_is_of_the_original(
    client: TestClient, store: InMemoryStore
) -> None:
    big = _encode(Image.new("RGB", (3000, 2000), (200, 30, 30)), "PNG")
    r = _post(client, big, "grande.png")
    assert r.status_code == 200
    body = r.json()
    assert body["image_sha256"] == hashlib.sha256(big).hexdigest()
    thumb = store.thumbnail_path(body["prediction_id"])
    assert thumb is not None
    with Image.open(thumb) as t:
        assert t.format == "JPEG" and t.size == (THUMBNAIL_SIZE, 341)


def test_api_decodes_the_original_once(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    opened: list[int] = []
    real = inference._open_bounded  # pyright: ignore[reportPrivateUsage]

    def spy(raw: bytes) -> Image.Image:
        opened.append(len(raw))
        return real(raw)

    monkeypatch.setattr(inference, "_open_bounded", spy)
    seen: list[int] = []
    real_thumb = make_thumbnail

    def thumb_spy(image_bytes: bytes) -> bytes:
        seen.append(len(image_bytes))
        return real_thumb(image_bytes)

    monkeypatch.setattr("bcs.store.memory.make_thumbnail", thumb_spy)
    assert _post(client, REAR).status_code == 200
    assert opened == [len(REAR)]
    # el store recibe la miniatura, que se guarda tal cual sin abrir la original otra vez
    assert len(seen) == 1 and seen[0] != len(REAR)


def test_make_thumbnail_keeps_small_jpeg_and_respects_exif() -> None:
    small = _encode(Image.new("RGB", (400, 300)), "JPEG")
    assert make_thumbnail(small) == small
    exif = Image.Exif()
    exif[0x0112] = 6  # rotar 90°: no se guarda tal cual
    rotated = _encode(Image.new("RGB", (400, 300)), "JPEG", exif=exif)
    with Image.open(io.BytesIO(make_thumbnail(rotated))) as t:
        assert t.size == (300, 400)


def test_make_thumbnail_refuses_pixel_bombs() -> None:
    bomb = _encode(Image.new("L", (10000, 8000)), "PNG")
    with pytest.raises(ValueError, match="MP"):
        make_thumbnail(bomb)


# ---------------------------------------------------------------- R4

NOT_IMAGES = {
    "texto": b"hello world",
    "pdf": b"%PDF-1.4\n1 0 obj\n<< /Type /Catalog >>\nendobj\ntrailer\n<<>>\n%%EOF\n",
    "svg": b'<svg xmlns="http://www.w3.org/2000/svg" width="10" height="10"/>',
    "heic": b"\x00\x00\x00\x18ftypheic\x00\x00\x00\x00mif1heic" + b"\x00" * 64,
    "jpeg truncada": REAR[: len(REAR) // 3],
}


@pytest.fixture
def ultralytics_patch(monkeypatch: pytest.MonkeyPatch) -> None:
    """El `Image.open` de ultralytics activo, y que falle si intenta instalar algo en runtime."""
    import ultralytics.utils.checks as checks
    import ultralytics.utils.patches  # noqa: F401  — reemplaza Image.open

    def no_install(*_: Any, **__: Any) -> None:
        raise AssertionError("intentó instalar un paquete desde una request")

    monkeypatch.setattr(checks, "check_requirements", no_install)


@pytest.mark.parametrize("name", NOT_IMAGES)
def test_not_an_image_is_422(
    client: TestClient, store: InMemoryStore, ultralytics_patch: None, name: str
) -> None:
    r = _post(client, NOT_IMAGES[name], f"archivo.{name}")
    assert r.status_code == 422, r.text
    detail = r.json()["detail"]
    assert "image" in detail and "JPEG" in detail
    assert store.latest_by_cow() == []


def test_to_image_raises_invalid_image_for_any_failure() -> None:
    for raw in NOT_IMAGES.values():
        with pytest.raises(inference.InvalidImage):
            inference._to_image(raw)  # pyright: ignore[reportPrivateUsage]
