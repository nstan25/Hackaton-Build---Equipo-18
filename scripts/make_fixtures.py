"""Genera las imágenes sintéticas de tests/fixtures/ (dibujos con PIL, sin datos de terceros).

Uso: uv run python scripts/make_fixtures.py   (determinista: mismo código → mismos bytes)
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw

OUT = Path(__file__).resolve().parents[1] / "tests" / "fixtures"
W, H = 640, 480
GRASS, SKY = (96, 140, 70), (180, 205, 230)


def _canvas() -> tuple[Image.Image, ImageDraw.ImageDraw]:
    img = Image.new("RGB", (W, H), SKY)
    d = ImageDraw.Draw(img)
    d.rectangle([0, H // 2, W, H], fill=GRASS)
    return img, d


def rear_cow() -> Image.Image:
    """Vaca Holando vista desde atrás y arriba: grupa, isquiones, ilíacos y cola."""
    img, d = _canvas()
    d.ellipse([170, 90, 470, 420], fill=(30, 30, 30))  # cuerpo
    d.ellipse([230, 150, 330, 260], fill=(240, 240, 240))  # mancha blanca
    d.ellipse([170, 110, 220, 160], fill=(60, 60, 60))  # ilíaco izq.
    d.ellipse([420, 110, 470, 160], fill=(60, 60, 60))  # ilíaco der.
    d.ellipse([270, 300, 300, 330], fill=(70, 70, 70))  # isquion izq.
    d.ellipse([340, 300, 370, 330], fill=(70, 70, 70))  # isquion der.
    d.rectangle([312, 260, 328, 440], fill=(20, 20, 20))  # cola
    d.rectangle([215, 400, 255, 470], fill=(30, 30, 30))  # patas
    d.rectangle([385, 400, 425, 470], fill=(30, 30, 30))
    return img


def side_cow() -> Image.Image:
    """Vaca de costado: cuerpo alargado, cuatro patas y cabeza."""
    img, d = _canvas()
    d.rounded_rectangle([140, 170, 470, 330], radius=50, fill=(30, 30, 30))
    d.ellipse([230, 190, 330, 280], fill=(240, 240, 240))
    d.polygon([(470, 190), (560, 160), (580, 220), (480, 260)], fill=(30, 30, 30))  # cabeza
    for x in (165, 215, 400, 445):
        d.rectangle([x, 320, x + 25, 440], fill=(30, 30, 30))
    d.line([(140, 190), (110, 300)], fill=(20, 20, 20), width=8)  # cola
    return img


def dog() -> Image.Image:
    """Perro marrón de costado, más chico que una vaca."""
    img, d = _canvas()
    d.rounded_rectangle([220, 260, 420, 340], radius=30, fill=(150, 95, 45))
    d.ellipse([400, 210, 480, 280], fill=(150, 95, 45))  # cabeza
    d.polygon([(410, 215), (425, 180), (440, 215)], fill=(110, 70, 30))  # oreja
    d.ellipse([455, 235, 465, 245], fill=(0, 0, 0))  # ojo
    for x in (235, 270, 365, 395):
        d.rectangle([x, 335, x + 15, 400], fill=(150, 95, 45))
    d.line([(220, 280), (180, 230)], fill=(150, 95, 45), width=10)  # cola
    return img


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for name, fn in {"rear_cow": rear_cow, "side_cow": side_cow, "dog": dog}.items():
        fn().save(OUT / f"{name}.jpg", format="JPEG", quality=90)
        print(OUT / f"{name}.jpg")


if __name__ == "__main__":
    main()
