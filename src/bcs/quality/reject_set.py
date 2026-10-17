"""Genera el set de rechazo de tests/fixtures/reject/ (dibujos y texturas con PIL/numpy).

Nada de datos ni imágenes de terceros (SPEC §4.1): todo sale de este código, con semillas fijas.
Cada imagen es algo que el filtro tiene que rechazar: no es una vaca, no es de atrás, o la foto
no sirve (oscura, quemada, borrosa). La tabla de qué es cada una queda en el README del directorio.

Uso: uv run python -m bcs.quality.reject_set   (determinista: mismo código → mismos bytes)
"""

from __future__ import annotations

import math
from collections.abc import Callable
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from bcs.paths import fixtures_dir

W, H = 320, 256
GRASS, SKY = (96, 140, 70), (180, 205, 230)
SEED = 20261017

Draw = ImageDraw.ImageDraw


def reject_dir() -> Path:
    return fixtures_dir() / "reject"


def _canvas(sky: tuple[int, int, int] = SKY, ground: tuple[int, int, int] = GRASS):
    img = Image.new("RGB", (W, H), sky)
    d = ImageDraw.Draw(img)
    d.rectangle([0, H * 3 // 5, W, H], fill=ground)
    return img, d


def _rng(k: int) -> np.random.Generator:
    return np.random.default_rng(SEED + k)


# --- animales y vacas que no son "vaca de atrás" -------------------------------------------


def side_cow() -> Image.Image:
    img, d = _canvas()
    d.rounded_rectangle([70, 90, 240, 170], radius=25, fill=(30, 30, 30))
    d.ellipse([120, 100, 170, 145], fill=(240, 240, 240))
    d.polygon([(240, 100), (285, 85), (295, 115), (245, 135)], fill=(30, 30, 30))
    for x in (82, 108, 200, 222):
        d.rectangle([x, 165, x + 12, 225], fill=(30, 30, 30))
    d.line([(70, 100), (55, 155)], fill=(20, 20, 20), width=4)
    return img


def cow_face() -> Image.Image:
    img, d = _canvas()
    d.polygon([(110, 40), (210, 40), (190, 200), (130, 200)], fill=(240, 240, 240))
    d.polygon([(110, 40), (160, 40), (150, 120), (118, 110)], fill=(25, 25, 25))
    d.ellipse([75, 45, 115, 70], fill=(25, 25, 25))  # orejas
    d.ellipse([205, 45, 245, 70], fill=(25, 25, 25))
    d.ellipse([128, 85, 142, 99], fill=(0, 0, 0))  # ojos
    d.ellipse([178, 85, 192, 99], fill=(0, 0, 0))
    d.ellipse([125, 165, 195, 215], fill=(230, 170, 170))  # hocico
    d.ellipse([140, 182, 152, 194], fill=(90, 50, 50))
    d.ellipse([168, 182, 180, 194], fill=(90, 50, 50))
    return img


def dog() -> Image.Image:
    img, d = _canvas()
    d.rounded_rectangle([100, 130, 210, 175], radius=18, fill=(150, 95, 45))
    d.ellipse([200, 100, 245, 140], fill=(150, 95, 45))
    d.polygon([(205, 105), (213, 82), (222, 105)], fill=(110, 70, 30))
    d.ellipse([228, 113, 234, 119], fill=(0, 0, 0))
    for x in (108, 128, 180, 196):
        d.rectangle([x, 170, x + 9, 205], fill=(150, 95, 45))
    d.line([(100, 140), (80, 112)], fill=(150, 95, 45), width=6)
    return img


def cat() -> Image.Image:
    img = Image.new("RGB", (W, H), (210, 190, 160))
    d = ImageDraw.Draw(img)
    d.ellipse([110, 110, 210, 220], fill=(230, 140, 50))  # cuerpo
    d.ellipse([125, 50, 195, 120], fill=(230, 140, 50))  # cabeza
    d.polygon([(130, 70), (135, 35), (155, 60)], fill=(230, 140, 50))
    d.polygon([(165, 60), (185, 35), (190, 70)], fill=(230, 140, 50))
    d.ellipse([142, 75, 152, 85], fill=(40, 120, 40))
    d.ellipse([168, 75, 178, 85], fill=(40, 120, 40))
    for dy in (-4, 4):
        d.line([(120, 98 + dy), (150, 96)], fill=(60, 40, 20), width=1)
        d.line([(170, 96), (200, 98 + dy)], fill=(60, 40, 20), width=1)
    d.arc([195, 150, 260, 220], start=180, end=320, fill=(230, 140, 50), width=8)
    return img


def bird() -> Image.Image:
    img = Image.new("RGB", (W, H), SKY)
    d = ImageDraw.Draw(img)
    d.line([(0, 190), (W, 175)], fill=(100, 70, 40), width=6)  # rama
    d.ellipse([120, 110, 200, 180], fill=(200, 40, 40))
    d.ellipse([180, 90, 225, 135], fill=(200, 40, 40))
    d.polygon([(222, 108), (245, 115), (222, 122)], fill=(240, 180, 40))
    d.ellipse([203, 103, 210, 110], fill=(0, 0, 0))
    d.polygon([(125, 135), (80, 120), (95, 150)], fill=(160, 30, 30))
    return img


def fish() -> Image.Image:
    img = Image.new("RGB", (W, H), (40, 110, 180))
    d = ImageDraw.Draw(img)
    d.ellipse([90, 90, 230, 170], fill=(250, 150, 30))
    d.polygon([(95, 130), (45, 90), (45, 170)], fill=(250, 150, 30))
    d.ellipse([195, 115, 207, 127], fill=(0, 0, 0))
    for x in (130, 160):
        d.line([(x, 95), (x, 165)], fill=(255, 255, 255), width=5)
    return img


def horse_side() -> Image.Image:
    img, d = _canvas()
    d.rounded_rectangle([80, 95, 230, 155], radius=22, fill=(120, 70, 35))
    d.polygon([(215, 105), (250, 50), (270, 55), (245, 120)], fill=(120, 70, 35))
    d.polygon([(250, 50), (290, 70), (285, 85), (255, 75)], fill=(120, 70, 35))
    for x in (90, 110, 195, 213):
        d.rectangle([x, 150, x + 9, 225], fill=(100, 60, 30))
    d.line([(80, 105), (60, 170)], fill=(40, 25, 15), width=6)
    return img


def sheep_flock() -> Image.Image:
    img, d = _canvas()
    rng = _rng(1)
    for _ in range(7):
        x, y = int(rng.integers(10, 260)), int(rng.integers(150, 210))
        d.ellipse([x, y, x + 50, y + 30], fill=(235, 235, 225))
        d.ellipse([x + 42, y + 2, x + 58, y + 16], fill=(40, 40, 40))
    return img


# --- personas, objetos y escenas ------------------------------------------------------------


def person() -> Image.Image:
    img, d = _canvas()
    d.ellipse([140, 30, 180, 70], fill=(230, 190, 160))
    d.rectangle([135, 70, 185, 150], fill=(40, 80, 200))
    d.rectangle([140, 150, 158, 220], fill=(50, 50, 70))
    d.rectangle([162, 150, 180, 220], fill=(50, 50, 70))
    d.line([(135, 80), (105, 130)], fill=(230, 190, 160), width=10)
    d.line([(185, 80), (215, 130)], fill=(230, 190, 160), width=10)
    return img


def face_selfie() -> Image.Image:
    img = Image.new("RGB", (W, H), (200, 210, 220))
    d = ImageDraw.Draw(img)
    d.ellipse([90, 30, 230, 220], fill=(225, 180, 150))
    d.chord([85, 15, 235, 140], start=180, end=360, fill=(60, 40, 25))
    d.ellipse([125, 100, 145, 115], fill=(255, 255, 255))
    d.ellipse([175, 100, 195, 115], fill=(255, 255, 255))
    d.ellipse([131, 103, 140, 112], fill=(60, 40, 20))
    d.ellipse([181, 103, 190, 112], fill=(60, 40, 20))
    d.arc([130, 140, 190, 185], start=20, end=160, fill=(150, 50, 50), width=4)
    return img


def car() -> Image.Image:
    img = Image.new("RGB", (W, H), SKY)
    d = ImageDraw.Draw(img)
    d.rectangle([0, 180, W, H], fill=(80, 80, 85))
    d.rounded_rectangle([50, 120, 270, 175], radius=12, fill=(200, 30, 30))
    d.polygon([(100, 120), (130, 85), (210, 85), (240, 120)], fill=(200, 30, 30))
    d.polygon([(110, 118), (135, 92), (168, 92), (168, 118)], fill=(170, 210, 240))
    d.polygon([(175, 118), (175, 92), (205, 92), (230, 118)], fill=(170, 210, 240))
    for x in (75, 205):
        d.ellipse([x, 155, x + 42, 197], fill=(20, 20, 20))
        d.ellipse([x + 12, 167, x + 30, 185], fill=(170, 170, 170))
    return img


def tractor() -> Image.Image:
    img, d = _canvas()
    d.rectangle([100, 100, 220, 170], fill=(30, 140, 50))
    d.rectangle([170, 55, 220, 105], outline=(20, 20, 20), width=5)
    d.rectangle([80, 120, 105, 140], fill=(30, 140, 50))
    d.ellipse([170, 130, 250, 210], fill=(25, 25, 25))
    d.ellipse([190, 150, 230, 190], fill=(230, 200, 40))
    d.ellipse([85, 160, 125, 200], fill=(25, 25, 25))
    d.rectangle([120, 70, 128, 100], fill=(60, 60, 60))
    return img


def house() -> Image.Image:
    img, d = _canvas()
    d.rectangle([90, 100, 230, 200], fill=(230, 210, 170))
    d.polygon([(80, 100), (160, 40), (240, 100)], fill=(160, 50, 40))
    d.rectangle([145, 145, 175, 200], fill=(110, 70, 40))
    d.rectangle([105, 120, 135, 145], fill=(170, 210, 240))
    d.rectangle([185, 120, 215, 145], fill=(170, 210, 240))
    return img


def barn_empty() -> Image.Image:
    img, d = _canvas(sky=(190, 200, 210), ground=(120, 100, 80))
    d.rectangle([50, 70, 270, 170], fill=(170, 40, 30))
    d.polygon([(40, 70), (160, 20), (280, 70)], fill=(90, 90, 90))
    d.rectangle([125, 100, 195, 170], fill=(240, 240, 240))
    d.line([(125, 100), (195, 170)], fill=(170, 40, 30), width=5)
    d.line([(195, 100), (125, 170)], fill=(170, 40, 30), width=5)
    for x in range(0, W, 28):
        d.rectangle([x, 195, x + 5, 245], fill=(140, 110, 70))
    d.line([(0, 205), (W, 205)], fill=(140, 110, 70), width=4)
    return img


def tree_landscape() -> Image.Image:
    img, d = _canvas()
    d.ellipse([240, 20, 290, 70], fill=(250, 220, 60))
    d.rectangle([90, 100, 110, 170], fill=(100, 70, 40))
    d.ellipse([50, 30, 150, 120], fill=(40, 120, 40))
    d.polygon([(180, 154), (230, 80), (280, 154)], fill=(130, 130, 140))
    return img


def flower() -> Image.Image:
    img = Image.new("RGB", (W, H), (120, 180, 90))
    d = ImageDraw.Draw(img)
    cx, cy = 160, 110
    for k in range(8):
        a = 2 * math.pi * k / 8
        x, y = cx + 45 * math.cos(a), cy + 45 * math.sin(a)
        d.ellipse([x - 25, y - 25, x + 25, y + 25], fill=(240, 80, 150))
    d.ellipse([cx - 25, cy - 25, cx + 25, cy + 25], fill=(250, 210, 40))
    d.line([(cx, cy + 50), (cx, H)], fill=(40, 110, 40), width=6)
    return img


def mug() -> Image.Image:
    img = Image.new("RGB", (W, H), (180, 140, 100))
    d = ImageDraw.Draw(img)
    d.rectangle([0, 180, W, H], fill=(130, 90, 60))
    d.rounded_rectangle([110, 80, 200, 200], radius=10, fill=(245, 245, 245))
    d.arc([180, 105, 235, 170], start=270, end=90, fill=(245, 245, 245), width=12)
    d.ellipse([115, 72, 195, 92], fill=(90, 50, 25))
    return img


def chair() -> Image.Image:
    img = Image.new("RGB", (W, H), (225, 220, 210))
    d = ImageDraw.Draw(img)
    d.rectangle([0, 200, W, H], fill=(160, 120, 80))
    d.rectangle([120, 40, 135, 210], fill=(110, 70, 40))
    d.rectangle([195, 40, 210, 210], fill=(110, 70, 40))
    d.rectangle([120, 40, 210, 70], fill=(110, 70, 40))
    d.rectangle([110, 130, 220, 145], fill=(110, 70, 40))
    return img


def food_plate() -> Image.Image:
    img = Image.new("RGB", (W, H), (140, 90, 50))
    d = ImageDraw.Draw(img)
    d.ellipse([60, 30, 260, 230], fill=(250, 250, 250))
    d.ellipse([110, 70, 180, 140], fill=(200, 40, 30))
    d.ellipse([150, 120, 220, 180], fill=(250, 200, 40))
    d.ellipse([100, 140, 150, 190], fill=(60, 150, 50))
    return img


# --- pantallas, documentos y texturas -------------------------------------------------------


def text_document() -> Image.Image:
    img = Image.new("RGB", (W, H), (250, 250, 245))
    d = ImageDraw.Draw(img)
    font = ImageFont.load_default()
    rng = _rng(2)
    words = "vaca condicion corporal informe tambo rodeo leche planilla fecha total".split()
    for i, y in enumerate(range(12, H - 10, 14)):
        line = " ".join(rng.choice(words, size=int(rng.integers(3, 7))))
        d.text((12, y), line.upper() if i == 0 else line, fill=(20, 20, 20), font=font)
    return img


def spreadsheet() -> Image.Image:
    img = Image.new("RGB", (W, H), (255, 255, 255))
    d = ImageDraw.Draw(img)
    font = ImageFont.load_default()
    rng = _rng(3)
    for x in range(0, W, 64):
        d.line([(x, 0), (x, H)], fill=(190, 190, 190))
    for y in range(0, H, 18):
        d.line([(0, y), (W, y)], fill=(190, 190, 190))
    d.rectangle([0, 0, W, 18], fill=(210, 230, 210))
    for y in range(20, H, 18):
        for x in range(4, W, 64):
            d.text((x, y + 2), f"{rng.uniform(1, 5):.2f}", fill=(0, 0, 0), font=font)
    return img


def phone_screenshot() -> Image.Image:
    img = Image.new("RGB", (W, H), (245, 245, 250))
    d = ImageDraw.Draw(img)
    font = ImageFont.load_default()
    d.rectangle([0, 0, W, 30], fill=(30, 100, 200))
    d.text((10, 10), "Mensajes", fill=(255, 255, 255), font=font)
    for i, y in enumerate(range(45, H - 30, 40)):
        x0 = 15 if i % 2 == 0 else 120
        fill = (225, 225, 230) if i % 2 == 0 else (60, 160, 90)
        d.rounded_rectangle([x0, y, x0 + 185, y + 28], radius=12, fill=fill)
        d.text((x0 + 10, y + 9), "hola, mando la foto", fill=(0, 0, 0), font=font)
    d.rectangle([0, H - 26, W, H], fill=(220, 220, 225))
    return img


def qr_like() -> Image.Image:
    rng = _rng(4)
    grid = rng.integers(0, 2, size=(25, 25)).astype(np.uint8) * 255
    for r, c in ((0, 0), (0, 18), (18, 0)):
        grid[r : r + 7, c : c + 7] = 0
        grid[r + 1 : r + 6, c + 1 : c + 6] = 255
        grid[r + 2 : r + 5, c + 2 : c + 5] = 0
    img = Image.fromarray(grid, mode="L").resize((220, 220), Image.Resampling.NEAREST)
    out = Image.new("RGB", (W, H), (255, 255, 255))
    out.paste(img.convert("RGB"), (50, 18))
    return out


def color_noise() -> Image.Image:
    arr = _rng(5).integers(0, 256, size=(H, W, 3), dtype=np.uint8)
    return Image.fromarray(arr, mode="RGB")


def gray_noise_blur() -> Image.Image:
    arr = _rng(6).integers(0, 256, size=(H // 8, W // 8), dtype=np.uint8)
    small = Image.fromarray(arr, mode="L").resize((W, H), Image.Resampling.BICUBIC)
    return small.convert("RGB")


def gradient() -> Image.Image:
    x = np.linspace(0, 1, W)[None, :, None]
    y = np.linspace(0, 1, H)[:, None, None]
    arr = np.concatenate([x * 255 + 0 * y, y * 255 + 0 * x, (1 - x) * 200 + 0 * y], axis=2)
    return Image.fromarray(arr.astype(np.uint8), mode="RGB")


def checkerboard() -> Image.Image:
    yy, xx = np.mgrid[0:H, 0:W]
    arr = (((xx // 32) + (yy // 32)) % 2 * 255).astype(np.uint8)
    return Image.fromarray(arr, mode="L").convert("RGB")


def stripes() -> Image.Image:
    xx = np.arange(W)[None, :].repeat(H, 0)
    r = ((np.sin(xx / 6.0) > 0) * 200 + 30).astype(np.uint8)
    arr = np.stack([r, np.full_like(r, 60), 255 - r], axis=2)
    return Image.fromarray(arr, mode="RGB")


def solid_color() -> Image.Image:
    return Image.new("RGB", (W, H), (70, 130, 180))


def concentric() -> Image.Image:
    img = Image.new("RGB", (W, H), (255, 255, 255))
    d = ImageDraw.Draw(img)
    for i, r in enumerate(range(120, 0, -12)):
        c = (230, 60, 60) if i % 2 == 0 else (255, 255, 255)
        d.ellipse([160 - r, 128 - r, 160 + r, 128 + r], fill=c)
    return img


def sky_only() -> Image.Image:
    img = Image.new("RGB", (W, H), (150, 190, 235))
    d = ImageDraw.Draw(img)
    for x, y in ((40, 50), (170, 90), (230, 30)):
        for dx in (0, 25, 50):
            d.ellipse([x + dx, y, x + dx + 45, y + 35], fill=(250, 250, 250))
    return img


def floor_concrete() -> Image.Image:
    arr = _rng(7).normal(130, 18, size=(H, W)).clip(0, 255).astype(np.uint8)
    img = Image.fromarray(arr, mode="L").convert("RGB").filter(ImageFilter.GaussianBlur(1))
    d = ImageDraw.Draw(img)
    for y in (60, 140, 220):
        d.line([(0, y), (W, y + 10)], fill=(90, 90, 90), width=3)
    return img


# --- fotos que no sirven (oscuras, quemadas, borrosas) -----------------------------------


def _rear_cow_drawing() -> Image.Image:
    """Vaca desde atrás y arriba (misma idea que tests/fixtures/rear_cow.jpg)."""
    img = Image.new("RGB", (W, H), (110, 110, 105))
    d = ImageDraw.Draw(img)
    d.ellipse([60, 10, 260, 250], fill=(30, 30, 30))
    d.ellipse([100, 50, 170, 120], fill=(240, 240, 240))
    d.ellipse([175, 140, 235, 200], fill=(240, 240, 240))
    d.ellipse([60, 30, 95, 65], fill=(60, 60, 60))
    d.ellipse([225, 30, 260, 65], fill=(60, 60, 60))
    d.rectangle([152, 130, 168, 256], fill=(20, 20, 20))
    return img


def black_frame() -> Image.Image:
    arr = _rng(8).integers(0, 6, size=(H, W, 3), dtype=np.uint8)
    return Image.fromarray(arr, mode="RGB")


def very_dark_cow() -> Image.Image:
    arr = np.asarray(_rear_cow_drawing(), dtype=np.float32) * 0.07
    return Image.fromarray(arr.astype(np.uint8), mode="RGB")


def night_noise() -> Image.Image:
    base = np.asarray(_rear_cow_drawing(), dtype=np.float32) * 0.05
    noise = _rng(9).normal(0, 4, size=base.shape)
    return Image.fromarray((base + noise).clip(0, 255).astype(np.uint8), mode="RGB")


def overexposed() -> Image.Image:
    arr = np.asarray(_rear_cow_drawing(), dtype=np.float32) * 0.15 + 235
    return Image.fromarray(arr.clip(0, 255).astype(np.uint8), mode="RGB")


def white_frame() -> Image.Image:
    return Image.new("RGB", (W, H), (252, 252, 252))


def blurred_cow() -> Image.Image:
    return _rear_cow_drawing().filter(ImageFilter.GaussianBlur(12))


def motion_blur_cow() -> Image.Image:
    img = _rear_cow_drawing()
    arr = np.asarray(img, dtype=np.float32)
    out = np.mean([np.roll(arr, s, axis=1) for s in range(-30, 31, 2)], axis=0)
    return Image.fromarray(out.astype(np.uint8), mode="RGB").filter(ImageFilter.GaussianBlur(4))


def lens_cap_finger() -> Image.Image:
    img = Image.new("RGB", (W, H), (200, 120, 100))
    d = ImageDraw.Draw(img)
    d.ellipse([-80, -40, 260, 330], fill=(170, 90, 75))
    return img.filter(ImageFilter.GaussianBlur(25))


GENERATORS: dict[str, Callable[[], Image.Image]] = {
    "side_cow": side_cow,
    "cow_face": cow_face,
    "dog": dog,
    "cat": cat,
    "bird": bird,
    "fish": fish,
    "horse_side": horse_side,
    "sheep_flock": sheep_flock,
    "person": person,
    "face_selfie": face_selfie,
    "car": car,
    "tractor": tractor,
    "house": house,
    "barn_empty": barn_empty,
    "tree_landscape": tree_landscape,
    "flower": flower,
    "mug": mug,
    "chair": chair,
    "food_plate": food_plate,
    "text_document": text_document,
    "spreadsheet": spreadsheet,
    "phone_screenshot": phone_screenshot,
    "qr_like": qr_like,
    "color_noise": color_noise,
    "gray_noise_blur": gray_noise_blur,
    "gradient": gradient,
    "checkerboard": checkerboard,
    "stripes": stripes,
    "solid_color": solid_color,
    "concentric": concentric,
    "sky_only": sky_only,
    "floor_concrete": floor_concrete,
    "black_frame": black_frame,
    "very_dark_cow": very_dark_cow,
    "night_noise": night_noise,
    "overexposed": overexposed,
    "white_frame": white_frame,
    "blurred_cow": blurred_cow,
    "motion_blur_cow": motion_blur_cow,
    "lens_cap_finger": lens_cap_finger,
}


def main() -> None:
    out = reject_dir()
    out.mkdir(parents=True, exist_ok=True)
    for name, fn in GENERATORS.items():
        fn().save(out / f"{name}.png", format="PNG")
    print(f"{len(GENERATORS)} imágenes en {out}")


if __name__ == "__main__":
    main()
