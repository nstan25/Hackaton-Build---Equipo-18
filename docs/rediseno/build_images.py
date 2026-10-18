"""Genera las imágenes de la web (`web/img/`) a partir de `logo.jpeg` y `hero.jpeg` (T18).

Uso: `uv run python docs/rediseno/build_images.py`. Determinista: mismas fuentes → mismos archivos.

- `logo-mark-*`: el símbolo (sol-ojo + campos) sin el texto, con el fondo de papel vuelto
  transparente; el nombre "Argos" lo pone la página con la tipografía de títulos.
- `hero-cow-*`: recorte de la mitad derecha del hero (la vaca y las líneas de escaneo, sin el logo,
  para no duplicarlo con el header). Las animaciones de la página se ubican en % de este recorte.
- `hero-full-*`: la imagen completa (con su logo), para "Sobre Argos".
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
OUT = HERE.parents[1] / "web" / "img"

# Cajas en píxeles de las imágenes originales (logo 1413×865, hero 1536×1024).
LOGO_MARK_BOX = (350, 40, 1112, 514)
HERO_COW_BOX = (704, 160, 1536, 1024)
PAPER = np.array([242, 242, 237])


def _transparent_paper(img: Image.Image) -> Image.Image:
    """Alfa por distancia al color del papel: el fondo desaparece y los bordes quedan suaves."""
    rgb = np.asarray(img.convert("RGB")).astype(np.int16)
    dist = np.abs(rgb - PAPER).sum(axis=2)
    alpha = np.clip((dist - 24) * 255 / 70, 0, 255).astype(np.uint8)
    return Image.fromarray(np.dstack([rgb.astype(np.uint8), alpha]), "RGBA")


def _save(img: Image.Image, stem: str, widths: list[int], jpeg: bool) -> None:
    for w in widths:
        h = round(img.height * w / img.width)
        out = img.resize((w, h), Image.Resampling.LANCZOS)
        out.save(OUT / f"{stem}-{w}.webp", quality=80, method=6)
        if jpeg:
            out.convert("RGB").save(
                OUT / f"{stem}-{w}.jpg", quality=80, optimize=True, progressive=True
            )
        else:
            out.save(OUT / f"{stem}-{w}.png", optimize=True)
        print(f"{stem}-{w}: {w}×{h}")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    logo = Image.open(HERE / "logo.jpeg")
    mark = _transparent_paper(logo.crop(LOGO_MARK_BOX))
    _save(mark, "logo-mark", [64, 128], jpeg=False)

    hero = Image.open(HERE / "hero.jpeg").convert("RGB")
    _save(hero.crop(HERO_COW_BOX), "hero-cow", [480, 720, 832], jpeg=True)
    _save(hero, "hero-full", [640, 1024, 1536], jpeg=True)


if __name__ == "__main__":
    main()
