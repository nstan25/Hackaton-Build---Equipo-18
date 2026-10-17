"""Genera el mini `dataset.rar` de fixture para T03 (SPEC §5.6.1).

Las imágenes son **sintéticas** (PIL): ningún dato de terceros en el repo (SPEC §1.1, §4.1).
Imitan lo que importa de `sdb_bcs`: estructura `dataset/<BCS>/<nombre>.jpg + .xml` (VOC con un
bbox), los esquemas de nombre `GS_<vaca>_<frame>`, `YM_<vaca>_<frame>`, `L-i<n>`, `R-i<n>`, frames
casi iguales dentro de un mismo grupo y el texto sobreimpreso (fecha arriba a la izquierda, "BCS"
abajo) en las posiciones del dataset real.

Ninguna herramienta instalada escribe `.rar` (libarchive solo lo lee), así que el archivo se arma
acá en formato RAR5 sin compresión ("stored"), según la nota técnica pública de RARLAB.

Uso: uv run python tests/fixtures/sdb_mini/make_mini_rar.py   (escribe dataset.rar y split.csv)
"""

from __future__ import annotations

import io
import random
import struct
import subprocess
import sys
import tempfile
import zlib
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
MTIME = 1_700_000_000  # fijo: mismo .rar byte a byte en cada corrida

CLASSES = ("3.25", "3.75", "4.25")
# (prefijo, vaca, frames) por clase; GS_7 y YM_7 comparten número (make_split los agrupa).
GSYM = {
    "3.25": [
        ("GS", 11, 2),
        ("GS", 12, 1),
        ("YM", 13, 1),
        ("GS", 14, 1),
        ("YM", 15, 1),
        ("GS", 16, 1),
        ("GS", 17, 1),
    ],
    "3.75": [
        ("GS", 21, 2),
        ("YM", 22, 1),
        ("GS", 7, 1),
        ("YM", 7, 1),
        ("GS", 23, 1),
        ("YM", 24, 1),
        ("GS", 25, 1),
    ],
    "4.25": [
        ("GS", 31, 2),
        ("GS", 32, 1),
        ("YM", 33, 1),
        ("GS", 34, 1),
        ("YM", 35, 1),
        ("GS", 36, 1),
        ("GS", 37, 1),
    ],
}
# tramos L/R (números consecutivos = un grupo; L-i<n> y R-i<n> comparten número)
LR = {
    "3.25": [(101, 2), (201, 1), (301, 1), (401, 1)],
    "3.75": [(501, 2), (601, 1), (701, 1), (801, 1)],
    "4.25": [(901, 2), (1001, 1), (1101, 1), (1201, 1)],
}
BIG = "GS_34"  # un grupo en 1364x768, como los 19 frames grandes del real


def _overlay(draw: ImageDraw.ImageDraw, family: str, w: int, h: int) -> None:
    """Texto blanco con borde negro en las posiciones del real (coordenadas de 1024x576)."""
    sx, sy = w / 1024, h / 576
    spots = {
        "GS": [((2, 4), "2023-07-06  THURSDAY  06:37:32 CH1"), ((934, 546), "BCS")],
        "YM": [
            ((2, 32), "2023-01-13  FRIDAY  08:29:41 CH02"),
            ((24, 496), "BCS2"),
            ((746, 512), "BCS1"),
        ],
    }[family]
    for (x, y), text in spots:
        font = ImageFont.load_default(size=int(20 * sy))  # ~ alto del texto real
        draw.text(
            (x * sx, y * sy), text, fill="white", font=font, stroke_width=1, stroke_fill="black"
        )


def _image(seed: int, frame: int, family: str, size: tuple[int, int]) -> tuple[bytes, tuple]:
    """Una 'vaca' (elipse con manchas propias del grupo) sobre fondo con ruido."""
    w, h = size
    rng = random.Random(seed)
    img = Image.new("RGB", size, tuple(rng.randrange(60, 120) for _ in range(3)))
    draw = ImageDraw.Draw(img)
    for _ in range(6):  # fondo: rejas
        x = rng.randrange(w)
        draw.line([(x, 0), (x + rng.randrange(-200, 200), h)], fill=(150, 150, 150), width=12)
    dx = frame * 6  # frames del mismo video: casi iguales
    x0, y0 = int(w * 0.26) + dx, int(h * 0.05)
    x1, y1 = int(w * 0.72) + dx, int(h * 0.95)
    draw.ellipse([x0, y0, x1, y1], fill=(190, 190, 185))
    for _ in range(6):
        cx, cy = rng.randrange(x0 + 20, x1 - 20), rng.randrange(y0 + 20, y1 - 20)
        r = rng.randrange(20, 70)
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(20, 20, 20))
    if family in ("GS", "YM"):
        _overlay(draw, family, w, h)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=75)
    return buf.getvalue(), (x0, y0, x1, y1)


def _xml(name: str, size: tuple[int, int], box: tuple) -> bytes:
    x0, y0, x1, y1 = box
    return (
        f"<annotation><filename>{name}</filename><size><width>{size[0]}</width>"
        f"<height>{size[1]}</height><depth>3</depth></size><object><name>cow</name><bndbox>"
        f"<xmin>{x0}</xmin><ymin>{y0}</ymin><xmax>{x1}</xmax><ymax>{y1}</ymax>"
        "</bndbox></object></annotation>\n"
    ).encode()


def build_files() -> dict[str, bytes]:
    files: dict[str, bytes] = {}
    for cls in CLASSES:
        items = [
            (f"{p}_{cow}_{f}", p, f"{p}_{cow}", f)
            for p, cow, n in GSYM[cls]
            for f in range(1, n + 1)
        ]
        for start, n in LR[cls]:
            side = "L" if start % 200 else "R"
            items += [
                (f"{side}-i{k}", "LR", f"LR{start}", k - start) for k in range(start, start + n)
            ]
        for stem, family, group, frame in items:
            size = (1364, 768) if group == BIG else (1024, 576)
            seed = zlib.crc32(f"{cls}/{group}".encode())
            jpg, box = _image(seed, frame, family, size)
            files[f"dataset/{cls}/{stem}.jpg"] = jpg
            files[f"dataset/{cls}/{stem}.xml"] = _xml(f"{stem}.jpg", size, box)
    return files


# --- RAR5 "stored" -------------------------------------------------------------------------


def _vint(n: int) -> bytes:
    out = bytearray()
    while True:
        b = n & 0x7F
        n >>= 7
        out.append(b | (0x80 if n else 0))
        if not n:
            return bytes(out)


def _block(body: bytes) -> bytes:
    """body = tipo + flags + campos. Antepone CRC32 y tamaño de encabezado."""
    head = _vint(len(body)) + body
    return struct.pack("<I", zlib.crc32(head)) + head


def rar5(files: dict[str, bytes]) -> bytes:
    out = bytearray(b"Rar!\x1a\x07\x01\x00")
    out += _block(_vint(1) + _vint(0) + _vint(0))  # main header, sin flags
    for name in sorted(files):
        data = files[name]
        enc = name.encode()
        body = (
            _vint(2)  # file header
            + _vint(0x02)  # tiene área de datos
            + _vint(len(data))
            + _vint(0x02 | 0x04)  # mtime + CRC32 presentes
            + _vint(len(data))  # tamaño descomprimido
            + _vint(0o644)  # atributos
            + struct.pack("<I", MTIME)
            + struct.pack("<I", zlib.crc32(data))
            + _vint(0)  # compresión: versión 0, método 0 = stored
            + _vint(1)  # host OS: Unix
            + _vint(len(enc))
            + enc
        )
        out += _block(body) + data
    out += _block(_vint(5) + _vint(0) + _vint(0))  # end of archive
    return bytes(out)


def main() -> None:
    files = build_files()
    rar = HERE / "dataset.rar"
    rar.write_bytes(rar5(files))
    # split.csv de referencia: make_split.py del equipo (sin tocar) sobre el mini dataset
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run(["bsdtar", "-xf", str(rar), "-C", tmp], check=True)
        subprocess.run(
            [
                sys.executable,
                str(REPO / "preprocesamiento" / "make_split.py"),
                str(Path(tmp) / "dataset"),
                str(HERE / "split.csv"),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
        )
    n = sum(k.endswith(".jpg") for k in files)
    print(f"{rar}: {n} imágenes, {rar.stat().st_size} bytes")


if __name__ == "__main__":
    main()
