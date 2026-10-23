"""Fetch de `zen_rgbd` (SPEC §1.2): `Dataset.xlsx` + frames RGB sueltos con remotezip.

Nunca baja los zips enteros (83,5 GB): lee su directorio central y extrae solo los frames del plan,
3 por (animal_id, fecha), con 8 hilos. Idempotente y reanudable (no rebaja lo que ya está con el
tamaño esperado; el crc32 se verifica al bajar). Escribe $BCS_DATA_DIR/sources.lock.zen_rgbd.json.

`--sample N` baja solo N frames por clase de BCS (rápido); sin él, el plan entero (~3.750 frames,
~12,7 GB).

Uso: python scripts/data/fetch_zen_rgbd.py [--sample N]
"""

from __future__ import annotations

import argparse

from bcs.data.sources.zen_rgbd import ZenRgbdSource


def main() -> None:
    ap = argparse.ArgumentParser(description="Fetch de frames de zen_rgbd")
    ap.add_argument("--sample", type=int, default=None, help="N frames por clase de BCS")
    args = ap.parse_args()
    src = ZenRgbdSource()
    stats = src.fetch(args.sample)
    print(
        f"zen_rgbd: {stats['downloaded']} frames bajados ahora, {stats['present']}/{stats['plan']} "
        f"del plan en disco; lock en {src.root / 'sources.lock.zen_rgbd.json'}"
    )


if __name__ == "__main__":
    main()
