"""Fetch de `sdb_bcs` (SPEC §1.2, §2.5): verifica el .rar en $BCS_DATA_DIR/raw/sdb_bcs/.

Si ya está con el tamaño y el sha256 esperados, no baja nada; si falta o está a medias, lo baja
(o reanuda) con curl. Escribe $BCS_DATA_DIR/sources.lock.sdb_bcs.json.

`--sample N` se acepta por contrato (§1.2) pero no cambia la descarga: el .rar es un único
archivo; el muestreo de N imágenes por clase lo hace `make preprocess`.

Uso: python scripts/data/fetch_sdb_bcs.py [--sample N]
"""

from __future__ import annotations

import argparse

from bcs.data.sources.sdb_bcs import SdbBcsSource


def main() -> None:
    ap = argparse.ArgumentParser(description="Fetch del .rar de sdb_bcs")
    ap.add_argument("--sample", type=int, default=None, help="N imágenes por clase (ver doc)")
    args = ap.parse_args()
    src = SdbBcsSource()
    src.fetch(args.sample)
    print(f"sdb_bcs: .rar verificado; lock en {src.root / 'sources.lock.sdb_bcs.json'}")


if __name__ == "__main__":
    main()
