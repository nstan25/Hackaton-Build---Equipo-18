"""Arma el artifact que sube `cpu-job.yml` (Tenki). El repo es PÚBLICO: solo salen
`report.json`, `config.json` y los PNG, sin rutas. Pesos, predicciones (etiquetas de terceros,
`image_id`) y cualquier otro archivo se quedan en el runner.

    python scripts/compute/public_artifact.py <salida de bcs.train> <carpeta a subir>
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from typing import Any

ALLOWED = ("report.json", "config.json")
ALLOWED_SUFFIXES = (".png",)
# claves de report["train"] con rutas o datos del runner
SCRUB_TRAIN = ("weights", "predictions", "host")


def _scrub_paths(obj: Any) -> Any:
    """Saca strings que parecen rutas absolutas (el config no debería traer, por las dudas)."""
    if isinstance(obj, dict):
        return {k: _scrub_paths(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_scrub_paths(v) for v in obj]
    if isinstance(obj, str) and obj.startswith(("/", "~")):
        return None
    return obj


def public_files(src: Path) -> list[Path]:
    return sorted(
        p
        for p in src.iterdir()
        if p.is_file() and (p.name in ALLOWED or p.suffix in ALLOWED_SUFFIXES)
    )


def build(src: Path, dst: Path) -> list[Path]:
    dst.mkdir(parents=True, exist_ok=True)
    out = []
    for f in public_files(src):
        target = dst / f.name
        if f.suffix == ".json":
            data = json.loads(f.read_text())
            if f.name == "report.json" and isinstance(data.get("train"), dict):
                for k in SCRUB_TRAIN:
                    data["train"][k] = None
                data["train"]["artifacts_note"] = "pesos y predicciones quedaron en el runner"
            target.write_text(json.dumps(_scrub_paths(data), indent=2, ensure_ascii=False))
        else:
            shutil.copy2(f, target)
        out.append(target)
    return out


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 2:
        print(__doc__, file=sys.stderr)
        return 2
    files = build(Path(args[0]), Path(args[1]))
    print(f"public_artifact: {[f.name for f in files]}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
