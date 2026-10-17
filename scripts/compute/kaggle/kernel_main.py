"""Kernel de Kaggle (SPEC §2.4). Plantilla: `launch.py` reemplaza los dos marcadores y la sube
con `kaggle kernels push`.

- `__BCS_PAYLOAD__`: tar.gz en base64 con `src/bcs`, `config/` y `tests/fixtures/` del repo.
- `__BCS_ARGS__`: JSON con los argumentos de `bcs.train`, el nombre del pack y el sha de git.

El pack llega como dataset privado `<user>/bcs-pack` en /kaggle/input (como .tar o ya
extraído). Sin pack (smoke) corre sobre las fixtures sintéticas. Salida en /kaggle/working/out.
"""

import base64
import io
import json
import os
import subprocess
import sys
import tarfile
from pathlib import Path

PAYLOAD = "__BCS_PAYLOAD__"
ARGS = json.loads("__BCS_ARGS__")
WORK = Path("/kaggle/working")
REPO = Path("/tmp/repo")
DATA = Path("/tmp/bcs-data")
OUT = WORK / "out"


def find_pack(name: str) -> Path:
    """Directorio con labels.csv del pack `name` (extrae el .tar si Kaggle no lo hizo)."""
    inputs = Path("/kaggle/input")
    for tar_path in inputs.rglob(name):
        with tarfile.open(tar_path) as tar:
            tar.extractall(DATA, filter="data")
        return DATA
    for manifest in inputs.rglob("manifest.json"):
        if json.loads(manifest.read_text()).get("pack_hash") in name:
            return manifest.parent
    raise SystemExit(f"no encontré el pack {name} en /kaggle/input")


def main() -> None:
    REPO.mkdir(parents=True, exist_ok=True)
    DATA.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(base64.b64decode(PAYLOAD)), mode="r:gz") as tar:
        tar.extractall(REPO, filter="data")
    # torch, torchvision, numpy, pandas, sklearn y pillow vienen en la imagen de Kaggle.
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "timm", "pyyaml", "open_clip_torch"],
        check=True,
    )
    data = find_pack(ARGS["pack"]) if ARGS["pack"] else DATA
    cmd = [sys.executable, "-m", "bcs.train", *ARGS["train_args"], "--data-dir", str(data)]
    cmd += ["--out-dir", str(OUT), "--weights-dir", str(OUT), "--backend", "kaggle"]
    env = {
        **os.environ,
        "PYTHONPATH": str(REPO / "src"),
        "BCS_GIT_SHA": ARGS["git_sha"],
        "BCS_DATA_DIR": str(data),
        "BCS_BACKEND": "kaggle",
    }
    subprocess.run(cmd, check=True, cwd=REPO, env=env)
    print((OUT / "report.json").read_text())


if __name__ == "__main__":
    main()
