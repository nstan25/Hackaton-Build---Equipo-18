"""Backend Modal (SPEC §2.4): un contenedor GPU por fold, datos en el Volume `bcs-data`.

No se llama a mano: `make train BACKEND=modal` (scripts/compute/launch.py) sube el pack con
`modal volume put` y corre `modal run scripts/compute/modal_app.py::main --spec '<json>'`.

Imagen: Python 3.12 + las versiones de `uv.lock`, con torch/torchvision de PyPI (build CUDA; el
lock tiene el build CPU). El código del repo (`src/`, `config/`, `tests/fixtures/`) se monta en
`/root/repo`. GPU (`BCS_MODAL_GPU`, L4 por defecto) y timeout (`BCS_MODAL_TIMEOUT_S`, 4 h) se
fijan al importar este módulo.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tomllib
from pathlib import Path

import modal

REMOTE_REPO = "/root/repo"
# Este módulo también se importa dentro del contenedor, donde no está el repo ni uv.lock.
REPO = Path(__file__).resolve().parents[2] if modal.is_local() else Path(REMOTE_REPO)
VOL_MOUNT = "/vol"
GPU = os.environ.get("BCS_MODAL_GPU", "L4")
TIMEOUT_S = int(os.environ.get("BCS_MODAL_TIMEOUT_S", 4 * 3600))
TRAIN_DEPS = (
    "torch",
    "torchvision",
    "timm",
    "scikit-learn",
    "numpy",
    "pillow",
    "pandas",
    "pyyaml",
    "opencv-python-headless",
    "open-clip-torch",
    "matplotlib",
)


def _locked(names: tuple[str, ...]) -> list[str]:
    """`name==version` según uv.lock (la versión más alta si hay varias; sin el `+cpu`)."""
    lock = tomllib.loads((REPO / "uv.lock").read_text())
    found: dict[str, list[tuple[int, ...]]] = {}
    raw: dict[tuple[str, tuple[int, ...]], str] = {}
    for pkg in lock.get("package", []):
        if pkg["name"] in names:
            version = pkg["version"].split("+")[0]
            key = tuple(int(x) for x in re.findall(r"\d+", version))
            found.setdefault(pkg["name"], []).append(key)
            raw[(pkg["name"], key)] = version
    return [f"{n}=={raw[(n, max(found[n]))]}" if n in found else n for n in names]


vol = modal.Volume.from_name(os.environ.get("BCS_MODAL_VOLUME", "bcs-data"), create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(*(_locked(TRAIN_DEPS) if modal.is_local() else TRAIN_DEPS))
    .env({"PYTHONPATH": f"{REMOTE_REPO}/src", "BCS_BACKEND": "modal"})
    .add_local_dir(REPO / "src", f"{REMOTE_REPO}/src", ignore=["**/__pycache__"])
    .add_local_dir(REPO / "config", f"{REMOTE_REPO}/config")
    .add_local_dir(REPO / "tests" / "fixtures", f"{REMOTE_REPO}/tests/fixtures")
)
app = modal.App("bcs-train", image=image)


@app.function(gpu=GPU, timeout=TIMEOUT_S, volumes={VOL_MOUNT: vol})
def train_fold(fold: str, args: list[str], pack: str, task: str, git_sha: str) -> dict:
    import tarfile

    data = Path("/tmp/bcs-data")  # vacío → fixtures sintéticas (solo smoke)
    data.mkdir(parents=True, exist_ok=True)
    if pack:
        with tarfile.open(f"{VOL_MOUNT}/packs/{pack}") as tar:
            tar.extractall(data, filter="data")
    run_id = f"{task}/{os.environ.get('MODAL_TASK_ID', 'run')}_{fold}"
    out = Path(VOL_MOUNT) / "runs" / run_id
    weights = Path(VOL_MOUNT) / "artifacts" / run_id
    cmd = [sys.executable, "-m", "bcs.train", *args, "--data-dir", str(data)]
    cmd += ["--out-dir", str(out), "--weights-dir", str(weights), "--backend", "modal"]
    env = {**os.environ, "BCS_GIT_SHA": git_sha, "BCS_DATA_DIR": str(data)}
    subprocess.run(cmd, check=True, cwd=REMOTE_REPO, env=env)
    vol.commit()
    report = json.loads((out / "report.json").read_text())
    files = [str(f).removeprefix(VOL_MOUNT) for f in out.iterdir() if f.is_file()]
    files.append(report["train"]["weights"].removeprefix(VOL_MOUNT))
    return {"files": files, "train": report["train"], "mae": report["mean"].get("mae")}


@app.local_entrypoint()
def main(spec: str) -> None:
    s = json.loads(spec)
    folds: list[str] = s["folds"]
    calls = [(f, s["train_args"][f], s["pack"], s["task"], s["git_sha"]) for f in folds]
    for fold, res in zip(folds, train_fold.starmap(calls), strict=True):
        dst = Path(s["out"]) / fold
        dst.mkdir(parents=True, exist_ok=True)
        for remote in res["files"]:
            with (dst / Path(remote).name).open("wb") as f:
                for chunk in vol.read_file(remote):
                    f.write(chunk)
        t = res["train"]
        print(f"modal: fold={fold} device={t['device']} mae={res['mae']} {t['seconds']}s")
