"""Preprocesamiento de `sdb_bcs` según `preprocesamiento/README.md` (SPEC §2.5).

Un paso del README = una función, en el mismo orden. Los scripts del equipo se llaman tal cual
(subproceso, sin copiarlos ni reescribirlos). Sus salidas van a `$BCS_DATA_DIR/raw/sdb_bcs_pre/`:
como usan rutas relativas al directorio actual, se corren con `cwd` = esa carpeta.

    extract_rar         .rar → raw/sdb_bcs/dataset/            (requisito, no es paso del README)
    step1_make_split    make_split.py   → split.csv (debe coincidir byte a byte con el versionado)
    step2_apply_split   apply_split.py  → dataset_split/
    step3_prep_crops    prep_crops.py   → crops/ + crops/manifest.csv
    step4_prep_yolo     prep_yolo.py build → yolo_data/   (`prep_yolo.py eval` no se ejecuta)

Mismo .rar + mismo README/scripts + misma semilla + mismo `--sample` = mismo resultado; si el
manifiesto ya existe con esa clave, no se rehace nada.

Uso: python -m bcs.data.preprocess_sdb [--sample N]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from bcs.paths import REPO_ROOT, data_dir, runs_dir

SOURCE_ID = "sdb_bcs"
RAR_SHA256 = "b99164f4141e080b63b82d6462d0db4b61626138e20b07d02b6644b8b70d8880"
SEED = 42  # la del README (split.csv) y la del muestreo de --sample
MANIFEST = "preprocess_manifest.json"
YOLO_EVAL_NOTE = "no ejecutado: diagnóstico del detector, requiere bajar pesos (decisión T03/ask)"


def known_leakage(rar_sha: str) -> dict[str, Any] | None:
    """Fuga conocida del split del README (G0 punto 7, opción A); solo para el .rar real.

    `pairs_file` lista los pares dHash <= 4 entre splits como [a, b, dist, corr]; el adaptador
    exige que los casi-duplicados que encuentra sean exactamente esos (uno nuevo → falla).
    """
    if rar_sha != RAR_SHA256:
        return None
    evidence = runs_dir() / "T03" / "leak_evidence"
    return {
        "decision": "G0 punto 7, opción A: split.csv del README tal cual, sin excluir imágenes",
        "effect": "las métricas de val/test pueden estar levemente infladas (< 0,5% del total)",
        "causes": [
            "tramos L-i/R-i cortados en huecos de la numeración: frames contiguos del mismo"
            " video quedan en grupos LR_seg distintos y caen en splits distintos",
            "vacas GS con dos IDs: la misma vaca filmada con dos números queda en dos grupos",
        ],
        "dhash_le_4": {"pairs": 784},
        "corr_ge_0.9": {"pairs": 109, "images": 75},
        "corr_ge_0.8": {"pairs": 247, "images": 256},
        "corr": "correlación de píxeles entre las dos imágenes procesadas (4.º campo)",
        "evidence_dir": str(evidence),
        "pairs_file": str(evidence / "dups.json"),
        "pairs_sha256": "be146fcc0b6151be67fb3e64d2499a73e4a0ccd43c29c8670c5cdb9a418ef4d2",
    }


class PreprocessError(RuntimeError):
    pass


@dataclass(frozen=True)
class Config:
    rar: Path
    raw_dir: Path  # donde se extrae el .rar (queda raw_dir/dataset/<clase>/...)
    out_dir: Path  # raw/sdb_bcs_pre
    scripts_dir: Path  # preprocesamiento/ (README + scripts)
    reference_split: Path  # split.csv versionado contra el que se compara el paso 1
    sample: int | None = None

    @property
    def dataset_dir(self) -> Path:
        return self.raw_dir / "dataset"


def default_config(sample: int | None = None) -> Config:
    raw = data_dir() / "raw" / SOURCE_ID
    scripts = REPO_ROOT / "preprocesamiento"
    return Config(
        rar=raw / "dataset.rar",
        raw_dir=raw,
        out_dir=data_dir() / "raw" / f"{SOURCE_ID}_pre",
        scripts_dir=scripts,
        reference_split=scripts / "split.csv",
        sample=sample,
    )


def sha256_file(path: Path, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def _archiver() -> list[str]:
    for exe in ("bsdtar", str(Path.home() / ".local" / "bin" / "bsdtar")):
        if found := shutil.which(exe):
            return [found, "-xf"]
    if found := shutil.which("7z"):
        return [found, "x", "-y"]
    raise PreprocessError("falta bsdtar o 7z para extraer el .rar")


def _run(script: str, args: list[str], cfg: Config, log: str) -> None:
    """Corre un script del equipo tal cual, con cwd = out_dir, y guarda su salida."""
    cmd = [sys.executable, str(cfg.scripts_dir / script), *args]
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}  # nada de __pycache__ en preprocesamiento/
    res = subprocess.run(cmd, cwd=cfg.out_dir, capture_output=True, text=True, env=env)
    (cfg.out_dir / "logs").mkdir(exist_ok=True)
    (cfg.out_dir / "logs" / log).write_text(res.stdout + res.stderr)
    if res.returncode != 0:
        raise PreprocessError(f"{script} falló ({res.returncode}):\n{res.stderr[-2000:]}")


def _by_class(paths: list[str]) -> dict[str, int]:
    """'<clase>/<archivo>' → imágenes por clase."""
    return dict(sorted(Counter(p.split("/")[0] for p in paths).items()))


def _list_images(root: Path) -> list[str]:
    return sorted(f"{p.parent.name}/{p.name}" for p in root.glob("*/*.jpg"))


def _discarded(missing: list[str], reason: str, limit: int = 50) -> dict[str, Any]:
    return {"count": len(missing), "reason": reason if missing else "", "files": missing[:limit]}


# --- pasos -------------------------------------------------------------------------------


def extract_rar(cfg: Config, rar_sha: str) -> dict[str, Any]:
    """Extrae el .rar completo (también con --sample: el paso 1 necesita todo el dataset)."""
    stamp = cfg.raw_dir / ".extracted.json"
    if not (stamp.exists() and json.loads(stamp.read_text()).get("sha256") == rar_sha):
        cfg.raw_dir.mkdir(parents=True, exist_ok=True)
        subprocess.run([*_archiver(), str(cfg.rar), "-C", str(cfg.raw_dir)], check=True)
        stamp.write_text(json.dumps({"sha256": rar_sha}) + "\n")
    images = _list_images(cfg.dataset_dir)
    if not images:
        raise PreprocessError(f"el .rar no trae dataset/<clase>/*.jpg: {cfg.rar}")
    return {"rar_sha256": rar_sha, "images": images}


def step1_make_split(cfg: Config, images: list[str]) -> dict[str, Any]:
    """README paso 1: `make_split.py <dataset>` → split.csv; debe reproducir el versionado."""
    regen = cfg.out_dir / "split_regen.csv"
    _run("make_split.py", [str(cfg.dataset_dir), regen.name], cfg, "step1_make_split.log")
    if regen.read_bytes() != cfg.reference_split.read_bytes():
        raise PreprocessError(
            f"make_split.py no reproduce {cfg.reference_split} byte a byte (ver {regen})"
        )
    df = pd.read_csv(cfg.reference_split, dtype={"cow": str})
    out_df = df
    if cfg.sample is not None:
        rng = random.Random(SEED)
        keep: list[str] = []
        for _, sub in df.groupby("label", sort=True):
            paths = sorted(sub["path"])
            keep += rng.sample(paths, min(cfg.sample, len(paths)))
        out_df = df[df["path"].isin(keep)]
    # split.csv de trabajo: el versionado (filtrado al sample si corresponde), mismo formato
    if cfg.sample is None:
        shutil.copyfile(cfg.reference_split, cfg.out_dir / "split.csv")
    else:
        out_df.to_csv(cfg.out_dir / "split.csv", index=False)
    listed = set(df["path"])
    kept = list(out_df["path"])
    return {
        "before": _by_class(images),
        "after": _by_class(kept),
        "after_by_split": {k: _by_class(list(v)) for k, v in out_df.groupby("split")["path"]},
        "discarded": _discarded([p for p in images if p not in listed], "no figura en split.csv"),
        "not_sampled": len(df) - len(out_df),
        "split_reproduced": True,
    }


def step2_apply_split(cfg: Config, kept: list[str]) -> dict[str, Any]:
    """README paso 2: `apply_split.py <dataset>` → dataset_split/{train,val,test}/<clase>/."""
    root = cfg.out_dir / "dataset_split"
    if root.exists():  # sin restos de una corrida con otro --sample
        shutil.rmtree(root)
    _run(
        "apply_split.py",
        [str(cfg.dataset_dir), "split.csv", "dataset_split"],
        cfg,
        "step2_apply_split.log",
    )
    got = sorted(f"{p.parent.name}/{p.name}" for p in root.glob("*/*/*.jpg"))
    return {
        "before": _by_class(kept),
        "after": _by_class(got),
        "discarded": _discarded(sorted(set(kept) - set(got)), "no se copió"),
    }


def step3_prep_crops(cfg: Config, kept: list[str]) -> dict[str, Any]:
    """README paso 3: `prep_crops.py` → crops/{split}/{clase}/ (+10% margen, lado máx 320)."""
    _run(
        "prep_crops.py",
        ["dataset_split", "split.csv", "crops"],
        cfg,
        "step3_prep_crops.log",
    )
    manifest = pd.read_csv(cfg.out_dir / "crops" / "manifest.csv")
    done = [p for p, c in zip(manifest["path"], manifest["crop"], strict=True)
            if (cfg.out_dir / c).exists()]  # fmt: skip
    return {
        "before": _by_class(kept),
        "after": _by_class(done),
        "discarded": _discarded(sorted(set(kept) - set(done)), "sin recorte"),
    }


def step4_prep_yolo(cfg: Config, kept: list[str]) -> dict[str, Any]:
    """README paso 4: `prep_yolo.py build` → yolo_data/ (muestra de train/val, 1 clase)."""
    yolo = cfg.out_dir / "yolo_data"
    if yolo.exists():  # la muestra depende de split.csv: se rearma entera
        shutil.rmtree(yolo)
    _run("prep_yolo.py", ["build"], cfg, "step4_prep_yolo.log")
    cls_of = {p.split("/")[1]: p for p in kept}
    used = sorted(cls_of[p.name] for p in yolo.glob("images/*/*.jpg"))
    return {
        "before": _by_class(kept),
        "after": _by_class(used),
        "after_by_split": {
            s: _by_class([cls_of[p.name] for p in (yolo / "images" / s).glob("*.jpg")])
            for s in ("train", "val")
        },
        "discarded": _discarded(
            sorted(set(kept) - set(used)),
            "fuera de la muestra del detector (train 6000 / val 1000; test no entra)",
        ),
        "eval": YOLO_EVAL_NOTE,
    }


STEPS = (
    (1, "make_split.py <dataset>", step1_make_split),
    (2, "apply_split.py <dataset>", step2_apply_split),
    (3, "prep_crops.py", step3_prep_crops),
    (4, "prep_yolo.py build", step4_prep_yolo),
)


# --- orquestación ------------------------------------------------------------------------


def _key(cfg: Config) -> dict[str, Any]:
    scripts = sorted(p for p in cfg.scripts_dir.glob("*.py"))
    return {
        "source": SOURCE_ID,
        "rar_sha256": sha256_file(cfg.rar),
        "readme_sha256": sha256_file(cfg.scripts_dir / "README.md"),
        "scripts_sha256": {p.name: sha256_file(p) for p in scripts},
        "reference_split_sha256": sha256_file(cfg.reference_split),
        "code_sha256": sha256_file(Path(__file__)),
        "seed": SEED,
        "sample": cfg.sample,
    }


def run(cfg: Config) -> dict[str, Any]:
    """Corre los pasos en orden y escribe el manifiesto; con la misma clave no hace nada."""
    manifest_path = cfg.out_dir / MANIFEST
    key = _key(cfg)
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text())
        if {k: old.get(k) for k in key} == key:
            return old

    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    ex = extract_rar(cfg, key["rar_sha256"])
    records: list[dict[str, Any]] = []
    images: list[str] = ex["images"]  # el paso 1 ve todo el dataset; los demás, lo que quedó
    for n, script, fn in STEPS:
        rec = fn(cfg, images)
        records.append({"step": n, "script": script, "function": fn.__name__, **rec})
        if n == 1:
            images = sorted(pd.read_csv(cfg.out_dir / "split.csv")["path"])

    manifest = {
        **key,
        "images_in_rar": _by_class(ex["images"]),
        "steps": records,
        "known_leakage": known_leakage(key["rar_sha256"]),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    return manifest


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    ap.add_argument("--sample", type=int, default=None, help="N imágenes por clase")
    args = ap.parse_args(argv)
    manifest = run(default_config(args.sample))
    for s in manifest["steps"]:
        print(f"paso {s['step']} {s['function']}: {s['before']} → {s['after']}")
    print(f"manifiesto: {default_config(args.sample).out_dir / MANIFEST}")


if __name__ == "__main__":
    main()
