"""`make pack-data` (SPEC §2.4): pack de entrenamiento para los backends remotos.

Lee `labels.csv` + `splits.csv` de `$BCS_DATA_DIR`, descarta el split `test` y todo grupo de
`holdout_groups.txt` (si falta ese archivo, no empaqueta nada), redimensiona cada imagen a
256 px de lado corto y escribe `$BCS_ARTIFACTS_DIR/packs/bcs_pack_<hash>.tar` con:

    images/<path original>    jpg redimensionados
    labels.csv                `path` reescrito a images/...
    splits.csv                image_id,split (solo train/val)
    manifest.json             pack_hash, conteos, parámetros, sha256 de holdout_groups.txt

El hash es de contenido (filas de labels/splits + sha256 de cada imagen + parámetros): mismos
datos = mismo nombre, y si el pack ya existe no se rehace. El tar es determinista.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import sys
import tarfile
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from pathlib import Path

import pandas as pd
from PIL import Image

from bcs.data.schema import LABEL_COLUMNS, load_labels
from bcs.paths import artifacts_dir, data_dir

PACK_VERSION = 1
SHORT_SIDE = 256
JPEG_QUALITY = 90


class PackError(RuntimeError):
    pass


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def resize_jpeg(src: Path, short_side: int = SHORT_SIDE) -> bytes:
    with Image.open(src) as im:
        im = im.convert("RGB")
        w, h = im.size
        scale = short_side / min(w, h)
        if scale < 1:
            im = im.resize((round(w * scale), round(h * scale)), Image.Resampling.BICUBIC)
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=JPEG_QUALITY)
    return buf.getvalue()


def select_rows(root: Path) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """(labels, splits) de train+val sin holdout, y el sha256 de holdout_groups.txt."""
    holdout_path = root / "holdout_groups.txt"
    if not holdout_path.exists():
        raise PackError(f"falta {holdout_path}: sin él no se puede garantizar que no entre holdout")
    holdout = {ln.strip() for ln in holdout_path.read_text().splitlines() if ln.strip()}
    labels = load_labels(root / "labels.csv")
    splits = pd.read_csv(root / "splits.csv", dtype=str, keep_default_na=False)
    splits = splits[["image_id", "split"]]
    keep_ids = set(splits.loc[splits["split"].isin(["train", "val"]), "image_id"])
    in_holdout = labels["group_id"].isin(holdout)
    if bool(in_holdout.any()):
        # labels.csv no debería traer holdout (make labels lo excluye): se descarta y se avisa
        print(f"pack-data: AVISO: {int(in_holdout.sum())} filas de holdout en labels.csv; "
              "se descartan", file=sys.stderr)  # fmt: skip
    labels = labels[labels["image_id"].isin(keep_ids) & ~in_holdout]
    labels = labels.sort_values("image_id").reset_index(drop=True)
    splits = splits[splits["image_id"].isin(set(labels["image_id"]))]
    splits = splits.sort_values("image_id").reset_index(drop=True)
    if labels.empty:
        raise PackError("no quedan imágenes de train/val para empaquetar")
    return labels, splits, _sha256_file(holdout_path)


@contextmanager
def _mapper(workers: int) -> Iterator[Callable[[Callable, Iterable], Iterable]]:
    """`map` en paralelo con procesos, o en serie si `workers == 1`."""
    if workers == 1:
        yield map
        return
    with ProcessPoolExecutor(max_workers=workers) as ex:
        yield lambda fn, items: ex.map(fn, items, chunksize=32)


def _hash_one(path: str) -> str:
    return _sha256_file(Path(path))


def content_hash(
    root: Path, labels: pd.DataFrame, splits: pd.DataFrame, workers: int
) -> tuple[str, list[str]]:
    paths = [str(root / p) for p in labels["path"]]
    with _mapper(workers) as pmap:
        digests = list(pmap(_hash_one, paths))
    h = hashlib.sha256()
    h.update(json.dumps({"v": PACK_VERSION, "short": SHORT_SIDE, "q": JPEG_QUALITY}).encode())
    h.update(labels.to_csv(index=False).encode())
    h.update(splits.to_csv(index=False).encode())
    for d in digests:
        h.update(d.encode())
    return h.hexdigest()[:16], digests


def _add_bytes(tar: tarfile.TarFile, name: str, data: bytes) -> None:
    info = tarfile.TarInfo(name)
    info.size = len(data)
    info.mtime = 0
    info.mode = 0o644
    tar.addfile(info, io.BytesIO(data))


def _resize_job(args: tuple[str, int]) -> bytes:
    return resize_jpeg(Path(args[0]), args[1])


def build_pack(
    root: Path | None = None, out_dir: Path | None = None, workers: int | None = None
) -> Path:
    root = root or data_dir()
    out_dir = out_dir or artifacts_dir() / "packs"
    workers = workers or os.cpu_count() or 1
    labels, splits, holdout_sha = select_rows(root)
    pack_hash, _ = content_hash(root, labels, splits, workers)
    out = out_dir / f"bcs_pack_{pack_hash}.tar"
    if out.exists():
        print(f"pack-data: ya existe {out}", file=sys.stderr)
        return out

    packed = labels.copy()
    packed["path"] = ["images/" + p for p in labels["path"]]
    manifest = {
        "pack_hash": pack_hash,
        "pack_version": PACK_VERSION,
        "short_side": SHORT_SIDE,
        "jpeg_quality": JPEG_QUALITY,
        "n_images": len(labels),
        "n_groups": int(labels["group_id"].nunique()),
        "by_split": {k: int(v) for k, v in splits["split"].value_counts().sort_index().items()},
        "by_bcs": {
            f"{k:.2f}": int(v) for k, v in labels["bcs"].value_counts().sort_index().items()
        },
        "sources": sorted(labels["source"].unique().tolist()),
        "holdout_groups_sha256": holdout_sha,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tar.tmp")
    jobs = [(str(root / p), SHORT_SIDE) for p in labels["path"]]
    with tarfile.open(tmp, "w", format=tarfile.PAX_FORMAT) as tar:
        _add_bytes(tar, "manifest.json", json.dumps(manifest, indent=2).encode())
        _add_bytes(
            tar, "labels.csv", packed.loc[:, list(LABEL_COLUMNS)].to_csv(index=False).encode()
        )
        _add_bytes(tar, "splits.csv", splits.to_csv(index=False).encode())
        with _mapper(workers) as pmap:
            for name, data in zip(packed["path"], pmap(_resize_job, jobs), strict=True):
                _add_bytes(tar, name, data)
    tmp.rename(out)
    out.with_suffix(".json").write_text(json.dumps(manifest, indent=2))
    print(
        f"pack-data: {out} ({len(labels)} imágenes, {out.stat().st_size / 1e9:.2f} GB)",
        file=sys.stderr,
    )
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--data-dir", type=Path, default=None)
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--workers", type=int, default=None)
    args = p.parse_args(argv)
    try:
        print(build_pack(args.data_dir, args.out_dir, args.workers))
    except (PackError, FileNotFoundError) as e:
        print(f"pack-data: {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
