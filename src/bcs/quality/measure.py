"""Mide el filtro de calidad y escribe `$BCS_ARTIFACTS_DIR/T08/quality_metrics.json`.

- `reject_negatives`: fracción rechazada del set de rechazo (`tests/fixtures/reject/`).
- `accept_rear`: null. No hay fotos traseras de celular (G0 §2: sin `phone_backview`).
- `reject_val`: fracción rechazada de las imágenes de val de `sdb_bcs` (nunca test).
- Extra (no lo usa `make eval`): val degradada a mano (oscura, quemada, borrosa), que debería
  rechazarse, para tener negativos con fotos reales sin meter imágenes de terceros en el repo.

Uso: uv run python -m bcs.quality.measure [--val-limit N] [--out DIR]
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageFilter

from bcs.contract import QualityReport
from bcs.paths import artifacts_dir, data_dir
from bcs.quality.clip_checker import (
    CLIP_MODEL,
    CLIP_PRETRAINED,
    ClipQualityChecker,
)
from bcs.quality.reject_set import reject_dir

SEED = 42
BATCH = 64
SOURCE = "sdb_bcs"


def val_image_paths(limit: int | None = None) -> list[Path]:
    """Imágenes de val de `sdb_bcs` según `splits_sdb_bcs.csv`. Las de test no se tocan."""
    root = data_dir()
    with (root / f"splits_{SOURCE}.csv").open() as f:
        ids = [r["image_id"] for r in csv.DictReader(f) if r["split"] == "val"]
    if limit is not None and limit < len(ids):
        ids = random.Random(SEED).sample(ids, limit)
    return [root / "images" / SOURCE / f"{i}.jpg" for i in sorted(ids)]


def reject_image_paths() -> list[Path]:
    return sorted(reject_dir().glob("*.png"))


def _darken(im: Image.Image) -> Image.Image:
    arr = np.asarray(im.convert("RGB"), dtype=np.float32) * 0.12
    return Image.fromarray(arr.astype(np.uint8), mode="RGB")


def _overexpose(im: Image.Image) -> Image.Image:
    arr = np.asarray(im.convert("RGB"), dtype=np.float32) * 0.2 + 225
    return Image.fromarray(arr.clip(0, 255).astype(np.uint8), mode="RGB")


def _blur(im: Image.Image) -> Image.Image:
    return im.convert("RGB").filter(ImageFilter.GaussianBlur(10))


DEGRADATIONS: dict[str, Callable[[Image.Image], Image.Image]] = {
    "dark": _darken,
    "overexposed": _overexpose,
    "blur": _blur,
}


def run_checker(
    checker: ClipQualityChecker,
    paths: Sequence[Path],
    transform: Callable[[Image.Image], Image.Image] | None = None,
) -> list[QualityReport]:
    reports: list[QualityReport] = []
    for i in range(0, len(paths), BATCH):
        images = []
        for p in paths[i : i + BATCH]:
            with Image.open(p) as im:
                im = im.convert("RGB")
                images.append(transform(im) if transform else im)
        reports += checker.check_batch(images)
    return reports


def summarize(paths: Sequence[Path], reports: Sequence[QualityReport]) -> dict[str, Any]:
    rejected = [not r.passed for r in reports]
    issues = Counter(i for r in reports for i in r.issues)
    return {
        "n": len(reports),
        "rejected": int(sum(rejected)),
        "reject_rate": round(float(np.mean(rejected)), 4) if reports else None,
        "issues": dict(issues.most_common()),
    }


def _rows(paths: Sequence[Path], reports: Sequence[QualityReport], passed: bool) -> list[dict]:
    rows = [
        {"image": p.stem, **asdict(r)}
        for p, r in zip(paths, reports, strict=True)
        if r.passed == passed
    ]
    return sorted(rows, key=lambda d: d["score"])


def contact_sheet(paths: Sequence[Path], out: Path, n: int = 24) -> None:
    if not paths:
        return
    cols, cell = 6, (220, 180)
    shown = list(paths)[:n]
    sheet = Image.new("RGB", (cols * cell[0], -(-len(shown) // cols) * cell[1]))
    for k, p in enumerate(shown):
        with Image.open(p) as im:
            im = im.convert("RGB")
            im.thumbnail(cell)
            sheet.paste(im, ((k % cols) * cell[0], (k // cols) * cell[1]))
    sheet.save(out, quality=85)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Mide el filtro de calidad (T08).")
    parser.add_argument("--val-limit", type=int, help="muestra de val (default: todas)")
    parser.add_argument("--degraded-n", type=int, default=300, help="imágenes de val a degradar")
    parser.add_argument("--out", type=Path, default=artifacts_dir() / "T08")
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)

    checker = ClipQualityChecker()

    rej_paths = reject_image_paths()
    rej_reports = run_checker(checker, rej_paths)
    val_paths = val_image_paths(args.val_limit)
    val_reports = run_checker(checker, val_paths)

    deg_paths = val_image_paths(args.degraded_n)
    degraded = {
        name: summarize(deg_paths, run_checker(checker, deg_paths, fn))
        for name, fn in DEGRADATIONS.items()
    }

    rej = summarize(rej_paths, rej_reports)
    val = summarize(val_paths, val_reports)
    metrics = {
        "reject_negatives": rej["reject_rate"],
        "accept_rear": None,
        "reject_val": val["reject_rate"],
        "accept_rear_note": "sin fotos traseras de celular (G0 §2: no se usa phone_backview)",
        "model": f"{CLIP_MODEL}/{CLIP_PRETRAINED}",
        "thresholds": asdict(checker.thresholds),
        "reject_set": rej,
        "val": {**val, "source": SOURCE, "split": "val"},
        "val_degraded": degraded,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    (args.out / "quality_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")

    details = {
        "reject_set_accepted": _rows(rej_paths, rej_reports, passed=True),
        "val_rejected": _rows(val_paths, val_reports, passed=False),
    }
    (args.out / "quality_details.json").write_text(json.dumps(details, indent=2) + "\n")
    rejected_val = [p for p, r in zip(val_paths, val_reports, strict=True) if not r.passed]
    contact_sheet(rejected_val, args.out / "val_rejected.jpg")

    print(json.dumps({k: metrics[k] for k in ("reject_negatives", "accept_rear", "reject_val")}))
    print(f"set de rechazo: {rej}\nval: {val}\nval degradada: {degraded}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
