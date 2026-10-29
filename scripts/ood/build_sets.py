"""Embeddings de todos los conjuntos del experimento OOD de T27 (CPU, cacheados en disco).

Cada conjunto queda en `$BCS_ARTIFACTS_DIR/T27/emb/<set>.npz` (`feats` [n, 1280] float32 y
`value`, el BCS continuo sin TTA) + `<set>.csv` (una fila por imagen, mismo orden). Los grandes se
guardan por tandas en `emb/parts/` y se retoman si la VM se cuelga. Un conjunto ya armado no se
recalcula (borrá su `.npz` para rehacerlo).

Conjuntos (nunca test ni holdout; solo `train` y `val` de `splits*.csv`):

- `bank`: recortes de `sdb_bcs` train (texto tapado), ≤ `bank.per_group` frames por grupo.
- `bank_zen` (T28): todos los recortes de `zen_rgbd` train (fuente de entrenamiento del modelo
  nuevo; ~3 por animal).
- `val_sdb`: todos los recortes de `sdb_bcs` val (ID de prueba, mismos recortes que T09/T25).
- `val_zen`: recortes de `zen_rgbd` val (cambio de dominio, MAE 0,48 con el modelo de producción).
- `pipe_val`: frames completos de `sdb_bcs` val → filtro de calidad → detector → recorte, como en
  producción (ID de referencia para los conjuntos que pasan por el pipeline).
- `deg_<tipo>`: los mismos frames de val degradados en memoria (semilla fija) → pipeline.
- `reject`: `tests/fixtures/reject/` (40 dibujos/texturas de T08) → pipeline.
- `coco`: muestra de COCO (ver `fetch_coco.py`) → pipeline.
- `zen_raw`: frames crudos de `zen_rgbd` val (sala de ordeñe entera, varias vacas) → pipeline.

En los conjuntos por pipeline, si el detector no encuentra vaca se embebe la foto entera (achicada
como un recorte) para poder puntuarla igual; `detected` y `quality_ok` dicen qué hubiera pasado.

    uv run python scripts/ood/build_sets.py [--only bank,val_sdb]
    # T28: el modelo de T25 (letterbox) → $BCS_ARTIFACTS_DIR/T28/emb/
    uv run python scripts/ood/build_sets.py --task T28 --weights models/bcs_efficientnet_b0_zen.pt
"""

from __future__ import annotations

import argparse
import io
import sys
import time
import xml.etree.ElementTree as ET
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

from bcs.ood.embed import FeatureExtractor
from bcs.paths import REPO_ROOT, artifacts_dir, data_dir, fixtures_dir

CHUNK = 512
MAX_SIDE = 320
TASK = "T27"  # carpeta de salida en $BCS_ARTIFACTS_DIR (--task); COCO siempre sale de T27


def emb_dir() -> Path:
    return artifacts_dir() / TASK / "emb"


def load_cfg() -> dict:
    return yaml.safe_load((REPO_ROOT / "config" / "ood.yaml").read_text())


# ---------------------------------------------------------------- listas de imágenes


def sdb_rows(split: str) -> pd.DataFrame:
    d = data_dir()
    labels = pd.read_csv(d / "labels.csv")
    splits = pd.read_csv(d / "splits.csv")
    df = labels.merge(splits, on="image_id")
    df = df[(df["source"] == "sdb_bcs") & (df["split"] == split)].copy()
    df["path"] = [str(d / p) for p in df["path"]]
    return df.sort_values("image_id").reset_index(drop=True)


def zen_rows(split: str) -> pd.DataFrame:
    d = data_dir()
    df = pd.read_csv(d / "labels_zen_rgbd.csv").merge(
        pd.read_csv(d / "splits_zen_rgbd.csv"), on="image_id"
    )
    df = df[df["split"] == split].copy()
    df["path"] = [str(d / p) for p in df["path"]]
    plan = pd.read_csv(d / "raw" / "zen_rgbd" / "plan.csv")[["image_id", "rgb_path"]]
    df = df.merge(plan, on="image_id", how="left")
    df["raw_path"] = [str(d / "raw" / "zen_rgbd" / "rgb" / p) for p in df["rgb_path"]]
    return df.sort_values("image_id").reset_index(drop=True)


def per_group(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """≤ n filas por `group_id`, al azar con semilla fija."""
    rng = np.random.default_rng(seed)
    shuffled = df.iloc[rng.permutation(len(df))]
    return shuffled.groupby("group_id", sort=True).head(n).sort_values("image_id")


def sdb_full_frames() -> dict[str, Path]:
    """`image_id` → frame completo de val (con su .xml) de `raw/sdb_bcs_pre/dataset_split/val`."""
    root = data_dir() / "raw" / "sdb_bcs_pre" / "dataset_split" / "val"
    return {f"sdb_bcs_{p.stem}": p for p in root.glob("*/*.jpg")}


def read_bbox(xml_path: Path) -> tuple[int, int, int, int]:
    box = ET.parse(xml_path).find("object/bndbox")
    assert box is not None
    x0, y0, x1, y1 = (int(float(box.findtext(k) or 0)) for k in ("xmin", "ymin", "xmax", "ymax"))
    return x0, y0, x1, y1


# ---------------------------------------------------------------- degradaciones (en memoria)

Degrade = Callable[[Image.Image, Path, np.random.Generator], Image.Image]


def _noise(img: Image.Image, _: Path, rng: np.random.Generator) -> Image.Image:
    a = np.asarray(img, dtype=np.float32)
    return Image.fromarray(np.clip(a + rng.normal(0, 60, a.shape), 0, 255).astype(np.uint8))


def _jpeg(img: Image.Image, _: Path, __: np.random.Generator) -> Image.Image:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=2)
    return Image.open(io.BytesIO(buf.getvalue())).convert("RGB")


def _partial(img: Image.Image, path: Path, rng: np.random.Generator) -> Image.Image:
    """Solo un costado de la vaca (40% del ancho de su caja), como una foto mal encuadrada."""
    x0, y0, x1, y1 = read_bbox(path.with_suffix(".xml"))
    w = x1 - x0
    left = bool(rng.integers(2))
    a, b = (x0, x0 + int(0.4 * w)) if left else (x1 - int(0.4 * w), x1)
    return img.crop((max(0, a), max(0, y0 - 20), min(img.width, b), min(img.height, y1 + 20)))


def _too_close(img: Image.Image, path: Path, rng: np.random.Generator) -> Image.Image:
    """Muy de cerca: un 30% central de la caja de la vaca, corrido al azar."""
    x0, y0, x1, y1 = read_bbox(path.with_suffix(".xml"))
    w, h = (x1 - x0) * 0.3, (y1 - y0) * 0.3
    cx = x0 + (x1 - x0) * rng.uniform(0.35, 0.65)
    cy = y0 + (y1 - y0) * rng.uniform(0.35, 0.65)
    return img.crop((int(cx - w / 2), int(cy - h / 2), int(cx + w / 2), int(cy + h / 2)))


def _offcenter(img: Image.Image, path: Path, rng: np.random.Generator) -> Image.Image:
    """Descentrada: la vaca en una esquina, con el resto del cuadro vacío (galpón/piso)."""
    x0, y0, x1, y1 = read_bbox(path.with_suffix(".xml"))
    cw, ch = img.width, img.height
    dx = int((x1 - x0) * 0.6) * (1 if rng.integers(2) else -1)
    dy = int((y1 - y0) * 0.4) * (1 if rng.integers(2) else -1)
    return ImageOps.expand(img, border=0).transform(
        (cw, ch), Image.Transform.AFFINE, (1, 0, dx, 0, 1, dy), fillcolor=(40, 40, 40)
    )


DEGRADATIONS: dict[str, Degrade] = {
    "dark": lambda im, _, __: ImageEnhance.Brightness(im).enhance(0.08),
    "overexposed": lambda im, _, __: ImageEnhance.Brightness(im).enhance(3.5),
    "blur": lambda im, _, __: im.filter(ImageFilter.GaussianBlur(8)),
    "noise": _noise,
    "jpeg": _jpeg,
    "rot90": lambda im, _, __: im.rotate(90, expand=True),
    "upside_down": lambda im, _, __: im.transpose(Image.Transpose.FLIP_TOP_BOTTOM),
    "partial": _partial,
    "too_close": _too_close,
    "offcenter": _offcenter,
}


# ---------------------------------------------------------------- pipeline y embeddings


class Pipe:
    """Filtro de calidad básico + detector de producción (solo lectura, de `bcs.inference`)."""

    def __init__(self) -> None:
        from bcs.inference import CowLocator
        from bcs.quality.basic import BasicQualityChecker

        self.quality = BasicQualityChecker()
        self.locate = CowLocator()

    def __call__(self, img: Image.Image) -> tuple[Image.Image, bool, bool, str]:
        q = self.quality.check(img)
        crop = self.locate(img)
        detected = crop is not None
        if crop is None:
            crop = img.copy()
            crop.thumbnail((MAX_SIDE, MAX_SIDE))
        return crop, bool(q.passed), detected, "; ".join(q.issues)


def _open(path: str | Path) -> Image.Image:
    with Image.open(path) as im:
        return ImageOps.exif_transpose(im).convert("RGB")


def embed_set(
    name: str,
    meta: pd.DataFrame,
    load: Callable[[pd.Series], Image.Image],
    fe: FeatureExtractor,
    pipe: Pipe | None = None,
) -> None:
    """Embebe `meta` (en orden) por tandas; con `pipe`, cada imagen pasa antes por el pipeline."""
    out = emb_dir()
    if (out / f"{name}.npz").exists():
        print(f"[{name}] ya está", file=sys.stderr)
        return
    parts = out / "parts" / name
    parts.mkdir(parents=True, exist_ok=True)
    meta = meta.reset_index(drop=True)
    t0 = time.monotonic()
    for k in range(0, len(meta), CHUNK):
        part = parts / f"{k:07d}.npz"
        if part.exists():
            continue
        rows = meta.iloc[k : k + CHUNK]
        images, extra = [], []
        for _, r in rows.iterrows():
            img = load(r)
            if pipe is not None:
                img, q_ok, det, issues = pipe(img)
                extra.append((q_ok, det, issues))
            images.append(img)
        feats, value = fe(images)
        arrays: dict[str, np.ndarray] = {"feats": feats, "value": value}
        if pipe is not None:
            arrays["quality_ok"] = np.array([e[0] for e in extra])
            arrays["detected"] = np.array([e[1] for e in extra])
            arrays["issues"] = np.array([e[2] for e in extra], dtype=object).astype(str)
        np.savez(part, **arrays)  # type: ignore[arg-type]
        done = min(k + CHUNK, len(meta))
        rate = (done - k) / max(time.monotonic() - t0, 1e-9)
        print(f"[{name}] {done}/{len(meta)} · {rate:.1f} img/s", file=sys.stderr)
        t0 = time.monotonic()
    loaded = [np.load(p, allow_pickle=False) for p in sorted(parts.glob("*.npz"))]
    merged = {key: np.concatenate([z[key] for z in loaded]) for key in loaded[0].files}
    assert len(merged["feats"]) == len(meta), (name, len(merged["feats"]), len(meta))
    for key in ("quality_ok", "detected", "issues"):
        if key in merged:
            meta[key] = merged.pop(key)
    meta["value"] = merged["value"]
    meta.to_csv(out / f"{name}.csv", index=False)
    np.savez(out / f"{name}.npz", feats=merged["feats"], value=merged["value"])
    print(f"[{name}] listo: {len(meta)} imágenes", file=sys.stderr)


def build(only: set[str] | None, weights: str | None = None) -> None:
    cfg = load_cfg()
    seed = int(cfg["seed"])
    emb_dir().mkdir(parents=True, exist_ok=True)
    fe = FeatureExtractor(weights) if weights else FeatureExtractor()

    def want(name: str) -> bool:
        return only is None or name in only or name.split("_")[0] + "_*" in only

    keep = ["image_id", "group_id", "bcs", "path"]

    def by_path(r: pd.Series) -> Image.Image:
        return _open(r["path"])

    if want("bank"):
        bank = per_group(sdb_rows("train"), int(cfg["bank"]["per_group"]), seed)
        embed_set("bank", bank[keep], by_path, fe)
    if want("bank_zen") and TASK != "T27":
        embed_set("bank_zen", zen_rows("train")[keep], by_path, fe)
    if want("val_sdb"):
        embed_set("val_sdb", sdb_rows("val")[keep], by_path, fe)
    zen = zen_rows("val")
    if want("val_zen"):
        embed_set("val_zen", zen[keep], by_path, fe)

    pipe_needed = any(
        want(n)
        for n in ("pipe_val", "reject", "coco", "zen_raw", *(f"deg_{d}" for d in DEGRADATIONS))
    )
    if not pipe_needed:
        return
    pipe = Pipe()
    frames = sdb_full_frames()
    val = sdb_rows("val")
    val = val[val["image_id"].isin(list(frames))].copy()
    val["path"] = [str(frames[i]) for i in val["image_id"]]
    pv = per_group(val, int(cfg["val_pipe_per_group"]), seed)
    if want("pipe_val"):
        embed_set("pipe_val", pv[keep], by_path, fe, pipe)
    # degradados: un frame por grupo, `degraded_n` grupos, los mismos para todos los tipos
    one = per_group(pv, 1, seed + 1)
    deg_rows = one.iloc[
        np.sort(np.random.default_rng(seed + 2).permutation(len(one))[: int(cfg["degraded_n"])])
    ]
    for i, (dname, fn) in enumerate(DEGRADATIONS.items()):
        if not want(f"deg_{dname}"):
            continue

        def load_deg(r: pd.Series, fn: Degrade = fn, i: int = i) -> Image.Image:
            rng = np.random.default_rng([seed, i, int(r.name)])  # type: ignore[arg-type]
            return fn(_open(r["path"]), Path(r["path"]), rng).convert("RGB")

        embed_set(f"deg_{dname}", deg_rows[keep].reset_index(drop=True), load_deg, fe, pipe)
    if want("reject"):
        rej = pd.DataFrame(
            {"path": sorted(str(p) for p in (fixtures_dir() / "reject").glob("*.png"))}
        )
        rej["image_id"] = [Path(p).stem for p in rej["path"]]
        embed_set("reject", rej, by_path, fe, pipe)
    if want("coco"):
        cdir = artifacts_dir() / "T27" / "ood" / "coco"
        coco = pd.read_csv(cdir / "manifest.csv")
        coco["path"] = [str(cdir / "images" / f) for f in coco["file"]]
        coco["image_id"] = [f"coco_{i}" for i in coco["coco_id"]]
        embed_set(
            "coco", coco[["image_id", "kind", "categories", "license", "path"]], by_path, fe, pipe
        )
    if want("zen_raw"):
        raw = zen[["image_id", "group_id", "bcs", "raw_path"]].rename(columns={"raw_path": "path"})
        embed_set("zen_raw", raw, by_path, fe, pipe)


def main(argv: list[str] | None = None) -> int:
    global TASK
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", default=None, help="conjuntos separados por coma (p. ej. bank,deg_*)")
    ap.add_argument("--task", default=TASK, help="carpeta de salida en $BCS_ARTIFACTS_DIR")
    ap.add_argument("--weights", default=None, help="`.pt` a embeber (default: producción)")
    args = ap.parse_args(argv)
    TASK = args.task
    build(set(args.only.split(",")) if args.only else None, args.weights)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
