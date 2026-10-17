"""`make labels` (SPEC §3.2, G0 punto 3): une las fuentes y arma el split train / val.

Escribe en $BCS_DATA_DIR:

    labels.csv           unión de los labels_<id>.csv de escala 1–5 (solo train + val)
    sources.lock.json    unión de los sources.lock.<id>.json
    holdout_groups.txt   group_id de test (bcs.data.splits)
    splits.csv           image_id,split (train / val)

Antes de escribir verifica que ningún grupo quede en dos splits y que ningún grupo del holdout
esté en `labels.csv`; si algo falla, no escribe nada. Los `holdout_labels_<id>.csv` no se leen.

`python -m bcs.data.merge report` calcula los números, gráficos y hojas de contactos del
reporte de datos (`eval/data_report.md`).

Uso: python -m bcs.data.merge [report]
"""

from __future__ import annotations

import argparse
import json
import random
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from sklearn.metrics import roc_auc_score

from bcs.data import splits as sp
from bcs.data.schema import LABEL_COLUMNS, load_labels
from bcs.data.sources.sdb_bcs import family
from bcs.paths import REPO_ROOT, data_dir, runs_dir
from bcs.scale import load_scale

SEED = 42
FIGURES = REPO_ROOT / "eval" / "figures"


def is_scale_1_5(name: str) -> bool:
    try:
        s = load_scale(name)
    except (KeyError, ValueError):
        return False
    return (s.min, s.max) == (1, 5)


def label_files(root: Path) -> list[Path]:
    """labels_<id>.csv de cada fuente (nunca los holdout_labels_<id>.csv)."""
    return sorted(root.glob("labels_*.csv"))


def merge_labels(root: Path) -> pd.DataFrame:
    frames = [load_labels(p) for p in label_files(root)]
    if not frames:
        raise FileNotFoundError(f"no hay labels_<id>.csv en {root}")
    df = pd.concat(frames, ignore_index=True)
    keep = [s for s in df["scale"].unique() if is_scale_1_5(str(s))]
    df = pd.DataFrame(df[df["scale"].isin(keep)])
    return df.sort_values(by="image_id", kind="stable").reset_index(drop=True)


def merge_locks(root: Path) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for p in sorted(root.glob("sources.lock.*.json")):
        for k, v in json.loads(p.read_text()).items():
            if k in out and out[k] != v:
                raise ValueError(f"{p}: la fuente {k} ya está con otra entrada")
            out[k] = v
    return out


def _write_text(path: Path, content: str) -> None:
    if not path.exists() or path.read_text() != content:  # no reescribir si no cambió
        path.write_text(content)


def run(root: Path | None = None, readme_split: Path = sp.README_SPLIT) -> dict[str, Any]:
    root = root or data_dir()
    labels = merge_labels(root)
    readme = sp.read_readme_split(readme_split)
    errors: list[str] = []
    for p in sorted(root.glob("splits_*.csv")):
        errors += [f"{p.name}: {e}" for e in sp.check_source_splits(readme, pd.read_csv(p))]
    groups = sp.holdout_groups(readme)
    if leak := sp.group_leakage(readme):
        errors.append(f"grupos en más de un split: {leak[:5]} ({len(leak)})")
    if bad := sp.holdout_violations(labels, groups):
        errors.append(f"grupos del holdout en labels.csv: {bad[:5]} ({len(bad)})")
    if errors:
        raise RuntimeError("make labels: " + "; ".join(errors))
    splits = sp.build_splits(labels, readme)

    _write_text(root / "labels.csv", labels.loc[:, list(LABEL_COLUMNS)].to_csv(index=False))
    _write_text(root / "splits.csv", splits.to_csv(index=False))
    _write_text(root / "holdout_groups.txt", "".join(f"{g}\n" for g in groups))
    locks = merge_locks(root)
    _write_text(root / "sources.lock.json", json.dumps(locks, indent=2, ensure_ascii=False) + "\n")
    return {
        "labels": len(labels),
        "by_split": {k: int(v) for k, v in splits["split"].value_counts().sort_index().items()},
        "holdout_groups": len(groups),
        "sources": sorted(locks),
    }


# --- reporte de datos ----------------------------------------------------------------------


def paint_fractions(path: Path) -> tuple[float, float]:
    """Fracción de píxeles de pintura rojo-naranja y amarilla saturada en el recorte.

    HSV de PIL (0–255). Rojo-naranja: tono < 18 (≈ 0–25°) o > 241 (≈ 340–360°), saturación
    > 128, valor > 77. Amarillo: tono 18–35 (≈ 25–50°), saturación > 150, valor > 100 (también
    agarra pelaje marrón claro y bosta). Las franjas tapadas en negro quedan afuera por el valor.
    """
    with Image.open(path) as im:
        hsv = np.asarray(im.convert("HSV"), dtype=np.uint8)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    red = ((h < 18) | (h > 241)) & (s > 128) & (v > 77)
    yellow = (h >= 18) & (h < 36) & (s > 150) & (v > 100)
    return float(red.mean()), float(yellow.mean())


def _paint_columns(root: Path, labels: pd.DataFrame, cache: Path) -> pd.DataFrame:
    if cache.exists():
        got = pd.read_csv(cache).set_index("image_id")
        if got.index.equals(pd.Index(labels["image_id"])) and "yellow_frac" in got:
            return got
    with ProcessPoolExecutor() as ex:
        vals = list(ex.map(paint_fractions, [root / p for p in labels["path"]], chunksize=256))
    out = pd.DataFrame(
        vals,
        index=pd.Index(labels["image_id"], name="image_id"),
        columns=["red_frac", "yellow_frac"],
    )
    cache.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(cache)
    return out


def contact_sheet(
    root: Path, rows: pd.DataFrame, out: Path, n: int = 30, tile: int = 160, seed: int = SEED
) -> Path:
    pick = rows.sample(min(n, len(rows)), random_state=seed)
    cols = 6
    sheet = Image.new("RGB", (cols * tile, -(-len(pick) // cols) * (tile + 14)), "white")
    draw = ImageDraw.Draw(sheet)
    for k, r in enumerate(pick.to_dict(orient="records")):
        with Image.open(root / str(r["path"])) as im:
            im = im.convert("RGB")
            im.thumbnail((tile, tile))
            x, y = (k % cols) * tile, (k // cols) * (tile + 14)
            sheet.paste(im, (x, y))
        draw.text((x + 2, y + tile + 1), f"{r['bcs']} {str(r['image_id'])[8:]}", fill="black")
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out, quality=90)
    return out


def _spearman(df: pd.DataFrame, a: str, b: str) -> float:
    ra = pd.Series(df[a].to_numpy()).rank().to_numpy()
    rb = pd.Series(df[b].to_numpy()).rank().to_numpy()
    return float(np.corrcoef(ra, rb)[0, 1])


def _md(df: pd.DataFrame) -> str:
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join([str(df.index.name or ""), *cols]) + " |"]
    lines.append("|" + "---|" * (len(cols) + 1))
    for idx, row in df.iterrows():
        cells = [f"{v:.4g}" if isinstance(v, float) else str(v) for v in row]
        lines.append("| " + " | ".join([str(idx), *cells]) + " |")
    return "\n".join(lines)


def _figures(df: pd.DataFrame) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIGURES.mkdir(parents=True, exist_ok=True)
    out = []

    ct = pd.crosstab(df["bcs"], df["split"])
    ax = ct.plot.bar(figsize=(7, 4), rot=0)
    ax.set(xlabel="BCS", ylabel="imágenes", title="Imágenes por clase y split (train + val)")
    out.append(FIGURES / "data_class_split.png")
    plt.tight_layout()
    plt.savefig(out[-1], dpi=110)
    plt.close()

    ct = pd.crosstab(df["bcs"], df["farm"])
    ax = ct.plot.bar(figsize=(7, 4), rot=0, stacked=True)
    ax.set(xlabel="BCS", ylabel="imágenes", title="Imágenes por clase y origen del nombre")
    out.append(FIGURES / "data_class_farm.png")
    plt.tight_layout()
    plt.savefig(out[-1], dpi=110)
    plt.close()

    per_group = df.groupby("group_id").size()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(per_group, bins=60, log=True)
    ax.set(xlabel="imágenes por grupo", ylabel="grupos (log)", title="Tamaño de los grupos")
    out.append(FIGURES / "data_group_sizes.png")
    plt.tight_layout()
    plt.savefig(out[-1], dpi=110)
    plt.close()

    fig, ax = plt.subplots(figsize=(7, 4))
    all_bcs, all_red = df["bcs"].to_numpy(dtype=float), df["red_frac"].to_numpy(dtype=float)
    farms = df["farm"].to_numpy(dtype=str)
    classes = np.unique(all_bcs)
    for farm in np.unique(farms):
        bcs, red = all_bcs[farms == farm], all_red[farms == farm]
        m = [100 * red[bcs == c].mean() if (bcs == c).any() else np.nan for c in classes]
        ax.plot(classes, m, marker="o", label=str(farm))
    ax.set(xlabel="BCS", ylabel="% píxeles rojo-naranja (media)", title="Marcas de pintura")
    ax.legend()
    out.append(FIGURES / "data_red_paint.png")
    plt.tight_layout()
    plt.savefig(out[-1], dpi=110)
    plt.close()
    return out


def report(root: Path | None = None) -> Path:
    """Números, gráficos y hojas de contactos de train + val; los deja en runs/T06/."""
    root = root or data_dir()
    run_dir = runs_dir() / "T06"
    labels = load_labels(root / "labels.csv")
    splits = pd.read_csv(root / "splits.csv")
    df = labels.merge(splits, on="image_id", validate="one_to_one")
    df["farm"] = df["image_id"].str.removeprefix("sdb_bcs_").map(family)
    paint = _paint_columns(root, labels, run_dir / "paint_frac.csv")
    df["red_frac"] = paint["red_frac"].to_numpy()
    df["yellow_frac"] = paint["yellow_frac"].to_numpy()
    df["painted"] = df["red_frac"] > 0.005
    df["yellow"] = df["yellow_frac"] > 0.005
    rng = random.Random(SEED)

    sections: list[str] = []
    ct = pd.crosstab(df["bcs"], df["split"], margins=True, margins_name="total")
    sections.append("## Imágenes por clase y split\n\n" + _md(ct))
    g = df.groupby("bcs").agg(
        images=("image_id", "size"),
        animals=("animal_id", "nunique"),
        groups=("group_id", "nunique"),
    )
    gs = df.groupby(["bcs", "split"])["group_id"].nunique().unstack()
    g = g.join(gs.add_prefix("groups_"))
    sections.append("## Animales y grupos por clase\n\n" + _md(g))
    multi = df.groupby("group_id")["bcs"].nunique()
    sections.append(
        f"Grupos: {df['group_id'].nunique()}, animales: {df['animal_id'].nunique()}; grupos con "
        f"más de una clase: {int((multi > 1).sum())}. Imágenes por grupo: mediana "
        f"{df.groupby('group_id').size().median():.0f}, máx {df.groupby('group_id').size().max()}."
    )
    sections.append(
        "## Origen del nombre × clase\n\n"
        + _md(pd.crosstab(df["farm"], df["bcs"], margins=True, margins_name="total"))
    )
    red = df.groupby(["farm", "bcs"]).agg(
        n=("image_id", "size"),
        red_mean_pct=("red_frac", lambda s: 100 * s.mean()),
        red_p90_pct=("red_frac", lambda s: 100 * s.quantile(0.9)),
        painted_pct=("painted", lambda s: 100 * s.mean()),
        yellow_mean_pct=("yellow_frac", lambda s: 100 * s.mean()),
        yellow_pct=("yellow", lambda s: 100 * s.mean()),
    )
    sections.append(
        "## Marcas de pintura (% de píxeles; 'painted' / 'yellow' = > 0,5% del recorte)\n\n"
        + _md(red.reset_index().set_index("farm"))
    )
    corr = {str(f): _spearman(g2, "red_frac", "bcs") for f, g2 in df.groupby("farm")}
    grp = df.groupby(["farm", "group_id"]).agg(bcs=("bcs", "first"), red=("red_frac", "mean"))
    corr_g = {
        str(f): _spearman(pd.DataFrame(g2), "red", "bcs") for f, g2 in grp.groupby(level="farm")
    }
    sections.append(
        "Spearman(red_frac, bcs) por imagen: "
        + ", ".join(f"{k} {v:+.3f}" for k, v in corr.items())
        + "; por grupo (media del grupo): "
        + ", ".join(f"{k} {v:+.3f}" for k, v in corr_g.items())
        + f"; global por imagen {_spearman(df, 'red_frac', 'bcs'):+.3f}."
    )
    auc = {
        str(f): float(roc_auc_score(g2["bcs"] <= 3.5, g2["red_frac"]))
        for f, g2 in df.groupby("farm")
    }
    sections.append(
        "AUC de red_frac sola para separar BCS ≤ 3.5 de ≥ 3.75: "
        + ", ".join(f"{k} {v:.3f}" for k, v in auc.items())
        + "."
    )

    sheets = []
    for b, rows in df.groupby("bcs"):
        sheets.append(contact_sheet(root, rows, run_dir / "contact_sheets" / f"bcs_{b}.jpg"))
        for farm, fr in rows.groupby("farm"):
            name = f"bcs_{b}_{farm}.jpg"
            sheets.append(contact_sheet(root, fr, run_dir / "contact_sheets" / name, n=18))
    top = df.sort_values(by="red_frac", ascending=False).head(36)
    sheets.append(contact_sheet(root, top, run_dir / "contact_sheets" / "red_top.jpg", n=36))
    painted = pd.DataFrame(df[df["painted"]])
    pick = painted.sample(min(36, len(painted)), random_state=rng.randrange(1 << 30))
    sheets.append(contact_sheet(root, pick, run_dir / "contact_sheets" / "painted_sample.jpg"))

    figs = _figures(df)
    sections.append("## Archivos\n\n" + "\n".join(f"- {p}" for p in [*figs, *sheets]))
    out = run_dir / "data_numbers.md"
    out.write_text("# T06 · números del reporte de datos\n\n" + "\n\n".join(sections) + "\n")
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="make labels: unión de fuentes y split train / val")
    ap.add_argument("cmd", nargs="?", choices=["labels", "report"], default="labels")
    args = ap.parse_args(argv)
    if args.cmd == "report":
        print(f"números: {report()}")
        return
    summary = run()
    print(f"make labels: {json.dumps(summary, ensure_ascii=False)}")


if __name__ == "__main__":
    main()
