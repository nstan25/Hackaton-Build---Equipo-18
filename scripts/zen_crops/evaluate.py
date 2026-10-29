"""T30: predicciones y métricas en **val** (nunca test) para producción, `dev` (T25) y candidatos.

Vistas (lo que ve el modelo):

- `uso` (**la métrica que importa**): zen val, la vaca etiquetada sola (`crops.load_solo`) →
  `models/cow_det.pt` → recorte (+10%, lado mayor 320) → modelo. Sin caja del detector la app
  rechaza: cuenta como rechazo y no entra en la MAE.
- `uso_up<k>`: igual, con la caja ampliada hacia arriba `k` veces su alto (recorte en inferencia,
  opción B).
- `t04`: los recortes de vaca entera vertical de T04 (`$BCS_DATA_DIR/images/zen_rgbd/`), los de
  entrenamiento de T25.
- `crudo`: el frame entero de la sala → detector → recorte (lo de T28 §5: suele ser otra vaca).
- `sdb`: recortes de sdb val completos (8.028; no regresión como T25, IC por grupo).
- `sdb_uso[_up<k>]`: foto cruda de sdb val (≤ 2 por grupo, `pipe_val` de T28) → detector →
  recorte, para ver que el cambio de recorte de B no rompa sdb.

    uv run python scripts/zen_crops/evaluate.py predict --model dev --view uso
    uv run python scripts/zen_crops/evaluate.py report --models prod,dev,A
    uv run python scripts/zen_crops/evaluate.py source --models dev,A
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import crops as zc  # noqa: E402

from bcs.paths import REPO_ROOT, data_dir, runs_dir  # noqa: E402

GRID = tuple(3.0 + 0.25 * i for i in range(9))  # dairy_zen
SDB_CLASSES = (3.25, 3.5, 3.75, 4.0, 4.25)
MODELS: dict[str, Path] = {
    "prod": REPO_ROOT / "models" / "bcs_efficientnet_b0.pt",
    "dev": REPO_ROOT / "models" / "bcs_efficientnet_b0_zen.pt",
}
_UP = re.compile(r"_up([0-9.]+)$")


def model_path(name: str) -> Path:
    if name in MODELS:
        return MODELS[name]
    p = zc.out_dir() / name / "best.pt"
    if not p.exists():
        raise FileNotFoundError(p)
    return p


def preds_path(model: str, view: str) -> Path:
    return zc.out_dir() / "preds" / f"{view}__{model}.csv"


def feats_path(model: str, view: str) -> Path:
    return zc.out_dir() / "preds" / f"{view}__{model}.npy"


def view_rows(view: str) -> pd.DataFrame:
    """Filas (val) de la vista con `image_id, source, group_id, date, y, kind` y cómo armar la
    imagen (`loader`)."""
    up = float(m.group(1)) if (m := _UP.search(view)) else 0.0
    base = _UP.sub("", view)
    if base in ("uso", "crudo", "sdb_uso"):
        name = {"uso": "zen_solo", "crudo": "zen_raw", "sdb_uso": "sdb_raw"}[base]
        df = zc.det_table(name)
        df = df.loc[df["split"] == "val"].copy()
        df["kind"] = np.where(df["det"] != "", "det", "rejected")
        df["loader"] = [(name, up)] * len(df)
    elif base == "t04":
        df = zc.zen_rows()
        df = df.loc[df["split"] == "val"].copy()
        df["kind"] = "crop"
        df["loader"] = [("t04", 0.0)] * len(df)
    elif base == "sdb":
        root = data_dir()
        df = pd.read_csv(root / "labels_sdb_bcs.csv").merge(
            pd.read_csv(root / "splits_sdb_bcs.csv"), on="image_id"
        )
        df = df.loc[df["split"] == "val"].copy()
        df["kind"] = "crop"
        df["loader"] = [("sdb", 0.0)] * len(df)
    else:
        raise ValueError(view)
    df["y"] = [min(GRID, key=lambda g, v=v: abs(g - v)) for v in df["bcs"].astype(float)]
    df["date"] = df.get("date", pd.Series([""] * len(df), index=df.index)).fillna("")
    return pd.DataFrame(df).reset_index(drop=True)


def load_view_image(row: Any) -> Image.Image | None:
    kind, up = row["loader"]
    if kind in ("t04", "sdb"):
        with Image.open(data_dir() / row["path"]) as im:
            return im.convert("RGB")
    if not row["det"]:
        return None
    src = zc.load_src(kind, row)
    return zc.crop_from_solo(src, row["det"], up=up)


def run_predict(model: str, view: str, force: bool = False, batch: int = 64) -> Path:
    from bcs.models.efficientnet import EfficientNetScorer

    dest = preds_path(model, view)
    if dest.exists() and not force:
        print(f"{view}/{model}: ya está ({dest})")
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    df = view_rows(view)
    sc = EfficientNetScorer(GRID, {"model": {"weights": str(model_path(model)), "tta": True}})
    t0 = time.time()
    keep = (df["kind"] != "rejected").to_numpy()
    vals = np.full(len(df), np.nan)
    feats: list[np.ndarray] = []
    idx = np.flatnonzero(keep)
    for i in range(0, len(idx), batch):
        ims = [load_view_image(df.iloc[j]) for j in idx[i : i + batch]]
        v, f = sc.predict_value_features([im for im in ims if im is not None])
        vals[idx[i : i + batch]] = v
        feats.append(f)
    out = df[["image_id", "source", "group_id", "date", "bcs", "y", "kind"]].copy()
    out["pred"] = vals
    out.to_csv(dest, index=False)
    if feats:
        np.save(feats_path(model, view), np.concatenate(feats))
    print(f"{view}/{model}: {keep.sum()} predicciones en {time.time() - t0:.0f}s → {dest}")
    return dest


# ---------------------------------------------------------------- métricas


def metrics(df: pd.DataFrame) -> dict[str, Any]:
    """MAE, exacta y ±1 paso (clase redondeada a 3.0–5.0), sesgo, sobre lo que no se rechazó."""
    n_all = len(df)
    ok = df.loc[df["kind"] != "rejected"]
    if len(ok) == 0:
        return {"n": 0, "n_all": n_all, "animals": int(df["group_id"].nunique())}
    y, p = ok["y"].to_numpy(dtype=float), ok["pred"].to_numpy(dtype=float)
    g = np.asarray(GRID)
    cls = g[np.abs(p[:, None] - g[None, :]).argmin(1)]
    return {
        "n": len(ok),
        "n_all": n_all,
        "animals": int(ok["group_id"].nunique()),
        "rejected": float(1 - len(ok) / n_all),
        "mae": float(np.abs(p - y).mean()),
        "exact": float(np.isclose(cls, y).mean()),
        "pm1": float((np.abs(cls - y) <= 0.25 + 1e-6).mean()),
        "bias": float((p - y).mean()),
        "mean_pred": float(p.mean()),
    }


def paired_delta(
    base: pd.DataFrame, cand: pd.DataFrame, reps: int = 1000, seed: int = 0
) -> dict[str, Any]:
    """MAE(cand) − MAE(base) en las mismas filas no rechazadas por ninguno, IC 95% bootstrap por
    `group_id` (como `train_zen.paired_delta`)."""
    m = base.merge(cand, on="image_id", suffixes=("_0", "_1"))
    m = m.loc[(m["kind_0"] != "rejected") & (m["kind_1"] != "rejected")]
    if len(m) == 0:
        return {}
    e0 = np.abs(m["pred_0"].to_numpy(dtype=float) - m["y_0"].to_numpy(dtype=float))
    e1 = np.abs(m["pred_1"].to_numpy(dtype=float) - m["y_0"].to_numpy(dtype=float))
    codes, groups = pd.factorize(m["group_id_0"])
    k = len(groups)
    sums = np.bincount(codes, weights=e1 - e0, minlength=k)
    counts = np.bincount(codes, minlength=k).astype(float)
    draws = np.random.default_rng(seed).integers(0, k, size=(reps, k))
    boot = sums[draws].sum(1) / counts[draws].sum(1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {"delta": float((e1 - e0).mean()), "lo": float(lo), "hi": float(hi), "animals": k}


def load_preds(model: str, view: str) -> pd.DataFrame:
    df = pd.read_csv(preds_path(model, view), dtype={"date": str})
    df["date"] = df["date"].fillna("")
    return df


def report(models: list[str], views: list[str], base: str = "dev") -> dict[str, Any]:
    rep: dict[str, Any] = {"models": models, "views": views, "by_view": {}, "by_class": {}}
    rep["delta"] = {}
    for v in views:
        rep["by_view"][v], rep["by_class"][v], rep["delta"][v] = {}, {}, {}
        for m in models:
            if not preds_path(m, v).exists():
                continue
            df = load_preds(m, v)
            rep["by_view"][v][m] = metrics(df)
            rep["by_class"][v][m] = {f"{c:.2f}": metrics(g) for c, g in df.groupby("y")}
            if m != base and preds_path(base, v).exists():
                b = load_preds(base, v)
                cuts = {"all": None} | {f"{c:.2f}": c for c in sorted(set(df["y"]))}
                rep["delta"][v][m] = {
                    k: paired_delta(b if c is None else b[b["y"] == c], df) for k, c in cuts.items()
                }
    return rep


def _fmt(m: dict[str, Any]) -> str:
    if not m.get("n"):
        return "– | – | – | – | –"
    return (
        f"{m['mae']:.3f} | {100 * m['exact']:.0f}% | {100 * m['pm1']:.0f}% | {m['bias']:+.3f} | "
        f"{100 * m.get('rejected', 0):.1f}%"
    )


def tables_md(rep: dict[str, Any], base: str = "dev") -> str:
    out = []
    head = "| vista | modelo | n (anim.) | MAE | exacta | ±1 paso | sesgo | rechazo |"
    out += ["### Por vista (val)", "", head, "|---|---|---|---|---|---|---|---|"]
    for v in rep["views"]:
        for m, mm in rep["by_view"][v].items():
            out.append(f"| {v} | {m} | {mm.get('n', 0)} ({mm.get('animals', 0)}) | {_fmt(mm)} |")
    out.append("")
    for v in rep["views"]:
        ms = list(rep["by_class"][v])
        if not ms:
            continue
        classes = sorted({c for m in ms for c in rep["by_class"][v][m]})
        out += [f"### {v}, por clase", "", head.replace("vista", "clase"), out[3]]
        for c in classes:
            for m in ms:
                mm = rep["by_class"][v][m].get(c, {})
                out.append(
                    f"| {c} | {m} | {mm.get('n', 0)} ({mm.get('animals', 0)}) | {_fmt(mm)} |"
                )
        out.append("")
    out += [
        f"### Δ MAE vs. {base} (mismas imágenes; IC 95% bootstrap por animal)",
        "",
        "| vista | modelo | corte | Δ MAE | IC 95% | animales |",
        "|---|---|---|---|---|---|",
    ]
    for v, per in rep["delta"].items():
        for m, cuts in per.items():
            for k, d in cuts.items():
                if d:
                    out.append(
                        f"| {v} | {m} | {k} | {d['delta']:+.3f} | "
                        f"[{d['lo']:+.3f}, {d['hi']:+.3f}] | {d['animals']} |"
                    )
    out.append("")
    return "\n".join(out)


# ---------------------------------------------------------------- indicador de fuente


def source_auc(model: str, views: tuple[str, str] = ("sdb_uso", "uso"), seed: int = 0) -> dict:
    """¿El embedding reconoce la fuente? Regresión logística sdb vs. zen sobre el pooling (vista
    sin flip) de las dos vistas de uso real, CV agrupada por animal (5 folds) → AUC. 0,5 = no
    distingue. También dentro de la misma banda de BCS (3.25–4.25), para que no sea la etiqueta."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    parts, feats = [], []
    for v in views:
        df = load_preds(model, v)
        df = df.loc[df["kind"] != "rejected"].reset_index(drop=True)
        f = np.load(feats_path(model, v))
        assert len(f) == len(df), (v, len(f), len(df))
        parts.append(df.assign(is_zen=int(v == views[1])))
        feats.append(f)
    df, x = pd.concat(parts, ignore_index=True), np.concatenate(feats)
    res = {}
    for name, mask in {
        "all": np.ones(len(df), bool),
        "3.25-4.25": df["y"].between(3.25, 4.25).to_numpy(),
    }.items():
        d, xx = df.loc[mask].reset_index(drop=True), x[mask]
        y, g = d["is_zen"].to_numpy(), d["group_id"].to_numpy()
        p = np.zeros(len(d))
        for tr, te in GroupKFold(5).split(xx, y, g):
            clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=2000))
            clf.fit(xx[tr], y[tr])
            p[te] = clf.predict_proba(xx[te])[:, 1]
        res[name] = {"auc": float(roc_auc_score(y, p)), "n_sdb": int((y == 0).sum()),
                     "n_zen": int((y == 1).sum())}  # fmt: skip
    return res


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="scripts/zen_crops/evaluate.py")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("predict")
    p.add_argument("--model", required=True)
    p.add_argument("--view", required=True)
    p.add_argument("--force", action="store_true")
    r = sub.add_parser("report")
    r.add_argument("--models", default="prod,dev")
    r.add_argument("--views", default="uso,t04,crudo,sdb,sdb_uso")
    r.add_argument("--base", default="dev")
    r.add_argument("--name", default="tables")
    s = sub.add_parser("source")
    s.add_argument("--models", default="dev")
    a = ap.parse_args(argv)
    if a.cmd == "predict":
        run_predict(a.model, a.view, a.force)
    elif a.cmd == "report":
        rep = report(a.models.split(","), a.views.split(","), a.base)
        d = runs_dir() / zc.TASK
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{a.name}.json").write_text(json.dumps(rep, indent=2))
        (d / f"{a.name}.md").write_text(tables_md(rep, a.base))
        print(tables_md(rep, a.base))
    elif a.cmd == "source":
        res = {m: source_auc(m) for m in a.models.split(",")}
        d = runs_dir() / zc.TASK
        (d / "source_auc.json").write_text(json.dumps(res, indent=2))
        print(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
