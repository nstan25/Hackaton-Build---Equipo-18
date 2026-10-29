"""Experimento OOD de T27: kNN vs. Mahalanobis (y combinación) sobre los embeddings cacheados.

Lee `$BCS_ARTIFACTS_DIR/T27/emb/` (lo arma `build_sets.py`) y escribe en `$BCS_RUNS_DIR/T27/`:
`results.json`, `REPORT.md` y gráficos. Solo CPU; no recalcula embeddings.

Protocolo:
- Banco de referencia (fit): `bank` (sdb_bcs train, ≤ 5 frames por grupo). Nada de val entra al fit.
- Los grupos de sdb_bcs val se parten (semilla fija) en *calibración* (z-scores de la
  combinación, umbrales) y *evaluación* (todas las métricas ID).
- Conjuntos de recortes (`val_zen`) se comparan contra `val_sdb` (evaluación); los que pasan por el
  pipeline (degradados, reject, COCO, zen crudo) contra `pipe_val` (evaluación), en dos versiones:
  todas las fotos ("all") y solo las que pasan filtro de calidad + detector ("reach", lo que hoy
  llega al modelo).

    uv run python scripts/ood/run_experiment.py
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from scipy.stats import spearmanr

from bcs.ood.methods import (
    FeatureNormScore,
    KNNScore,
    MahalanobisScore,
    ZScoreCombiner,
    aurc,
    auroc,
    fpr_at_tpr,
    selective_mae,
    subsample_bank,
)
from bcs.paths import REPO_ROOT, artifacts_dir, runs_dir

PIPE_SETS = [
    "deg_dark",
    "deg_overexposed",
    "deg_blur",
    "deg_noise",
    "deg_jpeg",
    "deg_rot90",
    "deg_upside_down",
    "deg_partial",
    "deg_too_close",
    "deg_offcenter",
    "reject",
    "coco",
    "coco_cow",
    "coco_animal",
    "coco_other",
    "zen_raw",
]
COMBO = ("knn_k10_all", "maha_class")  # elegidos a priori, antes de ver resultados OOD
SHOW = ["knn_k1_all", "knn_k10_all", "knn_k50_all", "maha", "maha_class", "maha_rel", "norm"]


def emb_dir() -> Path:
    return artifacts_dir() / "T27" / "emb"


@dataclass
class Set:
    meta: pd.DataFrame
    feats: np.ndarray

    @property
    def reach(self) -> np.ndarray:
        """Pasa filtro de calidad y detector (si el conjunto pasó por el pipeline)."""
        if "quality_ok" not in self.meta:
            return np.ones(len(self.meta), dtype=bool)
        return (self.meta["quality_ok"] & self.meta["detected"]).to_numpy(dtype=bool)

    def take(self, mask: np.ndarray) -> Set:
        return Set(self.meta[mask].reset_index(drop=True), self.feats[mask])


def load(name: str) -> Set:
    z = np.load(emb_dir() / f"{name}.npz")
    meta = pd.read_csv(emb_dir() / f"{name}.csv")
    assert len(meta) == len(z["feats"])
    return Set(meta, z["feats"].astype(np.float32))


def build_methods(bank: Set, cfg: dict) -> dict[str, Any]:
    seed = int(cfg["seed"])
    y = bank.meta["bcs"].to_numpy()
    groups = bank.meta["group_id"].astype(str).tolist()
    methods: dict[str, Any] = {}
    fit_s: dict[str, float] = {}
    for size in cfg["bank"]["sizes"]:
        idx = subsample_bank(bank.feats, size, seed, groups)
        tag = "all" if size is None else f"{size // 1000}k"
        for k in cfg["knn_k"]:
            t = time.perf_counter()
            methods[f"knn_k{k}_{tag}"] = KNNScore(k=k).fit(bank.feats[idx])
            fit_s[f"knn_k{k}_{tag}"] = time.perf_counter() - t
    for name, m in {
        "maha": MahalanobisScore(),
        "maha_class": MahalanobisScore(per_class=True),
        "maha_rel": MahalanobisScore(per_class=True, relative=True),
        "maha_l2": MahalanobisScore(normalize=True),
        "norm": FeatureNormScore(),
    }.items():
        t = time.perf_counter()
        methods[name] = m.fit(bank.feats, y)
        fit_s[name] = time.perf_counter() - t
    methods["_fit_s"] = fit_s
    return methods


def score_all(methods: dict[str, Any], feats: np.ndarray, combo: ZScoreCombiner) -> pd.DataFrame:
    out = {n: m.score(feats) for n, m in methods.items() if not n.startswith("_")}
    out["combo"] = combo.combine(np.column_stack([out[c] for c in COMBO]))
    return pd.DataFrame(out)


def latency(methods: dict[str, Any], feats: np.ndarray, n: int = 200) -> dict[str, float]:
    """ms por imagen, de a una (como la API), sobre `n` features de val."""
    out = {}
    for name, m in methods.items():
        if name.startswith("_"):
            continue
        m.score(feats[:1])
        t = time.perf_counter()
        for i in range(n):
            m.score(feats[i : i + 1])
        out[name] = 1000 * (time.perf_counter() - t) / n
    return out


def split_groups(groups: pd.Series, frac: float, seed: int) -> np.ndarray:
    """True = grupo de calibración."""
    uniq = np.array(sorted(groups.astype(str).unique()))
    rng = np.random.default_rng(seed)
    calib = set(uniq[rng.permutation(len(uniq))[: int(round(frac * len(uniq)))]])
    return groups.astype(str).isin(calib).to_numpy()


def prod_errors(meta: pd.DataFrame) -> np.ndarray:
    """|error| del modelo de producción (con TTA, de T25) por `image_id`."""
    p = pd.read_csv(artifacts_dir() / "T25" / "preds_val_prod.csv")
    err = (p["pred"] - p["y"]).abs()
    m = dict(zip(p["image_id"], err, strict=True))
    return np.array([m[i] for i in meta["image_id"]], dtype=float)


def _mean(x: np.ndarray) -> float:
    return float(x.mean()) if len(x) else float("nan")


def ood_metrics(id_s: pd.DataFrame, ood_s: pd.DataFrame, tpr: float) -> dict[str, dict]:
    return {
        c: {
            "auroc": auroc(id_s[c].to_numpy(), ood_s[c].to_numpy()),
            "fpr95": fpr_at_tpr(id_s[c].to_numpy(), ood_s[c].to_numpy(), tpr),
        }
        for c in id_s.columns
    }


def run() -> dict[str, Any]:
    cfg = yaml.safe_load((REPO_ROOT / "config" / "ood.yaml").read_text())
    seed, tpr = int(cfg["seed"]), float(cfg["tpr"])
    out_dir = runs_dir() / "T27"
    out_dir.mkdir(parents=True, exist_ok=True)

    bank = load("bank")
    methods = build_methods(bank, cfg)

    val_sdb, pipe_val, val_zen = load("val_sdb"), load("pipe_val"), load("val_zen")
    is_cal = split_groups(val_sdb.meta["group_id"], float(cfg["calib_frac"]), seed)
    cal_groups = set(val_sdb.meta["group_id"][is_cal])
    pipe_cal = pipe_val.meta["group_id"].isin(cal_groups).to_numpy()

    # combinación: z-scores con media y desvío en val (calibración), recortes del pipeline
    cal_raw = {n: m.score(pipe_val.feats[pipe_cal]) for n, m in methods.items() if n in COMBO}
    combo = ZScoreCombiner().fit(np.column_stack([cal_raw[c] for c in COMBO]))

    s_sdb = score_all(methods, val_sdb.feats, combo)
    s_pipe = score_all(methods, pipe_val.feats, combo)
    s_zen = score_all(methods, val_zen.feats, combo)

    id_sdb_eval = s_sdb[~is_cal].reset_index(drop=True)
    pipe_eval = ~pipe_cal
    pipe_reach = pipe_val.reach
    id_pipe_all = s_pipe[pipe_eval].reset_index(drop=True)
    id_pipe_reach = s_pipe[pipe_eval & pipe_reach].reset_index(drop=True)
    cal_reach = s_pipe[pipe_cal & pipe_reach].reset_index(drop=True)

    sets: dict[str, tuple[Set, pd.DataFrame]] = {}
    for name in PIPE_SETS:
        base = name if not name.startswith("coco_") else "coco"
        if not (emb_dir() / f"{base}.npz").exists():
            continue
        s = load(base)
        if name.startswith("coco_"):
            s = s.take((s.meta["kind"] == name.removeprefix("coco_")).to_numpy())
        sets[name] = (s, score_all(methods, s.feats, combo))

    res: dict[str, Any] = {
        "config": cfg,
        "combo": list(COMBO),
        "n": {
            "bank": len(bank.meta),
            "val_sdb_eval": int((~is_cal).sum()),
            "val_sdb_calib": int(is_cal.sum()),
            "pipe_val_eval": int(pipe_eval.sum()),
            "pipe_val_eval_reach": int((pipe_eval & pipe_reach).sum()),
            "val_zen": len(val_zen.meta),
        },
        "ood": {},
        "pipeline": {},
    }
    # ---- métricas OOD
    res["ood"]["val_zen (recortes)"] = ood_metrics(id_sdb_eval, s_zen, tpr)
    res["ood"]["pipe_val vs val_sdb"] = ood_metrics(id_sdb_eval, s_pipe[pipe_eval], tpr)
    for name, (s, sc) in sets.items():
        res["ood"][f"{name} (all)"] = ood_metrics(id_pipe_all, sc, tpr)
        r = s.reach
        res["pipeline"][name] = {
            "n": len(s.meta),
            "quality_ok": float(s.meta["quality_ok"].mean()),
            "detected": float(s.meta["detected"].mean()),
            "reach": float(r.mean()),
            "n_reach": int(r.sum()),
        }
        if r.sum() >= 5:
            res["ood"][f"{name} (reach)"] = ood_metrics(id_pipe_reach, sc[r], tpr)
    res["pipeline"]["pipe_val"] = {
        "n": len(pipe_val.meta),
        "quality_ok": float(pipe_val.meta["quality_ok"].mean()),
        "detected": float(pipe_val.meta["detected"].mean()),
        "reach": float(pipe_reach.mean()),
        "n_reach": int(pipe_reach.sum()),
    }

    pipe_err = np.abs(pipe_val.meta["value"] - pipe_val.meta["bcs"]).to_numpy()

    # ---- umbrales: percentiles de val sdb (calibración, fotos que pasan el pipeline) y, como
    # umbral "tolerante al dominio", percentiles de val_zen (vacas reales de otra granja/cámara)
    def rejection(c: str, t: float) -> dict[str, float]:
        rej = {
            "val_sdb (eval)": float((id_sdb_eval[c] > t).mean()),
            "pipe_val reach (eval)": float((id_pipe_reach[c] > t).mean()),
            "val_zen": float((s_zen[c] > t).mean()),
        }
        for name, (s, sc) in sets.items():
            r = s.reach
            rej[f"{name} reach"] = float((sc[c][r] > t).mean()) if r.any() else float("nan")
            # lo que deja pasar el sistema entero (filtro + detector + OOD) sobre todas
            rej[f"{name} pasa todo"] = float((r & (sc[c] <= t).to_numpy()).mean())
        return rej

    thr: dict[str, Any] = {}
    for ref, ref_scores in (("val", cal_reach), ("zen", s_zen)):
        for q in cfg["thresholds"]:
            key = f"{ref}_p{round(100 * q)}"
            thr[key] = {}
            for c in [*SHOW, "maha_l2", "combo"]:
                t = float(np.quantile(ref_scores[c], q))
                thr[key][c] = {"threshold": t, "rejected": rejection(c, t)}
    res["thresholds"] = thr

    # ---- error del modelo en los degradados que llegan al modelo, aceptados vs. rechazados
    deg: dict[str, Any] = {}
    for name, (s, sc) in sets.items():
        if not name.startswith("deg_"):
            continue
        r = s.reach
        err = np.abs(s.meta["value"] - s.meta["bcs"]).to_numpy()
        row: dict[str, Any] = {"n_reach": int(r.sum()), "mae_reach": _mean(err[r])}
        for key in ("val_p99", "zen_p99"):
            t = thr[key]["maha"]["threshold"]
            acc = r & (sc["maha"] <= t).to_numpy()
            row[f"maha_{key}"] = {
                "mae_accepted": _mean(err[acc]),
                "mae_rejected": _mean(err[r & ~acc]),
            }
        deg[name] = row
    clean = pipe_eval & pipe_reach
    deg["pipe_val (limpias)"] = {"n_reach": int(clean.sum()), "mae_reach": _mean(pipe_err[clean])}
    res["degraded_mae"] = deg

    # ---- predicción selectiva
    fr = [float(f) for f in cfg["selective_fractions"]]
    err_sdb, err_zen = prod_errors(val_sdb.meta), prod_errors(val_zen.meta)
    deg_parts = [(s, sc) for n, (s, sc) in sets.items() if n.startswith("deg_")]
    pool_sc = pd.concat(
        [s_pipe[pipe_eval & pipe_reach]] + [sc[s.reach] for s, sc in deg_parts], ignore_index=True
    )
    pool_err = np.concatenate(
        [pipe_err[pipe_eval & pipe_reach]]
        + [np.abs(s.meta["value"] - s.meta["bcs"]).to_numpy()[s.reach] for s, _ in deg_parts]
    )
    pools = {
        "val_sdb (eval)": (id_sdb_eval, err_sdb[~is_cal]),
        "val_zen": (s_zen, err_zen),
        "val_sdb (eval) + val_zen": (
            pd.concat([id_sdb_eval, s_zen], ignore_index=True),
            np.concatenate([err_sdb[~is_cal], err_zen]),
        ),
        "pipe_val + degradados que llegan al modelo": (pool_sc, pool_err),
    }
    if "zen_raw" in sets:
        zs, zsc = sets["zen_raw"]
        zerr = np.abs(zs.meta["value"] - zs.meta["bcs"]).to_numpy()
        pools["zen_raw que llega al modelo (sin TTA)"] = (
            zsc[zs.reach].reset_index(drop=True),
            zerr[zs.reach],
        )
    sel: dict[str, Any] = {}
    for pname, (sc, err) in pools.items():
        sel[pname] = {"n": len(err), "mae_all": float(err.mean())}
        for c in [*SHOW, "combo"]:
            sel[pname][c] = {
                "curve": selective_mae(sc[c].to_numpy(), err, fr),
                "aurc": aurc(sc[c].to_numpy(), err),
                "spearman": float(spearmanr(sc[c], err).statistic),  # type: ignore[union-attr]
            }
        sel[pname]["oracle"] = {
            "curve": selective_mae(err, err, fr),
            "aurc": aurc(err, err),
        }
    res["selective"] = sel

    # ---- costo
    lat = latency(methods, val_sdb.feats)
    res["cost"] = {
        n: {
            "bytes": int(m.nbytes),
            "mb": round(m.nbytes / 2**20, 2),
            "ms_per_image": round(lat[n], 3),
            "fit_s": round(methods["_fit_s"][n], 2),
        }
        for n, m in methods.items()
        if not n.startswith("_")
    }
    # versión float16 del banco kNN completo
    k16 = KNNScore(k=10, dtype="float16").fit(bank.feats)
    ref = methods["knn_k10_all"].score(val_sdb.feats[:2000])
    res["cost"]["knn_k10_all_fp16"] = {
        "bytes": k16.nbytes,
        "mb": round(k16.nbytes / 2**20, 2),
        "ms_per_image": round(latency({"x": k16}, val_sdb.feats)["x"], 3),
        "max_abs_diff_vs_fp32": float(np.abs(k16.score(val_sdb.feats[:2000]) - ref).max()),
    }

    plots(out_dir, s_sdb[~is_cal], s_zen, id_pipe_reach, sets, sel)
    (out_dir / "results.json").write_text(json.dumps(res, indent=2, ensure_ascii=False))
    save_scores(out_dir, pipe_val, s_pipe, sets)
    return res


def save_scores(out_dir: Path, pipe_val: Set, s_pipe: pd.DataFrame, sets: dict) -> None:
    """Puntajes por imagen de los métodos principales (para mirar casos a mano)."""
    rows = [pipe_val.meta.assign(set="pipe_val").join(s_pipe[[*SHOW, "combo"]])]
    for name, (s, sc) in sets.items():
        if name.startswith("coco_"):
            continue
        rows.append(s.meta.assign(set=name).join(sc[[*SHOW, "combo"]]))
    pd.concat(rows, ignore_index=True).drop(columns=["path"], errors="ignore").to_csv(
        out_dir / "scores_pipeline.csv", index=False
    )


def plots(
    out_dir: Path,
    id_sdb: pd.DataFrame,
    s_zen: pd.DataFrame,
    id_reach: pd.DataFrame,
    sets: dict,
    sel: dict,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for c in ["knn_k10_all", "maha_class", "combo"]:
        fig, ax = plt.subplots(figsize=(9, 5))
        lo, hi = np.quantile(np.concatenate([id_sdb[c], id_reach[c]]), [0.0, 1.0])
        allv = np.concatenate([sc[c] for _, sc in sets.values()] + [s_zen[c].to_numpy()])
        hi = max(hi, np.quantile(allv, 0.98))
        bins = np.linspace(lo, hi, 60)
        kw = {"bins": bins, "density": True, "histtype": "step", "linewidth": 1.5}
        ax.hist(id_sdb[c], label="ID val_sdb", color="black", **kw)
        ax.hist(id_reach[c], label="ID pipe_val", color="gray", linestyle="--", **kw)
        ax.hist(s_zen[c], label="val_zen", **kw)
        for name in ["coco", "reject", "zen_raw", "deg_blur", "deg_partial", "deg_too_close"]:
            if name in sets:
                ax.hist(sets[name][1][c], label=name, **kw)
        ax.set_title(f"T27 · puntaje OOD {c} (más alto = más raro)")
        ax.set_xlabel("puntaje")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out_dir / f"hist_{c}.png", dpi=110)
        plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, pname in zip(
        axes,
        ["val_sdb (eval)", "val_zen", "pipe_val + degradados que llegan al modelo"],
        strict=True,
    ):
        for c in ["knn_k10_all", "maha", "maha_class", "maha_rel", "norm", "combo", "oracle"]:
            curve = sel[pname][c]["curve"]
            ax.plot(
                [p["rejected"] for p in curve],
                [p["mae"] for p in curve],
                marker="o",
                markersize=3,
                label=c,
                linestyle=":" if c == "oracle" else "-",
            )
        ax.set_title(pname, fontsize=9)
        ax.set_xlabel("fracción rechazada")
        ax.set_ylabel("MAE de lo aceptado")
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_dir / "selective.png", dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    r = run()
    print(json.dumps(r["n"], indent=1))
