"""T28: banco OOD del modelo de T25, umbrales de los dos niveles y tablas (G6 opción 1).

    uv run python scripts/ood/build_sets.py --task T28 --weights models/bcs_efficientnet_b0_zen.pt
    uv run python -m bcs.ood.fit_zen          # → models/ood_mahalanobis_zen.npz + runs/T28/ood/

Embeddings: los de `build_sets.py` con el modelo nuevo (`$BCS_ARTIFACTS_DIR/T28/emb/`, pooling de
la vista sin flip con letterbox, igual que en `inference`). Nunca test ni holdout.

- **Banco**: `sdb_bcs` train (≤ 5 frames por grupo) + `zen_rgbd` train (todo), una gaussiana con
  Ledoit-Wolf (`MahalanobisGate.fit`).
- **ID de val**: recortes de `sdb_bcs` val y `zen_rgbd` val, y los frames de `sdb_bcs` val por el
  pipeline (`pipe_val`). Los grupos se parten 50/50 (semilla de `config/ood.yaml`) en
  *calibración* (umbrales) y *evaluación* (tasas ID y AUROC).
- **Umbral 1 (aviso)**: el p99 de la mitad de calibración, **por fuente**, y se toma el mayor (así
  ninguna de las dos fuentes de entrenamiento pierde más del ~1% de sus fotos buenas).
- **Umbral 2 (rechazo)**: zen ya es ID, así que no hay "vacas reales de otro dominio" para fijarlo
  (T27). Se toma el **máximo de todas las vacas reales de val** (recortes de las dos fuentes,
  calibración y evaluación, y `pipe_val`) por `integration.reject_margin`: ninguna vaca de val
  queda rechazada por construcción, y el margen deja lugar a fotos de celular algo más raras. Se
  reporta qué parte de COCO y de los dibujos de T08 rechaza, y los candidatos alternativos (p99,9 y
  el máximo sin margen).

T30 (`--section integration_grupa`): lo mismo para el modelo de grupa, con embeddings en
`$BCS_ARTIFACTS_DIR/T30/emb/` (`build_sets.py --task T30` + `scripts/zen_crops/embed_grupa.py`); el
banco y el val ID de zen son los recortes de grupa (lo que ve el modelo en uso real), según la
sección (`bank_sets`, `val_sets`).

Salidas: el artefacto (`integration.artifact`), `$BCS_RUNS_DIR/T28/ood/{results.json,tables.md}` y
`$BCS_ARTIFACTS_DIR/efficientnet_b0_zen/quality_metrics.json` (filtro de m4 punta a punta: filtro
básico + detector + OOD, sobre el set de rechazo de T08 y los frames de val).
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from bcs.models.efficientnet import sha256_file
from bcs.ood.gate import LEVEL_REJECT, LEVEL_WARN, MahalanobisGate
from bcs.ood.methods import auroc, fpr_at_tpr
from bcs.paths import REPO_ROOT, artifacts_dir, config_dir, runs_dir

DEGRADED = (
    "dark",
    "overexposed",
    "blur",
    "noise",
    "jpeg",
    "rot90",
    "upside_down",
    "partial",
    "too_close",
    "offcenter",
)
PIPE_SETS = ("pipe_val", "zen_raw", "reject", "coco", *(f"deg_{d}" for d in DEGRADED))
BANK_SETS = {"sdb_bcs_train": "bank", "zen_rgbd_train": "bank_zen"}  # nombre en el banco → conjunto
VAL_SETS = {"sdb_bcs": "val_sdb", "zen_rgbd": "val_zen"}  # fuente → conjunto de val ID
COCO_KINDS = {"coco_cow": "cow", "coco_animal": "animal", "coco_other": "other"}


def load_cfg() -> dict[str, Any]:
    return yaml.safe_load((config_dir() / "ood.yaml").read_text())


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
        return Set(pd.DataFrame(self.meta.loc[mask]).reset_index(drop=True), self.feats[mask])


def load_set(emb: Path, name: str) -> Set:
    z = np.load(emb / f"{name}.npz")
    meta = pd.read_csv(emb / f"{name}.csv", dtype={"image_id": str, "group_id": str})
    if len(meta) != len(z["feats"]):
        raise ValueError(f"{name}: {len(meta)} filas y {len(z['feats'])} embeddings")
    return Set(meta, z["feats"].astype(np.float32))


def split_groups(groups: Any, frac: float, seed: int) -> np.ndarray:
    """True = grupo de calibración (igual que T27)."""
    uniq = np.array(sorted(groups.astype(str).unique()))
    rng = np.random.default_rng(seed)
    calib = set(uniq[rng.permutation(len(uniq))[: int(round(frac * len(uniq)))]])
    return groups.astype(str).isin(calib).to_numpy()


def warn_threshold(calib_scores: dict[str, np.ndarray], q: float) -> float:
    """El mayor de los cuantiles `q` por fuente."""
    return max(float(np.quantile(s, q, method="higher")) for s in calib_scores.values())


def reject_threshold(id_scores: np.ndarray, margin: float) -> float:
    return float(np.max(id_scores)) * margin


def rates(levels: np.ndarray, passed: np.ndarray) -> dict[str, float | int]:
    """Sobre todo el conjunto: lo que el pipeline rechaza (filtro, detector u OOD), lo que llega
    con aviso y lo que recibe BCS sin aviso; y, entre lo que llega al modelo, el % de aviso y de
    rechazo OOD."""
    n, reach = len(levels), int(passed.sum())
    warn = passed & (levels == LEVEL_WARN)
    rej = passed & (levels == LEVEL_REJECT)
    return {
        "n": n,
        "reach": reach,
        "ood_warn_reach": float(warn.sum() / reach) if reach else float("nan"),
        "ood_reject_reach": float(rej.sum() / reach) if reach else float("nan"),
        "rejected_total": float(((~passed) | rej).mean()) if n else float("nan"),
        "warned_total": float(warn.mean()) if n else float("nan"),
        "clean_total": float((passed & (levels == 0)).mean()) if n else float("nan"),
    }


def _q(s: np.ndarray) -> dict[str, float]:
    return {
        "n": len(s),
        "p50": float(np.quantile(s, 0.5)),
        "p95": float(np.quantile(s, 0.95)),
        "p99": float(np.quantile(s, 0.99)),
        "p999": float(np.quantile(s, 0.999)),
        "max": float(np.max(s)),
    }


def run(
    cfg: dict[str, Any], write_artifact: bool = True, section: str = "integration"
) -> dict[str, Any]:
    icfg = cfg[section]
    task = str(icfg["task"])
    seed, frac = int(cfg["seed"]), float(cfg["calib_frac"])
    emb = artifacts_dir() / icfg["task"] / "emb"
    weights = REPO_ROOT / icfg["weights"]
    sha = sha256_file(weights)

    # T30: los conjuntos del banco y de val ID salen de la sección (default: los de T28)
    banks = {k: load_set(emb, n) for k, n in icfg.get("bank_sets", BANK_SETS).items()}
    gate = MahalanobisGate.fit(
        np.concatenate([b.feats for b in banks.values()]),
        sha,
        {
            "task": task,
            "weights": icfg["weights"],
            "bank": {k: len(b.feats) for k, b in banks.items()},
            "bank_rule": f"sdb_bcs train ≤ {cfg['bank']['per_group']} frames por grupo + "
            + icfg.get("bank_rule_zen", "zen_rgbd train (todo)"),
            "embedding": "pooling 1280-d de la vista sin flip (letterbox 224×320)",
        },
    )

    # ID de val: recortes de las dos fuentes y frames de sdb por el pipeline
    val = {s: load_set(emb, n) for s, n in icfg.get("val_sets", VAL_SETS).items()}
    pipe_val = load_set(emb, "pipe_val")
    calib = {s: split_groups(v.meta["group_id"], frac, seed) for s, v in val.items()}
    calib_pipe = split_groups(pipe_val.meta["group_id"], frac, seed)
    score = {s: gate.score(v.feats) for s, v in val.items()}
    score_pipe = gate.score(pipe_val.feats)
    pv_reach = pipe_val.reach

    warn = warn_threshold(
        {s: score[s][calib[s]] for s in val} | {"pipe_val": score_pipe[calib_pipe & pv_reach]},
        float(icfg["warn_quantile"]),
    )
    id_all = np.concatenate([*score.values(), score_pipe[pv_reach]])
    margin = float(icfg["reject_margin"])
    reject = reject_threshold(id_all, margin)
    gate = gate.with_thresholds(warn, reject)
    gate.info |= {
        "warn_rule": f"max por fuente del p{100 * float(icfg['warn_quantile']):g} de la mitad de "
        "calibración de val (sdb, zen, pipe_val)",
        "reject_rule": f"máximo de todas las vacas reales de val × {margin:g}",
        "thresholds": {"warn": warn, "reject": reject},
    }

    # conjuntos (embeddings del modelo nuevo)
    sets: dict[str, Set] = {n: load_set(emb, n) for n in PIPE_SETS}
    coco = sets["coco"]
    for name, kind in COCO_KINDS.items():
        sets[name] = coco.take((coco.meta["kind"] == kind).to_numpy())
    id_eval = np.concatenate([score[s][~calib[s]] for s in val])

    t27 = _t27_results()
    results: dict[str, Any] = {
        "task": task,
        "model_sha256": sha,
        "weights": icfg["weights"],
        "bank": gate.info["bank"],
        "thresholds": {"warn": warn, "reject": reject, "reject_margin": margin},
        "id_scores": {
            **{f"val_{s} (calib)": _q(score[s][calib[s]]) for s in val},
            **{f"val_{s} (eval)": _q(score[s][~calib[s]]) for s in val},
            "pipe_val (reach)": _q(score_pipe[pv_reach]),
            "todas las vacas de val": _q(id_all),
        },
        "reject_candidates": {
            "p99.9 vacas de val": float(np.quantile(id_all, 0.999)),
            "máximo vacas de val": float(np.max(id_all)),
            f"máximo × {margin:g} (elegido)": reject,
        },
        "id_rates": {},
        "ood": {},
        "levels": {},
    }
    for s in val:
        lv = gate.levels(score[s][~calib[s]])
        results["id_rates"][f"val_{s} (eval)"] = rates(lv, np.ones(len(lv), dtype=bool))
    lv = gate.levels(score_pipe[~calib_pipe])
    results["id_rates"]["pipe_val (eval)"] = rates(lv, pv_reach[~calib_pipe])

    for name, st in sets.items():
        sc = gate.score(st.feats)
        lv = gate.levels(sc)
        reach = st.reach
        results["levels"][name] = rates(lv, reach)
        for view, mask in (("all", np.ones(len(sc), dtype=bool)), ("reach", reach)):
            if name in ("pipe_val",) or mask.sum() == 0:
                continue
            key = f"{name} ({view})"
            results["ood"][key] = {
                "n": int(mask.sum()),
                "auroc": auroc(id_eval, sc[mask]),
                "fpr95": fpr_at_tpr(id_eval, sc[mask], float(cfg["tpr"])),
                "t27_maha": t27.get(key),
            }
    results["ood"]["val_zen (recortes)"] = {
        "n": int((~calib["zen_rgbd"]).sum()),
        "auroc": auroc(score["sdb_bcs"][~calib["sdb_bcs"]], score["zen_rgbd"][~calib["zen_rgbd"]]),
        "fpr95": fpr_at_tpr(
            score["sdb_bcs"][~calib["sdb_bcs"]],
            score["zen_rgbd"][~calib["zen_rgbd"]],
            float(cfg["tpr"]),
        ),
        "t27_maha": t27.get("val_zen (recortes)"),
        "note": "zen contra sdb (ahora los dos son ID: se espera ~0,5)",
    }
    results["quality_metrics"] = quality_metrics(results)

    out = runs_dir() / task / "ood"
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(results, indent=2, default=float) + "\n")
    (out / "tables.md").write_text(tables_md(results))
    qdir = artifacts_dir() / str(icfg.get("quality_dir", "efficientnet_b0_zen"))
    qdir.mkdir(parents=True, exist_ok=True)
    (qdir / "quality_metrics.json").write_text(
        json.dumps(results["quality_metrics"], indent=2) + "\n"
    )
    if write_artifact:
        dest = REPO_ROOT / icfg["artifact"]
        gate.save(dest)
        print(f"banco OOD → {dest} ({dest.stat().st_size / 2**20:.1f} MB)", file=sys.stderr)
    print(f"umbrales: aviso {warn:.1f} · rechazo {reject:.1f} → {out}", file=sys.stderr)
    return results


def quality_metrics(results: dict[str, Any]) -> dict[str, Any]:
    """Filtro de m4 del pipeline nuevo (filtro básico + detector a 0,15 + OOD), punta a punta."""
    lv = results["levels"]
    rej, pv, zr = lv["reject"], lv["pipe_val"], lv["zen_raw"]
    n_val = pv["n"] + zr["n"]
    reject_val = (pv["rejected_total"] * pv["n"] + zr["rejected_total"] * zr["n"]) / n_val
    return {
        "reject_negatives": rej["rejected_total"],
        "accept_rear": None,
        "reject_val": reject_val,
        "accept_rear_note": "sin fotos traseras de celular (G0 §2), como en G3",
        "note": "T28: filtro básico + detector (conf 0,15) + OOD Mahalanobis (rechazo = nivel 2); "
        "reject_negatives sobre los 40 de T08, reject_val sobre frames crudos de val "
        "(sdb pipe_val + zen_raw)",
        "by_source": {
            "sdb_bcs": {"reject_val": pv["rejected_total"], "n": pv["n"]},
            "zen_rgbd": {"reject_val": zr["rejected_total"], "n": zr["n"]},
        },
        "reject_negatives_or_warned": rej["rejected_total"] + rej["warned_total"],
    }


def _t27_results() -> dict[str, Any]:
    path = runs_dir() / "T27" / "results.json"
    if not path.exists():
        return {}
    ood = json.loads(path.read_text()).get("ood", {})
    return {k: v.get("maha") for k, v in ood.items() if isinstance(v, dict)}


def _pct(v: Any) -> str:
    return "–" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{100 * v:.1f}%"


def _f(v: Any, nd: int = 3) -> str:
    return "–" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.{nd}f}"


def tables_md(r: dict[str, Any]) -> str:
    t = r["thresholds"]
    lines = [
        f"# {r.get('task', 'T28')} · detección OOD con `{r.get('weights', '')}` "
        "(tablas generadas por `bcs.ood.fit_zen`)",
        "",
        f"Banco: {r['bank']}. Umbrales: **aviso {t['warn']:.1f}**, **rechazo {t['reject']:.1f}** "
        f"(máximo de las vacas de val × {t['reject_margin']:g}).",
        "",
        "## Puntajes de las vacas de val (ID)",
        "",
        "| conjunto | n | p50 | p95 | p99 | p99,9 | máx |",
        "|---|---|---|---|---|---|---|",
    ]
    for k, q in r["id_scores"].items():
        lines.append(
            f"| {k} | {q['n']} | {q['p50']:.0f} | {q['p95']:.0f} | {q['p99']:.0f} "
            f"| {q['p999']:.0f} | {q['max']:.0f} |"
        )
    lines += ["", "Candidatos para el umbral de rechazo:", ""]
    lines += [f"- {k}: {v:.0f}" for k, v in r["reject_candidates"].items()]
    lines += [
        "",
        "## AUROC / FPR@95 (ID = mitad de evaluación de val, sdb + zen)",
        "",
        "| conjunto | n | AUROC | FPR@95 | T27 (modelo viejo, ID = sdb) AUROC / FPR@95 |",
        "|---|---|---|---|---|",
    ]
    for k, m in r["ood"].items():
        old = m.get("t27_maha") or {}
        oldtxt = f"{_f(old.get('auroc'))} / {_pct(old.get('fpr95'))}" if old else "–"
        lines.append(f"| {k} | {m['n']} | {_f(m['auroc'])} | {_pct(m['fpr95'])} | {oldtxt} |")
    lines += [
        "",
        "## Qué pasa con cada conjunto (pipeline completo: filtro + detector + OOD)",
        "",
        "`… reach` = entre lo que llega al modelo. `total` = sobre todo el conjunto.",
        "",
        "| conjunto | n | llega | aviso (reach) | rechazo OOD (reach) | rechazada (total) "
        "| con aviso (total) | BCS sin aviso (total) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for k, v in {**r["id_rates"], **r["levels"]}.items():
        lines.append(
            f"| {k} | {v['n']} | {v['reach']} | {_pct(v['ood_warn_reach'])} "
            f"| {_pct(v['ood_reject_reach'])} | {_pct(v['rejected_total'])} "
            f"| {_pct(v['warned_total'])} | {_pct(v['clean_total'])} |"
        )
    q = r["quality_metrics"]
    lines += [
        "",
        "## Filtro de m4 (`quality_metrics.json`)",
        "",
        f"reject_negatives {_pct(q['reject_negatives'])} (rechazadas o con aviso "
        f"{_pct(q['reject_negatives_or_warned'])}) · reject_val {_pct(q['reject_val'])} "
        f"(sdb {_pct(q['by_source']['sdb_bcs']['reject_val'])}, zen crudo "
        f"{_pct(q['by_source']['zen_rgbd']['reject_val'])}) · accept_rear sin medir.",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m bcs.ood.fit_zen", description=__doc__)
    ap.add_argument("--dry-run", action="store_true", help="no escribe el artefacto en models/")
    ap.add_argument(
        "--section",
        default="integration",
        help="sección de config/ood.yaml (T30: integration_grupa)",
    )
    args = ap.parse_args(argv)
    run(load_cfg(), write_artifact=not args.dry_run, section=args.section)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
