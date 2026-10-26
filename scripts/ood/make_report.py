"""Tablas del reporte de T27 desde `$BCS_RUNS_DIR/T27/results.json` → `tables.md`.

`REPORT.md` (a mano) incluye estas tablas más el análisis y la recomendación.

    uv run python scripts/ood/make_report.py
"""

from __future__ import annotations

import json
from typing import Any

from bcs.paths import runs_dir

MAIN = [
    "knn_k1_all",
    "knn_k10_all",
    "knn_k50_all",
    "maha",
    "maha_class",
    "maha_rel",
    "norm",
    "combo",
]


def f(x: float, pct: bool = False) -> str:
    if x != x:  # nan
        return "–"
    return f"{100 * x:.1f}%" if pct else f"{x:.3f}"


def table(header: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


def ood_tables(res: dict[str, Any]) -> str:
    sets = list(res["ood"])
    parts = []
    for metric, pct in (("auroc", False), ("fpr95", True)):
        rows = [
            [s] + [f(res["ood"][s][m][metric], pct) for m in MAIN if m in res["ood"][s]]
            for s in sets
        ]
        title = (
            "AUROC (más alto = mejor)" if metric == "auroc" else "FPR@95% TPR (más bajo = mejor)"
        )
        parts.append(f"### {title}\n\n" + table(["conjunto", *MAIN], rows))
    return "\n\n".join(parts)


def knn_grid(res: dict[str, Any], sets: list[str]) -> str:
    ks = sorted({int(m.split("_")[1][1:]) for m in res["cost"] if m.startswith("knn_k")})
    rows = []
    for tag in ("2k", "5k", "all"):
        for k in ks:
            m = f"knn_k{k}_{tag}"
            if m not in res["cost"]:
                continue
            cells = [m, f"{res['cost'][m]['mb']:.1f}", f"{res['cost'][m]['ms_per_image']:.2f}"]
            for s in sets:
                cells.append(
                    f"{f(res['ood'][s][m]['auroc'])} / {f(res['ood'][s][m]['fpr95'], True)}"
                )
            rows.append(cells)
    for m in ("maha", "maha_class", "maha_rel", "maha_l2", "norm", "combo"):
        if m not in res["ood"][sets[0]]:
            continue
        c = res["cost"].get(m, {"mb": float("nan"), "ms_per_image": float("nan")})
        cells = [m, f"{c['mb']:.1f}", f"{c['ms_per_image']:.2f}"]
        cells += [
            f"{f(res['ood'][s][m]['auroc'])} / {f(res['ood'][s][m]['fpr95'], True)}" for s in sets
        ]
        rows.append(cells)
    return table(["método", "MB", "ms/img", *[f"{s} AUROC/FPR95" for s in sets]], rows)


def pipeline_table(res: dict[str, Any]) -> str:
    rows = [
        [
            name,
            str(p["n"]),
            f(p["quality_ok"], True),
            f(p["detected"], True),
            f(p["reach"], True),
        ]
        for name, p in res["pipeline"].items()
    ]
    return table(["conjunto", "n", "pasa calidad", "detector ve vaca", "llega al modelo"], rows)


def threshold_tables(res: dict[str, Any], methods: list[str]) -> str:
    parts = []
    for key, by_m in res["thresholds"].items():
        cols = list(next(iter(by_m.values()))["rejected"])
        rows = [
            [m, f"{by_m[m]['threshold']:.4g}"] + [f(by_m[m]["rejected"][c], True) for c in cols]
            for m in methods
        ]
        parts.append(
            f"#### Umbral {key} de val (calibración)\n\n"
            "Columnas `… reach`: % rechazado por el OOD entre las que llegan al modelo. "
            "`… pasa todo`: % del conjunto que pasa filtro + detector + OOD.\n\n"
            + table(["método", "umbral", *cols], rows)
        )
    return "\n\n".join(parts)


def selective_tables(res: dict[str, Any]) -> str:
    parts = []
    for pname, d in res["selective"].items():
        fr = [p["rejected"] for p in d["oracle"]["curve"]]
        rows = []
        for m in [*MAIN, "oracle"]:
            if m not in d:
                continue
            rows.append(
                [m]
                + [f"{p['mae']:.3f}" for p in d[m]["curve"]]
                + [f"{d[m]['aurc']:.4f}", f(d[m].get("spearman", float("nan")))]
            )
        parts.append(
            f"#### {pname} (n={d['n']}, MAE sin rechazo {d['mae_all']:.3f})\n\n"
            + table(["método", *[f"rech. {100 * x:.0f}%" for x in fr], "AURC", "Spearman"], rows)
        )
    return "\n\n".join(parts)


def main() -> int:
    res = json.loads((runs_dir() / "T27" / "results.json").read_text())
    key_sets = [s for s in res["ood"] if s.endswith("(reach)") or s.startswith("val_zen")]
    md = [
        "# T27 · tablas generadas (`scripts/ood/make_report.py`)",
        f"Tamaños: `{json.dumps(res['n'])}`",
        "## Qué llega hoy al modelo (filtro de calidad + detector conf 0,15)",
        pipeline_table(res),
        "## Comparación de métodos",
        ood_tables(res),
        "## kNN: k × tamaño del banco (AUROC / FPR@95), con costo",
        knn_grid(res, key_sets[:6]),
        "## Umbrales propuestos",
        threshold_tables(res, ["knn_k10_all", "knn_k50_all", "maha", "maha_class", "combo"]),
        "## Predicción selectiva (MAE de lo aceptado vs. fracción rechazada)",
        selective_tables(res),
        "## Costo",
        table(
            ["método", "MB", "ms/img", "fit s"],
            [
                [m, f"{c['mb']:.2f}", f"{c['ms_per_image']:.3f}", str(c.get("fit_s", "–"))]
                for m, c in res["cost"].items()
            ],
        ),
    ]
    (runs_dir() / "T27" / "tables.md").write_text("\n\n".join(md) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
