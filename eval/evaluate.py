"""`make eval MODEL=<name> STAGE=<m2|m3|m4>` (SPEC §3.2, G0: sobre val).

Lee las predicciones de val del modelo (formato en `bcs.eval.report`), calcula las métricas,
escribe `runs/<task>/<ts>/{report.json,config.json,reliability.png,confusion.png}` y compara
contra `eval/thresholds.yaml[STAGE]`. Exit 0 si pasa, 1 si no pasa, 2 si la entrada es inválida.

T28: con más de una fuente, además informa las mismas métricas y chequeos **por fuente**
(`report.json` → `by_source`; mismas métricas y umbrales, solo restringidos a las filas de esa
fuente). El veredicto (exit code) sigue siendo el de todas las fuentes juntas.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import yaml

from bcs.eval.report import (
    EvalInputError,
    build_report,
    check_walls,
    load_predictions,
    sha256_file,
    train_labels,
    write_run,
)
from bcs.paths import REPO_ROOT, artifacts_dir, runs_dir
from bcs.scale import load_scale

THRESHOLDS = Path(__file__).with_name("thresholds.yaml")


def _git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, check=True
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return out.stdout.strip()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--stage", required=True, choices=["m2", "m3", "m4"])
    parser.add_argument(
        "--predictions",
        type=Path,
        help="CSV de predicciones (default: $BCS_ARTIFACTS_DIR/<model>/predictions_val.csv)",
    )
    parser.add_argument(
        "--quality",
        type=Path,
        help="JSON del filtro de calidad para m4 (default: $BCS_ARTIFACTS_DIR/<model>/"
        "quality_metrics.json): reject_negatives, accept_rear, reject_val",
    )
    parser.add_argument("--task", help="carpeta en runs/ (default: $BCS_TASK_ID o el modelo)")
    parser.add_argument("--scale", help="preset de config/scale.yaml (default: el activo)")
    parser.add_argument("--thresholds", type=Path, default=THRESHOLDS)
    parser.add_argument("--runs-dir", type=Path, help="default: $BCS_RUNS_DIR o ~/bcs-shared/runs")
    args = parser.parse_args(argv)

    scale = load_scale(args.scale)
    thresholds = yaml.safe_load(args.thresholds.read_text())[args.stage]
    model_dir = artifacts_dir() / args.model
    pred_path: Path = args.predictions or model_dir / "predictions_val.csv"
    quality_path: Path = args.quality or model_dir / "quality_metrics.json"
    task = args.task or os.environ.get("BCS_TASK_ID") or args.model

    if not pred_path.exists():
        print(f"no existen las predicciones {pred_path}", file=sys.stderr)
        return 2
    try:
        df = load_predictions(pred_path, scale)
        check_walls(df)
        quality = json.loads(quality_path.read_text()) if quality_path.exists() else None
        report = build_report(
            df,
            scale,
            model=args.model,
            stage=args.stage,
            task=task,
            thresholds=thresholds,
            quality=quality,
            y_train=train_labels(),
        )
        by_source = {
            str(s): build_report(
                df.loc[df["source"] == s],
                scale,
                model=args.model,
                stage=args.stage,
                task=task,
                thresholds=thresholds,
                quality=quality,
                y_train=train_labels(),
            )
            for s in sorted(set(df["source"]))
        }
    except EvalInputError as e:
        print(f"predicciones inválidas: {e}", file=sys.stderr)
        return 2
    if len(by_source) > 1:
        report["by_source"] = {
            s: {"mean": r["mean"], "thresholds": r["thresholds"], "n": int(r["pooled"]["n"])}
            for s, r in by_source.items()
        }

    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_dir = (args.runs_dir or runs_dir()) / task / ts
    config = {
        "argv": sys.argv if argv is None else ["eval/evaluate.py", *argv],
        "model": args.model,
        "stage": args.stage,
        "task": task,
        "scale": scale.name,
        "predictions": str(pred_path),
        "predictions_sha256": sha256_file(pred_path),
        "quality": str(quality_path) if quality is not None else None,
        "thresholds_file": str(args.thresholds),
        "thresholds": thresholds,
        "git_commit": _git_commit(),
    }
    paths = write_run(report, run_dir, config)

    result = report["thresholds"]
    for c in result["checks"]:
        mark = "ok  " if c["passed"] else "FAIL"
        value = "n/d" if c["value"] is None else f"{c['value']:.4f}"
        print(f"[{mark}] {c['key']}: {value} {c['op']} {c['threshold']}", file=sys.stderr)
    for s, r in report.get("by_source", {}).items():
        marks = ", ".join(
            f"{c['key']} {'n/d' if c['value'] is None else format(c['value'], '.4f')}"
            f"{'' if c['passed'] else ' FAIL'}"
            for c in r["thresholds"]["checks"]
        )
        print(f"  [{s}, n={r['n']}] {marks}", file=sys.stderr)
    verdict = "PASA" if result["passed"] else "NO PASA"
    print(f"{args.stage} {verdict} · {paths['report']}", file=sys.stderr)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
