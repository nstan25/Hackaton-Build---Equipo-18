"""T02: `report.json`, umbrales y `eval/evaluate.py` de punta a punta (sin datos reales)."""

from __future__ import annotations

import importlib.util
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import yaml
from PIL import Image

from bcs.eval.metrics import snap_to_grid
from bcs.eval.report import (
    EvalInputError,
    build_report,
    check_thresholds,
    check_walls,
    dummy_baseline,
    synthetic_predictions,
)
from bcs.paths import REPO_ROOT
from bcs.scale import load_scale

SCALE = load_scale("dairy_1_5")
GRID = np.asarray(SCALE.grid)
STEP = SCALE.step
THRESHOLDS = yaml.safe_load((REPO_ROOT / "eval" / "thresholds.yaml").read_text())
Y = np.array([3.25, 3.5, 3.5, 3.75, 3.75, 3.75, 4.0, 4.0, 4.25, 4.25] * 2)
QUALITY_OK = {"reject_negatives": 0.9, "accept_rear": 0.9, "reject_val": 0.05}


def _load_cli() -> Callable[[list[str]], int]:
    spec = importlib.util.spec_from_file_location("evaluate_cli", REPO_ROOT / "eval/evaluate.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.main


def one_hot(values: np.ndarray) -> np.ndarray:
    p = np.zeros((len(values), len(GRID)))
    p[np.arange(len(values)), snap_to_grid(values, GRID)] = 1.0
    return p


def preds(shift_steps: int = 0, **cols: Any) -> pd.DataFrame:
    return synthetic_predictions(
        Y,
        one_hot(Y + shift_steps * STEP),
        SCALE,
        group_id=[f"g{i // 2}" for i in range(len(Y))],
        **cols,
    )


def uniform() -> pd.DataFrame:
    return synthetic_predictions(Y, np.full((len(Y), len(GRID)), 1 / len(GRID)), SCALE)


def nine_of_ten_covered() -> dict[str, Any]:
    lo = Y - STEP
    hi = Y + STEP
    lo[::10] = Y[::10] + STEP  # 2 de 20 afuera → cobertura 0.9
    return {"interval_lo": list(lo), "interval_hi": list(hi)}


def test_m3_perfect_passes_uniform_fails() -> None:
    ok = build_report(preds(), SCALE, model="m", stage="m3", thresholds=THRESHOLDS["m3"])
    assert ok["thresholds"]["passed"]
    assert {c["key"] for c in ok["thresholds"]["checks"]} == set(THRESHOLDS["m3"])
    bad = build_report(uniform(), SCALE, model="m", stage="m3", thresholds=THRESHOLDS["m3"])
    assert not bad["thresholds"]["passed"]


def test_m3_shift_one_step() -> None:
    """MAE 0.25 ≤ 0.35 y ±0.25/±0.5 al 100%: un sesgo de un paso pasa m3 (así son los umbrales)."""
    r = build_report(preds(1), SCALE, model="m", stage="m3", thresholds=THRESHOLDS["m3"])
    values = {c["key"]: c["value"] for c in r["thresholds"]["checks"]}
    assert values == {"mae_max": 0.25, "acc_within_0_25_min": 1.0, "acc_within_0_50_min": 1.0}
    r2 = build_report(preds(2), SCALE, model="m", stage="m3", thresholds=THRESHOLDS["m3"])
    assert not r2["thresholds"]["passed"]


def test_within_threshold_follows_scale_step() -> None:
    report = {"scale": {"step": 1.0}, "mean": {"accuracy": 0.3, "acc_within_1": 0.9}}
    res = check_thresholds(report, "m3", {"acc_within_0_25_min": 0.5})
    assert res["checks"][0]["value"] == 0.3 and not res["passed"]


def test_dummy_baseline_and_m2() -> None:
    d = dummy_baseline(np.array([3.25, 3.5, 3.5, 3.75]), SCALE.grid, np.array([3.5, 3.5, 4.0]))
    assert d["fit_on"] == "train" and d["value_mode"] == 3.5 and d["value_mean"] == 3.75
    assert d["mae_mode"] == pytest.approx(0.125) and d["mae_mean"] == pytest.approx(0.25)
    assert d["mae"] == pytest.approx(0.125)
    assert dummy_baseline(Y, SCALE.grid, None)["fit_on"] == "eval_set"

    good = build_report(preds(), SCALE, model="m", stage="m2", thresholds=THRESHOLDS["m2"])
    assert good["mae_improvement_vs_dummy"] == 1.0 and good["thresholds"]["passed"]
    # predecir siempre la moda de val es el propio baseline → mejora 0 → no pasa
    mode = np.full(len(Y), 3.75)
    df = synthetic_predictions(Y, one_hot(mode), SCALE)
    same = build_report(df, SCALE, model="m", stage="m2", thresholds=THRESHOLDS["m2"])
    assert same["mae_improvement_vs_dummy"] == pytest.approx(0) and not same["thresholds"]["passed"]


def test_m4_needs_quality_and_coverage() -> None:
    df = preds(**nine_of_ten_covered())
    r = build_report(df, SCALE, model="m", stage="m4", thresholds=THRESHOLDS["m4"])
    by_key = {c["key"]: c for c in r["thresholds"]["checks"]}
    assert by_key["coverage_90_min"]["value"] == pytest.approx(0.9)
    assert by_key["coverage_90_min"]["passed"] and by_key["coverage_90_max"]["passed"]
    assert by_key["ece_max"]["passed"]
    q = by_key["quality_filter.reject_negatives_min"]
    assert q["value"] is None and not q["passed"] and not r["thresholds"]["passed"]

    r = build_report(
        df, SCALE, model="m", stage="m4", thresholds=THRESHOLDS["m4"], quality=QUALITY_OK
    )
    assert r["thresholds"]["passed"]
    full = build_report(
        preds(interval_lo=list(Y - STEP), interval_hi=list(Y + STEP)),
        SCALE,
        model="m",
        stage="m4",
        thresholds=THRESHOLDS["m4"],
        quality=QUALITY_OK,
    )
    assert not full["thresholds"]["passed"]  # cobertura 1.0 > 0.95: intervalo demasiado ancho


def test_folds_mean_and_loso() -> None:
    a = preds(fold=["val"] * len(Y))
    b = preds(1, fold=["val_b"] * len(Y))
    c = preds(2, fold=["loso:zen_rgbd"] * len(Y))
    r = build_report(pd.concat([a, b, c]), SCALE, model="m")
    assert set(r["folds"]) == {"val", "val_b"} and r["mean"]["n_folds"] == 2
    assert r["mean"]["mae"] == pytest.approx(STEP / 2) and r["pooled"]["n"] == 2 * len(Y)
    assert r["loso"] is not None and r["loso"]["zen_rgbd"]["mae"] == pytest.approx(2 * STEP)
    single = build_report(preds(), SCALE, model="m")
    assert single["loso"] is None and "una sola fuente" in single["loso_note"]
    assert single["thresholds"] is None and single["schema_version"] == 1


def test_forbidden_folds_and_columns() -> None:
    with pytest.raises(EvalInputError, match="holdout"):
        build_report(preds(fold=["test"] * len(Y)), SCALE, model="m")
    with pytest.raises(EvalInputError, match="faltan columnas"):
        build_report(preds().drop(columns=["p_3.25"]), SCALE, model="m")
    with pytest.raises(EvalInputError, match="duplicado"):
        build_report(pd.concat([preds(), preds()]), SCALE, model="m")


def test_walls(tmp_path: Path) -> None:
    p = preds()
    check_walls(p, tmp_path)  # sin archivos en $BCS_DATA_DIR no hay nada que chequear
    (tmp_path / "holdout_groups.txt").write_text("g3\n")
    with pytest.raises(EvalInputError, match="holdout"):
        check_walls(p, tmp_path)
    (tmp_path / "holdout_groups.txt").write_text("otro\n")
    splits = pd.DataFrame({"image_id": p["image_id"], "split": ["val"] * len(p)})
    splits.loc[0, "split"] = "train"
    splits.to_csv(tmp_path / "splits.csv", index=False)
    with pytest.raises(EvalInputError, match="no son de val"):
        check_walls(p, tmp_path)


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    for name in ("data", "artifacts", "runs", "holdout"):
        (tmp_path / name).mkdir()
    monkeypatch.setenv("BCS_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("BCS_ARTIFACTS_DIR", str(tmp_path / "artifacts"))
    monkeypatch.setenv("BCS_RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setenv("BCS_HOLDOUT_DIR", str(tmp_path / "holdout"))
    monkeypatch.delenv("BCS_TASK_ID", raising=False)
    monkeypatch.delenv("BCS_SCALE", raising=False)
    return tmp_path


def test_cli_end_to_end(env: Path) -> None:
    main = _load_cli()
    model_dir = env / "artifacts" / "cand"
    model_dir.mkdir()
    preds(**nine_of_ten_covered()).to_csv(model_dir / "predictions_val.csv", index=False)
    labels = pd.DataFrame({"image_id": ["t1", "t2", "t3"], "bcs": [3.5, 3.5, 4.0]})
    labels.to_csv(env / "data" / "labels.csv", index=False)
    split = pd.DataFrame({"image_id": ["t1", "t2", "t3"], "split": ["train"] * 3})
    split.to_csv(env / "data" / "splits.csv", index=False)

    assert main(["--model", "cand", "--stage", "m3", "--task", "T99"]) == 0
    (run,) = (env / "runs" / "T99").iterdir()
    report = json.loads((run / "report.json").read_text())
    config = json.loads((run / "config.json").read_text())
    assert report["thresholds"]["passed"] and report["dummy"]["fit_on"] == "train"
    assert report["plots"] == {"reliability": "reliability.png", "confusion": "confusion.png"}
    assert config["stage"] == "m3" and len(config["predictions_sha256"]) == 64
    for png in report["plots"].values():
        with Image.open(run / png) as im:
            assert im.format == "PNG" and im.size[0] > 100

    # m4 sin métricas del filtro de calidad → no pasa; con ellas, pasa
    assert main(["--model", "cand", "--stage", "m4", "--task", "T99"]) == 1
    (model_dir / "quality_metrics.json").write_text(json.dumps(QUALITY_OK))
    assert main(["--model", "cand", "--stage", "m4", "--task", "T99"]) == 0


def test_cli_fails_and_walls(env: Path) -> None:
    main = _load_cli()
    bad = env / "uniform.csv"
    uniform().to_csv(bad, index=False)
    assert main(["--model", "u", "--stage", "m3", "--predictions", str(bad)]) == 1
    assert (env / "runs" / "u").is_dir()
    assert main(["--model", "u", "--stage", "m3", "--predictions", str(env / "nada.csv")]) == 2
    inside = env / "holdout" / "p.csv"
    preds().to_csv(inside, index=False)
    assert main(["--model", "u", "--stage", "m3", "--predictions", str(inside)]) == 2
