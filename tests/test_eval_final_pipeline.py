"""T21: `eval/evaluate_final.py` con datos sintéticos y un pipeline falso (nunca test)."""

from __future__ import annotations

import importlib.util
import json
from collections.abc import Sequence
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from bcs.calibration.stub import FullGridInterval, IdentityCalibrator
from bcs.contract import QualityReport
from bcs.eval.report import synthetic_predictions
from bcs.inference import Pipeline
from bcs.paths import REPO_ROOT
from bcs.scale import load_scale

SCALE = load_scale("dairy_sdb")
GRID = SCALE.grid
# (image_id, bcs, clase predicha, recorte pintado, foto cruda oscura, carpeta de la foto cruda)
ROWS = [
    ("sdb_bcs_GS_1_1", 3.25, 3.25, False, False, "3.25"),
    ("sdb_bcs_GS_1_2", 3.5, 3.75, True, False, "3.5"),
    ("sdb_bcs_GS_2_1", 4.25, 4.0, True, False, "4.25"),
    ("sdb_bcs_GS_2_2", 4.0, 4.0, False, True, "4.0"),
    ("sdb_bcs_YM_3_1", 3.75, 3.75, False, False, "3.25"),  # carpeta "equivocada" a propósito
    ("sdb_bcs_YM_3_2", 3.75, 3.5, False, False, "3.75"),
    ("sdb_bcs_L-i10", 4.0, 4.0, False, False, "4.0"),
    ("sdb_bcs_R-i11", 3.5, 3.5, False, False, None),  # sin foto cruda
]


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "evaluate_final", REPO_ROOT / "eval/evaluate_final.py"
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


EF = _load()


def _image(pred: float, painted: bool, dark: bool) -> Image.Image:
    """El verde codifica la clase que va a predecir el scorer falso."""
    if dark:
        return Image.new("RGB", (64, 48), (0, 0, 0))
    im = Image.new("RGB", (64, 48), (90, 60 + 20 * GRID.index(pred), 90))
    if painted:
        im.paste((220, 20, 20), (0, 0, 16, 12))
    return im


class GreenScorer:
    """Distribución con 0,8 en la clase codificada en el píxel (63, 47) y el resto repartido."""

    model_version = "fake-green"

    def predict_proba(self, images: Sequence[Image.Image]) -> np.ndarray:
        out = np.full((len(images), len(GRID)), 0.2 / (len(GRID) - 1))
        for i, im in enumerate(images):
            px = im.convert("RGB").getpixel((im.width - 1, im.height - 1))
            assert isinstance(px, tuple)
            out[i, min(max((px[1] - 60) // 20, 0), len(GRID) - 1)] = 0.8
        return out


class DarkChecker:
    def check(self, image: Image.Image) -> QualityReport:
        dark = float(np.asarray(image.convert("L")).mean()) < 20
        return QualityReport(
            is_cow_rear=True, score=1.0, issues=["imagen muy oscura"] if dark else []
        )


def fake_pipeline() -> Pipeline:
    return Pipeline(
        scale=SCALE,
        quality=DarkChecker(),  # type: ignore[arg-type]
        scorer=GreenScorer(),  # type: ignore[arg-type]
        calibrator=IdentityCalibrator(),
        interval=FullGridInterval(GRID),
    )


@pytest.fixture
def data(tmp_path: Path) -> Path:
    root = tmp_path / "data"
    (root / "images" / "sdb_bcs").mkdir(parents=True)
    for d in ("3.25", "3.5", "3.75", "4.0", "4.25"):
        (root / "raw" / "sdb_bcs" / "dataset" / d).mkdir(parents=True)
    labels = []
    for image_id, bcs, pred, painted, dark, folder in ROWS:
        path = f"images/sdb_bcs/{image_id}.jpg"
        _image(pred, painted, False).save(root / path, quality=95)
        if folder is not None:
            name = image_id.removeprefix("sdb_bcs_")
            raw = root / "raw" / "sdb_bcs" / "dataset" / folder / f"{name}.jpg"
            _image(pred, False, dark).save(raw, quality=95)
        group = f"sdb_bcs:{image_id.split('_')[2]}"
        labels.append({"image_id": image_id, "path": path, "source": "sdb_bcs", "animal_id": group,
                       "group_id": group, "bcs": bcs, "scale": "dairy_1_5"})  # fmt: skip
    pd.DataFrame(labels).to_csv(root / "val_labels.csv", index=False)
    splits = [{"image_id": r[0], "split": "val"} for r in ROWS] + [
        {"image_id": "sdb_bcs_GS_9_1", "split": "train"}
    ]
    pd.DataFrame(splits).to_csv(root / "splits.csv", index=False)
    return root


@pytest.fixture
def val_preds(tmp_path: Path) -> Path:
    y = np.array([3.25, 3.5, 3.75, 4.0, 4.25] * 4)
    p = np.full((len(y), len(GRID)), 0.05)
    p[np.arange(len(y)), np.arange(len(y)) % len(GRID)] = 0.8
    ids = [f"sdb_bcs_{o}_{i}" for i, o in enumerate(["GS_", "YM_", "L-i"] * 6 + ["GS_", "YM_"])]
    df = synthetic_predictions(
        y, p, SCALE, image_id=ids, interval_lo=(y - 0.25).tolist(), interval_hi=(y + 0.25).tolist()
    )
    path = tmp_path / "predictions_val.csv"
    df.to_csv(path, index=False)
    return path


def _run(data: Path, tmp_path: Path, val_preds: Path, *extra: str) -> tuple[int, Path | None]:
    runs = tmp_path / "runs"
    argv = ["--data-dir", str(data), "--runs-dir", str(runs), "--task", "T21",
            "--val-predictions", str(val_preds), *extra]  # fmt: skip
    code = EF.main(argv, pipe=fake_pipeline())
    found = sorted(runs.glob("T21/*/report.json"))
    return code, (found[-1].parent if found else None)


def test_end_to_end_writes_report(data: Path, tmp_path: Path, val_preds: Path) -> None:
    code, run = _run(data, tmp_path, val_preds, "--labels", str(data / "val_labels.csv"))
    assert code == 0 and run is not None
    for name in ("report.md", "config.json", "predictions_crops.csv", "predictions_e2e.csv",
                 "confusion_crops.png", "reliability_crops.png", "confusion_e2e.png",
                 "reliability_e2e.png"):  # fmt: skip
        assert (run / name).exists(), name
    rep = json.loads((run / "report.json").read_text())
    assert rep["is_holdout"] is False and rep["model_version"] == "fake-green"

    crops = rep["modes"]["crops"]
    assert crops["n_scored"] == len(ROWS) and crops["n_rejected"] == 0
    m = crops["metrics"]
    errs = np.array([abs(r[2] - r[1]) for r in ROWS])
    assert m["mae"] == pytest.approx(errs.mean())
    assert m["accuracy"] == pytest.approx((errs == 0).mean())
    assert m["coverage_90"] == 1.0
    assert set(m["by_source"]) == {"GS", "YM", "L-R"}
    assert m["by_class"]["4.25"] == pytest.approx(
        {"n": 1, "mae": 0.25, "bias": -0.25, "accuracy": 0.0, "acc_within_1": 1.0}
    )

    e2e = rep["modes"]["e2e"]
    # GS_2_2 tiene la foto cruda oscura; R-i11 no tiene foto cruda.
    assert (e2e["n_rejected"], e2e["n_missing"], e2e["n_scored"]) == (1, 1, len(ROWS) - 2)
    assert e2e["reject_rate"] == pytest.approx(1 / (len(ROWS) - 1))
    assert e2e["reject_issues"] == {"imagen muy oscura": 1}
    # 3.25 y 4.25 predichos → low_confidence por la regla de extremos de G3.
    assert e2e["n_low_confidence"] == 1
    assert rep["quality"]["reject_val"] == e2e["reject_rate"]

    paint = crops["paint"]
    assert paint["pintadas"]["n"] == 2 and paint["no pintadas"]["n"] == 2
    assert rep["val"]["n"] == 20

    checks = {(c["key"]): c for c in rep["thresholds"]["e2e"]["m4"]["checks"]}
    assert checks["quality_filter.reject_val_max"]["value"] == pytest.approx(e2e["reject_rate"])
    assert checks["quality_filter.accept_rear_min"]["value"] is None
    assert rep["thresholds"]["crops"]["m3"]["passed"] is True

    md = (run / "report.md").read_text()
    assert "Val vs. test" in md and "no son las de test" in md and "pintadas" in md


def test_limit_is_deterministic(data: Path, tmp_path: Path, val_preds: Path) -> None:
    a = EF.load_labels(data / "val_labels.csv", 3, data)
    b = EF.load_labels(data / "val_labels.csv", 3, data)
    assert len(a) == 3 and a["image_id"].tolist() == b["image_id"].tolist()
    code, run = _run(data, tmp_path, val_preds, "--labels", str(data / "val_labels.csv"),
                     "--limit", "3")  # fmt: skip
    assert code == 0 and run is not None
    rep = json.loads((run / "report.json").read_text())
    assert rep["n_labels"] == 3 and rep["modes"]["crops"]["n_total"] == 3


def test_raw_photo_found_without_label(data: Path) -> None:
    path = EF.raw_path(data, "sdb_bcs_YM_3_1")
    assert path is not None and path.parent.name == "3.25"
    assert EF.raw_path(data, "sdb_bcs_R-i11") is None


def test_rejects_train_images(data: Path, tmp_path: Path, val_preds: Path) -> None:
    bad = pd.read_csv(data / "val_labels.csv")
    bad.loc[0, "image_id"] = "sdb_bcs_GS_9_1"
    bad.to_csv(data / "bad.csv", index=False)
    code, run = _run(data, tmp_path, val_preds, "--labels", str(data / "bad.csv"))
    assert code == 2 and run is None


def test_rejects_missing_columns(data: Path, tmp_path: Path, val_preds: Path) -> None:
    pd.read_csv(data / "val_labels.csv").drop(columns="group_id").to_csv(data / "bad.csv")
    code, _ = _run(data, tmp_path, val_preds, "--labels", str(data / "bad.csv"))
    assert code == 2


def test_default_labels_is_holdout_file(data: Path, tmp_path: Path, val_preds: Path) -> None:
    # Directorio de datos sintético sin el archivo de holdout: el default lo busca ahí y sale 2.
    code, run = _run(data, tmp_path, val_preds)
    assert code == 2 and run is None
    assert EF.HOLDOUT_LABELS == "holdout_labels_sdb_bcs.csv"
