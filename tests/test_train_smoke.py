"""T16: `bcs.train`, `make pack-data`, lanzador de backends y bitácora (SPEC §2.4)."""

from __future__ import annotations

import ast
import csv
import hashlib
import importlib
import json
import sys
import tarfile
from pathlib import Path
from types import ModuleType

import pandas as pd
import pytest
import yaml
from PIL import Image

from bcs.data.schema import LABEL_COLUMNS
from bcs.eval.report import load_predictions
from bcs.paths import config_dir
from bcs.scale import load_scale
from bcs.train import TrainError, run

REPO = Path(__file__).resolve().parents[1]
COMPUTE = REPO / "scripts" / "compute"
BASE_CONFIG = config_dir() / "train.yaml"
REPORT_KEYS = {"schema_version", "task", "model", "scale", "mean", "pooled", "folds", "dummy"}
TRAIN_KEYS = {
    "model_version",
    "fold",
    "smoke",
    "synthetic",
    "backend",
    "device",
    "git_sha",
    "pack_hash",
    "n_train",
    "n_val",
    "seconds",
    "weights",
    "predictions",
}
# grupo → split; g5 y g6 son holdout (test)
GROUPS = {"g1": "train", "g2": "train", "g3": "train", "g4": "val", "g5": "test", "g6": "test"}
BCS = {"g1": 3.25, "g2": 3.5, "g3": 3.75, "g4": 4.0, "g5": 4.25, "g6": 3.5}


def _compute(name: str) -> ModuleType:
    if str(COMPUTE) not in sys.path:
        sys.path.insert(0, str(COMPUTE))
    return importlib.import_module(name)


@pytest.fixture
def shared(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    for var, sub in [("BCS_DATA_DIR", "data"), ("BCS_RUNS_DIR", "runs")]:
        (tmp_path / sub).mkdir()
        monkeypatch.setenv(var, str(tmp_path / sub))
    monkeypatch.setenv("BCS_ARTIFACTS_DIR", str(tmp_path / "artifacts"))
    monkeypatch.delenv("BCS_TASK", raising=False)
    return tmp_path


def _make_data(root: Path, *, leak_holdout_row: bool = True) -> None:
    """Datos canónicos de juguete: 6 grupos × 3 imágenes 600×400; g5/g6 en holdout."""
    rows, splits = [], []
    for g, split in GROUPS.items():
        for i in range(3):
            image_id = f"toy_{g}_{i}"
            rel = f"images/toy/{image_id}.jpg"
            (root / rel).parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (600, 400), (40 * i, 30 * int(g[1]), 90)).save(root / rel)
            in_labels = split != "test" or (leak_holdout_row and i == 0)
            if in_labels:
                rows.append(
                    {
                        "image_id": image_id,
                        "path": rel,
                        "source": "toy",
                        "animal_id": f"toy:{g}",
                        "group_id": f"toy:{g}",
                        "bcs": BCS[g],
                        "scale": "dairy_1_5",
                        "view": "rear_dorsal",
                        "scorer": "",
                        "date": "",
                        "bbox": "",
                    }
                )
            # una fila de holdout marcada como train: el pack la tiene que descartar igual
            splits.append(
                {"image_id": image_id, "split": "train" if in_labels and split == "test" else split}
            )
    pd.DataFrame(rows, columns=list(LABEL_COLUMNS)).to_csv(root / "labels.csv", index=False)
    pd.DataFrame(splits).to_csv(root / "splits.csv", index=False)
    (root / "holdout_groups.txt").write_text("toy:g5\ntoy:g6\n")


def _check_report(report: dict, *, synthetic: bool) -> dict:
    """Mismo esquema que `make eval` (T02) + `train`. Devuelve `report["train"]`."""
    assert REPORT_KEYS <= report.keys() and report["schema_version"] == 1
    train = report["train"]
    assert TRAIN_KEYS <= train.keys()
    assert train["synthetic"] is synthetic and train["fold"] == "val"
    assert list(report["folds"]) == ["val"]
    assert {"mae", "accuracy", "n"} <= report["mean"].keys()
    assert report["pooled"]["n"] == train["n_val"] > 0
    assert Path(train["weights"]).exists()
    run_dir = Path(report["report_path"]).parent
    on_disk = json.loads((run_dir / "report.json").read_text())
    assert on_disk["mean"] == report["mean"]
    assert {"config.json", "reliability.png", "confusion.png"} <= {
        p.name for p in run_dir.iterdir()
    }
    # el CSV de predicciones lo acepta `make eval` (T02)
    preds = load_predictions(Path(train["predictions"]), load_scale(report["scale"]["name"]))
    assert len(preds) == train["n_val"] and set(preds["fold"]) == {"val"}
    return train


def test_smoke_synthetic_without_labels(shared: Path) -> None:
    report = run(BASE_CONFIG, smoke=True, model="stub_uniform", task="T16")
    train = _check_report(report, synthetic=True)
    assert train["smoke"] and "warning" in train
    assert Path(report["report_path"]).is_relative_to(shared / "runs" / "T16")
    assert Path(train["weights"]).is_relative_to(shared / "artifacts" / "T16")
    assert Path(train["predictions"]).name == "predictions_val_smoke.csv"


def test_no_labels_without_smoke_fails(shared: Path) -> None:
    with pytest.raises(TrainError, match="labels.csv"):
        run(BASE_CONFIG, model="stub_uniform")


def test_single_split_only(shared: Path) -> None:
    with pytest.raises(TrainError, match="único split"):
        run(BASE_CONFIG, fold="3", smoke=True, model="stub_uniform")


def test_train_drops_test_and_holdout(shared: Path) -> None:
    data = shared / "data"
    _make_data(data)
    report = run(BASE_CONFIG, model="stub_uniform", task="T16")
    train = _check_report(report, synthetic=False)
    # g1–g3 train (9), g4 val (3); la fila de g5 marcada como train se descarta por holdout
    assert (train["n_train"], train["n_val"]) == (9, 3)
    assert Path(train["predictions"]) == shared / "artifacts" / "T16" / "predictions_val.csv"
    assert report["dummy"]["fit_on"] == "train"


def test_group_leakage_is_an_error(shared: Path) -> None:
    data = shared / "data"
    _make_data(data)
    splits = pd.read_csv(data / "splits.csv")
    splits.loc[splits["image_id"] == "toy_g1_0", "split"] = "val"
    splits.to_csv(data / "splits.csv", index=False)
    with pytest.raises(TrainError, match="train y val"):
        run(BASE_CONFIG, model="stub_uniform")


def test_pack_excludes_holdout_and_is_deterministic(shared: Path) -> None:
    pack_data = _compute("pack_data")
    data = shared / "data"
    _make_data(data)
    a = pack_data.build_pack(data, shared / "packs_a", workers=1)
    b = pack_data.build_pack(data, shared / "packs_b", workers=1)
    assert a.name == b.name and a.name.startswith("bcs_pack_")
    assert hashlib.sha256(a.read_bytes()).digest() == hashlib.sha256(b.read_bytes()).digest()
    assert pack_data.build_pack(data, shared / "packs_a", workers=1) == a  # idempotente

    with tarfile.open(a) as tar:
        names = tar.getnames()
        labels = pd.read_csv(tar.extractfile("labels.csv"))  # type: ignore[arg-type]
        splits = pd.read_csv(tar.extractfile("splits.csv"))  # type: ignore[arg-type]
        manifest = json.loads(tar.extractfile("manifest.json").read())  # type: ignore[union-attr]
        img = Image.open(tar.extractfile(labels["path"][0]))  # type: ignore[arg-type]
        assert min(img.size) == 256
    assert not any("g5" in n or "g6" in n for n in names)
    assert not set(labels["group_id"]) & {"toy:g5", "toy:g6"}
    assert set(splits["split"]) == {"train", "val"}
    assert manifest["n_images"] == len(labels) == 12
    assert manifest["pack_hash"] in a.name

    # el pack extraído entrena igual que $BCS_DATA_DIR
    root = shared / "unpacked"
    with tarfile.open(a) as tar:
        tar.extractall(root, filter="data")
    report = run(BASE_CONFIG, model="stub_uniform", data_root=root, task="T16")
    train = _check_report(report, synthetic=False)
    assert train["pack_hash"] == manifest["pack_hash"]
    assert (train["n_train"], train["n_val"]) == (9, 3)


def test_pack_changes_with_content(shared: Path) -> None:
    pack_data = _compute("pack_data")
    data = shared / "data"
    _make_data(data)
    a = pack_data.build_pack(data, shared / "packs", workers=1)
    Image.new("RGB", (600, 400), (1, 2, 3)).save(data / "images/toy/toy_g1_0.jpg")
    assert pack_data.build_pack(data, shared / "packs", workers=1).name != a.name


def test_pack_requires_holdout_groups(shared: Path) -> None:
    pack_data = _compute("pack_data")
    data = shared / "data"
    _make_data(data)
    (data / "holdout_groups.txt").unlink()
    with pytest.raises(pack_data.PackError, match="holdout_groups"):
        pack_data.build_pack(data, shared / "packs", workers=1)


def test_launch_local_smoke_writes_ledger(shared: Path) -> None:
    launch = _compute("launch")
    assert launch.main(["--model", "stub_uniform", "--backend", "local", "--smoke"]) == 0
    assert launch.main(["--model", "stub_uniform", "--backend", "local", "--smoke"]) == 0
    with (shared / "runs" / "compute_ledger.csv").open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2
    assert tuple(rows[0].keys()) == _compute("ledger").COLUMNS
    assert {r["backend"] for r in rows} == {"local"} and rows[0]["status"] == "ok"
    assert rows[0]["task"] == "stub_uniform" and rows[0]["smoke"] == "1"
    assert float(rows[0]["est_cost_usd"]) == 0.0
    assert Path(rows[0]["report"]).exists()


@pytest.mark.parametrize("backend", ["modal", "kaggle", "tenki"])
def test_remote_without_cli_or_credentials_is_unavailable(
    backend: str, shared: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    launch = _compute("launch")
    monkeypatch.setenv("HOME", str(shared))  # sin ~/.modal.toml ni ~/.kaggle
    for var in ("MODAL_TOKEN_ID", "MODAL_TOKEN_SECRET", "KAGGLE_USERNAME", "KAGGLE_KEY"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(launch.shutil, "which", lambda name: None)
    assert launch.main(["--model", "stub_uniform", "--backend", backend, "--smoke"]) == 3
    assert not (shared / "runs" / "compute_ledger.csv").exists()


def test_cost_estimate() -> None:
    ledger = _compute("ledger")
    assert ledger.estimate_cost("modal", 3600) == pytest.approx(0.80)
    assert ledger.estimate_cost("kaggle", 3600) == 0.0


def test_folds_are_a_single_split() -> None:
    launch = _compute("launch")
    assert launch.parse_folds(None) == launch.parse_folds("all") == ["val"]
    with pytest.raises(SystemExit):
        launch.parse_folds("5")


def test_kaggle_kernel_staging(tmp_path: Path) -> None:
    launch = _compute("launch")
    args = {"train_args": launch.train_args("stub_uniform", "T16", "val", True), "pack": ""}
    args["git_sha"] = 'a"b\\c'  # escapes
    launch.kaggle_stage_kernel(tmp_path, "someone", "bcs-train-t16", "someone/bcs-pack", args)
    meta = json.loads((tmp_path / "kernel-metadata.json").read_text())
    assert meta["id"] == "someone/bcs-train-t16" and meta["title"] == "bcs-train-t16"
    assert meta["enable_gpu"] is True and meta["is_private"] is True
    assert meta["dataset_sources"] == ["someone/bcs-pack"]
    tree = ast.parse((tmp_path / meta["code_file"]).read_text())
    consts = {
        n.targets[0].id: n.value
        for n in tree.body
        if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)
    }
    args_call = consts["ARGS"]
    assert isinstance(args_call, ast.Call) and isinstance(args_call.args[0], ast.Constant)
    assert json.loads(str(args_call.args[0].value)) == args
    payload = consts["PAYLOAD"]
    assert isinstance(payload, ast.Constant) and len(str(payload.value)) > 1000
    assert "--smoke" in args["train_args"] and "--model" in args["train_args"]


def test_collect_remote_output(shared: Path) -> None:
    """Lo que deja un backend remoto (como Kaggle/Tenki: todo en una carpeta) se trae a
    runs/<task>/<ts>_<fold>/ y artifacts/<task>/, con las rutas de `train` reescritas."""
    launch = _compute("launch")
    remote = shared / "remote_out"
    run(BASE_CONFIG, smoke=True, model="stub_uniform", out_dir=remote, weights_dir=remote)
    path = launch.collect(remote, "T16", "20261017T000000Z", "val")
    assert path == shared / "runs" / "T16" / "20261017T000000Z_val" / "report.json"
    train = json.loads(path.read_text())["train"]
    assert Path(train["weights"]) == shared / "artifacts" / "T16" / "stub_uniform_val_smoke.weights"
    assert Path(train["weights"]).exists() and Path(train["predictions"]).exists()
    assert (path.parent / "reliability.png").exists()


def test_train_requires_holdout_groups(shared: Path) -> None:
    data = shared / "data"
    _make_data(data)
    (data / "holdout_groups.txt").unlink()
    with pytest.raises(TrainError, match="holdout_groups"):
        run(BASE_CONFIG, model="stub_uniform")


def _all_strings(obj: object) -> list[str]:
    if isinstance(obj, dict):
        return [s for v in obj.values() for s in _all_strings(v)]
    if isinstance(obj, list):
        return [s for v in obj for s in _all_strings(v)]
    return [obj] if isinstance(obj, str) else []


def test_tenki_public_artifact_has_no_data(shared: Path) -> None:
    """El repo es público: el artifact de cpu-job.yml lleva solo report/config/PNG, sin rutas."""
    public_artifact = _compute("public_artifact")
    out = shared / "runner_out"
    _make_data(shared / "data")
    run(BASE_CONFIG, model="stub_uniform", out_dir=out, weights_dir=out)
    assert {p.name for p in out.iterdir()} >= {"predictions_val.csv", "stub_uniform_val.weights"}
    files = public_artifact.build(out, shared / "public")
    names = sorted(p.relative_to(shared / "public").as_posix() for p in files)
    assert names == ["config.json", "confusion.png", "reliability.png", "report.json"]
    assert sorted(p.name for p in (shared / "public").iterdir()) == names
    for name in ("report.json", "config.json"):
        text = (shared / "public" / name).read_text()
        assert "toy_g" not in text  # ningún image_id
        assert not [s for s in _all_strings(json.loads(text)) if s.startswith(("/", "~"))]
    train = json.loads((shared / "public" / "report.json").read_text())["train"]
    assert train["weights"] is None and train["predictions"] is None

    wf = yaml.safe_load((REPO / ".github" / "workflows" / "cpu-job.yml").read_text())
    steps = wf["jobs"]["train"]["steps"]
    uploads = [s for s in steps if str(s.get("uses", "")).startswith("actions/upload-artifact")]
    assert len(uploads) == 1
    assert uploads[0]["with"]["path"] == ".bcs/public/"
    assert uploads[0]["with"]["retention-days"] == 1
    assert any("public_artifact.py" in str(s.get("run", "")) for s in steps)


@pytest.mark.parametrize(
    "exc",
    [
        KeyError("x"),
        OSError("disco"),
        json.JSONDecodeError("mal", "doc", 0),
        RuntimeError("boom"),
        SystemExit("sin labels"),
        KeyboardInterrupt(),
    ],
)
def test_failed_run_writes_ledger(
    exc: BaseException, shared: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    launch = _compute("launch")

    def boom(*args: object) -> list[Path]:
        raise exc

    monkeypatch.setitem(launch.BACKENDS, "local", boom)
    argv = ["--model", "stub_uniform", "--backend", "local", "--smoke"]
    if isinstance(exc, Exception):
        assert launch.main(argv) == 1
    else:
        with pytest.raises(type(exc)):
            launch.main(argv)
    with (shared / "runs" / "compute_ledger.csv").open() as f:
        rows = list(csv.DictReader(f))
    assert [r["status"] for r in rows] == ["failed"]


def test_remote_timeout_is_noted(shared: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    launch = _compute("launch")

    def timeout(*args: object) -> list[Path]:
        raise launch.KernelTimeout("sigue corriendo")

    monkeypatch.setitem(launch.BACKENDS, "kaggle", timeout)
    assert launch.main(["--model", "stub_uniform", "--backend", "kaggle", "--smoke"]) == 1
    with (shared / "runs" / "compute_ledger.csv").open() as f:
        rows = list(csv.DictReader(f))
    assert rows[0]["status"] == "failed:timeout_may_still_run"


def test_kaggle_user_never_runs_the_cli(shared: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    launch = _compute("launch")
    monkeypatch.setenv("HOME", str(shared))
    monkeypatch.delenv("KAGGLE_USERNAME", raising=False)
    monkeypatch.delenv("KAGGLE_KEY", raising=False)

    def no_cli(*args: object, **kw: object) -> None:
        raise AssertionError("no se ejecuta ninguna CLI para obtener el usuario")

    monkeypatch.setattr(launch, "_run", no_cli)
    with pytest.raises(launch.Unavailable, match="usuario"):
        launch._kaggle_user()
    monkeypatch.setenv("KAGGLE_USERNAME", "someone")
    with pytest.raises(launch.Unavailable, match="credencial"):
        launch._kaggle_user()
    monkeypatch.setenv("KAGGLE_KEY", "x")
    assert launch._kaggle_user() == "someone"
