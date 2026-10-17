"""`make labels` (T06, G0 punto 3): unión, exclusión del holdout y split train / val.

Los chequeos corren sobre los archivos de $BCS_DATA_DIR que escribió `make labels`. Si todavía
no están (checkout sin datos), corren sobre una base sintética armada desde
`preprocesamiento/split.csv` con `merge.run`, así nunca quedan sin ejecutar.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import pytest

from bcs.data import merge
from bcs.data import splits as sp
from bcs.data.schema import LABEL_COLUMNS, LabelsSchemaError, load_labels, validate_labels
from bcs.paths import data_dir

README = sp.read_readme_split()


@dataclass(frozen=True)
class Files:
    root: Path
    labels: pd.DataFrame
    splits: pd.DataFrame
    holdout: list[str]


def _labels_from_readme(readme: pd.DataFrame) -> pd.DataFrame:
    tv = readme[readme["split"] != "test"]
    return pd.DataFrame(
        {
            "image_id": tv["image_id"],
            "path": "images/sdb_bcs/" + tv["image_id"] + ".jpg",
            "source": "sdb_bcs",
            "animal_id": tv["group_id"],
            "group_id": tv["group_id"],
            "bcs": 3.5,
            "scale": "dairy_1_5",
            "view": "rear_dorsal",
            "scorer": "",
            "date": "",
            "bbox": "",
        }
    ).loc[:, list(LABEL_COLUMNS)]


def _synthetic_root(root: Path, readme: pd.DataFrame = README) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    _labels_from_readme(readme).to_csv(root / "labels_sdb_bcs.csv", index=False)
    readme[["image_id", "split"]].to_csv(root / "splits_sdb_bcs.csv", index=False)
    (root / "sources.lock.sdb_bcs.json").write_text(json.dumps({"sdb_bcs": {"sha256": "x"}}))
    return root


def _load(root: Path) -> Files:
    return Files(
        root=root,
        labels=load_labels(root / "labels.csv"),
        splits=pd.read_csv(root / "splits.csv", dtype=str),
        holdout=(root / "holdout_groups.txt").read_text().split(),
    )


@pytest.fixture(scope="module")
def files(tmp_path_factory: pytest.TempPathFactory) -> Files:
    root = data_dir()
    if not (root / "labels.csv").exists():
        root = _synthetic_root(tmp_path_factory.mktemp("data"))
        merge.run(root)
    return _load(root)


# --- sobre la salida de make labels -------------------------------------------------------


def test_labels_schema(files: Files) -> None:
    assert validate_labels(files.labels) == []
    assert set(files.labels["scale"]) <= {"dairy_1_5"}
    assert files.labels["image_id"].is_monotonic_increasing


def test_no_group_leakage(files: Files) -> None:
    tv = files.labels[["image_id", "group_id"]].merge(files.splits, on="image_id")
    test = pd.DataFrame({"group_id": files.holdout, "split": "test"})
    assert sp.group_leakage(pd.concat([tv, test], ignore_index=True)) == []
    assert sp.group_leakage(README) == []


def test_holdout_excluded(files: Files) -> None:
    assert files.holdout, "holdout_groups.txt vacío"
    assert sp.holdout_violations(files.labels, files.holdout) == []
    assert files.holdout == sp.holdout_groups(README)
    test_ids = set(README.loc[README["split"] == "test", "image_id"])
    assert not test_ids & set(files.labels["image_id"])


def test_split_matches_readme(files: Files) -> None:
    assert list(files.splits.columns) == ["image_id", "split"]
    assert files.splits["image_id"].tolist() == files.labels["image_id"].tolist()
    assert set(files.splits["split"]) == {"train", "val"}
    readme = README.set_index("image_id")["split"]
    assert (readme.loc[files.splits["image_id"]].to_numpy() == files.splits["split"]).all()
    # todo train + val del README está en labels.csv (no se perdieron imágenes)
    assert (readme != "test").sum() == len(files.splits)


def test_sources_lock_merged(files: Files) -> None:
    lock = json.loads((files.root / "sources.lock.json").read_text())
    for p in files.root.glob("sources.lock.*.json"):
        for k, v in json.loads(p.read_text()).items():
            assert lock[k] == v


# --- unidades sobre bases sintéticas --------------------------------------------------------


def test_run_is_idempotent(tmp_path: Path) -> None:
    root = _synthetic_root(tmp_path)
    first = merge.run(root)
    mtimes = {p.name: p.stat().st_mtime_ns for p in root.iterdir()}
    assert merge.run(root) == first
    assert {p.name: p.stat().st_mtime_ns for p in root.iterdir()} == mtimes


def test_run_refuses_holdout_group_in_labels(tmp_path: Path) -> None:
    root = _synthetic_root(tmp_path)
    labels = pd.read_csv(root / "labels_sdb_bcs.csv", keep_default_na=False)
    test_row = README[README["split"] == "test"].iloc[0]
    extra = labels.iloc[[0]].assign(image_id=test_row["image_id"], group_id=test_row["group_id"])
    pd.concat([labels, extra]).to_csv(root / "labels_sdb_bcs.csv", index=False)
    with pytest.raises(RuntimeError, match="holdout"):
        merge.run(root)
    assert not (root / "labels.csv").exists()


def test_run_refuses_split_mismatch(tmp_path: Path) -> None:
    root = _synthetic_root(tmp_path)
    s = pd.read_csv(root / "splits_sdb_bcs.csv")
    i = s.index[s["split"] == "train"][0]
    s.loc[i, "split"] = "val"
    s.to_csv(root / "splits_sdb_bcs.csv", index=False)
    with pytest.raises(RuntimeError, match="otro split"):
        merge.run(root)


def test_group_leakage_detected() -> None:
    t = pd.DataFrame({"group_id": ["a", "a", "b"], "split": ["train", "val", "val"]})
    assert sp.group_leakage(t) == ["a"]


def test_leaky_readme_refused(tmp_path: Path) -> None:
    readme = README.copy()
    tv = readme.index[readme["split"] == "train"][0]
    group = readme.loc[readme["split"] == "test", "group_id"].iat[0]
    readme.loc[tv, "group_id"] = group  # una imagen de train en un grupo de test
    path = tmp_path / "split.csv"
    raw = pd.read_csv(sp.README_SPLIT, dtype=str)
    raw.loc[tv, "group"] = group.removeprefix("sdb_bcs:")
    raw.to_csv(path, index=False)
    root = _synthetic_root(tmp_path / "data", readme)
    with pytest.raises(RuntimeError, match="más de un split|holdout"):
        merge.run(root, readme_split=path)


def test_only_scale_1_5_and_no_holdout_labels(tmp_path: Path) -> None:
    root = _synthetic_root(tmp_path)
    other = _labels_from_readme(README).head(3)
    other = other.assign(
        image_id="criollo_" + other["image_id"],
        source="criollo",
        animal_id="criollo:" + other["animal_id"],
        scale="beef_1_8",
    )
    other.to_csv(root / "labels_criollo.csv", index=False)
    # un holdout_labels_<id>.csv roto: si merge lo leyera, fallaría el esquema
    (root / "holdout_labels_sdb_bcs.csv").write_text("no,es,un,csv\n")
    merge.run(root)
    labels = load_labels(root / "labels.csv")
    assert set(labels["source"]) == {"sdb_bcs"}


def test_bad_source_labels_fail_schema(tmp_path: Path) -> None:
    root = _synthetic_root(tmp_path)
    df = pd.read_csv(root / "labels_sdb_bcs.csv", keep_default_na=False)
    df.loc[0, "animal_id"] = "sin_prefijo"
    df.to_csv(root / "labels_sdb_bcs.csv", index=False)
    with pytest.raises(LabelsSchemaError):
        merge.run(root)


def test_lock_conflict(tmp_path: Path) -> None:
    root = _synthetic_root(tmp_path)
    (root / "sources.lock.otro.json").write_text(json.dumps({"sdb_bcs": {"sha256": "y"}}))
    with pytest.raises(ValueError, match="otra entrada"):
        merge.merge_locks(root)
