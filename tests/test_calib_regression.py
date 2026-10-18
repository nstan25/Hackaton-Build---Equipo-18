"""T11: distribución sobre la grilla, temperature scaling y conformal (sin datos reales)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from bcs.calibration.base import Calibrator, IntervalPredictor
from bcs.calibration.regression import (
    CalibratedRegression,
    ConformalInterval,
    GlobalSigma,
    ResidualDistribution,
    TemperatureScaling,
    cross_fit,
    group_folds,
    snap_index,
)
from bcs.calibration.run import main as run_main

GRID = (3.25, 3.5, 3.75, 4.0, 4.25)


def synthetic(n_groups: int = 300, per_group: int = 10, seed: int = 0):
    """Vacas con BCS latente; el modelo predice con ruido y encoge hacia el centro."""
    rng = np.random.default_rng(seed)
    latent = rng.uniform(3.1, 4.4, n_groups)
    groups = np.repeat([f"sdb_bcs:cow_{i}" for i in range(n_groups)], per_group)
    true = np.repeat(latent, per_group)
    y = np.asarray(GRID)[snap_index(true, GRID)]
    value = 3.75 + 0.8 * (y - 3.75) + rng.normal(0, 0.15, len(y))
    return value, y, groups


def test_protocols() -> None:
    assert isinstance(TemperatureScaling(), Calibrator)
    assert isinstance(ConformalInterval(GRID), IntervalPredictor)


def test_distribution_is_a_distribution_and_follows_value() -> None:
    dist = ResidualDistribution(GRID, degree=1)
    p = dist.predict_proba(np.array([3.0, 3.6, 3.75, 4.4, 5.0]))
    assert p.shape == (5, len(GRID))
    np.testing.assert_allclose(p.sum(axis=1), 1.0)
    assert (p >= 0).all()
    assert list(p.argmax(axis=1)) == [0, 1, 2, 4, 4]


def test_distribution_recovers_sigma() -> None:
    rng = np.random.default_rng(1)
    y = rng.choice(GRID, 20000)
    v = y + rng.normal(0, 0.2, len(y))
    dist = ResidualDistribution(GRID, degree=0).fit(v, snap_index(y, GRID))
    _, sigma = dist.params(np.array([3.75]))
    # y | v no es exactamente normal (prior discreto), pero σ queda en el orden del ruido
    assert 0.12 < sigma[0] < 0.3


def test_temperature_recovers_known_temperature() -> None:
    rng = np.random.default_rng(2)
    logits = rng.normal(0, 2, (5000, 5))
    true_p = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    y = np.array([rng.choice(5, p=row) for row in true_p])
    sharp = np.exp(logits * 2) / np.exp(logits * 2).sum(axis=1, keepdims=True)
    cal = TemperatureScaling()
    cal.fit(sharp, y)
    assert cal.temperature == pytest.approx(2.0, rel=0.15)
    out = cal.transform(sharp)
    np.testing.assert_allclose(out, true_p, atol=0.05)
    np.testing.assert_allclose(out.sum(axis=1), 1.0)


@pytest.mark.parametrize("by_class", [False, True])
def test_conformal_coverage_on_fresh_data(by_class: bool) -> None:
    v, y, groups = synthetic(seed=3)
    v2, y2, g2 = synthetic(seed=4)
    model = CalibratedRegression(GRID, by_class=by_class).fit(v, y, groups)
    iv = np.asarray(model.interval(model.predict_proba(v2)))
    covered = (y2 >= iv[:, 0]) & (y2 <= iv[:, 1])
    assert 0.86 <= covered.mean() <= 0.97
    assert (iv[:, 0] <= iv[:, 1]).all()
    assert set(iv.ravel()) <= set(GRID)
    if by_class:
        per_class = {g: float(covered[y2 == g].mean()) for g in GRID}
        assert min(per_class.values()) >= 0.8, per_class


def test_class_conditional_fixes_the_extremes() -> None:
    """El modelo encoge hacia el centro: el conformal marginal sub-cubre las puntas y el
    condicional a la clase no."""
    v, y, groups = synthetic(seed=5)
    v2, y2, _ = synthetic(seed=6)
    cov = {}
    for by_class in (False, True):
        model = CalibratedRegression(GRID, by_class=by_class).fit(v, y, groups)
        iv = np.asarray(model.interval(model.predict_proba(v2)))
        ok = (y2 >= iv[:, 0]) & (y2 <= iv[:, 1])
        cov[by_class] = min(ok[y2 == GRID[0]].mean(), ok[y2 == GRID[-1]].mean())
    assert cov[True] >= cov[False]
    assert cov[True] >= 0.8


def test_interval_always_contains_argmax() -> None:
    conf = ConformalInterval(GRID, thresholds=[0.0] * len(GRID))
    p = np.array([[0.1, 0.6, 0.1, 0.1, 0.1], [0.05, 0.05, 0.1, 0.3, 0.5]])
    assert conf.interval(p) == [(3.5, 3.5), (4.25, 4.25)]


def test_interval_is_hull_of_the_set() -> None:
    conf = ConformalInterval(GRID, thresholds=[0.75] * len(GRID))
    p = np.array([[0.3, 0.1, 0.3, 0.1, 0.2]])  # entran 3.25 y 3.75 → envolvente 3.25–3.75
    assert conf.interval(p) == [(3.25, 3.75)]


def test_conformal_with_few_cases_includes_the_class() -> None:
    conf = ConformalInterval(GRID)
    p = np.full((20, 5), 0.2)
    y = np.r_[np.zeros(19, dtype=int), 4]  # una sola imagen de 4.25
    conf.fit(p, y)
    assert conf.thresholds[4] == 1.0


def test_save_load_roundtrip(tmp_path) -> None:
    v, y, groups = synthetic(seed=7)
    model = CalibratedRegression(GRID, degree=1).fit(v, y, groups)
    path = tmp_path / "calibration.json"
    model.save(path)
    loaded = CalibratedRegression.load(path)
    np.testing.assert_allclose(loaded.predict_proba(v[:50]), model.predict_proba(v[:50]))
    p = model.predict_proba(v[:50])
    assert loaded.interval(p) == model.interval(p)


def test_group_folds_keep_groups_together() -> None:
    _, _, groups = synthetic(n_groups=57, per_group=7)
    folds = group_folds(groups, 5)
    per_group = pd.Series(folds).groupby(groups).nunique()
    assert (per_group == 1).all()
    assert set(folds) == set(range(5))
    np.testing.assert_array_equal(folds, group_folds(groups, 5))  # semilla fija


def test_cross_fit_never_fits_on_the_fold_it_predicts() -> None:
    v, y, groups = synthetic(n_groups=100, seed=8)
    seen: list[set[str]] = []

    class Spy(GlobalSigma):
        def fit(self, values, y, groups=None):
            seen.append(set(np.asarray(groups, dtype=str)))
            return super().fit(values, y, groups)

    probs, iv, folds = cross_fit(lambda: Spy(GRID), v, y, groups, k=5)
    assert probs.shape == (len(v), len(GRID))
    np.testing.assert_allclose(probs.sum(axis=1), 1.0)
    for f, fit_groups in enumerate(seen):
        assert fit_groups.isdisjoint(set(groups[folds == f]))


def test_run_writes_predictions_for_make_eval(tmp_path) -> None:
    v, y, groups = synthetic(n_groups=120, per_group=5, seed=9)
    ids = [f"sdb_bcs_GS_{i}" for i in range(len(v))]
    src = pd.DataFrame(
        {"image_id": ids, "group_id": groups, "source": "sdb_bcs", "bcs": y, "fold": "val"}
    )
    src["value"] = v
    pred_in = tmp_path / "t09.csv"
    src.to_csv(pred_in, index=False)
    out, runs = tmp_path / "art", tmp_path / "runs"
    assert (
        run_main(["--predictions", str(pred_in), "--out", str(out), "--runs-dir", str(runs)]) == 0
    )

    pred = pd.read_csv(out / "predictions_val.csv")
    p_cols = [c for c in pred.columns if c.startswith("p_")]
    assert p_cols == ["p_3.25", "p_3.50", "p_3.75", "p_4.00", "p_4.25"]
    np.testing.assert_allclose(pred[p_cols].sum(axis=1), 1.0, atol=1e-6)
    assert set(pred["fold"]) == {f"cf{k}" for k in range(5)}
    assert (pred["interval_lo"] <= pred["interval_hi"]).all()
    assert CalibratedRegression.load(out / "calibration.json").conformal.by_class
    assert len(list(runs.glob("T11/*/summary.md"))) == 1


def test_run_refuses_non_val_predictions(tmp_path) -> None:
    pred_in = tmp_path / "t09.csv"
    pd.DataFrame(
        {"image_id": ["a"], "group_id": ["g"], "source": ["s"], "bcs": [3.5], "fold": ["test"],
         "value": [3.5]}
    ).to_csv(pred_in, index=False)  # fmt: skip
    with pytest.raises(SystemExit):
        run_main(["--predictions", str(pred_in), "--out", str(tmp_path / "o")])
