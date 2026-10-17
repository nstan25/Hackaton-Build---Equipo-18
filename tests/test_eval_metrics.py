"""T02: métricas con predicciones sintéticas de resultado conocido."""

from __future__ import annotations

import inspect

import numpy as np
import pytest

from bcs.eval.metrics import (
    confidence_from_probs,
    evaluate,
    evaluate_with_intervals,
    snap_to_grid,
)
from bcs.scale import load_scale

SCALE = load_scale("dairy_1_5")
GRID = np.asarray(SCALE.grid)
STEP = SCALE.step
K = len(GRID)
# rango validado de sdb_bcs (G0): 3.25–4.25, con algunos repetidos
Y = np.array([3.25, 3.5, 3.5, 3.75, 3.75, 3.75, 4.0, 4.0, 4.25, 4.25])
GROUPS = [f"sdb_bcs:cow_{i // 2}" for i in range(len(Y))]
SOURCES = ["sdb_bcs"] * len(Y)


def one_hot(values: np.ndarray, grid: np.ndarray = GRID) -> np.ndarray:
    idx = snap_to_grid(values, grid)
    p = np.zeros((len(values), len(grid)))
    p[np.arange(len(values)), idx] = 1.0
    return p


def test_signature_frozen() -> None:
    assert list(inspect.signature(evaluate).parameters) == [
        "probs",
        "y",
        "grid",
        "groups",
        "sources",
    ]


def test_perfect() -> None:
    m = evaluate_with_intervals(
        one_hot(Y), Y, SCALE.grid, GROUPS, SOURCES, intervals=[(v, v) for v in Y]
    )
    assert m["n"] == 10 and m["n_groups"] == 5 and m["n_sources"] == 1
    assert m["mae"] == 0 and m["mae_expected"] == pytest.approx(0) and m["bias"] == 0
    assert m["accuracy"] == m["acc_within_1"] == m["acc_within_2"] == 1
    assert m["ece"] == pytest.approx(0) and m["mean_confidence"] == pytest.approx(1)
    assert m["nll"] == pytest.approx(0, abs=1e-9)
    assert m["mae_ci95"] == [0, 0] and m["mae_group_mean"] == 0
    assert m["coverage_90"] == 1 and m["interval_width_steps"] == 0
    cm = np.asarray(m["confusion"]["matrix"])
    assert cm.shape == (K, K) and np.trace(cm) == 10 and cm.sum() == 10
    assert m["confusion"]["labels"][snap_to_grid([3.25], GRID)[0]] == "3.25"
    assert set(m["by_source"]) == {"sdb_bcs"} and m["by_source"]["sdb_bcs"]["mae"] == 0


def test_uniform() -> None:
    """Uniforme: argmax = primer valor de la grilla; confianza = 2/K y nunca acierta."""
    m = evaluate(np.full((len(Y), K), 1 / K), Y, SCALE.grid, GROUPS, SOURCES)
    assert m["mae"] == pytest.approx(np.mean(Y - GRID[0]))
    assert m["mae_expected"] == pytest.approx(np.mean(np.abs(Y - GRID.mean())))
    assert m["accuracy"] == 0 and m["acc_within_1"] == 0
    assert m["mean_confidence"] == pytest.approx(2 / K)
    assert m["ece"] == pytest.approx(2 / K)
    assert m["nll"] == pytest.approx(np.log(K))
    assert m["coverage_90"] is None and m["interval_width_steps"] is None


def test_random_matches_chance() -> None:
    rng = np.random.default_rng(42)
    n = 20_000
    y = rng.choice(Y, size=n)
    probs = rng.dirichlet(np.ones(K), size=n)
    m = evaluate(probs, y, SCALE.grid, [f"g{i}" for i in range(n)], ["s"] * n)
    assert m["accuracy"] == pytest.approx(1 / K, abs=0.01)
    assert m["acc_within_1"] == pytest.approx(3 / K, abs=0.015)
    assert m["acc_within_2"] == pytest.approx(5 / K, abs=0.02)


def test_shifted_one_step() -> None:
    """Sesgo de un paso: MAE = paso, ninguna exacta, todas a ±1; confianza 1 y acierta → ECE 0."""
    m = evaluate(one_hot(Y + STEP), Y, SCALE.grid, GROUPS, SOURCES)
    assert m["mae"] == pytest.approx(STEP) and m["bias"] == pytest.approx(STEP)
    assert m["accuracy"] == 0 and m["acc_within_1"] == 1 and m["acc_within_2"] == 1
    assert m["ece"] == pytest.approx(0)
    cm = np.asarray(m["confusion"]["matrix"])
    assert np.trace(cm) == 0 and np.trace(cm, offset=1) == 10


def test_shifted_two_steps_is_fully_miscalibrated() -> None:
    m = evaluate(one_hot(Y - 2 * STEP), Y, SCALE.grid, GROUPS, SOURCES)
    assert m["mae"] == pytest.approx(2 * STEP) and m["bias"] == pytest.approx(-2 * STEP)
    assert m["acc_within_1"] == 0 and m["acc_within_2"] == 1
    assert m["ece"] == pytest.approx(1.0)
    assert np.trace(np.asarray(m["confusion"]["matrix"]), offset=-2) == 10


def test_ece_underconfident() -> None:
    """0.8 en el valor real y 0.2 lejos: confianza 0.8, acierto 1 → ECE 0.2 en el bin (0.7, 0.8]."""
    probs = 0.8 * one_hot(Y)
    probs[:, 0] += 0.2
    assert confidence_from_probs(probs) == pytest.approx(np.full(len(Y), 0.8))
    m = evaluate(probs, Y, SCALE.grid, GROUPS, SOURCES)
    assert m["ece"] == pytest.approx(0.2)
    filled = [b for b in m["reliability"] if b["count"]]
    assert len(filled) == 1 and filled[0]["lo"] == pytest.approx(0.7)
    assert filled[0]["accuracy"] == 1 and filled[0]["count"] == 10


def test_confidence_at_grid_edges() -> None:
    p = np.zeros((2, K))
    p[0, [0, 1, 2]] = [0.5, 0.3, 0.2]
    p[1, [K - 3, K - 2, K - 1]] = [0.2, 0.3, 0.5]
    assert confidence_from_probs(p) == pytest.approx([0.8, 0.8])


def test_coverage_and_width() -> None:
    half = len(Y) // 2
    intervals = [(v - STEP, v + STEP) for v in Y[:half]] + [
        (v + STEP, v + 2 * STEP) for v in Y[half:]
    ]
    m = evaluate_with_intervals(one_hot(Y), Y, SCALE.grid, GROUPS, SOURCES, intervals)
    assert m["coverage_90"] == pytest.approx(0.5)
    assert m["interval_width_steps"] == pytest.approx(1.5)


def test_by_source() -> None:
    probs = np.vstack([one_hot(Y), one_hot(Y + STEP)])
    y = np.concatenate([Y, Y])
    sources = ["a"] * len(Y) + ["b"] * len(Y)
    groups = [f"a{i}" for i in range(len(Y))] + [f"b{i}" for i in range(len(Y))]
    m = evaluate(probs, y, SCALE.grid, groups, sources)
    assert m["n_sources"] == 2 and m["mae"] == pytest.approx(STEP / 2)
    assert m["by_source"]["a"]["mae"] == 0 and m["by_source"]["b"]["mae"] == pytest.approx(STEP)
    assert "by_source" not in m["by_source"]["a"]
    lo, hi = m["mae_ci95"]
    assert 0 <= lo <= m["mae"] <= hi <= STEP


def test_other_scale_no_fixed_grid() -> None:
    """beef_1_8 (paso 1): las mismas métricas salen de la grilla recibida."""
    beef = load_scale("beef_1_8")
    grid = np.asarray(beef.grid)
    y = np.array([2.0, 3.0, 5.0, 7.0])
    m = evaluate(one_hot(y + 1, grid), y, beef.grid, ["g"] * 4, ["s"] * 4)
    assert m["mae"] == pytest.approx(1.0) and m["acc_within_1"] == 1
    assert m["confusion"]["labels"] == [f"{v:.2f}" for v in grid]


def test_collapsed_tails_snap() -> None:
    grid = np.asarray([3.0, 3.25, 3.5, 3.75, 4.0])  # p. ej. collapse_below=3.0
    assert snap_to_grid([2.0, 3.0, 4.5], grid).tolist() == [0, 0, 4]


def test_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="fuera de la grilla"):
        evaluate(one_hot(Y), Y + 0.1, SCALE.grid, GROUPS, SOURCES)
    with pytest.raises(ValueError, match="sumar 1"):
        evaluate(one_hot(Y) * 0.5, Y, SCALE.grid, GROUPS, SOURCES)
    with pytest.raises(ValueError, match=r"\[n, 17\]"):
        evaluate(one_hot(Y)[:, :5], Y, SCALE.grid, GROUPS, SOURCES)
    with pytest.raises(ValueError, match="groups"):
        evaluate(one_hot(Y), Y, SCALE.grid, GROUPS[:3], SOURCES)
