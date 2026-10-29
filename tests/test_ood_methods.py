"""Métodos OOD de T27 sobre features sintéticas (sin el modelo real)."""

from __future__ import annotations

import numpy as np
import pytest

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
    threshold_at_tpr,
)

DIM = 32


def _data(seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Banco ID (3 clases), ID de prueba y OOD corrido en direcciones que el ID no usa."""
    rng = np.random.default_rng(seed)
    centers = rng.normal(0, 3, (3, DIM)) * (np.arange(DIM) < 8)  # ID vive en 8 dimensiones
    y = rng.integers(0, 3, 600)
    bank = centers[y] + rng.normal(0, 1, (600, DIM)) * np.where(np.arange(DIM) < 8, 1.0, 0.2)
    yt = rng.integers(0, 3, 300)
    test = centers[yt] + rng.normal(0, 1, (300, DIM)) * np.where(np.arange(DIM) < 8, 1.0, 0.2)
    ood = centers[rng.integers(0, 3, 300)] + rng.normal(0, 1, (300, DIM))
    ood[:, 8:] += 2.0
    return bank, y, test, ood


SEPARATING = [
    KNNScore(k=1),
    KNNScore(k=10),
    MahalanobisScore(),
    MahalanobisScore(per_class=True),
    MahalanobisScore(normalize=True),
]
METHODS = [*SEPARATING, MahalanobisScore(per_class=True, relative=True)]


@pytest.mark.parametrize("method", SEPARATING, ids=repr)
def test_methods_separate_shifted_gaussians(method) -> None:
    bank, y, test, ood = _data()
    m = method.fit(bank, y)
    assert auroc(m.score(test), m.score(ood)) > 0.95
    assert fpr_at_tpr(m.score(test), m.score(ood)) < 0.2


@pytest.mark.parametrize("method", METHODS, ids=repr)
def test_methods_are_deterministic(method) -> None:
    bank, y, test, _ = _data()
    a = type(method)(**_params(method)).fit(bank, y).score(test)
    b = type(method)(**_params(method)).fit(bank.copy(), y.copy()).score(test.copy())
    np.testing.assert_array_equal(a, b)


def test_relative_mahalanobis_flags_extrapolation_but_not_background_shift() -> None:
    bank, y, test, ood = _data()
    m = MahalanobisScore(per_class=True, relative=True).fit(bank, y)
    far = bank[:300].copy()
    far[:, :8] *= 4.0  # lejos de las clases a lo largo de las direcciones que las separan
    assert auroc(m.score(test), m.score(far)) > 0.95
    # el corrimiento en dimensiones que no separan clases se cancela con el fondo
    assert auroc(m.score(test), m.score(ood)) < 0.5


def _params(m) -> dict:
    if isinstance(m, KNNScore):
        return {"k": m.k}
    return {"per_class": m.per_class, "relative": m.relative, "normalize": m.normalize}


def test_knn_matches_brute_force_cosine() -> None:
    bank, _, test, _ = _data()
    s = KNNScore(k=5).fit(bank).score(test)
    bn = bank / np.linalg.norm(bank, axis=1, keepdims=True)
    tn = test / np.linalg.norm(test, axis=1, keepdims=True)
    expected = 1.0 - np.sort(tn @ bn.T, axis=1)[:, -5]
    np.testing.assert_allclose(s, expected, atol=1e-5)


def test_knn_float16_bank_is_close_and_smaller() -> None:
    bank, _, test, _ = _data()
    a, b = KNNScore(k=5).fit(bank), KNNScore(k=5, dtype="float16").fit(bank)
    np.testing.assert_allclose(a.score(test), b.score(test), atol=1e-2)
    assert b.nbytes * 2 == a.nbytes


def test_knn_bank_smaller_than_k_fails() -> None:
    with pytest.raises(ValueError):
        KNNScore(k=50).fit(np.ones((10, 4)))


def test_mahalanobis_single_gaussian_zero_at_mean() -> None:
    bank, _, _, _ = _data()
    m = MahalanobisScore().fit(bank)
    assert m.score(bank.mean(axis=0, keepdims=True))[0] == pytest.approx(0.0, abs=1e-8)
    assert m.nbytes == 4 * (DIM + DIM * DIM)


def test_relative_requires_classes() -> None:
    bank, _, _, _ = _data()
    with pytest.raises(ValueError):
        MahalanobisScore(relative=True).fit(bank)
    with pytest.raises(ValueError):
        MahalanobisScore(per_class=True).fit(bank)


def test_feature_norm_direction() -> None:
    s = FeatureNormScore().fit(np.ones((2, 2))).score(np.array([[3.0, 4.0], [0.3, 0.4]]))
    assert s[1] > s[0]


def test_zscore_combiner_standardizes_on_id() -> None:
    rng = np.random.default_rng(1)
    id_scores = np.column_stack([rng.normal(5, 2, 500), rng.normal(-1, 0.1, 500)])
    c = ZScoreCombiner().fit(id_scores)
    z = c.combine(id_scores)
    assert abs(z.mean()) < 1e-9
    assert c.combine(np.array([[5 + 2 * 3, -1 + 0.1 * 3]]))[0] == pytest.approx(3.0, rel=0.05)


def test_auroc_and_fpr_extremes() -> None:
    assert auroc(np.zeros(5), np.ones(5)) == 1.0
    assert auroc(np.ones(5), np.zeros(5)) == 0.0
    assert auroc(np.zeros(5), np.zeros(5)) == 0.5
    id_s = np.arange(100.0)
    assert threshold_at_tpr(id_s, 0.95) == pytest.approx(95.0)
    assert fpr_at_tpr(id_s, np.array([10.0, 200.0])) == 0.5


def test_selective_mae_drops_when_score_tracks_error() -> None:
    rng = np.random.default_rng(2)
    err = rng.exponential(0.2, 1000)
    curve = selective_mae(err + rng.normal(0, 0.01, 1000), err, [0.0, 0.1, 0.5])
    maes = [c["mae"] for c in curve]
    assert maes[0] == pytest.approx(err.mean())
    assert maes[0] > maes[1] > maes[2]
    assert [c["n"] for c in curve] == [1000, 900, 500]
    assert aurc(err, err) < aurc(-err, err)


def test_subsample_bank_spreads_over_groups() -> None:
    groups = np.repeat(["a", "b", "c", "d"], [100, 5, 5, 5])
    idx = subsample_bank(np.zeros((115, 2)), 20, seed=0, groups=list(groups))
    assert len(idx) == 20 and len(set(idx)) == 20
    assert set(groups[idx]) == {"a", "b", "c", "d"}
    np.testing.assert_array_equal(idx, subsample_bank(np.zeros((115, 2)), 20, 0, list(groups)))
    assert len(subsample_bank(np.zeros((10, 2)), None)) == 10
