"""Puntajes OOD sobre features y métricas para compararlos (T27).

Todos los puntajes siguen la convención "más alto = más raro". Ninguno usa el `cow_id`.

- `KNNScore`: distancia coseno al k-ésimo vecino del banco de referencia, con features
  normalizadas a norma 1 (Sun et al. 2022, "Out-of-distribution detection with deep nearest
  neighbors").
- `MahalanobisScore`: distancia de Mahalanobis a una sola gaussiana (`per_class=False`) o la mínima
  a las medias de cada clase de BCS con covarianza compartida (Lee et al. 2018). Covarianza con
  shrinkage de Ledoit-Wolf. `relative=True` resta la distancia a la gaussiana de fondo
  (Ren et al. 2021, "A simple fix to Mahalanobis distance").
- `FeatureNormScore`: menos la norma de la feature (baseline).
- `ZScoreCombiner`: promedio de z-scores calibrados en un conjunto ID (media y desvío de cada
  puntaje en ese conjunto).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

KNN_CHUNK = 1024


def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), eps)


def subsample_bank(
    feats: np.ndarray, n: int | None, seed: int = 0, groups: Sequence[str] | None = None
) -> np.ndarray:
    """Índices de un submuestreo de `n` filas (todas si `n` es None o ≥ len). Con `groups`,
    reparte lo más parejo posible entre grupos (round-robin con orden aleatorio fijo)."""
    m = len(feats)
    if n is None or n >= m:
        return np.arange(m)
    rng = np.random.default_rng(seed)
    if groups is None:
        return np.sort(rng.choice(m, n, replace=False))
    g = np.asarray(groups)
    # rango de cada fila dentro de su grupo (orden aleatorio), y se toman los rangos bajos primero
    perm = rng.permutation(m)
    rank = np.empty(m, dtype=np.int64)
    seen: dict[str, int] = {}
    for i in perm.tolist():
        k = str(g[i])
        r = seen.get(k, 0)
        rank[i] = r
        seen[k] = r + 1
    order = np.lexsort((rng.random(m), rank))
    return np.sort(order[:n])


@dataclass
class KNNScore:
    """Distancia coseno (1 − similitud) al k-ésimo vecino más cercano del banco."""

    k: int = 10
    dtype: str = "float32"
    bank: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), dtype=np.float32))

    def fit(self, feats: np.ndarray, y: np.ndarray | None = None) -> KNNScore:
        if len(feats) < self.k:
            raise ValueError(f"el banco tiene {len(feats)} filas, menos que k={self.k}")
        self.bank = l2_normalize(feats).astype(self.dtype)
        return self

    def score(self, feats: np.ndarray) -> np.ndarray:
        q = l2_normalize(feats)
        bank = self.bank.astype(np.float32, copy=False)
        out = np.empty(len(q), dtype=np.float64)
        for i in range(0, len(q), KNN_CHUNK):
            sim = q[i : i + KNN_CHUNK] @ bank.T
            # k-ésima similitud más alta = (n-k)-ésima más baja
            kth = np.partition(sim, sim.shape[1] - self.k, axis=1)[:, sim.shape[1] - self.k]
            out[i : i + KNN_CHUNK] = 1.0 - kth
        return out

    @property
    def nbytes(self) -> int:
        return int(self.bank.nbytes)


def ledoit_wolf(x: np.ndarray) -> np.ndarray:
    """Covarianza con shrinkage de Ledoit-Wolf de `x` ya centrada (sklearn, sin re-centrar)."""
    from sklearn.covariance import ledoit_wolf as lw

    cov, _ = lw(x, assume_centered=True)
    return np.asarray(cov, dtype=np.float64)


@dataclass
class MahalanobisScore:
    """Mahalanobis con covarianza Ledoit-Wolf.

    `per_class=False`: una sola gaussiana. `per_class=True`: medias por clase (`y` en `fit`) y
    covarianza compartida (intra-clase); el puntaje es la distancia a la clase más cercana.
    `relative=True` (solo por clase): resta la distancia a la gaussiana de fondo (Ren et al.).
    Ojo: un corrimiento en direcciones donde fondo y clases tienen la misma varianza se cancela
    (la versión relativa solo mira lo que separa a las clases del fondo).
    `normalize=True` normaliza las features a norma 1 antes de todo.
    """

    per_class: bool = False
    relative: bool = False
    normalize: bool = False
    means: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    precision: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    mean0: np.ndarray = field(default_factory=lambda: np.zeros(0))
    precision0: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))

    def _prep(self, feats: np.ndarray) -> np.ndarray:
        x = np.asarray(feats, dtype=np.float64)
        return l2_normalize(x).astype(np.float64) if self.normalize else x

    def fit(self, feats: np.ndarray, y: np.ndarray | None = None) -> MahalanobisScore:
        if self.relative and not self.per_class:
            raise ValueError("relative=True requiere per_class=True")
        x = self._prep(feats)
        self.mean0 = x.mean(axis=0)
        if self.per_class:
            if y is None:
                raise ValueError("per_class=True requiere y")
            y = np.asarray(y)
            classes = np.unique(y)
            self.means = np.stack([x[y == c].mean(axis=0) for c in classes])
            centered = x - self.means[np.searchsorted(classes, y)]
        else:
            self.means = self.mean0[None, :]
            centered = x - self.mean0
        self.precision = np.linalg.inv(ledoit_wolf(centered))
        if self.relative:
            self.precision0 = np.linalg.inv(ledoit_wolf(x - self.mean0))
        return self

    @staticmethod
    def _dist(x: np.ndarray, means: np.ndarray, precision: np.ndarray) -> np.ndarray:
        """[n, c]: distancia de Mahalanobis al cuadrado de cada fila a cada media."""
        xp = x @ precision
        quad_x = np.einsum("ij,ij->i", xp, x)
        mp = means @ precision
        quad_m = np.einsum("ij,ij->i", mp, means)
        return quad_x[:, None] - 2.0 * (xp @ means.T) + quad_m[None, :]

    def score(self, feats: np.ndarray) -> np.ndarray:
        x = self._prep(feats)
        d = self._dist(x, self.means, self.precision)
        if self.relative:
            d = d - self._dist(x, self.mean0[None, :], self.precision0)
        return d.min(axis=1)

    @property
    def nbytes(self) -> int:
        """Bytes de parámetros en float32 (como se guardarían para producción)."""
        n = self.means.size + self.precision.size
        if self.relative:
            n += self.mean0.size + self.precision0.size
        return 4 * n


@dataclass
class FeatureNormScore:
    """Baseline: menos la norma L2 de la feature (features chicas = más raro)."""

    def fit(self, feats: np.ndarray, y: np.ndarray | None = None) -> FeatureNormScore:
        return self

    def score(self, feats: np.ndarray) -> np.ndarray:
        return -np.linalg.norm(np.asarray(feats, dtype=np.float64), axis=1)

    @property
    def nbytes(self) -> int:
        return 0


@dataclass
class ZScoreCombiner:
    """Promedio de z-scores: cada puntaje se estandariza con su media y desvío en un conjunto ID
    de calibración (que no se usa para evaluar)."""

    mu: np.ndarray = field(default_factory=lambda: np.zeros(0))
    sd: np.ndarray = field(default_factory=lambda: np.zeros(0))

    def fit(self, id_scores: np.ndarray) -> ZScoreCombiner:
        s = np.asarray(id_scores, dtype=np.float64)
        self.mu, self.sd = s.mean(axis=0), np.maximum(s.std(axis=0), 1e-12)
        return self

    def combine(self, scores: np.ndarray) -> np.ndarray:
        return ((np.asarray(scores, dtype=np.float64) - self.mu) / self.sd).mean(axis=1)


# ---------------------------------------------------------------- métricas


def auroc(id_scores: np.ndarray, ood_scores: np.ndarray) -> float:
    """P(puntaje OOD > puntaje ID), con empates a la mitad (positivo = OOD)."""
    from scipy.stats import rankdata

    a, b = np.asarray(id_scores, float), np.asarray(ood_scores, float)
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    r = rankdata(np.concatenate([a, b]))
    return float((r[len(a) :].sum() - len(b) * (len(b) + 1) / 2) / (len(a) * len(b)))


def threshold_at_tpr(id_scores: np.ndarray, tpr: float = 0.95) -> float:
    """Umbral que acepta (puntaje ≤ umbral) la fracción `tpr` de las fotos ID."""
    return float(np.quantile(np.asarray(id_scores, float), tpr, method="higher"))


def fpr_at_tpr(id_scores: np.ndarray, ood_scores: np.ndarray, tpr: float = 0.95) -> float:
    """Fracción de OOD aceptada (puntaje ≤ umbral) cuando se acepta `tpr` de las ID."""
    b = np.asarray(ood_scores, float)
    if len(b) == 0:
        return float("nan")
    return float((b <= threshold_at_tpr(id_scores, tpr)).mean())


def selective_mae(
    scores: np.ndarray, errors: np.ndarray, fractions: Sequence[float]
) -> list[dict[str, float]]:
    """MAE de lo que queda al rechazar la fracción `f` con puntaje más alto (orden estable)."""
    s, e = np.asarray(scores, float), np.abs(np.asarray(errors, float))
    order = np.argsort(s, kind="stable")
    out = []
    for f in fractions:
        keep = order[: max(1, round(len(s) * (1.0 - f)))]
        out.append({"rejected": float(f), "n": int(len(keep)), "mae": float(e[keep].mean())})
    return out


def aurc(scores: np.ndarray, errors: np.ndarray) -> float:
    """Área bajo la curva riesgo-cobertura (MAE vs. cobertura); más bajo es mejor."""
    s, e = np.asarray(scores, float), np.abs(np.asarray(errors, float))
    e_sorted = e[np.argsort(s, kind="stable")]
    risk = np.cumsum(e_sorted) / np.arange(1, len(e) + 1)
    return float(risk.mean())
