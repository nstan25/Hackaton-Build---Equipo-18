"""Distribución, calibración e intervalo sobre la salida continua de la EfficientNet-B0 (T11).

Tres piezas, en orden (SPEC §2.3):

1. `ResidualDistribution`: valor continuo `v` → distribución sobre la grilla. Normal discretizada
   heterocedástica (probit ordinal con cortes fijos en los puntos medios de la grilla y las puntas
   abiertas): media `v + paso · Σ a_j z^j` y `log σ = Σ b_j z^j`, con `z = (v - centro) / paso`.
   Se ajusta por máxima verosimilitud sobre las clases reales. Con grado ≥ 1, σ y el corrimiento
   dependen del valor predicho; en val no mejoró fuera de fold, así que el default es grado 0
   (σ y corrimiento globales, por ML). El sesgo de las puntas (3.25 sale alta, 4.25 baja) es de
   E[v | y], no de E[y | v]: lo corrige el conformal condicional a la clase, no la distribución.
2. `TemperatureScaling` (`Calibrator`): `p ∝ p^(1/T)`, T por mínima NLL.
3. `ConformalInterval` (`IntervalPredictor`): split conformal LAC (score `1 - p_y`), por defecto
   condicional a la clase real (un q̂ por clase); el conjunto se cierra a su envolvente (min, max)
   en la grilla.

`CalibratedRegression` junta las tres y se guarda/carga como JSON (lo que usa T12).
`cross_fit` arma las predicciones cross-fitted por `group_id` (lo que mide `make eval`).
Todo se ajusta con datos de val, nunca de train.
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import minimize, minimize_scalar
from scipy.special import log_ndtr, ndtr

_EPS = 1e-9
SEED = 42
Groups = Sequence[str] | np.ndarray
# grado elegido por cross-fitting en val (T11): 1 y 2 no mejoran NLL ni ECE fuera de fold
DEFAULT_DEGREE = 0


def snap_index(y: np.ndarray | Sequence[float], grid: Sequence[float]) -> np.ndarray:
    """Índice de la grilla más cercano a cada `y` (las colas van a las puntas)."""
    g = np.asarray(grid, dtype=float)
    return np.abs(np.asarray(y, dtype=float)[:, None] - g[None, :]).argmin(axis=1)


def _bounds(grid: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    g = np.asarray(grid, dtype=float)
    half = (g[1] - g[0]) / 2 if len(g) > 1 else 0.5
    return np.r_[-np.inf, g[1:] - half], np.r_[g[:-1] + half, np.inf]


def _normalize(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, _EPS, None)
    return p / p.sum(axis=1, keepdims=True)


def _nll(p: np.ndarray, y_idx: np.ndarray) -> float:
    return float(-np.log(np.clip(p[np.arange(len(p)), y_idx], _EPS, None)).mean())


class ResidualDistribution:
    """Normal discretizada con media y σ polinómicos en el valor predicho (ver docstring)."""

    def __init__(self, grid: Sequence[float], degree: int = 2) -> None:
        self.grid = tuple(float(v) for v in grid)
        self.degree = degree
        g = np.asarray(self.grid)
        self.center = float(g.mean())
        self.step = float(g[1] - g[0]) if len(g) > 1 else 1.0
        self.mean_coef = np.zeros(degree + 1)
        self.log_sigma_coef = np.r_[math.log(0.85), np.zeros(degree)]

    def _design(self, values: np.ndarray) -> np.ndarray:
        z = (np.asarray(values, dtype=float) - self.center) / self.step
        return np.vander(z, self.degree + 1, increasing=True)

    def params(self, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(μ, σ) por fila, en unidades de BCS."""
        x = self._design(values)
        mu = np.asarray(values, dtype=float) + self.step * (x @ self.mean_coef)
        sigma = self.step * np.exp(np.clip(x @ self.log_sigma_coef, -6.0, 3.0))
        return mu, sigma

    def predict_proba(self, values: np.ndarray) -> np.ndarray:
        lo, hi = _bounds(self.grid)
        mu, sigma = self.params(values)
        p = ndtr((hi[None, :] - mu[:, None]) / sigma[:, None]) - ndtr(
            (lo[None, :] - mu[:, None]) / sigma[:, None]
        )
        return _normalize(p)

    def fit(self, values: np.ndarray, y_idx: np.ndarray) -> ResidualDistribution:
        v = np.asarray(values, dtype=float)
        y_idx = np.asarray(y_idx, dtype=int)
        lo_all, hi_all = _bounds(self.grid)
        lo, hi = lo_all[y_idx], hi_all[y_idx]
        x = self._design(v)
        d = self.degree + 1
        resid = np.asarray(self.grid)[y_idx] - v
        start = np.r_[
            np.zeros(d), math.log(max(float(np.std(resid)), 1e-3) / self.step), np.zeros(d - 1)
        ]

        def nll(theta: np.ndarray) -> float:
            mu = v + self.step * (x @ theta[:d])
            sigma = self.step * np.exp(np.clip(x @ theta[d:], -6.0, 3.0))
            a, b = (lo - mu) / sigma, (hi - mu) / sigma
            # log(Φ(b) - Φ(a)) estable en las colas
            la, lb = log_ndtr(a), log_ndtr(b)
            ll = lb + np.log1p(-np.exp(np.minimum(la - lb, -1e-12)))
            return float(-ll.mean())

        res = minimize(nll, start, method="L-BFGS-B")
        self.mean_coef, self.log_sigma_coef = res.x[:d].copy(), res.x[d:].copy()
        return self

    def to_dict(self) -> dict[str, Any]:
        return {
            "grid": list(self.grid),
            "degree": self.degree,
            "center": self.center,
            "step": self.step,
            "mean_coef": self.mean_coef.tolist(),
            "log_sigma_coef": self.log_sigma_coef.tolist(),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ResidualDistribution:
        obj = cls(d["grid"], int(d["degree"]))
        obj.mean_coef = np.asarray(d["mean_coef"], dtype=float)
        obj.log_sigma_coef = np.asarray(d["log_sigma_coef"], dtype=float)
        return obj


class TemperatureScaling:
    """`Calibrator` (base.py): `p ∝ exp(log p / T)`, T por mínima NLL en val."""

    def __init__(self, temperature: float = 1.0) -> None:
        self.temperature = temperature

    def fit(self, probs: np.ndarray, y: np.ndarray) -> None:
        logp = np.log(np.clip(np.asarray(probs, dtype=float), _EPS, None))
        y_idx = np.asarray(y, dtype=int)

        def nll(log_t: float) -> float:
            return _nll(_normalize(np.exp(logp / math.exp(log_t))), y_idx)

        res: Any = minimize_scalar(nll, bounds=(math.log(0.05), math.log(20.0)), method="bounded")
        self.temperature = float(math.exp(res.x))

    def transform(self, probs: np.ndarray) -> np.ndarray:
        logp = np.log(np.clip(np.asarray(probs, dtype=float), _EPS, None)) / self.temperature
        logp -= logp.max(axis=1, keepdims=True)
        return _normalize(np.exp(logp))


def _conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """Cuantil `⌈(n+1)(1-α)⌉/n` de los scores; con muy pocos casos, 1 (la clase entra siempre)."""
    n = len(scores)
    if n == 0 or math.ceil((n + 1) * (1 - alpha)) > n:
        return 1.0
    return float(np.quantile(scores, math.ceil((n + 1) * (1 - alpha)) / n, method="higher"))


class ConformalInterval:
    """`IntervalPredictor` (base.py): split conformal LAC, envolvente del conjunto.

    Con `by_class=True` es condicional a la clase real (Mondrian por etiqueta): un umbral q̂_k por
    clase, calculado con los casos de calibración de esa clase, y la clase k entra si
    `1 - p_k ≤ q̂_k`. Garantiza cobertura ≥ 1 - α dentro de cada clase real, incluidas las puntas,
    donde el modelo encoge hacia el centro. Con `by_class=False`, un único q̂ (cobertura marginal).
    """

    def __init__(
        self,
        grid: Sequence[float],
        thresholds: Sequence[float] | None = None,
        by_class: bool = True,
    ) -> None:
        self.grid = tuple(float(v) for v in grid)
        self.by_class = by_class
        # q̂_k por clase (en el modo marginal, el mismo valor repetido)
        self.thresholds = np.ones(len(self.grid)) if thresholds is None else np.asarray(thresholds)
        self.alpha = 0.1

    def fit(self, probs: np.ndarray, y: np.ndarray, alpha: float = 0.1) -> None:
        p = np.asarray(probs, dtype=float)
        y_idx = np.asarray(y, dtype=int)
        scores = 1.0 - p[np.arange(len(p)), y_idx]
        if self.by_class:
            q = [_conformal_quantile(scores[y_idx == k], alpha) for k in range(len(self.grid))]
            self.thresholds = np.asarray(q)
        else:
            self.thresholds = np.full(len(self.grid), _conformal_quantile(scores, alpha))
        self.alpha = alpha

    def index_interval(self, probs: np.ndarray) -> np.ndarray:
        """[n, 2] índices (lo, hi) de la grilla; siempre incluye el argmax."""
        p = np.asarray(probs, dtype=float)
        keep = (1.0 - p) <= self.thresholds[None, :] + 1e-12
        keep[np.arange(len(p)), p.argmax(axis=1)] = True
        k = p.shape[1]
        lo = keep.argmax(axis=1)
        hi = k - 1 - keep[:, ::-1].argmax(axis=1)
        return np.stack([lo, hi], axis=1)

    def interval(self, probs: np.ndarray) -> list[tuple[float, float]]:
        g = self.grid
        return [(g[lo], g[hi]) for lo, hi in self.index_interval(probs)]


class CalibratedRegression:
    """Valor continuo → distribución calibrada + intervalo conformal (lo que carga T12)."""

    def __init__(
        self,
        grid: Sequence[float],
        degree: int = DEFAULT_DEGREE,
        alpha: float = 0.1,
        by_class: bool = True,
    ) -> None:
        self.grid = tuple(float(v) for v in grid)
        self.alpha = alpha
        self.dist = ResidualDistribution(self.grid, degree)
        self.calibrator = TemperatureScaling()
        self.conformal = ConformalInterval(self.grid, by_class=by_class)
        self.info: dict[str, Any] = {}

    def fit(
        self, values: np.ndarray, y: np.ndarray, groups: Groups | None = None
    ) -> CalibratedRegression:
        """Ajusta las tres piezas. Con `groups`, distribución + temperatura en la mitad de los
        grupos y el conformal en la otra mitad (split conformal honesto); sin `groups`, todo
        sobre las mismas filas."""
        v = np.asarray(values, dtype=float)
        y_idx = snap_index(y, self.grid)
        if groups is None:
            fit_m = conf_m = np.ones(len(v), dtype=bool)
        else:
            conf_m = group_folds(groups, 2, seed=SEED) == 1
            fit_m = ~conf_m
        self.dist.fit(v[fit_m], y_idx[fit_m])
        self.calibrator.fit(self.dist.predict_proba(v[fit_m]), y_idx[fit_m])
        self.conformal.fit(self.predict_proba(v[conf_m]), y_idx[conf_m], self.alpha)
        self.info = {"n_fit": int(fit_m.sum()), "n_conformal": int(conf_m.sum())}
        if groups is not None:
            self.info["n_groups"] = len(set(groups))
        return self

    def predict_proba(self, values: np.ndarray) -> np.ndarray:
        return self.calibrator.transform(self.dist.predict_proba(values))

    def interval(self, probs: np.ndarray) -> list[tuple[float, float]]:
        return self.conformal.interval(probs)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "calibrated_regression",
            "grid": list(self.grid),
            "alpha": self.alpha,
            "distribution": self.dist.to_dict(),
            "temperature": self.calibrator.temperature,
            "conformal_by_class": self.conformal.by_class,
            "conformal_thresholds": self.conformal.thresholds.tolist(),
            "info": self.info,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> CalibratedRegression:
        dist = ResidualDistribution.from_dict(d["distribution"])
        by_class = bool(d["conformal_by_class"])
        obj = cls(d["grid"], dist.degree, float(d["alpha"]), by_class)
        obj.dist = dist
        obj.calibrator = TemperatureScaling(float(d["temperature"]))
        obj.conformal = ConformalInterval(obj.grid, d["conformal_thresholds"], by_class)
        obj.conformal.alpha = obj.alpha
        obj.info = dict(d.get("info") or {})
        return obj

    def save(self, path: Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2) + "\n")

    @classmethod
    def load(cls, path: Path) -> CalibratedRegression:
        return cls.from_dict(json.loads(Path(path).read_text()))


class GlobalSigma(CalibratedRegression):
    """Referencia: la distribución provisoria de T09 (σ global = RMSE, sin corrimiento ni
    temperatura) + conformal sobre las mismas filas."""

    def __init__(self, grid: Sequence[float], alpha: float = 0.1, by_class: bool = False) -> None:
        super().__init__(grid, degree=0, alpha=alpha, by_class=by_class)

    def fit(
        self, values: np.ndarray, y: np.ndarray, groups: Groups | None = None
    ) -> CalibratedRegression:
        v = np.asarray(values, dtype=float)
        y_idx = snap_index(y, self.grid)
        rmse = float(np.sqrt(np.mean((np.asarray(self.grid)[y_idx] - v) ** 2)))
        self.dist.log_sigma_coef = np.array([math.log(max(rmse, 1e-3) / self.dist.step)])
        self.conformal.fit(self.predict_proba(v), y_idx, self.alpha)
        self.info = {"n_fit": len(v), "sigma": rmse}
        return self


def group_folds(groups: Groups, k: int, seed: int = SEED) -> np.ndarray:
    """Fold (0..k-1) por fila, asignando grupos enteros al azar (semilla fija) y balanceando
    la cantidad de filas por fold (grupos grandes primero)."""
    grp = np.asarray(groups, dtype=str)
    uniq, inv, counts = np.unique(grp, return_inverse=True, return_counts=True)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(uniq))
    order = order[np.argsort(-counts[order], kind="stable")]
    load = np.zeros(k)
    fold_of = np.empty(len(uniq), dtype=int)
    for gi in order:
        f = int(load.argmin())
        fold_of[gi] = f
        load[f] += counts[gi]
    return fold_of[inv]


def cross_fit(
    model_factory: Any,
    values: np.ndarray,
    y: np.ndarray,
    groups: Groups,
    k: int = 5,
    seed: int = SEED,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Predicciones fuera de fold: para cada fold de grupos, ajusta `model_factory()` con el resto
    y predice ese fold. Devuelve (probs [n, K], intervalos [n, 2] en BCS, fold [n])."""
    v = np.asarray(values, dtype=float)
    yv = np.asarray(y, dtype=float)
    grp = np.asarray(groups, dtype=str)
    folds = group_folds(grp, k, seed)
    probs: np.ndarray | None = None
    iv = np.zeros((len(v), 2))
    for f in range(k):
        test = folds == f
        model: CalibratedRegression = model_factory()
        model.fit(v[~test], yv[~test], grp[~test])
        p = model.predict_proba(v[test])
        if probs is None:
            probs = np.zeros((len(v), p.shape[1]))
        probs[test] = p
        iv[test] = np.asarray(model.interval(p), dtype=float)
    assert probs is not None
    return probs, iv, folds
