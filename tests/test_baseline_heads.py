"""T09: baseline tonto, cabezas lineales sobre embeddings, desaturación de rojo y Scorer.

Sin red ni datos reales: embeddings sintéticos y un embedder falso."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from bcs.data.merge import paint_fractions
from bcs.models import baseline as bl
from bcs.models.base import Scorer
from bcs.models.embeddings import Embedder, EmbeddingCache, desaturate_red, red_mask
from bcs.paths import config_dir
from bcs.scale import load_scale
from bcs.train import build_scorer, load_config

GRID = load_scale("dairy_sdb").grid


def _synthetic(n_groups: int = 60, per_group: int = 8, dim: int = 16, seed: int = 0):
    """Embeddings con señal lineal del BCS + ruido; un BCS por grupo."""
    rng = np.random.default_rng(seed)
    g = np.asarray(GRID)
    direction = rng.normal(size=dim)
    xs, ys, groups = [], [], []
    for k in range(n_groups):
        y = g[k % len(g)]
        base = (y - g.mean()) * 4 * direction + rng.normal(scale=0.5, size=dim)
        xs.append(base + rng.normal(scale=0.5, size=(per_group, dim)))
        ys += [y] * per_group
        groups += [f"g{k}"] * per_group
    return np.concatenate(xs), np.array(ys), groups


def test_group_weights_inverse_sqrt_mean_one() -> None:
    w = bl.group_weights(["a"] * 4 + ["b"])
    assert w.mean() == pytest.approx(1.0)
    assert w[0] / w[-1] == pytest.approx(0.5)  # 1/√4 vs 1/√1


def test_inner_folds_keep_groups_together_and_are_deterministic() -> None:
    _, _, groups = _synthetic()
    f1, f2 = bl.inner_folds(groups), bl.inner_folds(groups)
    assert (f1 == f2).all()
    df = pd.DataFrame({"g": groups, "f": f1})
    assert (df.groupby("g")["f"].nunique() == 1).all()
    assert set(f1) == set(range(bl.INNER_FOLDS))


def test_prior_head_predicts_mode() -> None:
    y = np.array([3.5, 3.5, 3.5, 4.0, 3.25])
    p = bl.PriorHead(GRID).fit(np.zeros((5, 2)), y, np.ones(5)).predict_proba(np.zeros((3, 2)))
    assert np.allclose(p.sum(1), 1)
    assert (np.asarray(GRID)[p.argmax(1)] == 3.5).all()


@pytest.mark.parametrize("name", ["ridge", "ordinal"])
def test_linear_heads_learn_signal(name: str) -> None:
    x, y, groups = _synthetic()
    folds = bl.inner_folds(groups)
    tr, te = folds != 0, folds == 0
    w = bl.group_weights([g for g, t in zip(groups, tr, strict=True) if t])
    head = bl.make_head(name, GRID).fit(x[tr], y[tr], w)
    p = head.predict_proba(x[te])
    assert p.shape == (int(te.sum()), len(GRID))
    assert np.allclose(p.sum(1), 1) and (p >= 0).all()
    dummy = bl.PriorHead(GRID).fit(x[tr], y[tr], w).predict_proba(x[te])
    assert bl._mae_argmax(p, y[te], GRID) < 0.5 * bl._mae_argmax(dummy, y[te], GRID)
    # la esperanza crece con el BCS real
    exp = p @ np.asarray(GRID)
    assert np.corrcoef(exp, y[te])[0, 1] > 0.8


def test_ordinal_thresholds_increasing() -> None:
    x, y, groups = _synthetic()
    head = bl.OrdinalHead(GRID).fit(x, y, bl.group_weights(groups))
    assert (np.diff(head.theta) > 0).all()


def test_select_and_fit_searches_inside_train() -> None:
    x, y, groups = _synthetic()
    head, info = bl.select_and_fit("ridge", GRID, x, None, y, groups, 0.0)
    assert [r["alpha"] for r in info["search"]] == list(bl.RIDGE_ALPHAS)
    assert info["params"]["alpha"] in bl.RIDGE_ALPHAS and info["params"]["sigma"] > 0
    assert isinstance(head, bl.RidgeHead)
    _, fixed = bl.select_and_fit("ordinal", GRID, x, None, y, groups, 0.0, {"lam": 0.01})
    assert fixed["search"] == [] and fixed["params"] == {"lam": 0.01}


def test_training_set_splits_weight_between_original_and_desaturated() -> None:
    x, y, w = np.ones((3, 2)), np.array([3.5, 3.75, 4.0]), np.array([1.0, 2.0, 3.0])
    xs, ys, ws = bl.training_set(x, 2 * x, y, w, 0.5)
    assert len(xs) == 6 and (ys[:3] == ys[3:]).all()
    assert ws.sum() == pytest.approx(w.sum()) and (ws[:3] == ws[3:]).all()
    assert bl.training_set(x, 2 * x, y, w, 0.0)[0] is x


def _painted_image() -> Image.Image:
    arr = np.full((20, 20, 3), 120, dtype=np.uint8)
    arr[5:10, 5:10] = (220, 30, 20)  # rojo-naranja saturado
    arr[12:15, 12:15] = (30, 30, 200)  # azul: no se toca
    return Image.fromarray(arr)


def test_desaturate_red_only_touches_red_pixels(tmp_path: Path) -> None:
    im = _painted_image()
    mask = red_mask(im)
    assert mask.sum() == 25
    path = tmp_path / "p.png"
    im.save(path)
    assert mask.mean() == pytest.approx(paint_fractions(path)[0])  # misma definición que T06
    out = np.asarray(desaturate_red(im)).astype(int)
    before = np.asarray(im).astype(int)
    assert not red_mask(Image.fromarray(out.astype(np.uint8))).any()
    red_px = out[mask]
    assert (red_px.max(1) - red_px.min(1) <= 2).all()  # gris
    assert np.abs(out[~mask] - before[~mask]).max() <= 2  # el resto queda igual


def _fake_embed(images: Sequence[Image.Image]) -> np.ndarray:
    """Embedder falso: color medio + dispersión de cada imagen."""
    out = []
    for im in images:
        a = np.asarray(im.convert("RGB"), dtype=float) / 255
        out.append(np.r_[a.mean((0, 1)), a.std((0, 1))])
    return np.asarray(out, dtype=np.float32)


def test_cache_reuses_original_for_images_without_red(tmp_path: Path) -> None:
    plain = tmp_path / "plain.png"
    Image.fromarray(np.full((8, 8, 3), 90, dtype=np.uint8)).save(plain)
    painted = tmp_path / "painted.png"
    _painted_image().save(painted)
    calls: list[int] = []

    def embed(images: Sequence[Image.Image]) -> np.ndarray:
        calls.append(len(images))
        return _fake_embed(images)

    cache = EmbeddingCache(tmp_path / "emb")
    ids, paths = ["plain", "painted"], [str(plain), str(painted)]
    orig = cache.get(ids, paths, "orig", embed)
    desat = cache.get(ids, paths, "desat", embed)
    assert calls == [2, 1]  # en desat solo se embebe la pintada
    assert np.array_equal(orig[0], desat[0]) and not np.array_equal(orig[1], desat[1])
    again = cache.get(ids, paths, "orig", embed)  # todo del cache
    assert calls == [2, 1] and np.array_equal(again, orig)


def test_cache_recomputes_truncated_shard(tmp_path: Path) -> None:
    paths = []
    for k in range(2):
        p = tmp_path / f"i{k}.png"
        Image.fromarray(np.full((8, 8, 3), 40 * (k + 1), dtype=np.uint8)).save(p)
        paths.append(str(p))
    cache = EmbeddingCache(tmp_path / "emb")
    first = cache.get(["i0"], paths[:1], "orig", _fake_embed)
    (tmp_path / "emb" / "orig" / "00001.npz").write_bytes(b"")  # escritura cortada
    (tmp_path / "emb" / "orig" / "00002.tmp.npz").write_bytes(b"")
    both = cache.get(["i0", "i1"], paths, "orig", _fake_embed)
    assert np.array_equal(both[0], first[0]) and both.shape == (2, 6)
    assert (tmp_path / "emb" / "orig" / "00001.bad").exists()
    assert set(cache.load("orig")) == {"i0", "i1"}


def test_train_m2_config_builds_embedding_scorer() -> None:
    cfg = load_config(config_dir() / "train_m2.yaml")
    scorer = build_scorer(cfg, GRID)
    assert isinstance(scorer, bl.EmbeddingScorer) and isinstance(scorer, Scorer)
    assert scorer.p_desat == 0.5 and scorer.head_name == "ordinal"
    assert scorer.embedder._model is None  # no baja pesos al construirse


def test_scorer_fit_predict_save_load(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BCS_ARTIFACTS_DIR", str(tmp_path / "artifacts"))
    monkeypatch.setattr(Embedder, "embed", lambda self, images: _fake_embed(images))
    rng = np.random.default_rng(0)
    rows = []
    for k in range(30):
        bcs = GRID[k % len(GRID)]
        level = int(40 + 40 * GRID.index(bcs))  # más brillo, más BCS
        for j in range(3):
            path = tmp_path / f"c{k}_{j}.png"
            arr = np.clip(level + rng.normal(0, 8, (12, 12, 3)), 0, 255).astype(np.uint8)
            Image.fromarray(arr).save(path)
            rows.append(
                {"image_id": f"c{k}_{j}", "path": str(path), "group_id": f"g{k}", "bcs": bcs}
            )
    df = pd.DataFrame(rows)
    cfg = {"model": {"head": "ridge"}, "augment": {"red_paint_desaturate": 0.5}}
    scorer = bl.EmbeddingScorer(GRID, cfg)
    scorer.fit(df)
    assert (tmp_path / "artifacts" / "T09" / "emb").exists()
    images = [Image.open(p).convert("RGB") for p in df["path"][:6]]
    p = scorer.predict_proba(images)
    assert p.shape == (6, len(GRID)) and np.allclose(p.sum(1), 1)

    weights = tmp_path / "w.npz"
    scorer.save(weights)
    other = bl.EmbeddingScorer(GRID, {})
    other.load(weights)
    assert other.head_name == "ridge" and other.model_version == scorer.model_version
    assert np.allclose(other.predict_proba(images), p)


def test_paint_report_measures_bias_and_delta() -> None:
    g = np.asarray(GRID)
    y = np.array([3.25, 3.25, 4.0, 4.0])
    p = np.eye(len(g))[[0, 0, 3, 3]].astype(float)
    p_desat = np.eye(len(g))[[1, 0, 3, 3]].astype(float)  # la 1.ª cambia al desaturar
    painted = np.array([True, False, True, False])
    r = bl.paint_report(y, p, p_desat, painted, GRID)
    assert r["painted"]["n"] == 2 and r["painted"]["mae"] == 0.0
    assert r["delta_expected_painted"] == pytest.approx(0.125)
    assert r["pred_changed_painted"] == pytest.approx(0.5)
    assert r["by_class"]["3.25"]["painted"]["n"] == 1
