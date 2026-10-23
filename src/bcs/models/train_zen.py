"""T25 (experimento): fine-tuning en CPU de la EfficientNet-B0 de producción con `sdb_bcs` +
`zen_rgbd`, grilla `dairy_zen` (3.0–5.0). No toca producción: todo sale a `$BCS_ARTIFACTS_DIR/T25/`.

Arranca de `models/bcs_efficientnet_b0.pt` con la misma arquitectura y regresión que
`train_cnn.py` (se importa `build_model`, `EVAL_TF` y `decode` sin editarlos): 1 salida =
(bcs - center) / 0.25. Entrada con letterbox a 224×320 (los recortes de zen son verticales).

Datos: solo train para entrenar y val para elegir y evaluar (nunca test ni `holdout_labels_*`).
`sdb_bcs` train se submuestrea a ≤ 5 frames por grupo; mezcla ~50/50 por fuente por época con
muestreo pesado. Dos variantes de zen: A sin la fecha 06.12.2024 (toma diagonal), B con todas y
06.12 con peso bajo (`config/train_zen.yaml`).

Checkpoint cada `ckpt_every_steps` pasos y al fin de cada época (`last.pt`); `train` reanuda
solo desde ahí. El índice de muestras de cada época sale de una semilla fija por época, así que
reanudar a mitad de época sigue con las mismas muestras.

    python -m bcs.models.train_zen manifest
    python -m bcs.models.train_zen train --variant A [--max-steps N] [--epochs E]
    python -m bcs.models.train_zen predict --model prod|A|B
    python -m bcs.models.train_zen report
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Sampler
from torchvision import transforms as T

from bcs.models.efficientnet import team_module
from bcs.paths import REPO_ROOT, artifacts_dir, config_dir, data_dir, runs_dir
from bcs.scale import load_scale

TASK = "T25"
SOURCES = ("sdb_bcs", "zen_rgbd")
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
PROD = "prod"
CONTRAST_DATE = "2025-03-27"
TAILS = (3.0, 4.5, 4.75, 5.0)


def load_config(path: Path | None = None) -> dict[str, Any]:
    return yaml.safe_load((path or config_dir() / "train_zen.yaml").read_text())


def out_dir() -> Path:
    return artifacts_dir() / TASK


def report_dir() -> Path:
    return runs_dir() / TASK


# ---------------------------------------------------------------- datos


def _even_take(df: pd.DataFrame, k: int) -> pd.DataFrame:
    """k filas repartidas parejo a lo largo de la secuencia (el grupo ya viene ordenado)."""
    if len(df) <= k:
        return df
    idx = np.unique(np.linspace(0, len(df) - 1, k).round().astype(int))
    return df.iloc[idx]


def _frame_no(image_id: str) -> int:
    tail = image_id.rsplit("_", 1)[-1]
    return int(tail) if tail.isdigit() else 0


def build_manifests(
    cfg: dict[str, Any], root: Path | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(train, val) de las dos fuentes. Lee solo `labels_<src>.csv` + `splits_<src>.csv`; las filas
    que no son train ni val se descartan (en `labels_*` no hay test, pero por las dudas)."""
    root = root or data_dir()
    grid = load_scale(cfg["scale"]).grid
    parts = []
    for src in SOURCES:
        labels = pd.read_csv(root / f"labels_{src}.csv", dtype={"date": str})
        splits = pd.read_csv(root / f"splits_{src}.csv")
        df = labels.merge(splits, on="image_id", how="inner")
        df = df.loc[df["split"].isin(["train", "val"])].copy()
        df["path"] = [str(root / p) for p in df["path"]]
        df["date"] = df["date"].fillna("")
        parts.append(df)
    df = pd.concat(parts, ignore_index=True)
    df["y"] = [min(grid, key=lambda g, v=v: abs(g - v)) for v in df["bcs"].astype(float)]
    leak = set(df.loc[df["split"] == "train", "group_id"]) & set(
        df.loc[df["split"] == "val", "group_id"]
    )
    if leak:
        raise ValueError(f"group_id en train y val: {sorted(leak)[:5]}")

    train = df.loc[df["split"] == "train"].copy()
    train["frame"] = [_frame_no(i) for i in train["image_id"]]
    train = train.sort_values(["source", "group_id", "frame", "image_id"])
    k = int(cfg["data"]["sdb_max_per_group"])
    sdb = [_even_take(g, k) for _, g in train[train["source"] == "sdb_bcs"].groupby("group_id")]
    train = pd.concat([*sdb, train[train["source"] != "sdb_bcs"]]).drop(columns="frame")
    val = df.loc[df["split"] == "val"].copy()
    return pd.DataFrame(train).reset_index(drop=True), pd.DataFrame(val).reset_index(drop=True)


def selection_rows(val: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    """Val para elegir época: `sdb_bcs` a ≤ k frames por grupo (es 15× más grande) + todo zen."""
    k = int(cfg["data"]["select_sdb_max_per_group"])
    v = val.assign(frame=[_frame_no(i) for i in val["image_id"]])
    v = v.sort_values(["source", "group_id", "frame", "image_id"])
    sdb = [_even_take(g, k) for _, g in v.loc[v["source"] == "sdb_bcs"].groupby("group_id")]
    out = pd.concat([*sdb, v[v["source"] != "sdb_bcs"]]).drop(columns="frame")
    return pd.DataFrame(out).reset_index(drop=True)


def variant_rows(train: pd.DataFrame, cfg: dict[str, Any], variant: str) -> pd.DataFrame:
    """Filas de train de la variante, con la columna `w` (peso de muestreo).

    Dentro de cada fuente: w ∝ peso_fecha / n_clase^power; después cada fuente suma su parte de
    `source_mix`."""
    v = cfg["variants"][variant]
    df = train[~train["date"].isin(v.get("drop_dates") or [])].copy()
    dw = v.get("date_weight") or {}
    w = np.array([float(dw.get(d, 1.0)) for d in df["date"]])
    out = np.zeros(len(df))
    for src, share in cfg["data"]["source_mix"].items():
        m = (df["source"] == src).to_numpy()
        if not m.any():
            continue
        power = float(cfg["data"]["class_balance_power"].get(src, 1.0))
        counts = df.loc[m, "y"].map(df.loc[m, "y"].value_counts()).to_numpy(dtype=float)
        ws = w[m] / counts**power
        out[m] = float(share) * ws / ws.sum()
    df["w"] = out / out.sum()
    return pd.DataFrame(df).reset_index(drop=True)


def write_manifests(cfg: dict[str, Any]) -> dict[str, Any]:
    train, val = build_manifests(cfg)
    d = out_dir()
    d.mkdir(parents=True, exist_ok=True)
    val.to_csv(d / "manifest_val.csv", index=False)
    summary: dict[str, Any] = {"val": _counts(val)}
    for name in cfg["variants"]:
        rows = variant_rows(train, cfg, name)
        rows.to_csv(d / f"manifest_train_{name}.csv", index=False)
        summary[f"train_{name}"] = _counts(rows) | {
            "w_by_source": {
                str(k): round(float(g["w"].to_numpy(dtype=float).sum()), 4)
                for k, g in rows.groupby("source")
            }
        }
    (d / "manifest_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def _counts(df: pd.DataFrame) -> dict[str, Any]:
    return {
        str(src): {"images": len(g), "groups": len(set(g["group_id"]))}
        for src, g in df.groupby("source")
    }


# ---------------------------------------------------------------- transformaciones


class Letterbox:
    """Escala para que entre en (alto, ancho) sin deformar y rellena con el gris de la media."""

    def __init__(self, size: Sequence[int]) -> None:
        self.h, self.w = int(size[0]), int(size[1])
        self.fill = tuple(round(255 * m) for m in MEAN)

    def __call__(self, im: Image.Image) -> Image.Image:
        r = min(self.h / im.height, self.w / im.width)
        nw, nh = max(1, round(im.width * r)), max(1, round(im.height * r))
        canvas = Image.new("RGB", (self.w, self.h), self.fill)
        canvas.paste(
            im.resize((nw, nh), Image.Resampling.BILINEAR), ((self.w - nw) // 2, (self.h - nh) // 2)
        )
        return canvas


def make_transforms(cfg: dict[str, Any]) -> tuple[Callable[..., Any], Callable[..., Any]]:
    a, size = cfg["augment"], cfg["size"]
    aff, cj = a["affine"], a["color_jitter"]
    train_tf = T.Compose(
        [
            Letterbox(size),
            T.RandomHorizontalFlip(0.5 if a.get("hflip", True) else 0.0),
            T.RandomAffine(
                degrees=aff["degrees"],
                translate=(aff["translate"], aff["translate"]),
                scale=tuple(aff["scale"]),
                fill=cast(Any, list(Letterbox(size).fill)),
            ),
            T.ColorJitter(cj["brightness"], cj["contrast"], cj["saturation"], cj["hue"]),
            T.RandomGrayscale(a["grayscale_p"]),
            T.ToTensor(),
            T.Normalize(MEAN, STD),
        ]
    )
    eval_tf = T.Compose([Letterbox(size), T.ToTensor(), T.Normalize(MEAN, STD)])
    return train_tf, eval_tf


class Rows(Dataset):
    def __init__(self, paths: Sequence[str], target: np.ndarray, tf: Callable[..., Any]) -> None:
        self.paths, self.target, self.tf = list(paths), target.astype(np.float32), tf

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, float]:
        with Image.open(self.paths[i]) as im:
            return self.tf(im.convert("RGB")), float(self.target[i])


class OffsetSampler(Sampler[int]):
    """Los índices de la época a partir de `start` (para reanudar a mitad de época)."""

    def __init__(self, indices: Sequence[int], start: int = 0) -> None:
        self.indices = list(indices)[start:]

    def __iter__(self) -> Iterator[int]:
        return iter(self.indices)

    def __len__(self) -> int:
        return len(self.indices)


def epoch_indices(weights: np.ndarray, n: int, seed: int, epoch: int) -> list[int]:
    g = torch.Generator().manual_seed(seed * 1000 + epoch)
    w = torch.tensor(np.array(weights, dtype=float))
    return torch.multinomial(w, n, replacement=True, generator=g).tolist()


# ---------------------------------------------------------------- modelo


def build_init_model(cfg: dict[str, Any]) -> torch.nn.Module:
    train_cnn = team_module("train_cnn")
    model = train_cnn.build_model(cfg["arch"], pretrained=False, loss="huber")
    if cfg.get("init_weights"):
        ck = torch.load(REPO_ROOT / cfg["init_weights"], map_location="cpu")
        model.load_state_dict(ck["state_dict"])
    return model


def load_candidate(path: Path) -> tuple[torch.nn.Module, dict[str, Any]]:
    ck = torch.load(path, map_location="cpu")
    model = team_module("train_cnn").build_model(ck["arch"], pretrained=False, loss="huber")
    model.load_state_dict(ck["state_dict"])
    return model.eval(), {k: v for k, v in ck.items() if k != "state_dict"}


@torch.inference_mode()
def predict_values(
    model: torch.nn.Module,
    paths: Sequence[str],
    tf: Callable[..., Any],
    center: float,
    step: float,
    tta: bool = True,
    batch_size: int = 64,
    num_workers: int = 3,
) -> np.ndarray:
    """BCS continuo; con `tta`, promedio con el flip horizontal (como producción)."""
    model.eval()
    loader = DataLoader(Rows(paths, np.zeros(len(paths)), tf), batch_size, num_workers=num_workers)
    out = []
    for x, _ in loader:
        p = model(x).squeeze(1)
        if tta:
            p = (p + model(torch.flip(x, dims=[3])).squeeze(1)) / 2
        out.append((p * step + center).numpy())
    return np.concatenate(out).astype(float) if out else np.zeros(0)


def _lr(step: int, total: int, warm: int, base: float) -> float:
    if step < warm:
        return base * (step + 1) / warm
    return base * 0.5 * (1 + math.cos(math.pi * (step - warm) / max(1, total - warm)))


def _val_score(val: pd.DataFrame, pred: np.ndarray) -> dict[str, float]:
    err = np.abs(pred - val["y"].to_numpy(dtype=float))
    res = {f"mae_{s}": float(err[(val["source"] == s).to_numpy()].mean()) for s in SOURCES}
    res = {k: v for k, v in res.items() if not math.isnan(v)}
    # selección: promedio de las MAE por fuente (cada fuente pesa igual, zen es 15× más chica)
    res["score"] = float(np.mean(list(res.values())))
    return res


def train(
    cfg: dict[str, Any],
    variant: str,
    *,
    train_rows: pd.DataFrame,
    val_rows: pd.DataFrame,
    dest: Path,
    max_steps: int | None = None,
    log: Callable[[str], None] = print,
) -> dict[str, Any]:
    """Entrena (o reanuda) la variante. Escribe `last.pt`, `best.pt` y `log.csv` en `dest`."""
    t = cfg["train"]
    seed = int(cfg["seed"])
    torch.set_num_threads(int(t["torch_threads"]))
    dest.mkdir(parents=True, exist_ok=True)
    center, step = float(cfg["center"]), float(cfg["step"])
    train_tf, eval_tf = make_transforms(cfg)
    bs, n_ep = int(t["batch_size"]), int(t["epochs"])
    steps_per_epoch = int(t["epoch_samples"]) // bs
    total = n_ep * steps_per_epoch

    model = build_init_model(cfg)
    opt = torch.optim.AdamW(
        model.parameters(), lr=float(t["lr"]), weight_decay=float(t["weight_decay"])
    )
    st: dict[str, Any] = {
        "epoch": 0,
        "batch": 0,
        "best": math.inf,
        "bad": 0,
        "done": False,
        "log": [],
    }
    last = dest / "last.pt"
    if last.exists():
        ck = torch.load(last, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        st = ck["state"]
        log(f"reanudo {variant}: época {st['epoch'] + 1}, lote {st['batch']}/{steps_per_epoch}")
    if st["done"]:
        log(f"{variant}: ya terminó")
        return st

    def save_last() -> None:
        tmp = last.with_suffix(".tmp")
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "state": st}, tmp)
        tmp.replace(last)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    ds = Rows(train_rows["path"].tolist(), ((train_rows["y"] - center) / step).to_numpy(), train_tf)
    weights = train_rows["w"].to_numpy(dtype=float)
    steps_run = 0
    while st["epoch"] < n_ep:
        ep = st["epoch"] + 1
        idx = epoch_indices(weights, steps_per_epoch * bs, seed, ep)
        loader = DataLoader(
            ds,
            bs,
            sampler=OffsetSampler(idx, st["batch"] * bs),
            num_workers=int(t["num_workers"]),
            drop_last=True,
            generator=torch.Generator().manual_seed(seed * 1000 + ep * 100 + st["batch"]),
        )
        model.train()
        t0, run, n = time.time(), 0.0, 0
        for x, y in loader:
            g = (ep - 1) * steps_per_epoch + st["batch"]
            for pg in opt.param_groups:
                pg["lr"] = _lr(g, total, int(t["warmup_steps"]), float(t["lr"]))
            out = model(x).squeeze(1)
            loss = F.huber_loss(out, y.float(), delta=float(t["huber_delta"]))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            run, n = run + loss.item(), n + 1
            st["batch"] += 1
            steps_run += 1
            if st["batch"] % 50 == 0:
                rate = n * bs / (time.time() - t0)
                log(
                    f"  {variant} ep{ep} {st['batch']}/{steps_per_epoch} "
                    f"loss={run / n:.4f} {rate:.1f} img/s"
                )
            if st["batch"] % int(t["ckpt_every_steps"]) == 0:
                save_last()
            if max_steps is not None and steps_run >= max_steps:
                save_last()
                log(
                    f"{variant}: corte por --max-steps ({steps_run} pasos, {time.time() - t0:.0f}s)"
                )
                return st
        train_sec = time.time() - t0
        t1 = time.time()
        pred = predict_values(
            model,
            val_rows["path"].tolist(),
            eval_tf,
            center,
            step,
            tta=False,
            batch_size=int(t["val_batch_size"]),
            num_workers=int(t["num_workers"]),
        )
        m = _val_score(val_rows, pred)
        row = {
            "epoch": ep,
            "train_loss": run / max(n, 1),
            **m,
            "train_sec": train_sec,
            "val_sec": time.time() - t1,
            "lr_end": opt.param_groups[0]["lr"],
        }
        st["log"].append(row)
        pd.DataFrame(st["log"]).to_csv(dest / "log.csv", index=False)
        log(f"{variant} ep{ep} " + " ".join(f"{k}={v:.4f}" for k, v in row.items() if k != "epoch"))
        if m["score"] < st["best"]:
            st["best"], st["bad"] = m["score"], 0
            torch.save(
                {
                    "arch": cfg["arch"],
                    "loss": "huber",
                    "state_dict": model.state_dict(),
                    "size": tuple(cfg["size"]),
                    "center": center,
                    "scale": step,
                    "letterbox": True,
                    "grid": list(load_scale(cfg["scale"]).grid),
                    "variant": variant,
                    "epoch": ep,
                    "val": m,
                },
                dest / "best.pt",
            )
        else:
            st["bad"] += 1
        st["epoch"], st["batch"] = ep, 0
        if st["bad"] >= int(t["patience"]):
            log(f"{variant}: early stopping en la época {ep}")
            st["done"] = True
        save_last()
        if st["done"]:
            break
    st["done"] = True
    save_last()
    return st


# ---------------------------------------------------------------- evaluación


def red_coat_fraction(im: Image.Image) -> float:
    """Heurística de pelaje colorado: fracción de píxeles rojizo-marrones (tono 0–25° o ≥ 340°,
    saturación ≥ 0,30, valor 0,12–0,90) en el centro del recorte (40% del ancho, del 20% al 80% del
    alto), donde está la grupa y no el brete ni las vecinas."""
    w, h = im.size
    c = im.convert("RGB").crop((int(0.3 * w), int(0.2 * h), int(0.7 * w), int(0.8 * h)))
    hsv = np.asarray(c.convert("HSV"), dtype=float)
    hue = hsv[..., 0] * 360 / 255
    s, v = hsv[..., 1] / 255, hsv[..., 2] / 255
    m = ((hue <= 25) | (hue >= 340)) & (s >= 0.30) & (v >= 0.12) & (v <= 0.90)
    return float(m.mean()) if m.size else 0.0


def metrics(
    y: np.ndarray, pred: np.ndarray, groups: np.ndarray, grid: Sequence[float]
) -> dict[str, Any]:
    """MAE del valor continuo; exacta / ±1 paso de la clase redondeada a la grilla; sesgo."""
    y, pred = np.asarray(y, dtype=float), np.asarray(pred, dtype=float)
    if len(y) == 0:
        return {"n": 0, "animals": 0}
    g = np.asarray(grid, dtype=float)
    cls = g[np.abs(pred[:, None] - g[None, :]).argmin(1)]
    step = float(g[1] - g[0]) if len(g) > 1 else 1.0
    return {
        "n": int(len(y)),
        "animals": int(len(set(groups))),
        "mae": float(np.abs(pred - y).mean()),
        "exact": float(np.isclose(cls, y).mean()),
        "pm1": float((np.abs(cls - y) <= step + 1e-6).mean()),
        "bias": float((pred - y).mean()),
        "mean_pred": float(pred.mean()),
    }


def predictions_path(model: str) -> Path:
    return out_dir() / f"preds_val_{model}.csv"


def run_predict(cfg: dict[str, Any], model_name: str) -> Path:
    val = pd.read_csv(out_dir() / "manifest_val.csv", dtype={"date": str}).fillna({"date": ""})
    torch.set_num_threads(int(cfg["train"]["torch_threads"]))
    t0 = time.time()
    if model_name == PROD:
        # producción tal cual: su transformación (estira a 224×320), su decodificación, TTA flip
        train_cnn = team_module("train_cnn")
        model = build_init_model(cfg | {"init_weights": "models/bcs_efficientnet_b0.pt"})
        pred = predict_values(model, val["path"].tolist(), train_cnn.EVAL_TF, 3.75, 0.25)
    else:
        model, meta = load_candidate(out_dir() / model_name / "best.pt")
        _, eval_tf = make_transforms(cfg)
        pred = predict_values(model, val["path"].tolist(), eval_tf, meta["center"], meta["scale"])
    out = val[["image_id", "source", "group_id", "date", "bcs", "y"]].copy()
    out["pred"] = pred
    dest = predictions_path(model_name)
    out.to_csv(dest, index=False)
    print(f"{model_name}: {len(out)} predicciones en {time.time() - t0:.0f}s → {dest}")
    return dest


def coat_table(val: pd.DataFrame, threshold: float) -> pd.DataFrame:
    cache = out_dir() / "coat_val.csv"
    if cache.exists():
        coat = pd.read_csv(cache)
    else:
        fr = []
        for p in val["path"]:
            with Image.open(p) as im:
                fr.append(red_coat_fraction(im))
        coat = pd.DataFrame({"image_id": val["image_id"], "red_frac": fr})
        coat.to_csv(cache, index=False)
    coat["red"] = coat["red_frac"] > threshold
    return coat


def build_report(cfg: dict[str, Any], models: Sequence[str]) -> dict[str, Any]:
    """Tablas (json) de actual vs. candidatos sobre las mismas imágenes de val."""
    grid = load_scale(cfg["scale"]).grid
    val = pd.read_csv(out_dir() / "manifest_val.csv", dtype={"date": str}).fillna({"date": ""})
    coat = coat_table(val, float(cfg["coat"]["threshold"]))
    preds = {}
    for m in models:
        p = pd.read_csv(predictions_path(m), dtype={"date": str}).fillna({"date": ""})
        preds[m] = p.merge(coat, on="image_id", how="left")

    def mt(df: pd.DataFrame) -> dict[str, Any]:
        return metrics(df["y"].to_numpy(), df["pred"].to_numpy(), df["group_id"].to_numpy(), grid)

    rep: dict[str, Any] = {
        "grid": list(grid),
        "models": list(models),
        "by_source": {},
        "by_class": {},
        "by_date": {},
        "contrast_2703": {},
        "coat_4.0-4.25": {},
        "sdb_animal": {},
    }
    for m, df in preds.items():
        rep["by_source"][m] = {s: mt(g) for s, g in df.groupby("source")} | {"all": mt(df)}
        rep["by_class"][m] = {
            s: {f"{c:.2f}": mt(g2) for c, g2 in g.groupby("y")} for s, g in df.groupby("source")
        }
        zen = df[df["source"] == "zen_rgbd"]
        rep["by_date"][m] = {d: mt(g) for d, g in zen.groupby("date")}
        d27 = zen[zen["date"] == CONTRAST_DATE]
        hi, mid = d27[d27["y"] >= 4.5], d27[d27["y"].isin([4.0, 4.25])]
        rep["contrast_2703"][m] = _contrast(hi, mid)
        band = df[df["y"].isin([4.0, 4.25])]
        rep["coat_4.0-4.25"][m] = {
            s: {("colorada" if r else "no colorada"): mt(g2) for r, g2 in g.groupby("red")}
            for s, g in band.groupby("source")
        }
        sdb = (
            df[df["source"] == "sdb_bcs"]
            .groupby("group_id")
            .agg(y=("y", "first"), pred=("pred", "mean"))
        )
        rep["sdb_animal"][m] = metrics(
            sdb["y"].to_numpy(), sdb["pred"].to_numpy(), sdb.index.to_numpy(), grid
        )
    rep["coat_counts"] = {
        s: {f"{c:.2f}": float(np.mean(g2["red"].to_numpy(dtype=float))) for c, g2 in g.groupby("y")}
        for s, g in val.merge(coat, on="image_id").groupby("source")
    }
    return rep


def _contrast(hi: pd.DataFrame, mid: pd.DataFrame) -> dict[str, Any]:
    """≥ 4.5 vs. 4.0–4.25 dentro de una misma fecha: diferencia de medias predicha vs. real y AUC
    (probabilidad de que una imagen ≥ 4.5 tenga predicción mayor que una de 4.0–4.25)."""
    if len(hi) == 0 or len(mid) == 0:
        return {}
    a, b = hi["pred"].to_numpy(dtype=float), mid["pred"].to_numpy(dtype=float)
    auc = float((a[:, None] > b[None, :]).mean() + 0.5 * (a[:, None] == b[None, :]).mean())
    return {
        "n_hi": int(len(a)),
        "animals_hi": len(set(hi["group_id"])),
        "n_mid": int(len(b)),
        "animals_mid": len(set(mid["group_id"])),
        "true_delta": float(hi["y"].mean() - mid["y"].mean()),
        "pred_delta": float(a.mean() - b.mean()),
        "auc": auc,
    }


# ---------------------------------------------------------------- CLI


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m bcs.models.train_zen")
    ap.add_argument("--config", type=Path, default=None)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("manifest")
    tr = sub.add_parser("train")
    tr.add_argument("--variant", required=True)
    tr.add_argument(
        "--max-steps", type=int, default=None, help="corta tras N pasos (estimar tiempos)"
    )
    tr.add_argument("--epochs", type=int, default=None)
    tr.add_argument("--dest", default=None, help="subcarpeta de T25/ (por defecto, la variante)")
    pr = sub.add_parser("predict")
    pr.add_argument("--model", required=True, help="prod o el nombre de la subcarpeta (A, B)")
    rp = sub.add_parser("report")
    rp.add_argument("--models", default="prod,A,B")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)

    if args.cmd == "manifest":
        print(json.dumps(write_manifests(cfg), indent=2))
    elif args.cmd == "train":
        if args.epochs is not None:
            cfg["train"]["epochs"] = args.epochs
        d = out_dir()
        rows = pd.read_csv(d / f"manifest_train_{args.variant}.csv", dtype={"date": str}).fillna(
            {"date": ""}
        )
        val = pd.read_csv(d / "manifest_val.csv", dtype={"date": str}).fillna({"date": ""})
        st = train(
            cfg,
            args.variant,
            train_rows=rows,
            val_rows=selection_rows(val, cfg),
            dest=d / (args.dest or args.variant),
            max_steps=args.max_steps,
            log=lambda s: print(s, flush=True),
        )
        print(json.dumps({k: v for k, v in st.items() if k != "log"}, default=str))
    elif args.cmd == "predict":
        run_predict(cfg, args.model)
    elif args.cmd == "report":
        rep = build_report(cfg, args.models.split(","))
        dest = report_dir()
        dest.mkdir(parents=True, exist_ok=True)
        (dest / "metrics.json").write_text(json.dumps(rep, indent=2))
        print(f"→ {dest / 'metrics.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
