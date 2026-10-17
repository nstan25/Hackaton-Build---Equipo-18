"""Baseline A: embeddings DINOv2 congelado + regresor (Ridge / LightGBM).

Uso: python baseline_embed.py [crops/manifest.csv] [vit_small_patch14_dinov2.lvd142m]
"""
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")  # conflicto OpenMP de Anaconda en Windows

import sys
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import timm
import torch
from PIL import Image
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

from metrics import report, summarize

SIZE = 224
TF = T.Compose([T.Resize((SIZE, SIZE)), T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


class Crops(Dataset):
    def __init__(self, paths):
        self.paths = list(paths)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return TF(Image.open(self.paths[i]).convert("RGB"))


def load_backbone(model_id, device):
    return timm.create_model(model_id, pretrained=True, img_size=SIZE, num_classes=0).half().to(device).eval()


def pool(model, x):
    """Embedding = token CLS + promedio de tokens de parche."""
    h = model.forward_features(x)
    return torch.cat([h[:, 0], h[:, model.num_prefix_tokens:].mean(1)], 1)


@torch.no_grad()
def embed(model, paths, device):
    out = []
    for x in DataLoader(Crops(paths), batch_size=64, num_workers=0):
        out.append(pool(model, x.to(device, torch.float16)).float().cpu())
    return torch.cat(out).numpy()


def main():
    manifest = sys.argv[1] if len(sys.argv) > 1 else "crops/manifest.csv"
    model_id = sys.argv[2] if len(sys.argv) > 2 else "vit_small_patch14_dinov2.lvd142m"
    tag = model_id.split(".")[0]
    Path("results").mkdir(exist_ok=True)
    Path("models").mkdir(exist_ok=True)
    df = pd.read_csv(manifest)

    cache = Path(f"results/emb_{tag}.npy")
    if cache.exists():
        X = np.load(cache)
    else:
        device = "cuda"
        X = embed(load_backbone(model_id, device), df["crop"], device)
        np.save(cache, X)
    print("embeddings", X.shape, flush=True)

    y = df["label"].to_numpy()
    idx = {s: (df["split"] == s).to_numpy() for s in ("train", "val", "test")}

    heads = {
        f"ridge_a{a}": make_pipeline(StandardScaler(), Ridge(alpha=a)) for a in (1, 10, 100, 1000)
    }
    heads["lgbm"] = lgb.LGBMRegressor(n_estimators=2000, learning_rate=0.03, num_leaves=63,
                                      subsample=0.8, subsample_freq=1, colsample_bytree=0.5, verbose=-1)
    best, best_mae = None, 1e9
    for name, h in heads.items():
        if name == "lgbm":
            h.fit(X[idx["train"]], y[idx["train"]], eval_set=[(X[idx["val"]], y[idx["val"]])],
                  callbacks=[lgb.early_stopping(100, verbose=False)])
        else:
            h.fit(X[idx["train"]], y[idx["train"]])
        m = summarize(y[idx["val"]], h.predict(X[idx["val"]]))
        print(f"{name:12s} val MAE={m['MAE']:.4f} exact={m['acc_exact']:.3f} +-0.25={m['acc_pm0.25']:.3f}")
        if m["MAE"] < best_mae:
            best, best_mae = name, m["MAE"]

    head = heads[best]
    print("mejor cabeza:", best)
    joblib.dump({"model_id": model_id, "head": head}, f"models/baseline_{tag}.joblib")
    test = df[idx["test"]].copy()
    test["pred"] = head.predict(X[idx["test"]])
    test.to_csv(f"results/preds_baseline_{tag}.csv", index=False)
    table, conf = report(test, name=f"Baseline {tag} + {best} (test)")
    table.to_csv(f"results/metrics_baseline_{tag}.csv")


if __name__ == "__main__":
    main()
