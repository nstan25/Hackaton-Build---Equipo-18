"""Calibra los umbrales de corte entre clases BCS sobre val, para que los extremos no se vayan al centro.

La regresion comprime las predicciones hacia la media; en vez de redondear al 0.25 mas cercano
(cortes fijos 3.375, 3.625, ...) buscamos 4 cortes que maximicen la exactitud balanceada en val
(cada clase pesa igual, asi 3.25 y 4.25 cuentan tanto como 3.5 y 3.75).

Uso: python calibrate.py [--run efficientnet_b0 | efficientnet_b0_coral]
"""
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")  # conflicto OpenMP de Anaconda en Windows

import argparse
import json

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from metrics import CLASSES
from train_cnn import EVAL_TF, Crops, build_model, predict

DEFAULT_CUTS = (CLASSES[:-1] + CLASSES[1:]) / 2


def apply_cuts(raw, cuts):
    return CLASSES[np.searchsorted(np.sort(cuts), raw)]


def balanced_acc(y, disc):
    return np.mean([np.mean(disc[y == c] == c) for c in CLASSES])


def fit_cuts(y, raw, rounds=4):
    """Busqueda por coordenadas de los 4 cortes, maximizando exactitud balanceada."""
    cuts = DEFAULT_CUTS.copy()
    grid = np.linspace(raw.min(), raw.max(), 400)
    for _ in range(rounds):
        for i in range(len(cuts)):
            lo = cuts[i - 1] if i else -np.inf
            hi = cuts[i + 1] if i < len(cuts) - 1 else np.inf
            cand = grid[(grid > lo) & (grid < hi)]
            scores = [balanced_acc(y, apply_cuts(raw, np.r_[cuts[:i], c, cuts[i + 1:]])) for c in cand]
            cuts[i] = cand[int(np.argmax(scores))]
    return cuts


def describe(name, y, disc):
    rec = {f"recall {c}": np.mean(disc[y == c] == c) for c in CLASSES}
    return {"metodo": name, "exacta": np.mean(disc == y), "balanceada": balanced_acc(y, disc),
            "±0.25": np.mean(np.abs(disc - y) <= 0.25 + 1e-6), "MAE_disc": np.mean(np.abs(disc - y)), **rec}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="efficientnet_b0", help="models/bcs_<run>.pt")
    args = ap.parse_args()

    df = pd.read_csv("crops/manifest.csv")
    va = df[df["split"] == "val"].reset_index(drop=True)
    val_cache = f"results/preds_val_{args.run}.csv"
    if os.path.exists(val_cache):
        va = pd.read_csv(val_cache)
    else:
        ck = torch.load(f"models/bcs_{args.run}.pt")
        loss = ck.get("loss", "huber")
        model = build_model(ck["arch"], pretrained=False, loss=loss)
        model.load_state_dict(ck["state_dict"])
        model.cuda()
        va["pred"] = predict(model, DataLoader(Crops(va, EVAL_TF), 64, num_workers=4), tta=True, loss=loss)
        va.to_csv(val_cache, index=False)

    te = pd.read_csv(f"results/preds_{args.run}.csv")
    cuts = fit_cuts(va["label"].to_numpy(), va["pred"].to_numpy())
    print("cortes por defecto:", DEFAULT_CUTS.round(3), "\ncortes calibrados: ", cuts.round(3))

    rows = []
    for split, d in (("val", va), ("test", te)):
        y, raw = d["label"].to_numpy(), d["pred"].to_numpy()
        rows.append({"split": split, **describe("redondeo", y, apply_cuts(raw, DEFAULT_CUTS))})
        rows.append({"split": split, **describe("calibrado", y, apply_cuts(raw, cuts))})
    table = pd.DataFrame(rows)
    print(table.round(3).to_string(index=False))
    table.to_csv(f"results/calibration_{args.run}.csv", index=False)

    y, disc = te["label"].to_numpy(), apply_cuts(te["pred"].to_numpy(), cuts)
    print("\nConfusion test (calibrado):")
    print(pd.crosstab(pd.Series(y, name="real"), pd.Series(disc, name="pred")).to_string())

    json.dump({"run": args.run, "cuts": cuts.tolist()}, open(f"models/cuts_{args.run}.json", "w"), indent=2)
    print(f"\nGuardado models/cuts_{args.run}.json")


if __name__ == "__main__":
    main()
