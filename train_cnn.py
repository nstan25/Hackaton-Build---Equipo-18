"""Modelo B: fine-tune de una CNN (timm) para BCS sobre los recortes.

Dos perdidas:
  huber -> regresion: 1 salida = (bcs - 3.75) / 0.25
  coral -> ordinal: 4 salidas = logit de P(bcs > 3.25), P(bcs > 3.5), P(bcs > 3.75), P(bcs > 4.0);
           bcs = 3.25 + 0.25 * suma de probabilidades. No tiene un "centro" al que encogerse.

Uso: python train_cnn.py [--arch efficientnet_b0] [--loss huber|coral] [--extreme-weight 2] [--select mae|balanced]
"""
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")  # conflicto OpenMP de Anaconda en Windows

import argparse
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import timm
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms as T

from metrics import report, summarize

SIZE = (224, 320)  # alto, ancho: los recortes son ~1.5 veces mas anchos que altos
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
CENTER, SCALE = 3.75, 0.25  # el modelo predice (bcs - 3.75) / 0.25

TRAIN_TF = T.Compose([
    T.Resize(SIZE),
    T.RandomHorizontalFlip(),
    T.RandomAffine(degrees=7, translate=(0.04, 0.04), scale=(0.92, 1.08)),
    T.ColorJitter(0.25, 0.25, 0.15, 0.02),
    T.ToTensor(), T.Normalize(MEAN, STD),
])
EVAL_TF = T.Compose([T.Resize(SIZE), T.ToTensor(), T.Normalize(MEAN, STD)])


class Crops(Dataset):
    def __init__(self, df, tf):
        self.paths, self.y, self.tf = df["crop"].tolist(), df["label"].to_numpy(np.float32), tf

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return self.tf(Image.open(self.paths[i]).convert("RGB")), (self.y[i] - CENTER) / SCALE


N_OUT = {"huber": 1, "coral": 4}


def build_model(arch, pretrained=True, loss="huber"):
    return timm.create_model(arch, pretrained=pretrained, num_classes=N_OUT[loss])


def decode(out, loss="huber"):
    """Salida cruda de la red (logits, ya promediados con TTA) -> BCS continuo."""
    out = out.float()
    if loss == "coral":
        return 3.25 + 0.25 * torch.sigmoid(out).sum(1)
    return out.squeeze(1) * SCALE + CENTER


def compute_loss(out, y_norm, loss, extreme_weight):
    """y_norm = (bcs - 3.75) / 0.25 en {-2..2}. Las clases extremas pesan extreme_weight."""
    k = (y_norm + 2).round()  # indice de clase 0..4
    w = torch.where((k == 0) | (k == 4), extreme_weight, 1.0)
    if loss == "coral":
        target = (k[:, None] > torch.arange(4, device=k.device)[None]).float()
        per = F.binary_cross_entropy_with_logits(out.float(), target, reduction="none").mean(1)
    else:
        per = F.huber_loss(out.squeeze(1).float(), y_norm, delta=1.0, reduction="none")
    return (per * w).sum() / w.sum()


@torch.no_grad()
def predict(model, loader, tta=False, loss="huber"):
    model.eval()
    out = []
    for x, _ in loader:
        x = x.cuda(non_blocking=True)
        with torch.autocast("cuda", dtype=torch.float16):
            p = model(x)
            if tta:
                p = (p + model(torch.flip(x, dims=[3]))) / 2
        out.append(decode(p, loss).cpu())
    return torch.cat(out).numpy()


def main():
    torch.backends.cudnn.benchmark = True
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="crops/manifest.csv")
    ap.add_argument("--arch", default="efficientnet_b0")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--loss", choices=list(N_OUT), default="huber")
    ap.add_argument("--extreme-weight", type=float, default=1.0, help="peso de 3.25 y 4.25 en la perdida")
    ap.add_argument("--select", choices=["mae", "balanced"], default="mae",
                    help="metrica de val para elegir el mejor checkpoint")
    args = ap.parse_args()
    run_name = args.arch if args.loss == "huber" else f"{args.arch}_{args.loss}"

    Path("models").mkdir(exist_ok=True)
    Path("results").mkdir(exist_ok=True)
    df = pd.read_csv(args.manifest)
    tr, va, te = (df[df["split"] == s].reset_index(drop=True) for s in ("train", "val", "test"))

    # muestreo balanceado por clase
    w = 1.0 / tr["label"].map(tr["label"].value_counts()).to_numpy()
    sampler = WeightedRandomSampler(torch.as_tensor(w, dtype=torch.double), len(tr), replacement=True)
    kw = dict(num_workers=args.workers, pin_memory=True, persistent_workers=True)
    dl_tr = DataLoader(Crops(tr, TRAIN_TF), args.bs, sampler=sampler, drop_last=True, **kw)
    dl_va = DataLoader(Crops(va, EVAL_TF), args.bs * 2, **kw)

    model = build_model(args.arch, loss=args.loss).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    total, warm = args.epochs * len(dl_tr), len(dl_tr) // 2
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / (total - warm))))
    scaler = torch.amp.GradScaler()

    ckpt = Path(f"models/bcs_{run_name}.pt")
    best, log = 1e9, []
    for ep in range(1, args.epochs + 1):
        model.train()
        t0, run = time.time(), 0.0
        for i, (x, y) in enumerate(dl_tr, 1):
            x = x.cuda(non_blocking=True)
            y = y.cuda(non_blocking=True)
            with torch.autocast("cuda", dtype=torch.float16):
                loss = compute_loss(model(x), y, args.loss, args.extreme_weight)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            run += loss.item()
            if i % 200 == 0:
                print(f"  ep{ep} it{i}/{len(dl_tr)} loss={run / i:.4f}", flush=True)
        m = summarize(va["label"], predict(model, dl_va, loss=args.loss))
        log.append({"epoch": ep, "train_loss": run / len(dl_tr), **{f"val_{k}": v for k, v in m.items()},
                    "sec": time.time() - t0})
        print(f"ep{ep} loss={run / len(dl_tr):.4f} val MAE={m['MAE']:.4f} exact={m['acc_exact']:.3f} "
              f"+-0.25={m['acc_pm0.25']:.3f} bal={m['acc_balanced']:.3f} ({time.time() - t0:.0f}s)", flush=True)
        pd.DataFrame(log).to_csv(f"results/train_log_{run_name}.csv", index=False)
        state = {"arch": args.arch, "loss": args.loss, "state_dict": model.state_dict(), "size": SIZE,
                 "center": CENTER, "scale": SCALE, "epoch": ep, "val": m}
        torch.save(state, ckpt.with_name(f"{ckpt.stem}_last.pt"))  # siempre, por si conviene calibrarlo
        score = m["MAE"] if args.select == "mae" else -m["acc_balanced"]
        if score < best:
            best = score
            torch.save(state, ckpt)

    model.load_state_dict(torch.load(ckpt)["state_dict"])
    te = te.copy()
    te["pred"] = predict(model, DataLoader(Crops(te, EVAL_TF), args.bs * 2, **kw), tta=True, loss=args.loss)
    te.to_csv(f"results/preds_{run_name}.csv", index=False)
    table, _ = report(te, name=f"{run_name} (test, TTA flip)")
    table.to_csv(f"results/metrics_{run_name}.csv")


if __name__ == "__main__":
    main()
