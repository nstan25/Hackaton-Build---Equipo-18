"""Pipeline end-to-end: imagen cruda -> detector de vaca -> recorte -> score BCS.

Uso como script: python bcs_pipeline.py [--cnn models/bcs_X.pt] [--no-calib] imagen.jpg [imagen2.jpg ...]
"""
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")  # conflicto OpenMP de Anaconda en Windows

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from metrics import CLASSES, to_bcs
from prep_crops import MAX_SIDE, expand_bbox

DET_WEIGHTS = ["models/cow_det.pt", "yolo11n.pt"]  # el primero que exista
CNN_WEIGHTS = "models/bcs_efficientnet_b0.pt"
BASELINE = "models/baseline_vit_small_patch14_dinov2.joblib"


def cuts_path(cnn_weights):
    """models/bcs_<run>.pt -> models/cuts_<run>.json (lo genera calibrate.py)."""
    p = Path(cnn_weights)
    return p.with_name(p.stem.replace("bcs_", "cuts_", 1) + ".json")


class BCSScorer:
    def __init__(self, cnn_weights=CNN_WEIGHTS, calibrated=True,
                 device="cuda" if torch.cuda.is_available() else "cpu"):
        from ultralytics import YOLO
        from prep_yolo import detect_cow
        self.device = device
        self.det_path = next(p for p in DET_WEIGHTS if Path(p).exists() or not p.startswith("models/"))
        self.det = YOLO(self.det_path)
        self._detect = detect_cow

        self.cuts = None
        if Path(cnn_weights).exists():
            import train_cnn
            ck = torch.load(cnn_weights, map_location=device)
            self.loss = ck.get("loss", "huber")
            self.model = train_cnn.build_model(ck["arch"], pretrained=False, loss=self.loss)
            self.model.load_state_dict(ck["state_dict"])
            self.model.to(device).eval()
            self._decode = train_cnn.decode
            self.tf, self.kind = train_cnn.EVAL_TF, f"CNN {ck['arch']} ({self.loss})"
            if calibrated and cuts_path(cnn_weights).exists():
                self.cuts = np.array(json.load(open(cuts_path(cnn_weights)))["cuts"])
                self.kind += " + cortes calibrados"
        else:
            import joblib
            import baseline_embed
            b = joblib.load(BASELINE)
            self.head = b["head"]
            self.model = baseline_embed.load_backbone(b["model_id"], device)
            self._pool = baseline_embed.pool
            self.tf, self.kind = baseline_embed.TF, f"baseline {b['model_id']}"

    @torch.no_grad()
    def score_crop(self, crop):
        x = self.tf(crop).unsqueeze(0).to(self.device)
        if self.kind.startswith("CNN"):
            out = (self.model(x) + self.model(torch.flip(x, dims=[3]))) / 2
            return float(self._decode(out, self.loss)[0])
        emb = self._pool(self.model, x.half()).float().cpu().numpy()
        return float(self.head.predict(emb)[0])

    def discretize(self, raw):
        if self.cuts is None:
            return float(to_bcs(raw))
        return float(CLASSES[np.searchsorted(self.cuts, raw)])

    def predict(self, image):
        img = image if isinstance(image, Image.Image) else Image.open(image)
        img = img.convert("RGB")
        bbox = self._detect(self.det, img)
        if bbox is None:
            return {"bbox": None, "bcs": None, "bcs_raw": None}
        box = expand_bbox([int(v) for v in bbox], *img.size)
        crop = img.crop(box)
        crop.thumbnail((MAX_SIDE, MAX_SIDE))
        raw = self.score_crop(crop)
        return {"bbox": [int(v) for v in bbox], "bcs": self.discretize(raw), "bcs_raw": round(raw, 3)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("images", nargs="+")
    ap.add_argument("--cnn", default=CNN_WEIGHTS)
    ap.add_argument("--no-calib", action="store_true")
    args = ap.parse_args()
    s = BCSScorer(args.cnn, calibrated=not args.no_calib)
    print(f"detector={s.det_path}  modelo={s.kind}")
    for p in args.images:
        print(p, s.predict(p))
