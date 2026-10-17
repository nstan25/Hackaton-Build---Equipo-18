"""Detector de vaca.

  python prep_yolo.py eval  [modelo.pt] [n]   -> IoU del detector vs bbox del XML en n imagenes de val
  python prep_yolo.py build [n_train] [n_val] -> arma yolo_data/ (formato YOLO, 1 clase) para entrenar

Entrenar despues con:
  yolo detect train data=yolo_data/data.yaml model=yolo11n.pt imgsz=640 epochs=20 batch=16 project=runs name=cow
"""
import shutil
import sys
from pathlib import Path

import pandas as pd
from PIL import Image

from prep_crops import read_bbox

COCO_COW = 19
SEED = 0


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)


def detect_cow(model, img, conf=0.1):
    """Mejor caja de vaca: si el modelo es COCO usa la clase cow, si no la unica clase. None si no detecta."""
    r = model.predict(img, conf=conf, verbose=False)[0]
    boxes, cls, sc = r.boxes.xyxy.cpu().numpy(), r.boxes.cls.cpu().numpy(), r.boxes.conf.cpu().numpy()
    if len(model.names) > 1:
        keep = cls == COCO_COW
        boxes, sc = boxes[keep], sc[keep]
    if not len(boxes):
        return None
    return boxes[sc.argmax()].tolist()


def sample(df, split, n):
    sub = df[df["split"] == split]
    return sub.groupby("source", group_keys=False).apply(
        lambda g: g.sample(min(len(g), max(1, round(n * len(g) / len(sub)))), random_state=SEED))


def cmd_eval(weights="yolo11n.pt", n=200):
    from ultralytics import YOLO
    model = YOLO(weights)
    df = sample(pd.read_csv("split.csv"), "val", int(n))
    res = []
    for split, path, src in zip(df["split"], df["path"], df["source"]):
        img = Path("dataset_split") / split / path
        det = detect_cow(model, str(img))
        res.append({"source": src, "found": det is not None,
                    "iou": iou(det, read_bbox(img.with_suffix(".xml"))) if det else 0.0})
    r = pd.DataFrame(res)
    print(f"{weights}: detectadas {r['found'].mean():.1%}  IoU media {r['iou'].mean():.3f}  "
          f"IoU>=0.7 {(r['iou'] >= 0.7).mean():.1%}")
    print(r.groupby("source")[["found", "iou"]].mean().round(3).to_string())


def cmd_build(n_train=6000, n_val=1000):
    df = pd.read_csv("split.csv")
    out = Path("yolo_data")
    for split, n in (("train", int(n_train)), ("val", int(n_val))):
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)
        for path in sample(df, split, n)["path"]:
            img = Path("dataset_split") / split / path
            name = img.name
            shutil.copy2(img, out / "images" / split / name)
            x0, y0, x1, y1 = read_bbox(img.with_suffix(".xml"))
            W, H = Image.open(img).size
            (out / "labels" / split / f"{img.stem}.txt").write_text(
                f"0 {(x0 + x1) / 2 / W:.6f} {(y0 + y1) / 2 / H:.6f} {(x1 - x0) / W:.6f} {(y1 - y0) / H:.6f}\n")
    (out / "data.yaml").write_text(f"path: {out.resolve().as_posix()}\ntrain: images/train\nval: images/val\n"
                                   "names:\n  0: cow\n")
    print("yolo_data listo")


if __name__ == "__main__":
    {"eval": cmd_eval, "build": cmd_build}[sys.argv[1]](*sys.argv[2:])
