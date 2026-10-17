"""Demo Gradio: subir una foto de la vaca y obtener su BCS.

Uso: python app.py   -> http://127.0.0.1:7860
"""
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")  # conflicto OpenMP de Anaconda en Windows

import random

import gradio as gr
import pandas as pd
from PIL import ImageDraw

from bcs_pipeline import CNN_WEIGHTS, BCSScorer

# BCS_CNN=models/bcs_<run>.pt elige el modelo; si existe models/cuts_<run>.json se usan los cortes calibrados
scorer = BCSScorer(os.environ.get("BCS_CNN", CNN_WEIGHTS))


def run(img):
    if img is None:
        return None, "Subí una imagen."
    r = scorer.predict(img)
    if r["bbox"] is None:
        return img, "No se detectó ninguna vaca."
    out = img.convert("RGB").copy()
    d = ImageDraw.Draw(out)
    d.rectangle(r["bbox"], outline=(255, 210, 0), width=5)
    d.text((r["bbox"][0] + 8, r["bbox"][1] + 8), f"BCS {r['bcs']:.2f}", fill=(255, 210, 0),
           font_size=36)
    return out, f"## BCS {r['bcs']:.2f}\nValor continuo: {r['bcs_raw']:.3f}  ·  modelo: {scorer.kind}"


def examples():
    df = pd.read_csv("split.csv")
    te = df[df["split"] == "test"]
    rnd = random.Random(1)
    return [f"dataset_split/test/{rnd.choice(g['path'].tolist())}" for _, g in te.groupby("label")]


with gr.Blocks(title="BCS vacas") as demo:
    gr.Markdown("# Body Condition Score\nSubí una foto desde la cámara superior trasera.")
    with gr.Row():
        inp = gr.Image(type="pil", label="Imagen")
        out = gr.Image(type="pil", label="Detección")
    txt = gr.Markdown()
    gr.Button("Calcular BCS", variant="primary").click(run, inp, [out, txt])
    inp.change(run, inp, [out, txt])  # calcula apenas se carga o cambia la imagen
    gr.Examples(examples(), inp, label="Ejemplos del test (uno por clase: 3.25 → 4.25)")

if __name__ == "__main__":
    demo.launch()
