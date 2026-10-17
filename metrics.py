"""Metricas compartidas para el scoring BCS."""
import numpy as np
import pandas as pd

CLASSES = np.array([3.25, 3.5, 3.75, 4.0, 4.25])


def to_bcs(pred):
    """Valor continuo -> BCS discreto (0.25 mas cercano dentro del rango)."""
    return np.clip(np.round(np.asarray(pred) * 4) / 4, CLASSES[0], CLASSES[-1])


def summarize(y_true, y_pred):
    y_true, raw = np.asarray(y_true), np.asarray(y_pred)
    disc = to_bcs(raw)
    return {
        "n": len(y_true),
        "MAE": float(np.mean(np.abs(raw - y_true))),
        "MAE_disc": float(np.mean(np.abs(disc - y_true))),
        "acc_exact": float(np.mean(np.isclose(disc, y_true))),
        "acc_pm0.25": float(np.mean(np.abs(disc - y_true) <= 0.25 + 1e-6)),
        # cada clase pesa igual: mide si los extremos (3.25 / 4.25) se aciertan
        "acc_balanced": float(np.mean([np.mean(np.isclose(disc[np.isclose(y_true, c)], c))
                                       for c in CLASSES if np.isclose(y_true, c).any()])),
    }


def report(df, pred_col="pred", name="modelo"):
    """df con columnas label, source, video, <pred_col>. Devuelve tabla imagen / video / por fuente."""
    rows = {"imagen": summarize(df["label"], df[pred_col])}
    vid = df.groupby("video").agg(label=("label", "first"), pred=(pred_col, "mean"))
    rows["video (prom. frames)"] = summarize(vid["label"], vid["pred"])
    for src, g in df.groupby("source"):
        rows[f"fuente {src}"] = summarize(g["label"], g[pred_col])
    table = pd.DataFrame(rows).T
    conf = pd.crosstab(df["label"], to_bcs(df[pred_col]), rownames=["real"], colnames=["pred"])
    print(f"\n=== {name} ===\n{table.round(3).to_string()}\n\nConfusion (imagen):\n{conf.to_string()}")
    return table, conf
