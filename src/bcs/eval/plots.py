"""Gráficos del reporte: reliability diagram de `confidence` y matriz de confusión (PNG)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def plot_reliability(metrics: dict[str, Any], path: Path, title: str = "") -> Path:
    bins = metrics["reliability"]
    fig, (ax, axh) = plt.subplots(
        2, 1, figsize=(5, 6), gridspec_kw={"height_ratios": [3, 1]}, sharex=True
    )
    ax.plot([0, 1], [0, 1], "--", color="0.6", label="calibración perfecta")
    filled = [b for b in bins if b["count"]]
    centers = [(b["lo"] + b["hi"]) / 2 for b in filled]
    width = bins[0]["hi"] - bins[0]["lo"]
    ax.bar(
        centers,
        [b["accuracy"] for b in filled],
        width=width * 0.9,
        color="tab:blue",
        alpha=0.7,
        label="acierto (|error| ≤ 1 paso)",
    )
    ax.plot(
        [b["mean_confidence"] for b in filled],
        [b["accuracy"] for b in filled],
        "o",
        color="tab:red",
        label="confianza media del bin",
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_ylabel("acierto")
    ax.set_title(f"{title}  ECE={metrics['ece']:.3f}  n={metrics['n']}".strip())
    ax.legend(loc="upper left", fontsize=8)
    axh.bar(
        [(b["lo"] + b["hi"]) / 2 for b in bins],
        [b["count"] for b in bins],
        width=width * 0.9,
        color="0.5",
    )
    axh.set_xlabel("confidence")
    axh.set_ylabel("casos")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_confusion(metrics: dict[str, Any], path: Path, title: str = "") -> Path:
    """Muestra solo el rango de la grilla con casos (reales o predichos) para que se lea."""
    labels = metrics["confusion"]["labels"]
    m = np.asarray(metrics["confusion"]["matrix"])
    used = np.flatnonzero(m.sum(axis=0) + m.sum(axis=1))
    lo, hi = (int(used.min()), int(used.max()) + 1) if len(used) else (0, len(labels))
    m = m[lo:hi, lo:hi]
    labels = labels[lo:hi]
    rows = m.sum(axis=1, keepdims=True)
    norm = np.divide(m, rows, out=np.zeros(m.shape), where=rows > 0)

    size = max(4.0, 0.55 * len(labels) + 2)
    fig, ax = plt.subplots(figsize=(size, size))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            if m[i, j]:
                color = "white" if norm[i, j] > 0.5 else "black"
                ax.text(j, i, str(m[i, j]), ha="center", va="center", fontsize=8, color=color)
    ax.set_xticks(range(len(labels)), labels, rotation=45)
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("BCS predicho")
    ax.set_ylabel("BCS real")
    ax.set_title(f"{title}  acc={metrics['accuracy']:.3f}  MAE={metrics['mae']:.3f}".strip())
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="fracción de la fila")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path
