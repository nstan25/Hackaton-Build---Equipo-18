"""Embeddings congelados de DINOv2 (SPEC §2.3 paso 2, T09) y su cache en disco.

Backbone: `vit_small_patch14_dinov2.lvd142m` de timm, sin cabeza, en CPU. El vector de cada
imagen es [CLS ‖ media de los tokens de parche] (768 valores). Entrada: la imagen recortada
llevada a 224×224 (bicúbica, sin recortar bordes) y normalizada como ImageNet.

Mitigación del atajo de pintura roja (G1 punto 2): `desaturate_red` lleva a saturación 0 los
píxeles rojo-naranja con la misma definición que T06 (`bcs.data.merge.paint_fractions`). La
variante `desat` del cache es la imagen con la pintura desaturada; las imágenes sin ningún píxel
rojo-naranja quedan iguales y reusan el embedding de `orig`.

Cache: `$BCS_ARTIFACTS_DIR/T09/emb/<backbone>_<size>/<variant>/NNNNN.npz` (`ids`, `emb`), por
tandas, así una extracción cortada sigue donde quedó. Los pesos se bajan una vez a
`$BCS_ARTIFACTS_DIR/T09/hf/hub`.
"""

from __future__ import annotations

import sys
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from bcs.paths import artifacts_dir

BACKBONE = "vit_small_patch14_dinov2.lvd142m"
IMG_SIZE = 224
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
VARIANTS = ("orig", "desat")
BATCH = 64
SHARD = 1024


def task_dir() -> Path:
    return artifacts_dir() / "T09"


def red_mask(im: Image.Image) -> np.ndarray:
    """Píxeles rojo-naranja (HSV de PIL, 0–255): tono < 18 o > 241, saturación > 128, valor > 77.
    Misma definición que `bcs.data.merge.paint_fractions` (T06)."""
    hsv = np.asarray(im.convert("HSV"), dtype=np.uint8)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    return ((h < 18) | (h > 241)) & (s > 128) & (v > 77)


def desaturate_red(im: Image.Image) -> Image.Image:
    """La misma imagen con los píxeles rojo-naranja en saturación 0 (gris del mismo valor)."""
    mask = red_mask(im)
    if not mask.any():
        return im.convert("RGB")
    hsv = np.array(im.convert("HSV"), dtype=np.uint8)
    hsv[..., 1][mask] = 0
    return Image.fromarray(hsv, mode="HSV").convert("RGB")


def to_tensor_batch(images: Sequence[Image.Image], size: int = IMG_SIZE) -> np.ndarray:
    """[n, 3, size, size] float32 normalizado (ImageNet)."""
    out = np.empty((len(images), 3, size, size), dtype=np.float32)
    for i, im in enumerate(images):
        arr = np.asarray(
            im.convert("RGB").resize((size, size), Image.Resampling.BICUBIC), dtype=np.float32
        )
        out[i] = ((arr / 255.0 - MEAN) / STD).transpose(2, 0, 1)
    return out


class Embedder:
    """Backbone congelado. `embed(images)` → [n, dim]. El modelo se carga recién al usarlo."""

    def __init__(self, backbone: str = BACKBONE, img_size: int = IMG_SIZE) -> None:
        self.backbone = backbone
        self.img_size = img_size
        self._model: Any = None

    @property
    def name(self) -> str:
        return f"{self.backbone}_{self.img_size}"

    def _load(self) -> Any:
        if self._model is None:
            import timm
            import torch

            torch.manual_seed(0)
            cache = task_dir() / "hf" / "hub"
            cache.mkdir(parents=True, exist_ok=True)
            self._model = timm.create_model(
                self.backbone,
                pretrained=True,
                num_classes=0,
                img_size=self.img_size,
                cache_dir=str(cache),
            ).eval()
        return self._model

    def embed(self, images: Sequence[Image.Image]) -> np.ndarray:
        import torch

        model = self._load()
        chunks = []
        with torch.inference_mode():
            for i in range(0, len(images), BATCH):
                x = torch.from_numpy(to_tensor_batch(images[i : i + BATCH], self.img_size))
                tokens = model.forward_features(x)  # [b, 1 + prefijos + parches, d]
                n_prefix = int(getattr(model, "num_prefix_tokens", 1))
                cls = tokens[:, 0]
                patches = tokens[:, n_prefix:].mean(dim=1)
                chunks.append(torch.cat([cls, patches], dim=1).numpy().astype(np.float32))
        return np.concatenate(chunks) if chunks else np.zeros((0, 0), dtype=np.float32)


def _open(path: str | Path) -> Image.Image:
    with Image.open(path) as im:
        return im.convert("RGB")


class EmbeddingCache:
    """Embeddings por `image_id` y variante, guardados por tandas en `root/<variant>/`."""

    def __init__(self, root: Path) -> None:
        self.root = root

    def load(self, variant: str) -> dict[str, np.ndarray]:
        out: dict[str, np.ndarray] = {}
        for shard in sorted((self.root / variant).glob("*.npz")):
            with np.load(shard) as z:
                out.update(zip(z["ids"].tolist(), z["emb"], strict=True))
        return out

    def _write(self, variant: str, ids: list[str], emb: np.ndarray) -> None:
        d = self.root / variant
        d.mkdir(parents=True, exist_ok=True)
        n = len(list(d.glob("*.npz")))
        tmp = d / f"{n:05d}.tmp.npz"
        np.savez(tmp, ids=np.asarray(ids), emb=emb.astype(np.float32))
        tmp.rename(d / f"{n:05d}.npz")

    def get(
        self,
        ids: Sequence[str],
        paths: Sequence[str],
        variant: str,
        embed: Callable[[Sequence[Image.Image]], np.ndarray],
        log: bool = False,
    ) -> np.ndarray:
        """[n, dim] en el orden de `ids`; calcula y guarda lo que falte."""
        if variant not in VARIANTS:
            raise ValueError(f"variante desconocida {variant!r}: {VARIANTS}")
        have = self.load(variant)
        todo = [(i, p) for i, p in zip(ids, paths, strict=True) if i not in have]
        orig = self.load("orig") if variant == "desat" and todo else {}
        t0 = time.monotonic()
        for k in range(0, len(todo), SHARD):
            part = todo[k : k + SHARD]
            images = [_open(p) for _, p in part]
            if variant == "desat":
                masks = [bool(red_mask(im).any()) for im in images]
                need = [j for j, m in enumerate(masks) if m or part[j][0] not in orig]
                new = embed([desaturate_red(images[j]) for j in need]) if need else None
                pos = {j: n for n, j in enumerate(need)}
                rows = [
                    new[pos[j]] if new is not None and j in pos else orig[img_id]
                    for j, (img_id, _) in enumerate(part)
                ]
                emb = np.stack(rows)
            else:
                emb = embed(images)
            self._write(variant, [i for i, _ in part], emb)
            have.update(zip([i for i, _ in part], emb, strict=True))
            if log:
                done = k + len(part)
                rate = done / max(time.monotonic() - t0, 1e-9)
                print(f"[emb {variant}] {done}/{len(todo)} · {rate:.1f} img/s", file=sys.stderr)
        return np.stack([have[i] for i in ids]) if len(ids) else np.zeros((0, 0), np.float32)
