"""Filtro de calidad (SPEC §2.3, G0 §5): CLIP zero-shot + brillo + nitidez (Laplaciano).

Una foto pasa si CLIP la ve como "vaca de atrás/arriba" y no está oscura, quemada ni borrosa.
Los umbrales salen de val de `sdb_bcs` y del set de rechazo (`tests/fixtures/reject/`); se revisan
en G1. El `cow_id` no pasa por acá: el filtro solo ve la imagen.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from bcs.contract import QualityReport
from bcs.paths import artifacts_dir

CLIP_MODEL = "ViT-B-32"
CLIP_PRETRAINED = "laion2b_s34b_b79k"

# Prompts en inglés (CLIP se entrenó así). Los positivos cubren cómo se ven las fotos de val:
# cámara fija desde arriba, vacas negras con pintura en la base de la cola, algo de barro y blur.
POSITIVE_PROMPTS: tuple[str, ...] = (
    "a photo of the rear of a cow seen from above",
    "a close-up photo of a cow's hindquarters and tail head",
    "a top-down photo of a dairy cow's back in a milking stall",
    "a photo of the back of a black and white Holstein cow",
    "a top-down photo of the rear of a black cow with paint on its tail head",
    "a blurry photo of a cow's rump from above",
    "a close-up photo of a dirty cow's back and tail from above",
)
NEGATIVE_PROMPTS: tuple[str, ...] = (
    "a photo of a dog",
    "a photo of a cat",
    "a photo of a bird",
    "a photo of a fish",
    "a photo of a horse",
    "a photo of sheep",
    "a photo of a person",
    "a photo of a human face",
    "a photo of a car",
    "a photo of a tractor",
    "a photo of a house",
    "a photo of a barn",
    "a landscape photo",
    "a photo of a flower",
    "a photo of a cup",
    "a photo of furniture",
    "a photo of food",
    "a document with text",
    "a screenshot of a phone screen",
    "a QR code",
    "random noise",
    "an abstract pattern",
    "a plain color background",
    "a photo of the sky",
    "a photo of a cow from the side",
    "a photo of a cow's face",
    "a cartoon drawing",
)

ISSUE_NOT_COW_REAR = "no parece una vaca de atrás"
ISSUE_DARK = "imagen muy oscura"
ISSUE_OVEREXPOSED = "imagen sobreexpuesta"
ISSUE_BLURRY = "imagen borrosa"

# Puntaje semántico: lista de PIL → probabilidad (0–1) de "vaca de atrás" por imagen.
SemanticScorer = Callable[[Sequence[Image.Image]], np.ndarray]


@dataclass(frozen=True)
class Thresholds:
    clip_min: float = 0.25  # masa de probabilidad de los prompts positivos
    brightness_min: float = 30.0  # luminancia media (0–255) sin el relleno negro del recorte
    brightness_max: float = 225.0
    saturated_max: float = (
        0.30  # fracción de píxeles quemados (≥ 250) a partir de la que es sobreexpuesta
    )
    sharpness_min: float = 6.0  # varianza del Laplaciano con el lado mayor a 256 px


@dataclass(frozen=True)
class ImageStats:
    brightness: float
    saturated: float
    sharpness: float


_STATS_SIDE = 256
_PAD_LEVEL = 8  # el preprocesamiento rellena con negro: esos píxeles no cuentan para el brillo


def image_stats(image: Image.Image) -> ImageStats:
    """Brillo, fracción quemada y nitidez, con la imagen llevada a un tamaño fijo."""
    gray = np.asarray(image.convert("L"))
    scale = _STATS_SIDE / max(gray.shape)
    size = (max(1, round(gray.shape[1] * scale)), max(1, round(gray.shape[0] * scale)))
    gray = cv2.resize(gray, size, interpolation=cv2.INTER_AREA)
    content = gray > _PAD_LEVEL
    if content.mean() < 0.05:
        brightness, saturated = 0.0, 0.0
    else:
        brightness = float(gray[content].mean())
        saturated = float((gray[content] >= 250).mean())
    sharpness = float(cv2.Laplacian(gray.astype(np.float64), cv2.CV_64F).var())
    return ImageStats(brightness=brightness, saturated=saturated, sharpness=sharpness)


def quality_issues(stats: ImageStats, t: Thresholds) -> list[str]:
    issues: list[str] = []
    if stats.brightness < t.brightness_min:
        issues.append(ISSUE_DARK)
    elif stats.brightness > t.brightness_max or stats.saturated > t.saturated_max:
        issues.append(ISSUE_OVEREXPOSED)
    if stats.sharpness < t.sharpness_min:
        issues.append(ISSUE_BLURRY)
    return issues


def clip_cache_dir() -> Path:
    return artifacts_dir() / "T08" / "clip"


class ClipScorer:
    """CLIP zero-shot: softmax sobre todos los prompts, suma de los positivos."""

    def __init__(self, cache_dir: Path | None = None, device: str = "cpu") -> None:
        import open_clip
        import torch

        self._torch = torch
        model, _, preprocess = open_clip.create_model_and_transforms(
            CLIP_MODEL, pretrained=CLIP_PRETRAINED, cache_dir=str(cache_dir or clip_cache_dir())
        )
        self._model: Any = model.to(device).eval()
        self._preprocess: Any = preprocess
        self._device = device
        tokens = open_clip.get_tokenizer(CLIP_MODEL)(list(POSITIVE_PROMPTS + NEGATIVE_PROMPTS))
        with torch.no_grad():
            text = self._model.encode_text(tokens.to(device))
        self._text = text / text.norm(dim=-1, keepdim=True)
        self._scale = float(self._model.logit_scale.exp().item())

    def __call__(self, images: Sequence[Image.Image]) -> np.ndarray:
        torch = self._torch
        if not images:
            return np.zeros(0)
        batch = torch.stack([self._preprocess(im.convert("RGB")) for im in images])
        with torch.no_grad():
            feats = self._model.encode_image(batch.to(self._device))
            feats = feats / feats.norm(dim=-1, keepdim=True)
            probs = (self._scale * feats @ self._text.T).softmax(dim=-1)
        return probs[:, : len(POSITIVE_PROMPTS)].sum(dim=1).cpu().numpy()


class ClipQualityChecker:
    """`QualityChecker` real. El modelo CLIP se carga en el primer uso (los pesos se cachean)."""

    def __init__(
        self, scorer: SemanticScorer | None = None, thresholds: Thresholds | None = None
    ) -> None:
        self._scorer = scorer
        self.thresholds = thresholds or Thresholds()

    @property
    def scorer(self) -> SemanticScorer:
        if self._scorer is None:
            self._scorer = ClipScorer()
        return self._scorer

    def check(self, image: Image.Image) -> QualityReport:
        return self.check_batch([image])[0]

    def check_batch(self, images: Sequence[Image.Image]) -> list[QualityReport]:
        scores = self.scorer(images)
        return [self._report(im, float(s)) for im, s in zip(images, scores, strict=True)]

    def _report(self, image: Image.Image, clip_score: float) -> QualityReport:
        is_cow_rear = clip_score >= self.thresholds.clip_min
        issues = [] if is_cow_rear else [ISSUE_NOT_COW_REAR]
        issues += quality_issues(image_stats(image), self.thresholds)
        return QualityReport(is_cow_rear=is_cow_rear, score=round(clip_score, 4), issues=issues)
