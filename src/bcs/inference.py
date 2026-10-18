"""`predict(image, cow_id) -> Prediction`: el contrato entre modelo y app (SPEC §2.2).

El `cow_id` solo viaja con la predicción: nunca se le pasa al `Scorer` ni a ningún componente.
`run_pipeline` compone las interfaces de §2.0. `predict` elige el predictor con `BCS_PREDICTOR`:

- `dummy` (default; tests y `make browser`): el fixture con el `cow_id` recibido normalizado.
- `model` (Docker/Railway): el pipeline de producción de G3 (`model_pipeline`): filtro de calidad
  CLIP (T08) sobre la foto → detector `models/cow_det.pt` → recorte (+10%, lado mayor 320) →
  EfficientNet-B0 con TTA (T09) → distribución, temperatura y conformal de T11
  (`models/calibration_efficientnet_b0.json`) → reglas de status con la de extremos de G3.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import threading
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from functools import cache
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps

from bcs.calibration.base import Calibrator, IntervalPredictor
from bcs.calibration.stub import FullGridInterval, IdentityCalibrator
from bcs.contract import Prediction, QualityReport, decide_status, is_edge, normalize_cow_id
from bcs.models.base import Scorer
from bcs.models.stub import UniformScorer
from bcs.paths import REPO_ROOT, fixtures_dir
from bcs.quality.base import QualityChecker
from bcs.quality.stub import AcceptAllChecker
from bcs.scale import Scale, grid_key, load_scale

ImageInput = bytes | Image.Image
# Foto → recorte de la vaca (lo que ve el `Scorer`), o None si no hay vaca.
Locator = Callable[[Image.Image], Image.Image | None]

PREDICTOR_ENV = "BCS_PREDICTOR"
DETECTOR = REPO_ROOT / "models" / "cow_det.pt"
WEIGHTS = REPO_ROOT / "models" / "bcs_efficientnet_b0.pt"
CALIBRATION = REPO_ROOT / "models" / "calibration_efficientnet_b0.json"
MODEL_SCALE = "dairy_sdb"
ISSUE_NO_COW = "no se encontró la vaca en la foto"
# Texto de G3 para `low_confidence` en una punta de la grilla (lo muestra la UI).
EDGE_LOW = "posible condición baja, verificar a mano"
EDGE_HIGH = "posible condición alta, verificar a mano"


@dataclass
class Pipeline:
    scale: Scale
    quality: QualityChecker
    scorer: Scorer
    calibrator: Calibrator
    interval: IntervalPredictor
    locate: Locator | None = None  # sin locator, el scorer ve la foto entera


def stub_pipeline(scale: Scale | None = None) -> Pipeline:
    scale = scale or load_scale()
    return Pipeline(
        scale=scale,
        quality=AcceptAllChecker(),
        scorer=UniformScorer(scale.grid),
        calibrator=IdentityCalibrator(),
        interval=FullGridInterval(scale.grid),
    )


def _to_image(image: ImageInput) -> tuple[Image.Image, bytes]:
    """La foto derecha (orientación EXIF aplicada, como la ve el celular) en RGB y sus bytes."""
    if isinstance(image, Image.Image):
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        return ImageOps.exif_transpose(image).convert("RGB"), buf.getvalue()
    return ImageOps.exif_transpose(Image.open(io.BytesIO(image))).convert("RGB"), image


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _new_meta(raw: bytes) -> dict[str, str]:
    return {
        "prediction_id": str(uuid.uuid4()),
        "created_at": _now(),
        "image_sha256": hashlib.sha256(raw).hexdigest(),
    }


def _rejected(
    raw: bytes, cow_id: str, scale: Scale, quality: QualityReport, version: str
) -> Prediction:
    return Prediction(
        **_new_meta(raw),
        cow_id=cow_id,
        status="rejected",
        bcs=None,
        bcs_expected=None,
        confidence=None,
        interval=None,
        distribution={k: 0.0 for k in scale.keys},
        quality=quality,
        scale=scale.name,
        model_version=version,
    )


def run_pipeline(pipeline: Pipeline, image: ImageInput, cow_id: str) -> Prediction:
    """Filtro de calidad → scorer → calibración → intervalo → reglas de status."""
    cow_id = normalize_cow_id(cow_id)
    img, raw = _to_image(image)
    scale = pipeline.scale
    grid = np.asarray(scale.grid)
    quality = pipeline.quality.check(img)
    version = pipeline.scorer.model_version
    if not quality.passed:
        if not quality.issues:
            quality = replace(quality, issues=["no parece una vaca de atrás"])
        return _rejected(raw, cow_id, scale, quality, version)

    crop = pipeline.locate(img) if pipeline.locate is not None else img
    if crop is None:
        quality = replace(quality, issues=[*quality.issues, ISSUE_NO_COW])
        return _rejected(raw, cow_id, scale, quality, version)

    probs = pipeline.calibrator.transform(pipeline.scorer.predict_proba([crop]))[0]
    top = int(np.argmax(probs))
    confidence = float(probs[max(top - 1, 0) : top + 2].sum())
    interval = pipeline.interval.interval(probs[None, :])[0]
    bcs = float(grid[top])
    edges = (float(grid[0]), float(grid[-1]))
    return Prediction(
        **_new_meta(raw),
        cow_id=cow_id,
        status=decide_status(quality, confidence, interval, scale.step, bcs=bcs, edges=edges),
        bcs=bcs,
        bcs_expected=float((grid * probs).sum()),
        confidence=confidence,
        interval=interval,
        distribution={grid_key(v): float(p) for v, p in zip(grid, probs, strict=True)},
        quality=quality,
        scale=scale.name,
        model_version=version,
    )


# ---------------------------------------------------------------- modelo de producción (G3)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# Umbral de confianza del detector (el equipo usa 0.1; 0.15 cumple m4 sin CLIP, ver tasks/G3.md). Sin CLIP, el detector es el único filtro de
# "es una vaca": un umbral más alto rechaza más negativos y también más fotos reales.
DETECTOR_CONF = float(os.environ.get("BCS_DET_CONF", "0.15"))


class CowLocator:
    """Detector YOLO11n del equipo + recorte del README (`prep_yolo.detect_cow`,
    `prep_crops.expand_bbox`, lado mayor `MAX_SIDE`), como `bcs_pipeline.BCSScorer.predict`."""

    def __init__(self, weights: Path = DETECTOR) -> None:
        self.weights = weights
        self._model: Any = None
        self._lock = threading.Lock()

    def __call__(self, image: Image.Image) -> Image.Image | None:
        from bcs.models.efficientnet import team_module

        prep_yolo, prep_crops = team_module("prep_yolo"), team_module("prep_crops")
        with self._lock:  # el predictor de ultralytics no es thread-safe
            if self._model is None:
                from ultralytics import YOLO

                self._model = YOLO(str(self.weights))
            bbox = prep_yolo.detect_cow(self._model, image, conf=DETECTOR_CONF)
        if bbox is None:
            return None
        crop = image.crop(prep_crops.expand_bbox([int(v) for v in bbox], *image.size))
        crop.thumbnail((prep_crops.MAX_SIDE, prep_crops.MAX_SIDE))
        return crop


class CalibratedEfficientNet:
    """`Scorer`: valor continuo de la EfficientNet-B0 (TTA) → distribución de T11 sobre la
    grilla, sin la temperatura (esa la aplica el `Calibrator` del pipeline). No se entrena acá."""

    def __init__(self, effnet: Any, dist: Any, model_version: str) -> None:
        self.effnet = effnet
        self.dist = dist
        self.grid: tuple[float, ...] = tuple(dist.grid)
        self.model_version = model_version

    def fit(self, df: Any) -> None:
        raise NotImplementedError("el modelo de producción no se entrena desde la app")

    def predict_proba(self, images: Sequence[Image.Image]) -> np.ndarray:
        return self.dist.predict_proba(self.effnet.predict_value(images))

    def save(self, path: Path) -> None:
        raise NotImplementedError

    def load(self, path: Path) -> None:
        raise NotImplementedError


@cache
def model_version() -> str:
    """`efficientnet_b0-<sha .pt>+cal-<sha calibration.json>+det-<sha cow_det.pt>` (8 hex c/u)."""
    return (
        f"efficientnet_b0-{_sha256(WEIGHTS)[:8]}"
        f"+cal-{_sha256(CALIBRATION)[:8]}+det-{_sha256(DETECTOR)[:8]}"
    )


def model_pipeline(scale: Scale | None = None) -> Pipeline:
    """El pipeline de G3. Los pesos (EfficientNet, YOLO) se cargan en el primer uso.

    Filtro de calidad: `BasicQualityChecker` (sin CLIP, decisión del 18/10 por la RAM de Railway);
    `BCS_QUALITY=clip` vuelve al `ClipQualityChecker` de T08.
    """
    from bcs.calibration.regression import CalibratedRegression
    from bcs.models.efficientnet import EfficientNetScorer
    from bcs.quality.basic import BasicQualityChecker
    from bcs.quality.clip_checker import ClipQualityChecker

    scale = scale or load_scale(os.environ.get("BCS_SCALE") or MODEL_SCALE)
    cal = CalibratedRegression.load(CALIBRATION)
    if tuple(scale.grid) != cal.grid:
        raise ValueError(
            f"la escala {scale.name} {scale.grid} no es la grilla de la calibración {cal.grid}"
            f" (usá BCS_SCALE={MODEL_SCALE})"
        )
    effnet = EfficientNetScorer(scale.grid, {"model": {"weights": str(WEIGHTS), "tta": True}})
    return Pipeline(
        scale=scale,
        quality=ClipQualityChecker()
        if os.environ.get("BCS_QUALITY") == "clip"
        else BasicQualityChecker(),
        scorer=CalibratedEfficientNet(effnet, cal.dist, model_version()),
        calibrator=cal.calibrator,
        interval=cal.conformal,
        locate=CowLocator(),
    )


def edge_note(pred: Prediction, step: float) -> str | None:
    """Texto de G3 si `pred` es `low_confidence` en una punta de su grilla, si no None."""
    if pred.status != "low_confidence" or pred.bcs is None:
        return None
    keys = [float(k) for k in pred.distribution]
    lo, hi = min(keys), max(keys)
    if not is_edge(pred.bcs, (lo, hi), step):
        return None
    return EDGE_LOW if abs(pred.bcs - lo) < step / 2 else EDGE_HIGH


# ---------------------------------------------------------------- predict


def use_model() -> bool:
    value = os.environ.get(PREDICTOR_ENV, "dummy").strip().lower()
    if value not in ("dummy", "model"):
        raise ValueError(f"{PREDICTOR_ENV} debe ser 'dummy' o 'model', no {value!r}")
    return value == "model"


_model_lock = threading.Lock()


@cache
def _model_pipeline() -> Pipeline:
    return model_pipeline()


@cache
def _fixture() -> Prediction:
    data = json.loads((fixtures_dir() / "prediction_example.json").read_text())
    return Prediction.from_dict(data)


def predict_dummy(image: ImageInput, cow_id: str) -> Prediction:
    """Dummy de T00: el fixture con el `cow_id` normalizado y metadatos propios de esta foto."""
    cow_id = normalize_cow_id(cow_id)
    _, raw = _to_image(image)
    return replace(_fixture(), cow_id=cow_id, **_new_meta(raw))


def predict(image: ImageInput, cow_id: str) -> Prediction:
    """El contrato de §2.2: dummy o modelo de producción según `BCS_PREDICTOR`."""
    if not use_model():
        return predict_dummy(image, cow_id)
    with _model_lock:  # torch ya usa todos los núcleos: de a una predicción por vez
        return run_pipeline(_model_pipeline(), image, cow_id)


def predict_model_version() -> str:
    return model_version() if use_model() else _fixture().model_version


def warmup() -> None:
    """Carga CLIP, YOLO y la EfficientNet y corre una pasada, para que la primera foto no pague
    la carga (~12 s en CPU). Toma el mismo lock que `predict`: una foto que llega antes espera."""
    with _model_lock:
        pipe = _model_pipeline()
        img = Image.new("RGB", (480, 320), (90, 80, 70))
        pipe.quality.check(img)
        if pipe.locate is not None:
            pipe.locate(img)
        pipe.scorer.predict_proba([img])


def start_warmup() -> threading.Thread | None:
    """Con `BCS_PREDICTOR=model`, `warmup()` en un thread de fondo (la API arranca y responde
    `/health` mientras tanto). Lo llama este módulo al importarse, es decir al levantar la app."""
    if not use_model():
        return None
    thread = threading.Thread(target=warmup, name="bcs-warmup", daemon=True)
    thread.start()
    return thread


start_warmup()
