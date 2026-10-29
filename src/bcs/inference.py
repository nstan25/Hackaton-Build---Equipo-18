"""`predict(image, cow_id) -> Prediction`: el contrato entre modelo y app (SPEC §2.2).

El `cow_id` solo viaja con la predicción: nunca se le pasa al `Scorer` ni a ningún componente.
`run_pipeline` compone las interfaces de §2.0. `predict` elige el predictor con `BCS_PREDICTOR`:

- `dummy` (default; tests y `make browser`): el fixture con el `cow_id` recibido normalizado.
- `model` (Docker/Railway): el pipeline de producción de G3 (`model_pipeline`): filtro de calidad
  básico (brillo, sobreexposición y nitidez de T08; sin CLIP desde el 18/10, `BCS_QUALITY=clip`
  lo vuelve a CLIP) sobre la foto → detector `models/cow_det.pt` (también hace de "es una vaca")
  → recorte (+10%, lado mayor 320) → EfficientNet-B0 con TTA (T09) → distribución, temperatura y
  conformal de T11 (`models/calibration_efficientnet_b0.json`) → reglas de status con la
  de extremos de G3.

T28 (rama `dev`): `BCS_MODEL` elige el modelo (`MODELS`). `zen` (default) es el de T25
(`models/bcs_efficientnet_b0_zen.pt`, letterbox, grilla `dairy_zen` 3.0–5.0) con su calibración
(`models/calibration_efficientnet_b0_zen.json`) y la detección OOD de T27 en dos niveles
(`models/ood_mahalanobis_zen.npz`, sobre el pooling de la misma pasada, después del detector y
antes de `decide_status`): aviso → `low_confidence` con `ISSUE_OOD_WARN`, rechazo → `rejected`
con `ISSUE_NOT_COW_REAR`. `sdb` es el de producción (`main`), sin OOD, en `dairy_sdb`.

`predict_raw` es lo mismo que `predict` pero devuelve también la foto decodificada, para que la API
arme la miniatura sin decodificar la original otra vez (R3 de eval/review.md).
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import sys
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
from bcs.contract import (
    ISSUE_NOT_COW_REAR,
    ISSUE_OOD_WARN,
    Prediction,
    QualityReport,
    decide_status,
    is_edge,
    normalize_cow_id,
)
from bcs.models.base import Scorer
from bcs.models.stub import UniformScorer
from bcs.ood.gate import LEVEL_REJECT, LEVEL_WARN, FeatureScorer, MahalanobisGate
from bcs.paths import REPO_ROOT, fixtures_dir
from bcs.quality.base import QualityChecker
from bcs.quality.stub import AcceptAllChecker
from bcs.scale import Scale, grid_key, load_scale

ImageInput = bytes | Image.Image
# Foto → recorte de la vaca (lo que ve el `Scorer`), o None si no hay vaca.
Locator = Callable[[Image.Image], Image.Image | None]

PREDICTOR_ENV = "BCS_PREDICTOR"
MODEL_ENV = "BCS_MODEL"
OOD_ENV = "BCS_OOD"  # "off" apaga la detección OOD (comparaciones); default prendida
DETECTOR = REPO_ROOT / "models" / "cow_det.pt"
WEIGHTS = REPO_ROOT / "models" / "bcs_efficientnet_b0.pt"
CALIBRATION = REPO_ROOT / "models" / "calibration_efficientnet_b0.json"
WEIGHTS_ZEN = REPO_ROOT / "models" / "bcs_efficientnet_b0_zen.pt"
CALIBRATION_ZEN = REPO_ROOT / "models" / "calibration_efficientnet_b0_zen.json"
OOD_ZEN = REPO_ROOT / "models" / "ood_mahalanobis_zen.npz"


@dataclass(frozen=True)
class ModelSpec:
    name: str  # prefijo de `model_version`
    weights: Path
    calibration: Path
    scale: str  # preset de config/scale.yaml (la grilla de la calibración)
    ood: Path | None = None


MODELS: dict[str, ModelSpec] = {
    "sdb": ModelSpec("efficientnet_b0", WEIGHTS, CALIBRATION, "dairy_sdb"),
    "zen": ModelSpec("efficientnet_b0_zen", WEIGHTS_ZEN, CALIBRATION_ZEN, "dairy_zen", OOD_ZEN),
}
DEFAULT_MODEL = "zen"


def model_spec(model: str | None = None) -> ModelSpec:
    """El modelo pedido, o el de `BCS_MODEL` (default `DEFAULT_MODEL`)."""
    key = (model or os.environ.get(MODEL_ENV) or DEFAULT_MODEL).strip().lower()
    if key not in MODELS:
        raise ValueError(f"{MODEL_ENV} debe ser uno de {sorted(MODELS)}, no {key!r}")
    return MODELS[key]


def model_scale(model: str | None = None) -> str:
    """La grilla del modelo de producción: `BCS_SCALE` si está, si no la del modelo."""
    return os.environ.get("BCS_SCALE") or model_spec(model).scale


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
    ood: MahalanobisGate | None = None  # T28: requiere un scorer `FeatureScorer`


def stub_pipeline(scale: Scale | None = None) -> Pipeline:
    scale = scale or load_scale()
    return Pipeline(
        scale=scale,
        quality=AcceptAllChecker(),
        scorer=UniformScorer(scale.grid),
        calibrator=IdentityCalibrator(),
        interval=FullGridInterval(scale.grid),
    )


# Tope de píxeles de la foto, se chequea antes de decodificar (R3 de eval/review.md: una PNG de un
# color de 80 MP pesa 250 KB y decodificada ocupa más de 1 GB). Una JPEG más grande (p. ej. 48 MP de
# celular) se decodifica achicada con `draft()`, que es casi gratis.
MAX_IMAGE_PIXELS = 40_000_000
FORMATS_HINT = "usá una foto JPEG, PNG o WebP"


class InvalidImage(ValueError):
    """Los bytes no son una imagen que se pueda abrir (la API responde 422)."""


class ImageTooLarge(InvalidImage):
    """La imagen tiene más de `MAX_IMAGE_PIXELS` (la API responde 413)."""


def _pil_open(fp: io.BytesIO) -> Image.Image:
    # ultralytics reemplaza `Image.open` por uno que ante un formato desconocido intenta instalar
    # pi-heif en runtime y levanta ModuleNotFoundError (R4): se usa el `Image.open` original.
    patches = sys.modules.get("ultralytics.utils.patches")
    return getattr(patches, "_image_open", Image.open)(fp)


def _open_bounded(raw: bytes) -> Image.Image:
    try:
        img = _pil_open(io.BytesIO(raw))
    except Image.DecompressionBombError as e:
        raise ImageTooLarge(f"la imagen supera {MAX_IMAGE_PIXELS // 10**6} MP") from e
    except Exception as e:  # noqa: BLE001 — cualquier cosa que no abre es un 422, nunca un 500
        raise InvalidImage(f"image no es una imagen válida ({FORMATS_HINT})") from e
    w, h = img.size
    if w * h > MAX_IMAGE_PIXELS and img.format == "JPEG":
        # la menor reducción (1/2, 1/4, 1/8) que deja la foto bajo el tope
        k = 2 ** math.ceil(math.log2(math.sqrt(w * h / MAX_IMAGE_PIXELS)))
        if k <= 8:
            img.draft("RGB", (math.ceil(w / k), math.ceil(h / k)))
    w, h = img.size
    if w * h > MAX_IMAGE_PIXELS:
        raise ImageTooLarge(
            f"la imagen tiene {w}×{h} píxeles; el máximo es {MAX_IMAGE_PIXELS // 10**6} MP"
        )
    return img


def _to_image(image: ImageInput) -> tuple[Image.Image, bytes]:
    """La foto derecha (orientación EXIF aplicada, como la ve el celular) en RGB y sus bytes.

    Con bytes, el tamaño se chequea antes de decodificar; si no abren levanta `InvalidImage`
    (`ImageTooLarge` si superan `MAX_IMAGE_PIXELS`).
    """
    if isinstance(image, Image.Image):
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        return ImageOps.exif_transpose(image).convert("RGB"), buf.getvalue()
    img = _open_bounded(image)
    try:
        return ImageOps.exif_transpose(img).convert("RGB"), image
    except Exception as e:  # noqa: BLE001 — JPEG truncada, datos corruptos, modos raros
        raise InvalidImage(f"image no es una imagen válida ({FORMATS_HINT})") from e


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
    return _run_decoded(pipeline, img, raw, cow_id)


def _run_decoded(pipeline: Pipeline, img: Image.Image, raw: bytes, cow_id: str) -> Prediction:
    scale = pipeline.scale
    grid = np.asarray(scale.grid)
    quality = pipeline.quality.check(img)
    version = pipeline.scorer.model_version
    if not quality.passed:
        if not quality.issues:
            quality = replace(quality, issues=[ISSUE_NOT_COW_REAR])
        return _rejected(raw, cow_id, scale, quality, version)

    crop = pipeline.locate(img) if pipeline.locate is not None else img
    if crop is None:
        quality = replace(quality, issues=[*quality.issues, ISSUE_NO_COW])
        return _rejected(raw, cow_id, scale, quality, version)

    if pipeline.ood is None:
        raw_probs = pipeline.scorer.predict_proba([crop])
    else:
        # OOD sobre el pooling de la misma pasada (vista sin flip), sin correr el modelo otra vez
        if not isinstance(pipeline.scorer, FeatureScorer):
            raise TypeError("la detección OOD necesita un scorer con predict_proba_features")
        raw_probs, feats = pipeline.scorer.predict_proba_features([crop])
        level = pipeline.ood.level(feats[0])
        if level == LEVEL_REJECT:
            quality = replace(
                quality, is_cow_rear=False, issues=[*quality.issues, ISSUE_NOT_COW_REAR]
            )
            return _rejected(raw, cow_id, scale, quality, version)
        if level == LEVEL_WARN:
            quality = replace(quality, issues=[*quality.issues, ISSUE_OOD_WARN])
    probs = pipeline.calibrator.transform(raw_probs)[0]
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


# Umbral de confianza del detector: el equipo usa 0.1; 0.15 cumple m4 sin CLIP (tasks/G3.md).
# Sin CLIP, el detector es el único filtro de "es una vaca": más alto rechaza más negativos
# y también más fotos reales.
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

    def predict_proba_features(
        self, images: Sequence[Image.Image]
    ) -> tuple[np.ndarray, np.ndarray]:
        """`predict_proba` y el pooling de la vista sin flip, de la misma pasada (T28, OOD)."""
        values, feats = self.effnet.predict_value_features(images)
        return self.dist.predict_proba(values), feats

    def save(self, path: Path) -> None:
        raise NotImplementedError

    def load(self, path: Path) -> None:
        raise NotImplementedError


def ood_enabled() -> bool:
    return os.environ.get(OOD_ENV, "on").strip().lower() not in ("off", "0", "false", "no")


@cache
def _model_version(spec: ModelSpec, ood: bool) -> str:
    version = (
        f"{spec.name}-{_sha256(spec.weights)[:8]}"
        f"+cal-{_sha256(spec.calibration)[:8]}+det-{_sha256(DETECTOR)[:8]}"
    )
    if spec.ood is not None and ood:
        version += f"+ood-{_sha256(spec.ood)[:8]}"
    return version


def model_version(model: str | None = None) -> str:
    """`<modelo>-<sha .pt>+cal-<sha calibración>+det-<sha cow_det.pt>[+ood-<sha banco>]` (8 hex
    c/u), p. ej. `efficientnet_b0-5a5b28d6+cal-…+det-…` (sdb) o `efficientnet_b0_zen-bd938c68+…`."""
    return _model_version(model_spec(model), ood_enabled())


def model_pipeline(scale: Scale | None = None, model: str | None = None) -> Pipeline:
    """El pipeline de G3 con el modelo `model` (default `BCS_MODEL`, ver `MODELS`). Los pesos
    (EfficientNet, YOLO) se cargan en el primer uso; el banco OOD, acá (se niega si es de otro
    `.pt`).

    Filtro de calidad: `BasicQualityChecker` (sin CLIP, decisión del 18/10 por la RAM de Railway);
    `BCS_QUALITY=clip` vuelve al `ClipQualityChecker` de T08.
    """
    from bcs.calibration.regression import CalibratedRegression
    from bcs.models.efficientnet import EfficientNetScorer
    from bcs.quality.basic import BasicQualityChecker
    from bcs.quality.clip_checker import ClipQualityChecker

    spec = model_spec(model)
    scale = scale or load_scale(model_scale(model))
    cal = CalibratedRegression.load(spec.calibration)
    if tuple(scale.grid) != cal.grid:
        raise ValueError(
            f"la escala {scale.name} {scale.grid} no es la grilla de la calibración {cal.grid}"
            f" (usá BCS_SCALE={spec.scale})"
        )
    effnet = EfficientNetScorer(scale.grid, {"model": {"weights": str(spec.weights), "tta": True}})
    use_ood = spec.ood is not None and ood_enabled()
    ood = (
        MahalanobisGate.load(spec.ood, model_sha256=_sha256(spec.weights))
        if spec.ood is not None and use_ood
        else None
    )
    return Pipeline(
        scale=scale,
        quality=ClipQualityChecker()
        if os.environ.get("BCS_QUALITY") == "clip"
        else BasicQualityChecker(),
        scorer=CalibratedEfficientNet(effnet, cal.dist, model_version(model)),
        calibrator=cal.calibrator,
        interval=cal.conformal,
        locate=CowLocator(),
        ood=ood,
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
    return _predict_dummy_decoded(image, cow_id)[0]


def _predict_dummy_decoded(image: ImageInput, cow_id: str) -> tuple[Prediction, Image.Image]:
    cow_id = normalize_cow_id(cow_id)
    img, raw = _to_image(image)
    return replace(_fixture(), cow_id=cow_id, **_new_meta(raw)), img


def predict(image: ImageInput, cow_id: str) -> Prediction:
    """El contrato de §2.2: dummy o modelo de producción según `BCS_PREDICTOR`."""
    if not use_model():
        return predict_dummy(image, cow_id)
    with _model_lock:  # torch ya usa todos los núcleos: de a una predicción por vez
        return run_pipeline(_model_pipeline(), image, cow_id)


def predict_raw(raw: bytes, cow_id: str) -> tuple[Prediction, Image.Image]:
    """`predict` sobre los bytes originales, que devuelve también la foto ya decodificada (derecha,
    RGB) para armar la miniatura sin decodificar dos veces. `image_sha256` es el de `raw`."""
    if not use_model():
        return _predict_dummy_decoded(raw, cow_id)
    cow_id = normalize_cow_id(cow_id)
    with _model_lock:
        img, raw = _to_image(raw)
        return _run_decoded(_model_pipeline(), img, raw, cow_id), img


def predict_model_version() -> str:
    return model_version() if use_model() else _fixture().model_version


def warmup() -> None:
    """Carga el filtro de calidad, YOLO y la EfficientNet y corre una pasada, para que la primera
    foto no pague la carga (~12 s en CPU). Toma el mismo lock que `predict`: una foto que llega
    antes espera."""
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
