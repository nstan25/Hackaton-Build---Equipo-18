"""`predict(image, cow_id) -> Prediction`: el contrato entre modelo y app (SPEC §2.2).

El `cow_id` solo viaja con la predicción: nunca se le pasa al `Scorer` ni a ningún componente.
Hasta T12, `predict` es el dummy: devuelve el fixture con el `cow_id` recibido normalizado.
`run_pipeline` compone las interfaces de §2.0 y es lo que T12 conecta con el modelo real.
"""

from __future__ import annotations

import hashlib
import io
import json
import uuid
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from functools import cache

import numpy as np
from PIL import Image

from bcs.calibration.base import Calibrator, IntervalPredictor
from bcs.calibration.stub import FullGridInterval, IdentityCalibrator
from bcs.contract import Prediction, decide_status, normalize_cow_id
from bcs.models.base import Scorer
from bcs.models.stub import UniformScorer
from bcs.paths import fixtures_dir
from bcs.quality.base import QualityChecker
from bcs.quality.stub import AcceptAllChecker
from bcs.scale import Scale, grid_key, load_scale

ImageInput = bytes | Image.Image


@dataclass
class Pipeline:
    scale: Scale
    quality: QualityChecker
    scorer: Scorer
    calibrator: Calibrator
    interval: IntervalPredictor


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
    if isinstance(image, Image.Image):
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        return image.convert("RGB"), buf.getvalue()
    return Image.open(io.BytesIO(image)).convert("RGB"), image


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _new_meta(raw: bytes) -> dict[str, str]:
    return {
        "prediction_id": str(uuid.uuid4()),
        "created_at": _now(),
        "image_sha256": hashlib.sha256(raw).hexdigest(),
    }


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

    probs = pipeline.calibrator.transform(pipeline.scorer.predict_proba([img]))[0]
    top = int(np.argmax(probs))
    confidence = float(probs[max(top - 1, 0) : top + 2].sum())
    interval = pipeline.interval.interval(probs[None, :])[0]
    return Prediction(
        **_new_meta(raw),
        cow_id=cow_id,
        status=decide_status(quality, confidence, interval, scale.step),
        bcs=float(grid[top]),
        bcs_expected=float((grid * probs).sum()),
        confidence=confidence,
        interval=interval,
        distribution={grid_key(v): float(p) for v, p in zip(grid, probs, strict=True)},
        quality=quality,
        scale=scale.name,
        model_version=version,
    )


@cache
def _fixture() -> Prediction:
    data = json.loads((fixtures_dir() / "prediction_example.json").read_text())
    return Prediction.from_dict(data)


def predict(image: ImageInput, cow_id: str) -> Prediction:
    """Dummy de T00: el fixture con el `cow_id` normalizado y metadatos propios de esta foto."""
    cow_id = normalize_cow_id(cow_id)
    _, raw = _to_image(image)
    return replace(_fixture(), cow_id=cow_id, **_new_meta(raw))


def predict_model_version() -> str:
    return _fixture().model_version
