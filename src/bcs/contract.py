"""Contrato de inferencia (SPEC §2.2). Se congela en G0; cambiarlo requiere un gate."""

from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass, field, fields
from typing import Any, Literal, get_args

from bcs.scale import grid_key

Status = Literal["ok", "low_confidence", "rejected"]
STATUSES: tuple[str, ...] = get_args(Status)

COW_ID_PATTERN = re.compile(r"^[A-Z0-9][A-Z0-9._-]{0,63}$")
LOW_CONFIDENCE_THRESHOLD = 0.5
MAX_INTERVAL_STEPS = 4

# T28 (G6): motivos de la detección OOD. El de rechazo es el de siempre. El de aviso es el único
# "issue" que no rechaza: la foto sigue con BCS pero queda `low_confidence` (sin campos nuevos, así
# que un cliente viejo lee igual el JSON y muestra el texto en la lista de motivos).
ISSUE_NOT_COW_REAR = "no parece una vaca de atrás"
ISSUE_OOD_WARN = "la foto no se parece a las del entrenamiento; verificá encuadre y luz"
WARNING_ISSUES: frozenset[str] = frozenset({ISSUE_OOD_WARN})


class InvalidCowId(ValueError):
    """`cow_id` ausente o con formato inválido. La API lo traduce a 422."""


@dataclass
class QualityReport:
    is_cow_rear: bool
    score: float
    issues: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        """Es una vaca de atrás y no hay ningún motivo de rechazo (los avisos no rechazan)."""
        return self.is_cow_rear and all(i in WARNING_ISSUES for i in self.issues)

    @property
    def warned(self) -> bool:
        """Trae algún aviso (`WARNING_ISSUES`): la predicción es a lo sumo `low_confidence`."""
        return any(i in WARNING_ISSUES for i in self.issues)


@dataclass
class Prediction:
    prediction_id: str  # uuid4, lo genera el sistema
    cow_id: str  # el ID que ingresó el usuario, normalizado
    created_at: str  # ISO 8601 en UTC
    image_sha256: str  # hash de la foto original
    status: Status
    bcs: float | None  # valor de la grilla de la escala
    bcs_expected: float | None  # esperanza de la distribución (continuo)
    confidence: float | None  # P(BCS real ∈ [bcs - paso, bcs + paso]), calibrada
    interval: tuple[float, float] | None  # conjunto conformal al 90% (min, max)
    distribution: dict[str, float]  # prob por clase de la grilla
    quality: QualityReport
    scale: str
    model_version: str

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["interval"] = list(self.interval) if self.interval is not None else None
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Prediction:
        errors = validate_prediction_dict(d)
        if errors:
            raise ValueError("; ".join(errors))
        interval = d["interval"]
        return cls(
            prediction_id=d["prediction_id"],
            cow_id=d["cow_id"],
            created_at=d["created_at"],
            image_sha256=d["image_sha256"],
            status=d["status"],
            bcs=d["bcs"],
            bcs_expected=d["bcs_expected"],
            confidence=d["confidence"],
            interval=(float(interval[0]), float(interval[1])) if interval is not None else None,
            distribution={str(k): float(v) for k, v in d["distribution"].items()},
            quality=QualityReport(**d["quality"]),
            scale=d["scale"],
            model_version=d["model_version"],
        )


PREDICTION_FIELDS: tuple[str, ...] = tuple(f.name for f in fields(Prediction))
QUALITY_FIELDS: tuple[str, ...] = tuple(f.name for f in fields(QualityReport))


def normalize_cow_id(raw: str | None) -> str:
    """Quita espacios en los extremos, pasa a mayúsculas y valida el formato."""
    if raw is None:
        raise InvalidCowId("cow_id es obligatorio")
    if not raw.isascii():  # "ß1".upper() == "SS1": sin esto, otro ID pasaría la validación
        raise InvalidCowId(
            "cow_id inválido: usá solo letras A-Z sin acentos, números, '.', '_' o '-'"
        )
    cow_id = raw.strip().upper()
    if not cow_id:
        raise InvalidCowId("cow_id es obligatorio")
    if not COW_ID_PATTERN.fullmatch(cow_id):
        raise InvalidCowId(
            "cow_id inválido: usá de 1 a 64 caracteres A-Z, 0-9, '.', '_' o '-', "
            "empezando por letra o número (p. ej. el número de caravana)"
        )
    return cow_id


def decide_status(
    quality: QualityReport,
    confidence: float | None,
    interval: tuple[float, float] | None,
    step: float,
    *,
    bcs: float | None = None,
    edges: tuple[float, float] | None = None,
) -> Status:
    """Reglas de `status` (SPEC §2.2).

    G3: si se pasan `bcs` y `edges` (extremos de la grilla del modelo), un BCS en un extremo
    también es `low_confidence` (SPEC §1.2: posible condición baja/alta, verificar a mano).
    T28: un aviso en `quality.issues` (`WARNING_ISSUES`, la detección OOD) también.
    """
    if not quality.passed:
        return "rejected"
    if quality.warned:
        return "low_confidence"
    if confidence is None or confidence < LOW_CONFIDENCE_THRESHOLD:
        return "low_confidence"
    if interval is None or interval_steps(interval, step) > MAX_INTERVAL_STEPS:
        return "low_confidence"
    if bcs is not None and edges is not None and is_edge(bcs, edges, step):
        return "low_confidence"
    return "ok"


def is_edge(bcs: float, edges: tuple[float, float], step: float) -> bool:
    """`bcs` cae en el extremo inferior o superior de la grilla (tolerancia de medio paso)."""
    return abs(bcs - edges[0]) < step / 2 or abs(bcs - edges[1]) < step / 2


def interval_steps(interval: tuple[float, float], step: float) -> int:
    return round((interval[1] - interval[0]) / step)


def _is_number(x: Any) -> bool:
    return isinstance(x, int | float) and not isinstance(x, bool) and math.isfinite(x)


def validate_prediction_dict(
    d: dict[str, Any], grid_keys: tuple[str, ...] | None = None, step: float | None = None
) -> list[str]:
    """Valida un JSON de `Prediction` contra el contrato. Devuelve la lista de errores."""
    errors: list[str] = []
    missing = set(PREDICTION_FIELDS) - d.keys()
    extra = d.keys() - set(PREDICTION_FIELDS)
    if missing:
        errors.append(f"faltan campos: {sorted(missing)}")
    if extra:
        errors.append(f"campos de más: {sorted(extra)}")
    if errors:
        return errors

    for name in ("prediction_id", "cow_id", "created_at", "image_sha256", "scale", "model_version"):
        if not isinstance(d[name], str) or not d[name]:
            errors.append(f"{name} debe ser un string no vacío")
    if isinstance(d["cow_id"], str) and not COW_ID_PATTERN.fullmatch(d["cow_id"]):
        errors.append("cow_id no está normalizado o es inválido")
    if isinstance(d["image_sha256"], str) and not re.fullmatch(r"[0-9a-f]{64}", d["image_sha256"]):
        errors.append("image_sha256 debe ser un sha256 hex en minúsculas")
    if d["status"] not in STATUSES:
        errors.append(f"status inválido: {d['status']!r}")

    q = d["quality"]
    if not isinstance(q, dict) or set(q) != set(QUALITY_FIELDS):
        errors.append(f"quality debe tener exactamente {list(QUALITY_FIELDS)}")
        return errors
    if not isinstance(q["is_cow_rear"], bool) or not _is_number(q["score"]):
        errors.append("quality.is_cow_rear debe ser bool y quality.score un número")
    if not isinstance(q["issues"], list) or not all(isinstance(i, str) for i in q["issues"]):
        errors.append("quality.issues debe ser una lista de strings")

    dist = d["distribution"]
    if not isinstance(dist, dict) or not all(_is_number(v) for v in dist.values()):
        errors.append("distribution debe ser un dict clase → probabilidad")
        return errors
    if grid_keys is not None and set(dist) != set(grid_keys):
        errors.append("distribution no cubre exactamente la grilla de la escala")

    if d["status"] == "rejected":
        for name in ("bcs", "bcs_expected", "confidence", "interval"):
            if d[name] is not None:
                errors.append(f"status=rejected no devuelve {name}")
        if not q["issues"]:
            errors.append("status=rejected debe traer al menos un issue")
        return errors

    if dist and abs(sum(dist.values()) - 1.0) > 1e-3:
        errors.append("distribution debe sumar 1")
    for name in ("bcs", "bcs_expected", "confidence"):
        if not _is_number(d[name]):
            errors.append(f"{name} debe ser un número si status != rejected")
    iv = d["interval"]
    if not (isinstance(iv, list | tuple) and len(iv) == 2 and all(_is_number(x) for x in iv)):
        errors.append("interval debe ser [min, max] si status != rejected")
    if errors:
        return errors

    if not 0.0 <= d["confidence"] <= 1.0:
        errors.append("confidence fuera de [0, 1]")
    if iv[0] > iv[1]:
        errors.append("interval con min > max")
    if grid_keys is not None and grid_key(d["bcs"]) not in grid_keys:
        errors.append("bcs no es un valor de la grilla")
    expected = sum(float(k) * v for k, v in dist.items())
    if abs(expected - d["bcs_expected"]) > 1e-2:
        errors.append("bcs_expected no coincide con la esperanza de distribution")
    if step is not None:
        quality = QualityReport(**q)
        edges = (
            (float(min(grid_keys, key=float)), float(max(grid_keys, key=float)))
            if grid_keys
            else None
        )
        want = decide_status(
            quality, d["confidence"], (iv[0], iv[1]), step, bcs=d["bcs"], edges=edges
        )
        if want != d["status"]:
            errors.append(f"status={d['status']!r} no respeta las reglas (debería ser {want!r})")
    return errors
