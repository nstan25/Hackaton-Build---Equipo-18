"""Explicación del puntaje con un modelo multimodal de OpenAI (T17).

Dos pasos, para que el modelo no avale el puntaje por defecto:

1. **Lectura ciega** (con la foto, sin el BCS predicho): el modelo describe cada región según los
   criterios de referencia y da su propio rango (`photo_estimate`: baja/media/alta/no_se_puede).
2. **Comparación** (solo texto, sin la foto): `agreement` sale en código de comparar ese rango con
   el del BCS predicho; después se le pide al modelo que lo explique para el productor a partir de
   su propia descripción ciega. No vuelve a mirar la foto, así que no puede acomodar la lectura.

La clave se lee **solo** de `OPENAI_API_KEY` y nunca se loguea ni se devuelve. Se llama a la
Responses API con `httpx` (sin el SDK) y salida estructurada (`text.format` = `json_schema`).
"""

from __future__ import annotations

import base64
import json
import os
import time
from datetime import UTC, datetime
from typing import Any, Literal

import httpx

from bcs.contract import Prediction

OPENAI_URL = "https://api.openai.com/v1/responses"
# Default: modelo multimodal vigente de la familia GPT-6 (docs de OpenAI, 18/10/2026). Se pisa con
# OPENAI_MODEL; p. ej. `gpt-6-luna` es ~20 veces más barato. Con razonamiento "low" responde mucho
# más rápido (medido con fotos de val) y los dos pasos entran en el timeout de 30 s.
DEFAULT_MODEL = "gpt-6.1-sol"
# OPENAI_REASONING_EFFORT; vacío = no se manda (modelos sin razonamiento)
DEFAULT_REASONING_EFFORT = "low"
MAX_TIMEOUT_S = 30.0  # para los dos pasos juntos
MAX_OUTPUT_TOKENS = 4000  # incluye el razonamiento interno del modelo, si lo tiene
MAX_TRAITS = 12
MAX_TEXT = 2000
# Versión del prompt: entra en la cache, así una explicación vieja se regenera al cambiarlo.
PROMPT_VERSION = "2-ciego"

Agreement = Literal["coincide", "sugiere_mas_baja", "sugiere_mas_alta", "no_se_puede"]
ESTIMATES: tuple[str, ...] = ("baja", "media", "alta", "no_se_puede")
_ORDER = {"baja": 0, "media": 1, "alta": 2}
_now = time.monotonic  # reloj del plazo compartido; los tests lo reemplazan

DISCLAIMER = (
    "Texto generado por IA (OpenAI) a partir de la foto; puede equivocarse. "
    "No reemplaza la evaluación de un técnico."
)

# Criterios de referencia: escala lechera 1–5 de Elanco (Edmonson et al., 1989, J Dairy Sci
# 72:68–78) y el diagrama de decisión de Ferguson, Galligan y Thomsen (1994, J Dairy Sci
# 77:2695–2703): V/U del anca, isquiones, ligamentos sacro y de la base de la cola, plano entre
# ilíacos e isquiones. Son guía para mirar, no texto para copiar.
CRITERIA = """\
Criterios de referencia (escala lechera 1–5, Elanco / Ferguson et al. 1994), mirando desde atrás \
y arriba:
- Condición BAJA (BCS ≈ 3.0–3.25 o menos): anca en V (≤ 3.0) o apenas redondeada (3.25). Ilíacos \
(puntas de cadera) e isquiones (puntas de nalga) marcados, con bordes que se ven; isquiones con \
poca cobertura. Ligamento sacro y ligamento de la base de la cola visibles como cuerdas. Hueco \
profundo a cada lado de la base de la cola. Apófisis de la columna (espinosas y transversas) que \
se insinúan como una fila; lomo con escalón hacia los costados.
- Condición MEDIA (BCS ≈ 3.5–3.75): anca en U, ilíacos e isquiones redondeados, con algo de \
grasa. Ligamento sacro visible (3.5) o apenas (3.75); ligamento de la base de la cola apenas \
visible (3.5) o no visible (3.75). Hueco junto a la cola poco profundo. Apófisis cubiertas, lomo \
redondeado.
- Condición ALTA (BCS ≈ 4.0–4.25 o más): ligamentos sacro y de la base de la cola no se ven. Zona \
entre ilíacos e isquiones plana (4.0). Isquiones tapados por grasa, casi no se distinguen (4.25). \
Hueco de la cola relleno o casi; pliegues o rollos de grasa alrededor de la cola. Lomo plano y \
ancho, sin apófisis; ilíacos poco visibles (≥ 4.5).
Lo que más separa los rangos es: profundidad del hueco a los costados de la base de la cola, \
cuánto se marcan isquiones e ilíacos y si se ven los ligamentos. "Redondeado" solo no alcanza: \
decí qué tan profundo, qué tan marcado, qué se ve y qué no."""

DESCRIBE_PROMPT = f"""\
Sos un técnico en condición corporal (BCS) de vacas lecheras. Te muestran una foto de una vaca \
tomada desde atrás y arriba. No sabés qué puntaje le dio nadie: describí y estimá vos.

{CRITERIA}

Reglas:
- Si hay varias vacas, hablá solo de la vaca central vista desde atrás.
- traits: una entrada por región, en este orden: ilíacos; isquiones; base de la cola y hueco; \
ligamentos sacro y de la cola; lomo y apófisis; forma del anca (V/U). En cada una describí lo \
concreto (qué tan profundo, qué tan marcado, qué se ve y qué no). Si una región no se ve \
(tapada, fuera de cuadro, oscura, borrosa), decilo y no la uses para estimar.
- photo_estimate: tu rango según los criterios: "baja" (≤ 3.25), "media" (3.5–3.75), "alta" \
(≥ 4.0), o "no_se_puede" si la foto no deja ver lo suficiente (ninguna de las regiones clave: \
hueco de la cola, isquiones, ligamentos).
- estimate_reason: 1 o 2 oraciones con las señales concretas que decidieron el rango.
- Español rioplatense (vos), claro, para un productor. Sin adjetivos vacíos."""

COMPARE_PROMPT = """\
Sos un técnico que le explica a un productor de leche un puntaje de condición corporal (BCS) \
calculado por un sistema a partir de una foto de su vaca tomada desde atrás.

Te paso: (1) la descripción de la foto que hizo otro técnico SIN conocer el puntaje, con su rango \
estimado; (2) el BCS que dio el sistema; (3) si ese rango y el del BCS coinciden.

Escribí explanation (2 a 4 oraciones) con estas reglas:
- Basate solo en la descripción ciega; no agregues rasgos que no estén ahí.
- Si coinciden, explicá el puntaje con las señales concretas de la descripción (hueco de la cola, \
isquiones, ilíacos, ligamentos, forma del anca).
- Si no coinciden, decí claramente que la foto sugiere una condición más baja (o más alta) que \
el puntaje y por qué, con esas señales. No lo suavices ni avales el puntaje por defecto.
- Si la foto no deja ver lo suficiente, decilo y qué faltó ver.
- No hables del sistema, de su confianza, de su estado ni de verificaciones; no digas \
"confianza", "verificar a mano" ni "compatible con el puntaje dado".
- Español rioplatense (vos), claro y breve, para un productor."""

TRAITS_SCHEMA: dict[str, Any] = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "region": {"type": "string"},
            "observation": {"type": "string"},
        },
        "required": ["region", "observation"],
        "additionalProperties": False,
    },
}

# Orden de las propiedades = orden de generación: primero describe, después estima.
DESCRIBE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "traits": TRAITS_SCHEMA,
        "estimate_reason": {"type": "string"},
        "photo_estimate": {"type": "string", "enum": list(ESTIMATES)},
    },
    "required": ["traits", "estimate_reason", "photo_estimate"],
    "additionalProperties": False,
}

COMPARE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"explanation": {"type": "string"}},
    "required": ["explanation"],
    "additionalProperties": False,
}

ESTIMATE_TEXT = {
    "baja": "condición baja (BCS ≤ 3.25)",
    "media": "condición media (BCS 3.5–3.75)",
    "alta": "condición alta (BCS ≥ 4.0)",
    "no_se_puede": "la foto no deja ver lo suficiente",
}
AGREEMENT_TEXT = {
    "coincide": "coincide con el puntaje",
    "sugiere_mas_baja": "la foto sugiere una condición más baja que el puntaje",
    "sugiere_mas_alta": "la foto sugiere una condición más alta que el puntaje",
    "no_se_puede": "no se puede comparar con el puntaje",
}


class ExplainError(Exception):
    """Error al explicar; `status_code` es el que devuelve la API (nunca 500)."""

    status_code = 502

    def __init__(self, detail: str) -> None:
        super().__init__(detail)
        self.detail = detail


class ExplainUnavailable(ExplainError):
    status_code = 503


class ExplainTimeout(ExplainError):
    status_code = 504


class ExplainBadResponse(ExplainError):
    status_code = 502


def model_name() -> str:
    return os.environ.get("OPENAI_MODEL", "").strip() or DEFAULT_MODEL


def reasoning_effort() -> str:
    return os.environ.get("OPENAI_REASONING_EFFORT", DEFAULT_REASONING_EFFORT).strip()


def timeout_s() -> float:
    try:
        value = float(os.environ.get("OPENAI_TIMEOUT", MAX_TIMEOUT_S))
    except ValueError:
        value = MAX_TIMEOUT_S
    return min(max(value, 1.0), MAX_TIMEOUT_S)


def bcs_range(bcs: float) -> str:
    """Rango del BCS predicho, con los mismos cortes que `photo_estimate`."""
    if bcs <= 3.25 + 1e-9:
        return "baja"
    if bcs <= 3.75 + 1e-9:
        return "media"
    return "alta"


def agreement(photo_estimate: str, bcs: float) -> Agreement:
    """Se decide en código, no lo elige el modelo: así no puede avalar el puntaje."""
    if photo_estimate not in _ORDER:
        return "no_se_puede"
    diff = _ORDER[photo_estimate] - _ORDER[bcs_range(bcs)]
    return "coincide" if diff == 0 else "sugiere_mas_baja" if diff < 0 else "sugiere_mas_alta"


def _request(
    model: str, system: str, content: list[dict[str, Any]], name: str, schema: dict[str, Any]
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": model,
        "input": [
            {"role": "system", "content": system},
            {"role": "user", "content": content},
        ],
        "text": {"format": {"type": "json_schema", "name": name, "schema": schema, "strict": True}},
        "max_output_tokens": MAX_OUTPUT_TOKENS,
        "store": False,
    }
    if effort := reasoning_effort():
        body["reasoning"] = {"effort": effort}
    return body


def describe_request(thumbnail: bytes, model: str) -> dict[str, Any]:
    """Paso 1: la foto sola. Ni el BCS, ni el estado, ni el `cow_id`."""
    image_url = "data:image/jpeg;base64," + base64.b64encode(thumbnail).decode("ascii")
    content = [
        {"type": "input_text", "text": "Describí esta vaca y estimá su rango de condición."},
        {"type": "input_image", "image_url": image_url, "detail": "high"},
    ]
    return _request(model, DESCRIBE_PROMPT, content, "bcs_lectura_ciega", DESCRIBE_SCHEMA)


def compare_request(
    blind: dict[str, Any], bcs: float, agreed: Agreement, model: str
) -> dict[str, Any]:
    """Paso 2: solo texto, con la descripción ciega, el BCS y el acuerdo ya calculado."""
    traits = "\n".join(f"- {t['region']}: {t['observation']}" for t in blind["traits"])
    text = (
        f"Descripción ciega de la foto:\n{traits}\n"
        f"Rango estimado por el técnico: {ESTIMATE_TEXT[blind['photo_estimate']]}. "
        f"Por qué: {blind['estimate_reason']}\n\n"
        f"BCS dado por el sistema: {bcs:.2f} ({ESTIMATE_TEXT[bcs_range(bcs)]}).\n"
        f"Comparación: {AGREEMENT_TEXT[agreed]}."
    )
    content = [{"type": "input_text", "text": text}]
    return _request(model, COMPARE_PROMPT, content, "bcs_explicacion", COMPARE_SCHEMA)


def _output_text(body: dict[str, Any]) -> str:
    if body.get("status") == "incomplete":
        raise ExplainBadResponse("la respuesta de OpenAI quedó incompleta")
    if isinstance(body.get("output_text"), str):
        return body["output_text"]
    for item in body.get("output") or []:
        if not isinstance(item, dict) or item.get("type") != "message":
            continue
        for part in item.get("content") or []:
            if not isinstance(part, dict):
                continue
            if part.get("type") == "refusal":
                raise ExplainBadResponse("OpenAI se negó a explicar esta foto")
            if part.get("type") == "output_text" and isinstance(part.get("text"), str):
                return part["text"]
    raise ExplainBadResponse("la respuesta de OpenAI no trae texto")


def _json_output(body: Any) -> dict[str, Any]:
    if not isinstance(body, dict):
        raise ExplainBadResponse("la respuesta de OpenAI no es JSON")
    try:
        data = json.loads(_output_text(body))
    except json.JSONDecodeError as e:
        raise ExplainBadResponse("la respuesta de OpenAI no es JSON válido") from e
    if not isinstance(data, dict):
        raise ExplainBadResponse("la respuesta de OpenAI no tiene el formato esperado")
    return data


def _clip(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ExplainBadResponse("la respuesta de OpenAI no tiene el formato esperado")
    return value.strip()[:MAX_TEXT]


def parse_describe(body: Any) -> dict[str, Any]:
    """{traits, estimate_reason, photo_estimate} del paso 1. `ExplainBadResponse` si no sirve."""
    data = _json_output(body)
    if not isinstance(data.get("traits"), list) or data.get("photo_estimate") not in ESTIMATES:
        raise ExplainBadResponse("la respuesta de OpenAI no tiene el formato esperado")
    traits = []
    for t in data["traits"][:MAX_TRAITS]:
        if not isinstance(t, dict):
            raise ExplainBadResponse("la respuesta de OpenAI no tiene el formato esperado")
        traits.append(
            {"region": _clip(t.get("region")), "observation": _clip(t.get("observation"))}
        )
    if not traits:
        raise ExplainBadResponse("la respuesta de OpenAI no tiene el formato esperado")
    return {
        "traits": traits,
        "estimate_reason": _clip(data.get("estimate_reason")),
        "photo_estimate": data["photo_estimate"],
    }


def parse_compare(body: Any) -> str:
    return _clip(_json_output(body).get("explanation"))


def _call(client: httpx.Client, api_key: str, body: dict[str, Any], deadline: float) -> Any:
    remaining = deadline - _now()
    if remaining < 1.0:
        raise ExplainTimeout("OpenAI tardó demasiado en responder; probá de nuevo")
    try:
        r = client.post(
            OPENAI_URL,
            json=body,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=remaining,
        )
    except httpx.TimeoutException as e:
        raise ExplainTimeout("OpenAI tardó demasiado en responder; probá de nuevo") from e
    except httpx.HTTPError as e:
        raise ExplainBadResponse("no se pudo conectar con OpenAI") from e
    # El cuerpo de un error de OpenAI puede citar parte de la clave: no se reenvía ni se loguea.
    if r.status_code in (401, 403):
        raise ExplainUnavailable("explicación no disponible: OpenAI rechazó la clave configurada")
    if r.status_code == 429:
        raise ExplainUnavailable("explicación no disponible: límite de uso de OpenAI alcanzado")
    if r.status_code != 200:
        raise ExplainBadResponse(f"OpenAI respondió con error {r.status_code}")
    try:
        return r.json()
    except ValueError as e:
        raise ExplainBadResponse("la respuesta de OpenAI no es JSON") from e


def explain(pred: Prediction, thumbnail: bytes, client: httpx.Client) -> dict[str, Any]:
    """Pide la explicación a OpenAI. Devuelve el JSON del endpoint `/predictions/{id}/explain`."""
    if pred.bcs is None or pred.status == "rejected":
        raise ValueError("solo se explican predicciones con BCS")
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise ExplainUnavailable("explicación no disponible: falta configurar OPENAI_API_KEY")
    model = model_name()
    deadline = _now() + timeout_s()
    body = _call(client, api_key, describe_request(thumbnail, model), deadline)
    blind = parse_describe(body)
    agreed = agreement(blind["photo_estimate"], pred.bcs)
    explanation = parse_compare(
        _call(client, api_key, compare_request(blind, pred.bcs, agreed, model), deadline)
    )
    return {
        "prediction_id": pred.prediction_id,
        "bcs": pred.bcs,
        "photo_estimate": blind["photo_estimate"],
        "agreement": agreed,
        "explanation": explanation,
        "traits": blind["traits"],
        "model": str(body.get("model") or model),
        "generated_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "disclaimer": DISCLAIMER,
        "prompt_version": PROMPT_VERSION,
    }
