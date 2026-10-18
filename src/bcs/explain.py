"""Qué se observa en la foto, según un modelo multimodal de OpenAI (T17).

Una sola llamada **ciega**: el modelo ve la foto y una guía de qué mirar (ilíacos, isquiones,
hueco de la cola, ligamentos, lomo, forma del anca), pero no el BCS predicho, ni el estado, ni el
`cow_id`. Devuelve una observación por región y un resumen neutral. No compara con el puntaje ni da
un rango propio (G3, decisión del humano en la vuelta 3: la lectura ciega acertaba el rango real
solo en 3/6 fotos de val y contradecía predicciones correctas).

La clave se lee **solo** de `OPENAI_API_KEY` y nunca se loguea ni se devuelve. Se llama a la
Responses API con `httpx` (sin el SDK) y salida estructurada (`text.format` = `json_schema`).

Latencia (T19, 6 fotos de val): la llama la domina la salida (~40 tokens/s; la imagen son ~770
tokens con cualquier `detail` y el razonamiento "low" casi no genera tokens). Pedir una oración
breve por región bajó la media de 12,8 s a 9,5 s sin perder regiones; `gpt-6-luna` tarda 3,5–4 s
pero repite la misma descripción en todas las fotos (degrada), y `detail: "low"` no ayuda.
"""

from __future__ import annotations

import base64
import json
import os
import threading
from datetime import UTC, datetime
from typing import Any

import httpx

from bcs.contract import Prediction

OPENAI_URL = "https://api.openai.com/v1/responses"
# Default: modelo multimodal vigente de la familia GPT-6 (docs de OpenAI, 18/10/2026). Se pisa con
# OPENAI_MODEL; `gpt-6-luna` es ~20 veces más barato y ~2,5 veces más rápido, pero sus textos son
# genéricos (T19). Con razonamiento "low" responde lejos del timeout de 30 s.
DEFAULT_MODEL = "gpt-6.1-sol"
# OPENAI_REASONING_EFFORT; vacío = no se manda (modelos sin razonamiento). "low" es el mínimo que
# admite gpt-6.1-sol ("none" y "minimal" dan 400).
DEFAULT_REASONING_EFFORT = "low"
MAX_TIMEOUT_S = 30.0
MAX_OUTPUT_TOKENS = 4000  # incluye el razonamiento interno del modelo, si lo tiene
MAX_TRAITS = 12
MAX_TEXT = 2000
# Versión del prompt: entra en la cache, así una explicación vieja se regenera al cambiarlo.
PROMPT_VERSION = "4-breve"
# A lo sumo tantas llamadas a OpenAI a la vez (las de fondo de `/predict` y las del endpoint).
MAX_CONCURRENT_CALLS = 2
_call_slots = threading.BoundedSemaphore(MAX_CONCURRENT_CALLS)

DISCLAIMER = (
    "Texto generado por IA (OpenAI) a partir de la foto; puede equivocarse. "
    "No reemplaza la evaluación de un técnico."
)

# Qué mirar y con qué gradaciones describirlo: son los signos de la escala lechera de Elanco
# (Edmonson et al., 1989, J Dairy Sci 72:68–78) y del diagrama de Ferguson, Galligan y Thomsen
# (1994, J Dairy Sci 77:2695–2703), sin la correspondencia con puntajes: el modelo describe, no
# puntúa.
GUIDE = """\
Qué mirar, desde atrás y arriba, y cómo describirlo con precisión:
- Ilíacos (puntas de cadera): ¿se ven con borde marcado, redondeados o casi tapados?
- Isquiones (puntas de nalga): ¿sobresalen con el borde a la vista, están redondeados con algo de \
cobertura, o tapados y casi no se distinguen?
- Base de la cola y hueco a cada lado: ¿hueco profundo, poco profundo o relleno? ¿Hay pliegues o \
rollos alrededor de la cola?
- Ligamento sacro y ligamento de la base de la cola: ¿se ven como cuerdas, apenas, o no se ven?
- Lomo y apófisis de la columna: ¿se insinúa una fila de apófisis o un escalón hacia los costados, \
o el lomo se ve cubierto, redondeado o plano y ancho?
- Forma del anca vista desde atrás: ¿más en V (angulosa) o más en U (redondeada)? ¿La zona entre \
cadera y nalga se ve hundida o plana?
"Redondeado" solo no alcanza: decí qué tan profundo, qué tan marcado, qué se ve y qué no."""

SYSTEM_PROMPT = f"""\
Sos un técnico en condición corporal de vacas lecheras. Te muestran una foto de una vaca tomada \
desde atrás y arriba. Tu tarea es describir lo que se observa, no puntuar.

{GUIDE}

Reglas:
- Si hay varias vacas, hablá solo de la vaca central vista desde atrás.
- traits: una entrada por región, en este orden: ilíacos; isquiones; base de la cola y hueco; \
ligamentos sacro y de la cola; lomo y apófisis; forma del anca. Si una región no se ve (tapada, \
fuera de cuadro, oscura, borrosa), decilo en su observación. Cada observación es UNA oración \
concreta de 25 palabras como máximo: qué se ve y qué tan marcado.
- summary: 2 oraciones neutrales y breves con lo más relevante que se observa (huesos, huecos de \
la cola, forma del anca, cobertura) y lo que no se puede ver.
- No des ningún puntaje, número de BCS ni rango, y no califiques la condición en general (nada de \
"condición baja/media/alta/buena", "flaca" o "gorda"): solo lo observable.
- Español rioplatense (vos), claro, para un productor. Sin adjetivos vacíos."""

RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "traits": {
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
        },
        "summary": {"type": "string"},
    },
    "required": ["traits", "summary"],
    "additionalProperties": False,
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


def build_request(thumbnail: bytes, model: str) -> dict[str, Any]:
    """La foto sola: ni el BCS, ni el estado, ni el `cow_id`."""
    image_url = "data:image/jpeg;base64," + base64.b64encode(thumbnail).decode("ascii")
    body: dict[str, Any] = {
        "model": model,
        "input": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "Describí lo que se observa en esta vaca."},
                    {"type": "input_image", "image_url": image_url, "detail": "high"},
                ],
            },
        ],
        "text": {
            "format": {
                "type": "json_schema",
                "name": "bcs_observacion",
                "schema": RESPONSE_SCHEMA,
                "strict": True,
            }
        },
        "max_output_tokens": MAX_OUTPUT_TOKENS,
        "store": False,
    }
    if effort := reasoning_effort():
        body["reasoning"] = {"effort": effort}
    return body


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


def _clip(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ExplainBadResponse("la respuesta de OpenAI no tiene el formato esperado")
    return value.strip()[:MAX_TEXT]


def parse_response(body: Any) -> tuple[str, list[dict[str, str]]]:
    """(summary, traits) de la respuesta de la Responses API.

    `ExplainBadResponse` si no tiene el formato pedido."""
    if not isinstance(body, dict):
        raise ExplainBadResponse("la respuesta de OpenAI no es JSON")
    try:
        data = json.loads(_output_text(body))
    except json.JSONDecodeError as e:
        raise ExplainBadResponse("la respuesta de OpenAI no es JSON válido") from e
    if not isinstance(data, dict) or not isinstance(data.get("traits"), list):
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
    return _clip(data.get("summary")), traits


def configured() -> bool:
    return bool(os.environ.get("OPENAI_API_KEY", "").strip())


def explain(pred: Prediction, thumbnail: bytes, client: httpx.Client) -> dict[str, Any]:
    """Pide la observación a OpenAI. Devuelve el JSON del endpoint `/predictions/{id}/explain`."""
    if pred.bcs is None or pred.status == "rejected":
        raise ValueError("solo se explican predicciones con BCS")
    return for_prediction(observe(thumbnail, client), pred)


def for_prediction(observation: dict[str, Any], pred: Prediction) -> dict[str, Any]:
    """El JSON del endpoint: la observación ciega + el `prediction_id` y el BCS que explica."""
    return {"prediction_id": pred.prediction_id, "bcs": pred.bcs, **observation}


def observe(thumbnail: bytes, client: httpx.Client) -> dict[str, Any]:
    """La observación ciega de la foto (solo la imagen: sirve antes de tener la predicción)."""
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise ExplainUnavailable("explicación no disponible: falta configurar OPENAI_API_KEY")
    model = model_name()
    try:
        with _call_slots:
            r = client.post(
                OPENAI_URL,
                json=build_request(thumbnail, model),
                headers={"Authorization": f"Bearer {api_key}"},
                timeout=timeout_s(),
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
        body = r.json()
    except ValueError as e:
        raise ExplainBadResponse("la respuesta de OpenAI no es JSON") from e
    summary, traits = parse_response(body)
    return {
        "explanation": summary,
        "traits": traits,
        "model": str(body.get("model") or model),
        "generated_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "disclaimer": DISCLAIMER,
        "prompt_version": PROMPT_VERSION,
    }
