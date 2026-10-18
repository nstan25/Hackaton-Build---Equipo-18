"""Explicación del puntaje con un modelo multimodal de OpenAI (T17).

A partir de la miniatura guardada de una predicción y de su `bcs`, se le pide al modelo que describa
qué se ve en la foto (ilíacos, isquiones, base de la cola, cobertura de grasa, etc.) y cómo se
relaciona con ese puntaje. El modelo no puntúa: el BCS es el de la predicción.

La clave se lee **solo** de `OPENAI_API_KEY` y nunca se loguea ni se devuelve. Se llama a la
Responses API con `httpx` (sin el SDK) y salida estructurada (`text.format` = `json_schema`).
"""

from __future__ import annotations

import base64
import json
import os
from datetime import UTC, datetime
from typing import Any

import httpx

from bcs.contract import Prediction

OPENAI_URL = "https://api.openai.com/v1/responses"
# Default: modelo multimodal vigente de la familia GPT-6 (docs de OpenAI, 18/10/2026). Se pisa con
# OPENAI_MODEL; p. ej. `gpt-6-luna` es ~20 veces más barato.
DEFAULT_MODEL = "gpt-6.1-sol"
MAX_TIMEOUT_S = 30.0
MAX_OUTPUT_TOKENS = 4000  # incluye el razonamiento interno del modelo, si lo tiene
MAX_TRAITS = 12
MAX_TEXT = 2000

DISCLAIMER = (
    "Texto generado por IA (OpenAI) a partir de la foto; puede equivocarse. "
    "No reemplaza la evaluación de un técnico."
)

SYSTEM_PROMPT = """\
Sos un asistente que ayuda a productores de leche a entender un puntaje de condición corporal \
(BCS) que ya calculó otro sistema a partir de una foto de la vaca tomada desde atrás y arriba.

Reglas:
- Describí solo lo que se ve en la foto. No inventes detalles que no se distinguen.
- Relacioná lo que ves con el puntaje dado: qué rasgos son coherentes con ese BCS y cuáles, si \
alguno, lo contradicen.
- Mirá, si se ven: ilíacos (puntas de cadera, hook bones), isquiones (pin bones), base de la cola \
y su hueco, ligamento sacro, apófisis de la columna (espinosas y transversas), línea del lomo, \
cobertura de grasa, forma del anca vista desde atrás (en V = más flaca, en U = más gorda) y postura.
- Si una región no se ve (tapada, fuera de cuadro, borrosa, oscura), decilo en su observación. \
Si la foto no sirve para juzgar la condición corporal, decilo en la explicación.
- No des otro puntaje ni digas que el BCS "debería ser" otro número. Si lo que ves no coincide \
con el puntaje, decí qué rasgo no coincide y sugerí verificarlo a mano.
- Escribí en español rioplatense (vos), claro y breve, para un productor; sin jerga innecesaria.
- explanation: 2 a 4 oraciones. traits: una entrada por región que comentás (entre 3 y 8)."""

RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "explanation": {"type": "string"},
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
    },
    "required": ["explanation", "traits"],
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


def timeout_s() -> float:
    try:
        value = float(os.environ.get("OPENAI_TIMEOUT", MAX_TIMEOUT_S))
    except ValueError:
        value = MAX_TIMEOUT_S
    return min(max(value, 1.0), MAX_TIMEOUT_S)


def user_prompt(pred: Prediction) -> str:
    """Datos de la predicción para el modelo. El `cow_id` no se manda."""
    assert pred.bcs is not None
    lines = [
        f"Puntaje BCS dado por el sistema: {pred.bcs:.2f} (escala {pred.scale}; 1 = muy flaca, "
        "5 = muy gorda en la escala lechera 1–5).",
    ]
    if pred.interval is not None:
        lo, hi = pred.interval
        lines.append(f"Rango plausible según el sistema: {lo:.2f}–{hi:.2f}.")
    if pred.status == "low_confidence":
        lines.append("El sistema marcó este resultado como de confianza baja.")
    lines.append(
        "Explicá, mirando la foto, por qué es razonable (o no) ese puntaje. "
        "Respondé en el JSON pedido."
    )
    return "\n".join(lines)


def build_request(pred: Prediction, thumbnail: bytes, model: str) -> dict[str, Any]:
    image_url = "data:image/jpeg;base64," + base64.b64encode(thumbnail).decode("ascii")
    return {
        "model": model,
        "input": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": user_prompt(pred)},
                    {"type": "input_image", "image_url": image_url, "detail": "high"},
                ],
            },
        ],
        "text": {
            "format": {
                "type": "json_schema",
                "name": "bcs_explanation",
                "schema": RESPONSE_SCHEMA,
                "strict": True,
            }
        },
        "max_output_tokens": MAX_OUTPUT_TOKENS,
        "store": False,
    }


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
    """(explanation, traits) de la respuesta de la Responses API.

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
    return _clip(data.get("explanation")), traits


def explain(pred: Prediction, thumbnail: bytes, client: httpx.Client) -> dict[str, Any]:
    """Pide la explicación a OpenAI. Devuelve el JSON del endpoint `/predictions/{id}/explain`."""
    if pred.bcs is None or pred.status == "rejected":
        raise ValueError("solo se explican predicciones con BCS")
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise ExplainUnavailable("explicación no disponible: falta configurar OPENAI_API_KEY")
    model = model_name()
    try:
        r = client.post(
            OPENAI_URL,
            json=build_request(pred, thumbnail, model),
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
    explanation, traits = parse_response(body)
    return {
        "prediction_id": pred.prediction_id,
        "bcs": pred.bcs,
        "explanation": explanation,
        "traits": traits,
        "model": str(body.get("model") or model),
        "generated_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "disclaimer": DISCLAIMER,
    }
