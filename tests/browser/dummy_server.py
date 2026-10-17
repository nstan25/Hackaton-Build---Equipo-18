"""API dummy para `make browser`: la app real (SqliteStore, token, `web/`) con un `predict` de test.

Hasta T12, `inference.predict` devuelve siempre el fixture `ok`. Para ver en el navegador los tres
estados, este servidor lo envuelve según la foto de fixture que llega (por sha256):
- `dog.jpg` → `rejected` (pasa por `run_pipeline` con un filtro de calidad que rechaza);
- `side_cow.jpg` → `low_confidence` (distribución ancha armada a mano, validada contra el contrato);
- cualquier otra → el dummy de T00 sin cambios.

Uso: `python tests/browser/dummy_server.py <puerto>` con `BCS_APP_TOKEN`, `BCS_DB_PATH` y
`BCS_MEDIA_DIR` en el entorno. Agrega `/__harness` (sin token, solo acá): un iframe de 390 px.
"""

from __future__ import annotations

import hashlib
import sys
from dataclasses import replace
from pathlib import Path

import uvicorn
from fastapi.responses import HTMLResponse
from PIL import Image

from bcs import inference
from bcs.api import create_app
from bcs.contract import Prediction, QualityReport, validate_prediction_dict
from bcs.scale import load_scale

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
VIEWPORT = (390, 844)


def _sha(name: str) -> str:
    return hashlib.sha256((FIXTURES / name).read_bytes()).hexdigest()


DOG_SHA = _sha("dog.jpg")
SIDE_SHA = _sha("side_cow.jpg")
LOW_DIST = {"3.00": 0.1, "3.25": 0.27, "3.50": 0.2, "3.75": 0.2, "4.00": 0.13, "4.25": 0.1}


class RejectChecker:
    def check(self, image: Image.Image) -> QualityReport:
        return QualityReport(is_cow_rear=False, score=0.03, issues=["no parece una vaca de atrás"])


def _low_confidence(base: Prediction) -> Prediction:
    scale = load_scale()
    dist = {k: LOW_DIST.get(k, 0.0) for k in scale.keys}
    pred = replace(
        base,
        status="low_confidence",
        bcs=3.25,
        bcs_expected=round(sum(float(k) * p for k, p in dist.items()), 4),
        confidence=0.57,
        interval=(3.0, 4.25),  # 5 pasos > 4 → low_confidence (SPEC §2.2)
        distribution=dist,
    )
    errors = validate_prediction_dict(pred.to_dict(), scale.keys, scale.step)
    assert not errors, errors
    return pred


_dummy_predict = inference.predict


def browser_predict(image: inference.ImageInput, cow_id: str) -> Prediction:
    sha = hashlib.sha256(image).hexdigest() if isinstance(image, bytes) else ""
    if sha == DOG_SHA:
        return inference.run_pipeline(
            replace(inference.stub_pipeline(), quality=RejectChecker()), image, cow_id
        )
    pred = _dummy_predict(image, cow_id)
    return _low_confidence(pred) if sha == SIDE_SHA else pred


HARNESS = f"""<!doctype html><meta charset="utf-8"><title>harness</title>
<style>body{{margin:0;background:#888}}iframe{{border:0;display:block;background:#fff}}</style>
<iframe id="app" width="{VIEWPORT[0]}" height="{VIEWPORT[1]}"></iframe>"""


def make_app():
    inference.predict = browser_predict
    app = create_app()
    app.add_api_route("/__harness", lambda: HTMLResponse(HARNESS), include_in_schema=False)
    # Delante del comodín de `web/` para que no lo tape.
    app.router.routes.insert(0, app.router.routes.pop())
    return app


if __name__ == "__main__":
    uvicorn.run(make_app(), host="127.0.0.1", port=int(sys.argv[1]), log_level="warning")
