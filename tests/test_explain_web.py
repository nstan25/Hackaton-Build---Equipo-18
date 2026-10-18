"""La web ofrece "Qué se observa en la foto" (T17): botón, título, resumen y regiones."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

LABEL = "Texto generado por IA (OpenAI) a partir de la foto; puede equivocarse."


@pytest.mark.parametrize("page", ["index.html", "rodeo.html"])
def test_page_has_explain_button(client: TestClient, page: str) -> None:
    r = client.get(f"/{page}")
    assert r.status_code == 200
    html = r.text
    assert "Ver qué se observa en la foto" in html
    assert "Qué se observa en la foto</p>" in html or '"Qué se observa en la foto")' in html
    assert "/explain`" in html
    assert LABEL not in html and "ai-label" not in html  # el humano sacó la etiqueta de IA
    # decisión del humano (vuelta 3): no se muestra ningún acuerdo ni rango de la foto
    for word in ("agreement", "photo_estimate", "sugiere", "¿Por qué este puntaje?"):
        assert word not in html


def test_rodeo_hides_button_for_rejected(client: TestClient) -> None:
    html = client.get("/rodeo.html").text
    rejected = html.index('if (p.status === "rejected") {')
    assert html.index("explainButton(p)", rejected) > rejected
    assert "} else {\n          info.append(explainButton(p));" in html
