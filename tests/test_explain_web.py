"""La web ofrece la explicación (T17): botón, endpoint y etiqueta de texto generado por IA."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

LABEL = "Texto generado por IA (OpenAI) a partir de la foto; puede equivocarse."


@pytest.mark.parametrize("page", ["index.html", "rodeo.html"])
def test_page_has_explain_button(client: TestClient, page: str) -> None:
    r = client.get(f"/{page}")
    assert r.status_code == 200
    html = r.text
    assert "¿Por qué este puntaje?" in html
    assert "/explain`" in html
    assert LABEL in html
    # el acuerdo entre la foto y el puntaje, en una línea
    assert "agreementText(data)" in html
    for key in ("coincide", "sugiere_mas_baja", "sugiere_mas_alta"):
        assert f"{key}:" in html


def test_rodeo_hides_button_for_rejected(client: TestClient) -> None:
    html = client.get("/rodeo.html").text
    rejected = html.index('if (p.status === "rejected") {')
    assert html.index("explainButton(p)", rejected) > rejected
    assert "} else {\n          info.append(explainButton(p));" in html
