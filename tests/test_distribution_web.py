"""El rodeo muestra la distribución de condición corporal (T20): textos, estados y SVG propio."""

from __future__ import annotations

import re

from fastapi.testclient import TestClient


def test_rodeo_has_distribution_section(client: TestClient) -> None:
    html = client.get("/rodeo.html").text
    for text in (
        "Distribución de condición corporal del rodeo",
        "Cantidad de animales según su condición corporal estimada",
        "Animales evaluados",
        "BCS promedio",
        "Cargando distribución…",
        "Todavía no hay animales con una evaluación válida",
        "No se pudo cargar la distribución.",
        "BCS promedio de las evaluaciones de cada vaca",  # tooltip: es el promedio por vaca
    ):
        assert text in html, text
    assert 'api("/cows/distribution")' in html
    # se recarga con la lista: loadCows() pide también la distribución
    load = html.index("async function loadCows()")
    assert html.index("loadDistribution();", load) < html.index("await api", load)


def test_distribution_without_chart_libraries(client: TestClient) -> None:
    html = client.get("/rodeo.html").text
    scripts = re.findall(r"<script[^>]*\bsrc=", html)
    assert scripts == []  # SVG propio, sin Chart.js/D3/etc.
    css = client.get("/argos.css").text
    assert ".dist-chart" in css and ".dist-tip" in css
    # la animación de entrada solo corre sin "reducir movimiento"
    motion = css.index("@media (prefers-reduced-motion: no-preference) {\n  .dist-chart.animate")
    assert css.index("dist-grow", motion) > motion
