# Fixtures

Todo lo que hay acá es **generado**: no hay imágenes ni datos de terceros (SPEC §1.1, §4.1).

| Archivo | Qué es | Origen |
|---|---|---|
| `rear_cow.jpg` | dibujo de una vaca Holando desde atrás (grupa, ilíacos, isquiones, cola) | `scripts/make_fixtures.py` (PIL), dominio público |
| `side_cow.jpg` | dibujo de una vaca de costado | ídem |
| `dog.jpg` | dibujo de un perro de costado | ídem |
| `prediction_example.json` | ejemplo del contrato `Prediction` (SPEC §2.2); `image_sha256` es el de `rear_cow.jpg` | escrito a mano en T00 |
| `labels_fixture.csv` | 11 filas con el esquema de `labels.csv` (SPEC §1.1), fuente `synthetic`; `path` relativo a `tests/` | escrito a mano en T00 |

Regenerar las imágenes: `uv run python scripts/make_fixtures.py`. Si cambian los bytes de
`rear_cow.jpg`, hay que actualizar `image_sha256` en `prediction_example.json`.

`prediction_example.json` es el contrato entre modelo y app: se congela en G0.
