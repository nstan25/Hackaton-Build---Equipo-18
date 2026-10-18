# T13-fix-1 · Robustez de la API (hallazgos R3, R4, R5, R6 y R16 de eval/review.md)
Rol: worker. Leé SPEC.md §0–§4 y §6, `eval/review.md` (R3–R6, R16) y `tasks/G3.md`.

## Objetivo
- **R3 (alta):** bomba de memoria. Limitar píxeles, no solo bytes: chequear `img.size` justo después de `Image.open` y antes de `load()`/`exif_transpose` (tope sugerido 40 MP; para JPEG grandes podés usar `draft()` para decodificar reducido), responder 413 o 422 con mensaje claro. Generar la miniatura desde la imagen ya decodificada (una sola decodificación) y que no corra fuera de control de memoria en paralelo. Test con una PNG de un color de 10000×8000 (se genera en el test, ~250 KB): responde 413/422 y el RSS no se dispara.
- **R4 (media):** un archivo que no es imagen (texto, PDF, HEIC sin soporte) tiene que dar **422** con mensaje claro, nunca 500 (ver el parche de ultralytics a `Image.open` descrito en R4). Tests.
- **R5 (media):** el token `?token=` queda en el access log y la web lo pone en cada URL de miniatura. Redactá el token en los logs de acceso (filtro de logging de uvicorn que reemplace `token=...` por `token=***`) y en la web cargá las miniaturas con `fetch` + header `X-App-Token` (blob URL) en vez de `?token=` en el `src`. El link compartible con `?token=` sigue funcionando para entrar; la web lo pasa a header/almacenamiento local y lo saca de la barra de direcciones con `history.replaceState`.
- **R6 (media):** chequear el token **antes** de leer/guardar el cuerpo (dependencia de auth antes de parsear el multipart) y cortar cuerpos > 20 MB sin guardarlos en disco. Test: sin token y 50 MB → 401 sin escribir en disco.
- **R16 (info):** docstrings que todavía hablan de CLIP como filtro activo → actualizar (G3 punto 5).

## Owns (solo podés editar esto)
- src/bcs/api_predict.py, src/bcs/auth.py, src/bcs/api.py (solo para el filtro de logging o el orden de dependencias)
- src/bcs/inference.py (solo `_to_image` y docstrings), src/bcs/store/memory.py, src/bcs/store/sqlite.py (solo miniaturas)
- web/** (solo carga de miniaturas y manejo del token)
- tests/test_api_robustness_*, tests/test_api_predict_*

## Restricciones
- No cambiar el contrato `Prediction` ni las reglas de status. No tocar `tests/test_contract.py` ni `eval/thresholds.yaml`.
- **No corras Firefox / make browser** (cuelga la VM). Validá la web con `make smoke` y tests de API.
- Commits chicos y frecuentes (la VM se cuelga a veces). Máx. 5 vueltas.

## Validaciones (las vuelve a correr el orquestador)
    make check
    make smoke
    make smoke-model
    uv run pytest -q tests/test_api_robustness_*

## Entregable
Los cuatro arreglos con tests y la medición de RSS antes/después del caso de R3.

## Notas del orquestador (se agregan si hay re-despacho)
- (vacío)
