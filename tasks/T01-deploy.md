# T01-deploy · Deploy en Railway con volumen
Rol: worker. Leé SPEC.md §0–§4 y §6 antes de empezar.

> **Decisiones de G0 (mandan sobre SPEC.md): leé `tasks/G0.md` antes de empezar.**

## Objetivo
- Deployar la app de `main` (Dockerfile + `railway.json` de T01, con `uv sync --no-cache`) en el proyecto Railway **ya creado y linkeado** por el humano: `hopeful-cooperation` (https://railway.com/project/13008ce1-3729-47f5-87a3-b6be335f75f2). La CLI `railway` 5.70 está en `~/.local/bin` con sesión iniciada.
- Crear el servicio, un **volumen montado en `/data`** (base SQLite + miniaturas) y un dominio público (`railway domain`).
- Dejar la URL pública en el reporte (el orquestador la anota en `tasks/_state.json`).

## Owns (solo podés editar esto)
- Nada del repo, salvo que un ajuste mínimo de `Dockerfile` o `railway.json` sea imprescindible para que el deploy funcione: en ese caso, pedilo con `ask` antes.
- Recursos del proyecto Railway `hopeful-cooperation` (servicio, volumen, dominio).

## Lee (contexto, no editar)
- Dockerfile, railway.json, src/bcs/api.py, tasks/G0.md

## Restricciones
- **No definas `BCS_APP_TOKEN` ni ninguna otra variable secreta:** la define el humano con `railway variables --set BCS_APP_TOKEN=...` cuando se lo pidas. No leas `~/.railway/` ni ningún archivo de credenciales.
- `BCS_DB_PATH=/data/bcs.sqlite` y `BCS_MEDIA_DIR=/data/media` ya vienen en el Dockerfile.
- No migres a `.railway/railway.ts` (`railway config migrate`): `railway.json` funciona hasta el 01/12/2026.
- No instales MCPs ni `railway setup agent`.
- Nada de datos, imágenes de terceros ni artefactos de modelos en la imagen.
- Máximo 5 vueltas. No mergeás, no pusheás.

## Validaciones (las vuelve a correr el orquestador)
    curl -s -o /dev/null -w '%{http_code}' <url-pública>/health                                   # → 200
    # después de que el humano defina BCS_APP_TOKEN y se redeploye:
    curl -s -o /dev/null -w '%{http_code}' -H "X-App-Token: $TOKEN" -F image=@tests/fixtures/rear_cow.jpg <url>/predict   # → 422 (sin cow_id)
    curl -s -o /dev/null -w '%{http_code}' -F cow_id=A1 -F image=@tests/fixtures/rear_cow.jpg <url>/predict              # → 401
    railway volume list   # volumen montado en /data

## Entregable
URL pública, nombre del servicio y del volumen, tiempo de build y tamaño de la imagen según Railway. Si hace falta que el humano haga algo (token, plan), pedilo con `ask` y esperá.

## Notas del orquestador (se agregan si hay re-despacho)
- (vacío)
