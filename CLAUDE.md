# BCS · instrucciones para agentes

1. Leé `SPEC.md` (worker: §0–§4 y §6) y tu brief `tasks/<ID>.md`. Solo editás los archivos de *Owns*.
2. Comandos:
   - `make check` — `ruff format --check`, `ruff check`, `pyright`, `pytest -q`. Tiene que pasar antes de terminar cada turno (hook Stop).
   - `make smoke` — levanta la API con base temporal y prueba `/predict`, `/cows`, token y `cow_id`.
   - `make eval MODEL=<name> STAGE=<m2|m3|m4>` — CV agrupada contra `eval/thresholds.yaml`.
   - Dependencias con `uv` (Python fijado en `.python-version`); no corras `uv add`: pedila.
3. Paredes (SPEC §4):
   - Nunca leer, listar ni evaluar `$BCS_HOLDOUT_DIR` (`~/bcs-shared/holdout`).
   - Nunca editar `eval/thresholds.yaml`, `tests/test_contract.py` ni `preprocesamiento/`.
   - Nunca debilitar, saltear (`skip`/`xfail`) ni cambiar una métrica para que algo pase.
   - Splits siempre por `group_id`. Semillas fijas. Ningún dato ni imagen de terceros en el repo.
   - Archivos calientes (`pyproject.toml`, `uv.lock`, `Makefile`, `config/`, los `base.py`, `src/bcs/contract.py`,
     `src/bcs/api.py`, `src/bcs/inference.py`, `CLAUDE.md`, `.claude/`): solo el orquestador.
   - El `cow_id` nunca llega al modelo.
4. Protocolo de worker (SPEC §6): implementar → validar → iterar (máx. 5 vueltas). Preguntas con
   `orca orchestration ask`. Commit en tu rama `<ID>: <resumen>`; no mergees, no pushees.
   Al terminar, `worker_done` con el reporte de §6.1.
