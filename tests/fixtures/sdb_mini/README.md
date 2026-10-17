# sdb_mini

Mini `dataset.rar` de fixture para T03 (`tests/data/test_sdb_*`, `tests/data/test_preprocess_sdb_*`).

- **Sintético** (PIL): no hay imágenes de terceros. Imita la estructura de `sdb_bcs`
  (`dataset/<BCS>/<nombre>.jpg + .xml`), los nombres `GS_`/`YM_`/`L-i`/`R-i`, el bbox VOC y el texto
  sobreimpreso en las posiciones del dataset real. 39 imágenes, 3 clases (3.25, 3.75, 4.25).
- RAR5 sin compresión, escrito por `make_mini_rar.py` (nada instalado escribe `.rar`).
- `split.csv`: salida de `make_split.py` del equipo (sin tocar) sobre este mini dataset; es la
  referencia contra la que el paso 1 se compara byte a byte en los tests.

Regenerar: `uv run python tests/fixtures/sdb_mini/make_mini_rar.py` (determinístico).
`dataset.rar` está en `.gitignore` del repo; este se agregó con `git add -f`.
