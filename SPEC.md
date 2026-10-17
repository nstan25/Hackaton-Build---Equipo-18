# SPEC · Estimador de Condición Corporal (BCS) desde fotos traseras de vacas

> Para agentes que corren dentro de **Orca** (ADE). Este archivo es la **fuente de verdad compartida**.
> Hay dos roles:
> - **Orquestador** (un agente en el worktree principal): planifica, despacha tareas a otros agentes, verifica, integra y escala al humano. Lee §0–§5.
> - **Worker** (un agente por tarea, cada uno en su worktree): implementa **una** tarea en loop. Lee §0–§4, §6 y su `tasks/<ID>.md`.
>
> Todo trabajo sigue el loop **Objetivo → Restricciones → Validaciones**: se itera hasta que las validaciones pasan, se reporta y se para.

---

## 0. Contexto del producto

**Qué hace:** el usuario ingresa el **ID de la vaca** (por ejemplo, el número de caravana) y saca o sube una foto de esa vaca **desde atrás**. La app devuelve, asociado a ese ID:

1. el **BCS estimado** (Body Condition Score),
2. una **confianza calibrada** (qué tan probable es que el valor sea correcto),
3. un **intervalo** de valores plausibles,
4. y un **rechazo explícito** cuando la foto no sirve (no es una vaca, no es de atrás, borrosa, muy oscura).

Cada evaluación queda **registrada con el ID de la vaca, la foto y la fecha**. Así, en la vista de rodeo (dashboard) se ve qué vaca es cuál, su último BCS y su evolución.

**El ID no lo infiere el sistema: lo ingresa el usuario.** El modelo **nunca** usa el ID como variable de entrada; solo viaja con la predicción para registrarla y mostrarla.

**Para qué:** herramienta de apoyo a productores y técnicos para monitorear reservas corporales. No reemplaza al técnico; la UI lo dice.

**Contexto:** hackatón de fin de semana. Prioridad: algo que funcione, que se pueda mostrar en una URL pública y cuyas métricas sean honestas.

### 0.1 Escala de BCS (configurable)

La escala depende de los datos que tengamos. Se define en `config/scale.yaml` y **ningún código asume una escala fija**:

| Preset | Rango | Paso | Uso típico |
|---|---|---|---|
| `dairy_1_5` (default) | 1.0 – 5.0 | 0.25 | Lechería (Holando) |
| `beef_1_8` | 1 – 8 | 1 | Cría vacuna en Uruguay |
| `beef_1_9` | 1 – 9 | 1 | Carne (escala NRC/EE.UU.) |

Si las colas de la distribución tienen pocos ejemplos (p. ej. < 15 imágenes), se pueden **agrupar** en el config (`collapse_below`, `collapse_above`). El modelo predice sobre la grilla resultante.

### 0.2 Protocolo de foto (lo muestra la UI y lo usa el filtro de calidad)

Todos los datasets públicos etiquetados (sección 1.2) son de **cámara fija, desde atrás y arriba** (vista trasera-dorsal), no fotos de celular a la altura de la cadera. Para reducir el salto de dominio, la UI pide:

- Vaca de atrás, cámara **por encima de la cadera** (brazo en alto, o desde arriba del tubo/brete), apuntando hacia abajo unos 30–60°.
- Se ven: base de la cola, isquiones (pin bones), ilíacos (hook bones) y, si se puede, la línea del lomo.
- Una sola vaca centrada, sin cola tapando la zona, con buena luz.

El salto de dominio igual existe; se mide explícitamente con el set `phone_backview` (sección 1.2) y se reporta.

---

## 1. Datos

### 1.1 Formato canónico

Todo dataset, venga de donde venga, se convierte con un adaptador a esta estructura. En este spec, **`data/` significa `$BCS_DATA_DIR`**: un directorio compartido **fuera del repo**, para que todos los worktrees vean los mismos datos sin duplicar gigas (ver §3.0).

```
$BCS_DATA_DIR/
  raw/<source_id>/          # lo que se descargó, sin tocar (gitignored)
  images/<source_id>/       # imágenes normalizadas: recortadas a la vaca, jpg, lado mayor 768px
  labels.csv                # image_id, path, source, animal_id, group_id, bcs, scale, view, scorer, date, bbox
  sources.lock.json         # por fuente: URL, versión/DOI, tamaño, sha256, fecha de descarga, licencia
```

El holdout **no** está acá: vive en `$BCS_HOLDOUT_DIR`, que solo toca el humano.

- `animal_id` es **obligatorio** y siempre se prefija con la fuente (`zen_rgbd:230211`). `group_id` agrupa lo que no puede separarse entre splits (mismo animal; si no hay ID, misma secuencia de video).
- De videos o ráfagas se toman frames **temporalmente separados**; nunca 50 copias casi iguales del mismo animal.
- **Ningún dato vive en el repo.** Nunca se commitean imágenes de terceros.

### 1.2 Fuentes de datos (relevadas y probadas el 17/10/2026)

Todas se descargan con scripts, sin intervención manual, salvo donde se indica. Lo marcado ✔ fue probado desde un sandbox (listado, descarga parcial o completa).

| `source_id` | Qué es | Vista | Escala / rango | Tamaño | Licencia | Descarga | Uso |
|---|---|---|---|---|---|---|---|
| `sdb_bcs` | ScienceDB, *Dairy cow BCS target detection data set* V3 (Huang et al.), DOI `10.57760/sciencedb.16704` | Trasera-dorsal cercana, cámara fija en granja; **timestamp y marca "BCS" sobreimpresos** | 1–5 · solo 3.25, 3.5, 3.75, 4.0, 4.25 | 53.566 jpg (1024×576) + xml VOC con bbox; ~4,4 GB en un `.rar` | Pública; **verificar licencia en la página** y citar | ✔ URL directa sin login, admite `Range` | Entrenamiento principal |
| `zen_rgbd` | Zenodo, *RGB-D Image Dataset for Automated BCS of Dairy Cows in Rotary Milking Parlors* (Ruchay, Kolpakov, Riazanov), DOI `10.5281/zenodo.20290988` | Trasera-dorsal oblicua desde arriba en sala rotativa; **varias vacas por frame** | 1–5 (Ferguson) · 2.75–5.00 en pasos de 0.25 | 25.700 frames RGB+depth, **1.025 animales**; 5 zips (83,5 GB) + `Dataset.xlsx` | CC BY 4.0 | ✔ API Zenodo + `remotezip` para bajar solo los frames elegidos | Entrenamiento principal + evaluación por animal |
| `iplab` | BCS Database (Corfilac / IPLAB / Penn Vet), iplab.dmi.unict.it/BCS | Dorsal desde arriba, salida de robot de ordeñe (Italia) | 1–5 · **dos técnicos por imagen** (BCS1, BCS2) | 207 imágenes, 29 vacas | Uso académico, **sin redistribución** | ✔ links directos en `dataset.html` (el BCS de cada técnico está en el nombre del `.txt`) | **Test externo** + acuerdo entre técnicos |
| `criollo_dge` | Winkler & Boucheron, Criollo cattle BCS, Zenodo `8108223` / Dryad `10.5061/dryad.tqjq2bw4s` / espejo HF `Jybbs/cattle-bcs-dge` | Imágenes de profundidad + gris + bordes (DGE), ganado de carne | 1–9 · sesgado a flaco | `Total_sorted_DGE_images.zip` 178 MB (carpetas `grado/vaca/frames`) | CC0 | ✔ Zenodo API o `hf_hub_download` | Smoke test del pipeline; **no mezclar** con la escala 1–5 |
| `phone_backview` | Mendeley Data `h2s22wr5py` v3, *Cattle side view and back view dataset* (Bai et al.) | **Trasera con celular** (4032×3024) y lateral | Sin BCS (trae medidas corporales) | 72 traseras + 72 laterales | CC BY 4.0 | ✔ API pública de Mendeley | Positivos/negativos del filtro de calidad + **set de dominio celular** para puntuar a mano |
| `cowdb_beef` | HF `ganader-ia-developers/cowdb` (Angus y Hereford) | Izquierda, derecha y arriba | Sin BCS (peso) | 876 imágenes, 112 vacas | Sin licencia declarada | ✔ HF datasets | Negativos "no es vista trasera" del filtro; solo uso interno |
| `rf_bcs` (opcional) | Roboflow Universe `bcs-g5nxo/bcs-4wm9w` v5 | Cenital | Clases 5–9, escala sin documentar | 1.311 imágenes | CC BY 4.0 | Requiere `ROBOFLOW_API_KEY` gratuita (la pone el humano en `.env`) | Solo si sobra tiempo; **no mezclar** |

**Descartados:** Roboflow `reagan-dmuqj/body-condition-scores-2` (capturas de pantalla, sin versión exportable); los datasets de los papers de *AgriEngineering* 2025 y *Vet. Sciences* 2024 (no encontré descarga pública); Kaggle (no hay datasets de imágenes con BCS).

#### Recetas de descarga (verificadas)

```bash
# sdb_bcs — 1) obtener URL, 2) bajar (4,4 GB, reanudable), 3) extraer con bsdtar (libarchive) o 7z
curl -s "https://www.scidb.cn/api/sdb-filetree-service/getAllUrl?dataSetId=16b8bdaf31ee4c8b9891fc7e9df6e41c&type=journalDataset&version=V3&global=en"
#   → https://download.scidb.cn/download?fileId=4516a6866954512e08446bc3504b32c9&path=/V3/dataset.rar&fileName=dataset.rar
curl -L -C - -A "Mozilla/5.0" -o data/raw/sdb_bcs/dataset.rar "<url de arriba>"
bsdtar -xf data/raw/sdb_bcs/dataset.rar -C data/raw/sdb_bcs/
#   estructura: dataset/<BCS>/<GRANJA>_<ID>_<n>.jpg + .xml  (ej. dataset/3.25/GS_1003_1.jpg)
#   4) preprocesar según preprocesamiento/README.md → $BCS_DATA_DIR/raw/sdb_bcs_pre/  (ver §2.5)

# zen_rgbd — metadata completa (1,2 MB) y frames sueltos sin bajar los zips enteros
curl -L -o data/raw/zen_rgbd/Dataset.xlsx "https://zenodo.org/api/records/20290988/files/Dataset.xlsx/content"
#   columnas: N, animal_id, BCS, data (fecha = nombre del zip), frame (subcarpeta), RGB_path, depth_path (con "\\")
#   zips: https://zenodo.org/api/records/20290988/files/<fecha>.zip/content   (fechas: 06.12.2024, 20.02.2025, 20.03.2025, 27.03.2025.1, 27.03.2025.2)
#   en Python: remotezip.RemoteZip(url).extract("06.12.2024/10/rgb/14.16.41.375.png", dest)
#   ~2,9 MB y ~4 s por frame → usar ThreadPool(8) y muestrear 3 frames por (animal_id, fecha)

# criollo_dge
curl -L -o data/raw/criollo_dge/dge.zip "https://zenodo.org/api/records/8108223/files/Total_sorted_DGE_images.zip/content"

# iplab — parsear los href de https://iplab.dmi.unict.it/BCS/dataset.html
#   DatasetImagesCows/<id>_cow.jpg  +  DatasetDataShapes/<id>_Anatomical_Points_BCS1_<a>_BCS2_<b>.txt
#   el primer número de <id> es la vaca (29 vacas distintas)

# phone_backview — API pública de Mendeley
#   carpetas: https://data.mendeley.com/public-api/datasets/h2s22wr5py/folders/3?version=3
#   archivos: https://data.mendeley.com/public-api/datasets/h2s22wr5py/files?folder_id=<id>&version=3  → content_details.download_url

# cowdb_beef
#   datasets.load_dataset("ganader-ia-developers/cowdb", "resized")  (columnas: image, cow_id, breed, orientation)
```

Cada fuente tiene su script `scripts/data/fetch_<source_id>.py`. Todos son **idempotentes** (si el archivo ya está y el sha256 coincide, no bajan nada), reanudables y escriben en `sources.lock.json`. Tienen un flag `--sample N` que baja solo N imágenes por clase, para iterar rápido.

#### Lo que estos datos NO cubren (decirlo en la UI y el README)

- **No hay vacas flacas etiquetadas en escala 1–5.** El mínimo es 2.75 en `zen_rgbd` y 3.25 en `sdb_bcs`. El modelo no puede detectar BCS < 2.75, que es justamente el caso de mayor riesgo. La UI muestra "rango validado: 2.75–5.0" y, si la distribución se apila en la clase más baja, devuelve `low_confidence` con el texto "posible condición baja, verificar a mano".
- **No hay fotos de celular etiquetadas.** El set `phone_backview` sirve para medir el salto de dominio solo si alguien lo puntúa. Plan: el humano consigue un técnico que puntúe esas 72 fotos (o fotos propias) en la hackatón y las guarda en `data/manual/phone_labels.csv`. Hasta entonces, el reporte dice "sin evaluación en dominio celular".
- Razas lecheras (Holando y cruzas). Cero datos de razas carniceras en escala 1–5.

#### Trampas conocidas (y cómo se neutralizan)

1. **Texto sobreimpreso en `sdb_bcs`** (fecha/hora arriba a la izquierda, "BCS" abajo a la derecha). La fecha puede correlacionar con la etiqueta. Siempre se **recorta al bbox del xml** y además se tapa la franja del texto. Hay un test que verifica que ninguna imagen procesada de `sdb_bcs` conserva esas regiones.
2. **Varias vacas por frame en `zen_rgbd`.** Hipótesis: la vaca etiquetada es la del box central. Antes de entrenar, el agente arma una hoja de contactos con 25 frames al azar y **la mira** (abriendo la imagen). El recorte se hace con un detector de vacas (clase `cow` de un YOLO preentrenado), eligiendo la caja más cercana al centro. Si la hipótesis no se sostiene, se para y se reporta.
3. **Frames casi idénticos y la misma vaca en varias fechas** (en `zen_rgbd`, 163 animales tienen más de un BCS). Siempre se agrupa por `group_id`.
4. **La fuente predice la etiqueta.** `sdb_bcs` nunca baja de 3.25. Además de la CV agrupada, se reporta **leave-one-source-out** (entrenar en una fuente, evaluar en la otra), que es la métrica más honesta de generalización.
5. **Escalas distintas.** Solo se mezclan fuentes de escala 1–5 (`sdb_bcs`, `zen_rgbd`, `iplab`). `criollo_dge` y `rf_bcs` nunca entran al mismo modelo.

### 1.3 Splits

- **Group K-Fold por `group_id`** (k=5), estratificado por BCS, sobre `sdb_bcs` + `zen_rgbd`.
- **Holdout** (lo crea el humano con `make holdout` en el gate GH, semilla fija). `make holdout` baja `iplab` directamente a `$BCS_HOLDOUT_DIR`, elige los animales y escribe dos archivos **visibles** en `$BCS_DATA_DIR`: `holdout_groups.txt` (los `group_id` a excluir) y `human_agreement.json` (el acuerdo entre los dos técnicos de `iplab`): 15% de los animales de `zen_rgbd` y de `sdb_bcs`, más **todo `iplab`** como test externo de otra granja, otro país y otra época. Solo se usa con `make eval-final`.
- Referencia humana: el acuerdo entre BCS1 y BCS2 de `iplab` (MAE y % dentro de ±0.25), tomado de `human_agreement.json`, se reporta junto a las métricas del modelo como techo razonable.
- Si en T03–T05 una descarga falla, se sigue con lo que haya y se reporta. Si no hay ninguna fuente 1–5 disponible, se usa un dataset sintético de fixtures para que el pipeline corra, marcando que las métricas no significan nada.

---

## 2. Arquitectura

```
bcs/
  config/            scale.yaml, train.yaml
  src/bcs/
    data/            adaptadores, dataset, splits, transforms
    models/          backbones, cabezas ordinales
    calibration/     temperature scaling, conformal
    quality/         filtro "¿es una vaca de atrás y la foto sirve?"
    inference.py     predict(image, cow_id) -> Prediction   ← CONTRATO
    store/           registro de evaluaciones (SQLite) + miniaturas de las fotos
    api.py           app FastAPI: solo monta los routers (lo crea T00)
    api_predict.py   POST /predict
    api_cows.py      GET /cows, /cows/{cow_id}, /predictions/{id}, /media/{id}.jpg
  web/               index.html (evaluar) + rodeo.html (dashboard); estáticos, mobile-first, servidos por FastAPI
  eval/
    evaluate.py      genera eval/report.json + gráficos
    thresholds.yaml  umbrales del loop (NO editable por el agente)
  tests/
  scripts/           stop_gate.sh, lint_hook.sh, data/fetch_<source_id>.py
  tasks/             <ID>.md — un brief por tarea, lo escribe el orquestador
  preprocesamiento/  README.md (+ scripts, si los hay) — lo provee el equipo; define cómo se preprocesa dataset.rar. Solo lectura
  Makefile
  CLAUDE.md          instrucciones para Claude Code
  AGENTS.md          symlink a CLAUDE.md (Codex y otros agentes leen AGENTS.md)
  .claude/settings.json
  .env.example       BCS_DATA_DIR, BCS_ARTIFACTS_DIR, BCS_HOLDOUT_DIR, BCS_DB_PATH, BCS_MEDIA_DIR, BCS_APP_TOKEN
```

### 2.0 Interfaces internas (se congelan en T00 para poder paralelizar)

Cada módulo se programa contra un `Protocol` en un `base.py`, con un stub que funciona. Así los workers trabajan en paralelo sin esperarse:

| Archivo | Interfaz | Stub de T00 |
|---|---|---|
| `src/bcs/data/schema.py` | `LabelRow`, `load_labels(path) -> DataFrame`, validación del esquema de `labels.csv` | lee un CSV de fixtures |
| `src/bcs/data/sources/base.py` | `Source`: `fetch(sample: int \| None)`, `adapt() -> Iterable[LabelRow]` | fuente sintética |
| `src/bcs/models/base.py` | `Scorer`: `fit(df)`, `predict_proba(images) -> ndarray[n, K]`, `save()`, `load()` | probabilidades uniformes |
| `src/bcs/calibration/base.py` | `Calibrator.fit(probs, y)` / `.transform(probs)`; `IntervalPredictor.fit(...)` / `.interval(probs)` | identidad / grilla completa |
| `src/bcs/quality/base.py` | `QualityChecker.check(image) -> QualityReport` | acepta todo |
| `src/bcs/eval/metrics.py` | `evaluate(probs, y, grid, groups, sources) -> dict` | MAE y accuracy |
| `src/bcs/store/base.py` | `Store`: `save(prediction, image_bytes) -> None`, `latest_by_cow(filters) -> list[CowSummary]`, `history(cow_id) -> list[Prediction]`, `get(prediction_id)`, `thumbnail_path(prediction_id)` | en memoria |
| `src/bcs/inference.py` | `predict(image, cow_id) -> Prediction` (compone todo lo anterior; `cow_id` **no** se le pasa al modelo) | devuelve el fixture con el `cow_id` recibido |

### 2.1 Stack

- Python 3.11+, **uv** para dependencias.
- PyTorch + **timm** (backbones preentrenados), scikit-learn, numpy, pillow.
- **FastAPI** + uvicorn para la API; el front es un `index.html` estático servido por la misma app (un solo deploy).
- Inferencia en producción en **CPU** (exportar a ONNX + onnxruntime si el tiempo da; si no, PyTorch CPU).
- ruff (lint + format), pyright (tipos), pytest.
- Deploy: **Railway** (API + front juntos). Alternativa: Hugging Face Spaces con Docker.

### 2.2 Contrato de inferencia (se congela en G0; cambiarlo requiere un gate)

```python
@dataclass
class Prediction:
    prediction_id: str             # uuid4, lo genera el sistema
    cow_id: str                    # el ID que ingresó el usuario, normalizado (ver abajo)
    created_at: str                # ISO 8601 en UTC
    image_sha256: str              # hash de la foto original: vincula la predicción con su foto y detecta duplicados
    status: Literal["ok", "low_confidence", "rejected"]
    bcs: float | None              # valor de la grilla de la escala
    bcs_expected: float | None     # esperanza de la distribución (continuo)
    confidence: float | None       # P(BCS real ∈ [bcs - paso, bcs + paso]), calibrada
    interval: tuple[float, float] | None   # conjunto conformal al 90% (min, max)
    distribution: dict[str, float] # prob por clase de la grilla
    quality: QualityReport         # is_cow_rear: bool, score: float, issues: list[str]
    scale: str
    model_version: str
```

**Definición de confianza** (importante, es lo que se muestra al usuario): probabilidad, calibrada en validación, de que el BCS real esté a **un paso o menos** del valor reportado (±0.25 en escala 1–5). Así "80% de confianza" significa algo verificable.

Reglas de `status`:
- `rejected` si el filtro de calidad falla → no se devuelve BCS, sí se devuelven los `issues` ("no parece una vaca de atrás", "imagen muy oscura", etc.).
- `low_confidence` si `confidence < 0.5` o el intervalo abarca más de 4 pasos.
- `ok` en el resto.

**Reglas del `cow_id`:**
- Obligatorio. Se normaliza: se quitan espacios en los extremos y se pasa a mayúsculas.
- Formato válido: `^[A-Z0-9][A-Z0-9._-]{0,63}$` (por ejemplo, el número de caravana o el ID interno del establecimiento). Si no cumple, la API responde **422** con un mensaje claro, sin evaluar la foto.
- No se usa como feature del modelo, no se loguea fuera del registro y no se mezcla con el `animal_id` de los datasets de entrenamiento (son cosas distintas).

**API:**

| Método y ruta | Entrada | Salida |
|---|---|---|
| `GET /health` | — | `{"status": "ok", "model_version": …}` |
| `POST /predict` | multipart: `image` (obligatorio), `cow_id` (obligatorio), `taken_at` (opcional, ISO 8601), `notes` (opcional, ≤ 280 caracteres) | `Prediction` (y la guarda en el registro) |
| `GET /cows` | filtros opcionales: `status`, `bcs_max`, `bcs_min`, `since` | una fila por vaca: `cow_id`, último BCS válido, su confianza e intervalo, fecha, cantidad de evaluaciones, `prediction_id` de la última foto |
| `GET /cows/{cow_id}` | — | historial completo de esa vaca, de la más reciente a la más vieja |
| `GET /predictions/{prediction_id}` | — | una `Prediction` |
| `GET /media/{prediction_id}.jpg` | — | miniatura de la foto (lado mayor 512 px) |

**Registro:**
- Se guardan **todas** las evaluaciones, incluidas las `rejected`, para que el usuario vea el intento.
- En `/cows`, el "último BCS" sale solo de evaluaciones `ok` o `low_confidence`. Si la última fue `rejected`, se indica aparte.
- La foto se guarda como miniatura en `$BCS_MEDIA_DIR` (no la original) y la base es SQLite en `$BCS_DB_PATH`. En Railway, ambas rutas van en un volumen persistente.

**Protección mínima:** la URL es pública y el registro tiene datos de un establecimiento. Todas las rutas salvo `/health` exigen el token `BCS_APP_TOKEN` (header `X-App-Token`, o `?token=` en el link que se comparte). No es autenticación de usuarios; es lo mínimo razonable para una demo.

Fixture de ejemplo en `tests/fixtures/prediction_example.json` (lo crea T00 y es el contrato entre modelo y app).

### 2.3 Enfoque de modelado (guía, no dogma)

Hay decenas de miles de frames pero muchos menos animales distintos, y un rango de BCS estrecho. El orden es:

1. **Baseline tonto:** predecir siempre la moda/media. Es la vara mínima.
2. **Embeddings congelados + cabeza chica:** extraer features de un backbone fuerte preentrenado (DINOv2 vía timm o equivalente) y entrenar encima regresión ridge y una regresión logística ordinal. Rápido, corre en CPU, suele ser muy competitivo con poco dato.
3. **Fine-tuning** (si hay GPU y datos suficientes): ConvNeXt-Tiny / EfficientNetV2-S con cabeza **ordinal** (CORAL o clasificación con *label smoothing gaussiano* σ = 1 paso, que además absorbe el ruido entre puntuadores).
4. Augmentations: flip horizontal (la vista trasera es ~simétrica), recortes leves, jitter de brillo/contraste, blur leve. **Nada de flip vertical ni rotaciones grandes.**
5. Recorte de región de interés opcional: detector de vaca (YOLO preentrenado, clase `cow`) para recortar antes de clasificar.

**Confianza:**
- Temperature scaling sobre logits usando el fold de validación.
- **Conformal prediction** (split conformal, APS o LAC) para el intervalo al 90%.
- Filtro de calidad: CLIP zero-shot ("rear view of a cow" vs. otros prompts) **o** distancia en el espacio de embeddings al set de entrenamiento, + chequeos simples de brillo y nitidez (varianza del Laplaciano).

**Extra para la demo (si sobra tiempo):** Grad-CAM sobre la imagen y "las 3 vacas más parecidas del set de entrenamiento con su BCS" (kNN sobre embeddings). Ayuda mucho a que el jurado confíe en el número.

### 2.4 Cómputo: dónde corre cada cosa

**Dataset de trabajo:** `sdb_bcs`, preprocesado a partir de `dataset.rar` **según `preprocesamiento/README.md`** (ver §2.5). El resultado va en `$BCS_DATA_DIR/raw/sdb_bcs_pre/` y T03 lo adapta al formato canónico (ver los chequeos extra en §5.6.1). Las otras fuentes de §1.2 se usan solo si el humano lo confirma en G0/G1. Si se entrena solo con `sdb_bcs`, el rango validado del modelo es **3.25–4.25** y la UI lo dice así.

**Un solo código de entrenamiento, varios backends.** `python -m bcs.train --config config/<modelo>.yaml --fold <k|all> [--smoke]` corre igual en local, en Modal, en Kaggle o en Tenki. Cada backend solo empaqueta y lanza:

```
make pack-data                                  # crops ya recortados, redimensionados a 256 px de lado corto,
                                                # + labels.csv + folds → $BCS_ARTIFACTS_DIR/packs/bcs_pack_<hash>.tar
make train MODEL=<cand> BACKEND=<modal|kaggle|tenki|local> [FOLDS=all] [SMOKE=1]
```

- El pack **nunca incluye el holdout** (se filtra con `holdout_groups.txt` antes de empaquetar) y queda en ~1–2 GB en vez de 4,4. Redimensionar una vez evita que la decodificación de JPEG sea el cuello de botella en máquinas con pocos CPU.
- Toda corrida remota escribe en `runs/<task>/<ts>/` el mismo `report.json` que `make eval`, y los pesos en `$BCS_ARTIFACTS_DIR/<task>/`.
- **Smoke primero:** cada backend se prueba con `SMOKE=1` (1 fold, 1 época, 500 imágenes, < 5 min) antes de lanzar la corrida completa.
- **Bitácora de gasto:** cada corrida agrega una línea a `runs/compute_ledger.csv` (backend, tarea, tipo de máquina, segundos, costo estimado). El orquestador la resume en cada gate.

| Backend | Para qué | Por qué | Cómo lo maneja el agente | Presupuesto del finde |
|---|---|---|---|---|
| **Modal** (principal GPU) | T10a, T10b; los 5 folds en paralelo | Se maneja 100% desde Python/CLI, cobra por segundo y permite varias GPUs a la vez, así que los dos candidatos y sus folds corren en paralelo | `modal run scripts/compute/modal_app.py::train --model cand_a --folds all`; datos en un Volume `bcs-data` (`modal volume put`) | ≤ $25 de los $30 de crédito mensual gratis; GPU L4 por defecto |
| **Kaggle** (respaldo GPU, gratis) | Re-correr un candidato completo si Modal se queda sin crédito o falla | Cuota semanal gratuita de GPU, manejable por CLI | Dataset privado `bcs-pack` (`kaggle datasets create/version`); kernel en `scripts/compute/kaggle/` (`kaggle kernels push` → `status` → `output`) | ≤ 20 h de la cuota semanal |
| **Tenki** (CPU, los $150 que ya hay) | T09 (embeddings congelados + cabezas lineales), T11 (calibración), `make eval` pesados, CI de `make check` en cada PR y re-entrenamiento programado | Son runners de GitHub Actions: CPU barata y sin GPU. Alcanza para todo lo que no es fine-tuning | Workflow `.github/workflows/cpu-job.yml` con `workflow_dispatch`; `gh workflow run` → `gh run watch` → `gh run download` | ≤ $40 |
| **Local** | Tests, smoke y debugging | — | `make train BACKEND=local SMOKE=1` | — |

**Descartados para este finde:**
- **Colab** gratis: necesita que una persona maneje la interfaz y se desconecta. Ningún agente puede operarlo solo.
- **Lightning AI**: tiene créditos gratis útiles, pero agrega otra cuenta y otra forma de trabajo. Queda como plan C.
- **GPUs por hora** (RunPod, Vast, etc.): no hacen falta con lo de arriba.

**Estimación gruesa** (verificar con el smoke): ConvNeXt-Tiny a 224 px sobre unas 30–40 mil imágenes recortadas, ~15 épocas, da del orden de 1–3 h de GPU por fold en una T4/L4, o sea ~5–15 GPU-h por candidato con 5 folds. Si el smoke indica más, se baja a 3 folds o se sub-muestrean frames por vaca, y se avisa en el reporte. Los embeddings congelados de T09 son una sola pasada hacia adelante: minutos u horas pocas en CPU.

**Credenciales** (las configura el humano en G0; ningún agente las lee):
- `modal token new` (queda en `~/.modal.toml`)
- `~/.kaggle/kaggle.json`
- La app de Tenki instalada en el repo de GitHub, con el label del runner en el workflow
- `gh auth login`


### 2.5 Preprocesamiento de `dataset.rar`

**La fuente de verdad es `preprocesamiento/README.md`.** Ese archivo lo escribió el equipo y define los pasos (cómo se ordena, cómo se balancea, etc.). Este spec no los repite a propósito, para que no haya dos versiones que se contradigan. Se lee entero antes de escribir código.

**Entrada:** `$BCS_DATA_DIR/raw/sdb_bcs/dataset.rar`. Si el humano ya lo copió ahí, no se descarga de nuevo (se verifica tamaño y sha256); si no está, se baja con la receta de §1.2.

**Salida:**
- `$BCS_DATA_DIR/raw/sdb_bcs_pre/` con las imágenes y anotaciones preprocesadas.
- `$BCS_DATA_DIR/raw/sdb_bcs_pre/preprocess_manifest.json` con:
  - el sha256 del `dataset.rar` y del `README.md` usados;
  - la versión del código y la semilla;
  - por cada paso del README: nombre, imágenes por clase antes y después, y archivos descartados con el motivo.

**Cómo se implementa:**
1. **Pasos en orden.** Cada paso del README se implementa en ese orden en `src/bcs/data/preprocess_sdb.py`, con una función por paso, y se ejecuta con `make preprocess SOURCE=sdb_bcs` (que `make data SOURCE=sdb_bcs` llama solo).
2. **Scripts del equipo.** Si el README menciona scripts que están en `preprocesamiento/`, se llaman tal cual desde ese módulo, sin copiarlos ni reescribirlos.
3. **Reproducible.** Mismo `.rar` + mismo README + misma semilla = mismo resultado byte a byte. Si el manifiesto ya existe con los mismos hashes, no se rehace nada.
4. **Pasos ambiguos.** Si un paso del README no está claro (un umbral sin valor, un orden que se puede leer de dos maneras), el worker **pregunta** con `orchestration ask` en vez de elegir una interpretación. El orquestador escala al humano si el spec no lo resuelve.

**Conflictos con las reglas de este spec.** El README manda en *qué* se hace. Las restricciones de §1.2 y §4 mandan en lo que no se puede romper:
- que el `group_id` se pueda reconstruir para cada imagen;
- que no queden copias de una misma foto en folds distintos;
- que no entren datos del holdout;
- que se quite el texto sobreimpreso.

Si aplicar un paso tal cual rompe alguna de esas reglas, el worker **no lo cambia por su cuenta**. Para y escala con la evidencia. El caso típico: un balanceo que duplica o aumenta imágenes *antes* de separar en folds. La salida probable es aplicar ese paso dentro de cada fold de entrenamiento (en `bcs.train`) en vez de en el preprocesamiento, pero lo decide el humano.

---

## 3. Arnés (lo crea T00)

### 3.0 Directorios compartidos fuera del repo

Cada worktree de Orca es un checkout separado. Si los datos vivieran en `data/` dentro del repo, cada worktree tendría su copia (gigas repetidos) o no tendría nada. Por eso:

```
~/bcs-shared/
  data/        → $BCS_DATA_DIR       (lo escriben solo las tareas de datos; después de G1 queda de solo lectura)
  artifacts/   → $BCS_ARTIFACTS_DIR  (pesos, embeddings cacheados; cada tarea en su subcarpeta <task_id>/)
  runs/        → resultados de cada `make eval`, con prefijo <task_id>/
  holdout/     → $BCS_HOLDOUT_DIR    (solo el humano)
```

El código lee estas rutas de variables de entorno, con esos valores por defecto. `.env.example` las documenta.

### 3.1 `CLAUDE.md` / `AGENTS.md`

Cortos. Dicen:
- Leé `SPEC.md` y tu `tasks/<ID>.md`.
- Cómo correr `make check`, `make eval`, `make smoke`.
- Las restricciones de §4.
- El protocolo de worker de §6.

`AGENTS.md` es un symlink a `CLAUDE.md`, para que Codex y otros agentes lean lo mismo.

### 3.2 `Makefile` (targets obligatorios)

| Target | Qué hace | Lo corre |
|---|---|---|
| `make preprocess SOURCE=sdb_bcs` | aplica `preprocesamiento/README.md` sobre `dataset.rar` → `raw/sdb_bcs_pre/` + `preprocess_manifest.json`; idempotente | T03 |
| `make data SOURCE=<id>` | `fetch_<id>` (+ `preprocess` si corresponde) + adaptador → `$BCS_DATA_DIR/images/<id>/` y `labels_<id>.csv`; idempotente | worker de datos |
| `make data-sample SOURCE=<id>` | igual, con `--sample 60` por clase (< 10 min) | worker de datos |
| `make labels` | une los `labels_<id>.csv` en `labels.csv`, excluye los `group_id` del holdout y arma los folds | T06 |
| `make check` | `ruff format --check`, `ruff check`, `pyright`, `pytest -q` (rápidos, sin datos reales) | todos, hook |
| `make eval MODEL=<name> STAGE=<m2\|m3\|m4>` | CV agrupada → `runs/<task>/<ts>/report.json`; compara contra `eval/thresholds.yaml[STAGE]`; exit ≠ 0 si no pasa | workers de modelo |
| `make pack-data` | arma el pack de entrenamiento (§2.4), sin holdout, con hash de contenido | T16 |
| `make train MODEL=<m> BACKEND=<b> [SMOKE=1]` | lanza `bcs.train` en el backend elegido, trae `report.json` y pesos, anota en la bitácora | workers de modelo |
| `make smoke` | levanta la API con una base temporal; manda las fixtures (vaca trasera, vaca de costado, perro) con `cow_id`; valida el JSON contra el contrato; verifica que `cow_id` inválido o ausente da 422, que sin token da 401 y que `/cows` y `/cows/{id}` reflejan lo enviado | todos los de app |
| `make browser` | a 390px con `agent-browser` (o Playwright): ingresa un `cow_id`, sube la fixture y verifica BCS, confianza, intervalo y el ID en pantalla; repite con el perro y ve el rechazo; abre `rodeo.html` y encuentra la vaca con su miniatura | T07, T15, T12 |
| `make holdout` / `make eval-final` | separar el holdout / evaluar sobre él | **humano** |
| `make deploy` | deploy a Railway | con permiso del humano |

### 3.3 `.claude/settings.json`

Lo hereda cada worktree, porque está commiteado:

```json
{
  "permissions": {
    "allow": [
      "Bash(make data *)",
      "Bash(make data-sample *)",
      "Bash(make preprocess *)",
      "Bash(make labels)",
      "Bash(make check)",
      "Bash(make eval *)",
      "Bash(make smoke)",
      "Bash(make browser)",
      "Bash(make pack-data)",
      "Bash(make train *)",
      "Bash(modal run *)",
      "Bash(modal volume *)",
      "Bash(kaggle kernels *)",
      "Bash(kaggle datasets *)",
      "Bash(gh workflow run *)",
      "Bash(gh run *)",
      "Bash(uv run *)",
      "Bash(git status)",
      "Bash(git diff *)",
      "Bash(git add *)",
      "Bash(git commit *)"
    ],
    "ask": [
      "Bash(make deploy)",
      "Bash(git push *)",
      "Bash(uv add *)"
    ],
    "deny": [
      "Read(~/bcs-shared/holdout/**)",
      "Edit(~/bcs-shared/holdout/**)",
      "Edit(./eval/thresholds.yaml)",
      "Edit(./tests/test_contract.py)",
      "Edit(./preprocesamiento/**)",
      "Bash(make eval-final)",
      "Bash(make holdout)",
      "Read(~/.modal.toml)",
      "Read(~/.kaggle/**)",
      "Read(./.env)",
      "Read(./.env.*)"
    ]
  },
  "hooks": {
    "PostToolUse": [
      {
        "matcher": "Edit|Write",
        "hooks": [{ "type": "command", "command": "./scripts/lint_hook.sh" }]
      }
    ],
    "Stop": [
      {
        "hooks": [{ "type": "command", "command": "./scripts/stop_gate.sh" }]
      }
    ]
  }
}
```

`uv add` queda en `ask`: las dependencias nuevas las aprueba el orquestador (ver §4.2). Si la ruta compartida no es `~/bcs-shared`, ajustá las reglas `~/...`; para rutas absolutas se usa `//ruta`.

**Importante:** hooks y permisos son de Claude Code. Un worker Codex no los tiene. Por eso el orquestador **siempre** vuelve a correr las validaciones por su cuenta (§5.5) y no confía en el `worker_done`.

### 3.4 `scripts/lint_hook.sh`

Lee el JSON del hook por stdin y saca `tool_input.file_path` con `jq`. Si es `.py`, corre `ruff format` y `ruff check --fix` sobre ese archivo. Si quedan errores, los imprime a stderr y sale con código 2.

### 3.5 `scripts/stop_gate.sh`

Lee el JSON por stdin. Si `stop_hook_active` es `true`, sale con 0 (evita loops infinitos). Si no, corre `make check`; si falla, imprime el resumen a stderr y sale con **2**: el agente no puede terminar el turno con su worktree roto.

### 3.6 CLIs y MCPs

Preferí CLIs (no gastan contexto):
- Generales: `orca` (en Linux, `orca-ide`; ver §5.1), `uv`, `git`, `gh`, `jq`, `curl`, `bsdtar`, `railway`, `agent-browser`.
- Python para datos: `remotezip`, `openpyxl`, `huggingface_hub`, `datasets`.

Como máximo 1–2 MCPs.

---

## 4. Restricciones globales (paredes)

### 4.1 Datos y evaluación
1. **Nunca** leer, copiar, listar ni evaluar sobre `$BCS_HOLDOUT_DIR`. Es del humano.
2. **Nunca** modificar `eval/thresholds.yaml` ni `tests/test_contract.py`. Si un umbral parece irreal, se escala.
3. **Nunca** "arreglar" una validación debilitando el test, salteándolo (`skip`, `xfail`) o cambiando la métrica.
4. Splits siempre por `group_id` (`test_no_group_leakage`).
5. Semillas fijas; cada `make eval` guarda config + métricas en `runs/<task_id>/<ts>/`.
6. Licencias: `iplab` es solo académico y sin redistribución; `cowdb_beef` no declara licencia. Ninguna de esas imágenes va al deploy ni al repo. El README cita todas las fuentes.
7. Disco: el `.rar` de `sdb_bcs` ocupa ~4,4 GB (más del doble extraído). De `zen_rgbd` solo se bajan frames muestreados, nunca los zips enteros.

### 4.2 Trabajo en paralelo
8. **Propiedad de archivos:** un worker solo edita los archivos de la lista `owns` de su tarea. Si necesita tocar otro, lo pide con `orchestration ask`.
9. **Archivos calientes** (solo los edita el orquestador después de T00): `pyproject.toml`, `uv.lock`, `Makefile`, `config/`, todos los `base.py` de §2.0, `src/bcs/api.py`, `src/bcs/inference.py` hasta T12, `tests/test_contract.py`, `CLAUDE.md`, `.claude/`. `preprocesamiento/` no lo edita **nadie**, ni siquiera el orquestador: es del equipo. Para una dependencia nueva, el worker la pide; el orquestador la agrega en `main` y el worker hace `git merge main` en su rama.
10. Un worker **commitea en su rama** y nada más: no mergea, no pushea, no rebasea `main`, no borra worktrees.
11. Las tareas que escriben en `$BCS_DATA_DIR` usan solo su subcarpeta `<source_id>/`. Después de G1, el orquestador lo deja de solo lectura (`chmod -R a-w`).
12. Tope de **5 vueltas** por tarea. A la quinta sin pasar, el worker reporta `failed` con lo que intentó.

---

## 5. Rol: Orquestador

### 5.1 Qué hacés y qué no

**Hacés:** crear el run y el DAG de tareas, escribir los briefs `tasks/<ID>.md`, crear worktrees, despachar, esperar mensajes, **verificar de forma independiente**, mergear a `main`, mantener los archivos calientes, abrir gates para el humano y reportarle el estado.

**No hacés:** implementar tareas (salvo arreglos triviales de integración al mergear), cambiar contratos o umbrales sin un gate, ni ver el holdout.

**Antes de cualquier comando de Orca:**

```bash
orca status --json                  # confirma que el runtime está arriba
orca skills get orca-cli            # referencia de la CLI que corresponde a TU versión del binario
orca skills get orchestration       # referencia de runs, tasks, dispatch, gates
```

Los comandos de esta sección muestran la **forma esperada** (tomada de la CLI de Orca a agosto de 2026). Si la guía que imprime el binario dice otra cosa, **manda el binario**: no adivines flags. En Linux, `orca` puede ser el lector de pantalla de GNOME. Usá `$ORCA_CLI_COMMAND` si está definido, o `orca-ide` fuera de una terminal de Orca. Si el ejecutable falla, reportá el error exacto y pará; no pruebes otro binario.

### 5.2 Setup (una vez)

1. Crear `~/bcs-shared/{data,artifacts,runs}` (no `holdout/`; ese lo crea el humano).
2. Escribir `tasks/<ID>.md` para cada tarea de §5.6, con la plantilla de §5.4, y commitearlos en `main` antes de crear worktrees, para que todos los hereden.
3. Crear el run y las tareas con sus dependencias:

```bash
orca orchestration run-create --objective "BCS desde fotos traseras: modelo calibrado + app desplegada" --json
orca orchestration task-create --spec "T00 · ver tasks/T00.md" --json                       # → task_id
orca orchestration task-create --spec "T03 · ver tasks/T03.md" --deps '["<task_id T00>"]' --json
# … una por tarea del DAG, incluidos los gates
```

Cada gate humano (G0, GH, G1–G4) es **una tarea propia, sin worker**. Las tareas que esperan la decisión dependen de ella. Cuando el gate se resuelve, el orquestador marca esa tarea como completada, con el mecanismo que indique `orca skills get orchestration`.

Guardá el mapeo `ID → task_id → worktree → terminal` en `tasks/_state.json`. Lo actualizás vos y lo commiteás en `main` con cada cambio de estado.

### 5.3 Loop de despacho

```
mientras queden tareas sin completar:
  para cada tarea en estado ready, por orden de prioridad (§5.6), mientras haya cupo (§5.8):
      orca worktree create --name <id-slug> --parent-worktree active \
           --agent <agente> --prompt "<prompt de §5.4>" --json        # guardar result.agentTerminalHandle
      orca orchestration dispatch --task <task_id> --to <terminal> --inject --json
  orca orchestration check --wait --types "worker_done,escalation" --timeout-ms 900000 --json
  por cada mensaje recibido: procesarlo (§5.5) y recién después hacer ack
  cada ~20 min: orca worktree ps --json → detectar workers trabados esperando input
```

La entrega es *at-least-once*: un mismo mensaje puede llegar dos veces. Procesá de forma idempotente (si la tarea ya está mergeada, ignorá el duplicado) y hacé ack solo después de procesar.

### 5.4 Plantilla de brief (`tasks/<ID>.md`) y prompt de arranque

```markdown
# <ID> · <título>
Rol: worker. Leé SPEC.md §0–§4 y §6 antes de empezar.
## Objetivo
## Owns (solo podés editar esto)
## Lee (contexto, no editar)
## Restricciones
## Validaciones (las vuelve a correr el orquestador; tienen que pasar en tu worktree)
    <comandos exactos, uno por línea>
## Entregable
## Notas del orquestador (se agregan si hay re-despacho)
```

Prompt al crear el worktree (`--prompt`):

> Sos un worker. Tu tarea es `<ID>`. Leé `SPEC.md` (§0–§4 y §6) y `tasks/<ID>.md`. Trabajá solo en este worktree y solo sobre los archivos de *Owns*. Iterá hasta que pasen las Validaciones (máximo 5 vueltas). Después commiteá y reportá según §6.

### 5.5 Al recibir `worker_done`

1. **Verificación independiente:** en el worktree de la tarea (`orca worktree list --json` da el path), corré vos mismo cada comando de *Validaciones* del brief más `make check`. Si es un worker Codex, esto es obligatorio y además revisás con `git diff main...` que no haya tocado archivos fuera de *Owns*.
2. **Revisión de diff:** confirmá que no hay `skip`/`xfail` nuevos, que no se tocaron umbrales, que no hay rutas al holdout y que no se commitearon datos. En las tareas marcadas *review: sí*, además lanzá un subagente revisor de solo lectura que busque fugas de datos y métricas infladas.
3. **Si algo falla:** mandale al worker la salida exacta (`orca terminal send --terminal <t> --text "<feedback>" --enter`), anotalo en *Notas del orquestador* y sumá una vuelta. A la quinta, abrí un gate al humano.
4. **Si todo pasa:** `git merge --no-ff <rama>` en `main`. Si hay conflicto en `uv.lock`, se regenera con `uv lock`. Si hay conflicto en archivos calientes, no lo resolvés eligiendo la versión del worker: lo devolvés. Después corré `make check` en `main`; si se rompe, revertí el merge y re-despachá.
5. Marcá la tarea como completada, actualizá `_state.json` y borrá el worktree (`orca worktree rm --worktree name:<slug> --json`), salvo que un gate pendiente lo necesite.
6. Si la tarea desbloquea a otras que ya corren (por ejemplo, un merge de `main` que necesitan), avisales: `git merge main` en su rama.

**Al recibir `escalation`/`ask`:**
- Si la respuesta está en el spec, contestás.
- Si implica cambiar un contrato, la escala, un umbral, una licencia o el alcance, abrís un gate.
- Mientras tanto seguís con el resto del DAG.

### 5.6 DAG de tareas

Prioridad = orden de la tabla. "Agente" es una sugerencia: si no hay Codex, usá Claude.

| ID | Tarea | Deps | Agente | Owns | Review |
|---|---|---|---|---|---|
| **T00** | Esqueleto, arnés (§3), contrato (§2.2), interfaces y stubs (§2.0), fixtures, `test_contract`, `predict` dummy | — | claude | repo inicial | sí |
| **G0** | *Humano: aprobar contrato e interfaces* | T00 | — | — | — |
| T03 | Fuente `sdb_bcs`: fetch del `.rar`, **preprocesamiento según `preprocesamiento/README.md`** (§2.5) y adaptador | G0 | claude | `src/bcs/data/sources/sdb_bcs.py`, `src/bcs/data/preprocess_sdb.py`, `scripts/data/fetch_sdb_bcs.py`, `tests/data/test_sdb_*`, `tests/data/test_preprocess_sdb_*` | sí |
| T04 | Fuente `zen_rgbd` (muestreo remoto + recorte de la vaca central) | G0 | claude | `…/zen_rgbd.py`, `fetch_zen_rgbd.py`, `tests/data/test_zen_*` | sí |
| T05 | Fuentes chicas: `criollo_dge`, `phone_backview`, `cowdb_beef`, más el fetcher de `iplab` (escrito y testeado con HTML de fixture, **nunca ejecutado** sobre datos reales) | G0 | codex | `…/sources/{iplab,criollo_dge,phone_backview,cowdb_beef}.py`, sus fetch y tests | no |
| T02 | Arnés de evaluación: métricas, ECE, cobertura, leave-one-source-out, chequeo de umbrales, gráficos | G0 | codex | `src/bcs/eval/**`, `eval/evaluate.py`, `tests/test_eval_*` | no |
| T01 | `POST /predict` con el `predict` dummy, validación de `cow_id`, token, guardado en el `Store` + deploy a Railway con volumen | G0 | claude | `src/bcs/api_predict.py`, `src/bcs/auth.py`, `Dockerfile`, `railway.json`, `tests/test_api_predict_*` | no |
| T16 | Backends de cómputo (§2.4): `bcs.train` con `--smoke`, `make pack-data`, `make train` para Modal, Kaggle, Tenki y local, bitácora de gasto | G0 | claude | `src/bcs/train.py`, `scripts/compute/**`, `.github/workflows/cpu-job.yml`, `tests/test_train_smoke*` | sí |
| T15 | Registro y endpoints de rodeo: `Store` en SQLite + miniaturas, `/cows`, `/cows/{id}`, `/predictions/{id}`, `/media/…` | G0 | codex | `src/bcs/store/sqlite.py`, `src/bcs/api_cows.py`, `tests/test_store_*`, `tests/test_api_cows_*` | sí |
| **GH** | *Humano: correr `make holdout`* | T03, T04, T05 | — | — | — |
| T06 | `make labels`: unión, exclusión del holdout, folds, `eval/data_report.md`, hojas de contactos | GH | claude | `src/bcs/data/{merge,splits}.py`, `eval/data_report.md`, `tests/test_splits_*` | sí |
| **G1** | *Humano: distribuciones, grilla, colas, fuentes a usar* | T06 | — | — | — |
| T07 | Web UI de evaluación (con `cow_id`) + vista de rodeo, contra la API dummy | T01, T15 | claude | `web/**`, `tests/browser/**` | no |
| T08 | Filtro de calidad | T05 | codex | `src/bcs/quality/**`, `tests/test_quality_*` | no |
| T09 | Baseline: tonto + embeddings congelados (ridge y logística ordinal); cache de embeddings en `artifacts/T09/`. Corre en Tenki o local | G1, T02, T16 | claude | `src/bcs/models/{baseline,embeddings}.py`, `config/train_m2.yaml`* | sí |
| **G2** | *Humano: revisar métricas del baseline y fijar umbrales de m3/m4* | T09 | — | — | — |
| T10a | Candidato A: fine-tune ConvNeXt-Tiny con cabeza ordinal (CORAL), en Modal | G2 | claude | `src/bcs/models/cand_a.py`, `config/cand_a.yaml`* | sí |
| T10b | Candidato B: DINOv2 + cabeza MLP con label smoothing gaussiano, en Modal | G2 | codex | `src/bcs/models/cand_b.py`, `config/cand_b.yaml`* | sí |
| T11 | Calibración (temperature scaling) + conformal; se desarrolla sobre las salidas de T09 | T09 | claude | `src/bcs/calibration/**`, `tests/test_calib_*` | sí |
| **G3** | *Humano: elegir modelo (tabla comparativa A vs B vs baseline, por fold y por fuente)* | T10a, T10b, T11 | — | — | — |
| T12 | Integración: modelo elegido + calibración + filtro en `inference.py`, export ONNX, `make smoke`/`make browser` con el modelo real, deploy | G3, T07, T08, T15 | claude | `src/bcs/inference.py`, `src/bcs/export.py`, `Dockerfile` | sí |
| T13 | Review adversarial (solo lectura): fugas de datos, métricas infladas, confianza que miente, formas de romper la API → `eval/review.md` | T12 | claude | `eval/review.md` | — |
| T13-fix-n | Un arreglo por hallazgo que lo amerite (los crea el orquestador) | T13 | cualquiera | según hallazgo | sí |
| T14 | README (problema, uso, métricas con desvío, leave-one-source-out, acuerdo humano de `iplab`, limitaciones, atribución de datos) + guion de demo | T13-fix-* | claude | `README.md`, `docs/demo.md` | no |
| **G4** | *Humano: `make eval-final` y OK para el deploy final* | T14 | — | — | — |

\* Excepción a la regla 9: cada tarea puede crear **su propio** archivo de config nuevo en `config/`, pero no tocar los existentes.

**T10a/T10b son un fan-out a propósito:** dos enfoques en paralelo, se elige uno en G3. Si el humano lo pide, se puede lanzar el mismo brief con dos agentes distintos.

### 5.6.1 Criterios por tarea (el orquestador los copia a cada brief)

**T00 · Esqueleto, arnés y contratos**
- Estructura de §2, arnés de §3 y directorios compartidos de §3.0.
- Interfaces y stubs de §2.0. El pipeline completo corre de punta a punta con stubs.
- `src/bcs/api.py` que solo crea la app, monta los routers de `api_predict.py` y `api_cows.py` (con stubs que respetan las rutas de §2.2) y sirve `web/`.
- `tests/fixtures/`: `prediction_example.json` (con `cow_id`, `prediction_id`, `created_at`, `image_sha256`), `rear_cow.jpg`, `side_cow.jpg`, `dog.jpg`, `labels_fixture.csv`.
- `test_contract.py` también cubre la normalización y validación de `cow_id`.
- Verificar que existe `preprocesamiento/README.md`. Si no está en el repo, se avisa en G0: T03 no puede arrancar sin él.
- `tests/test_contract.py` valida `Prediction`, el fixture y las reglas de `status`.
- `eval/thresholds.yaml` con las secciones `m2`, `m3` y `m4` (valores provisorios de abajo).
- `pyproject.toml` con **todas** las dependencias previsibles: torch, timm, scikit-learn, numpy, pillow, opencv-python-headless, fastapi, uvicorn, python-multipart, onnx, onnxruntime, open_clip_torch, ultralytics, remotezip, openpyxl, huggingface_hub, datasets, pandas, matplotlib, ruff, pyright, pytest. Así se evitan pedidos de dependencias después.
- Validaciones:
  - `make check`
  - `make smoke` (contra el dummy)
  - Romper un test a propósito y mostrar que el Stop hook bloquea; después revertir.
  - `AGENTS.md` es symlink.

**T01 · API + deploy temprano**
- `POST /predict` sobre el `predict` dummy:
  - Valida `cow_id` (422 si falta o es inválido) y el token (401).
  - Calcula `image_sha256`, guarda la predicción y la miniatura con el `Store`.
- Deploy en Railway con volumen para `$BCS_DB_PATH` y `$BCS_MEDIA_DIR`; `BCS_APP_TOKEN` en las variables de Railway.
- Validaciones:
  - `make smoke` (la parte de `/predict`)
  - `curl` sin `cow_id` → 422; sin token → 401
  - `curl <url-pública>/health` → 200
  - La URL queda en `tasks/_state.json`.

**T16 · Backends de cómputo**
- `bcs.train` lee un config, entrena un fold o todos y escribe `report.json` (formato de T02) y los pesos. Con `--smoke`: 1 fold, 1 época, 500 imágenes.
- Backends:
  - Modal: imagen con las dependencias del `pyproject.toml`, Volume `bcs-data`, GPU configurable (L4 por defecto), `timeout` por función, un contenedor por fold.
  - Kaggle: kernel script + `kernel-metadata.json` con GPU activada y el dataset privado como input.
  - Tenki: workflow `workflow_dispatch` con el label del runner, `timeout-minutes` y subida de `runs/` como artifact.
- Validaciones:
  - `make train BACKEND=local SMOKE=1`
  - `make train BACKEND=modal SMOKE=1`
  - `make train BACKEND=tenki SMOKE=1`
  - Kaggle smoke si las credenciales ya están
  - Los tres `report.json` tienen el mismo esquema y la bitácora tiene las líneas
  - Si falta una credencial, se reporta y se sigue con los demás backends

**T15 · Registro y endpoints de rodeo**
- Implementación SQLite del `Store` (tabla `predictions` con todos los campos de `Prediction` + `notes`, `taken_at` y ruta de la miniatura; índice por `cow_id, created_at`).
- Miniaturas JPG de lado mayor 512 px en `$BCS_MEDIA_DIR`.
- Endpoints de §2.2 en `api_cows.py`.
- Validaciones:
  - Tests del `Store`: guardar y leer, el último BCS válido ignora los `rejected`, el historial está ordenado, los filtros funcionan, la misma foto dos veces se detecta por `image_sha256`, persiste al reiniciar.
  - `make smoke` (la parte de `/cows`)

**T02 · Arnés de evaluación**
- Por fold y en promedio: MAE, accuracy exacta, ±1 paso y ±2 pasos, matriz de confusión, ECE de `confidence`, cobertura del intervalo, métricas por fuente y leave-one-source-out.
- Reliability diagram y matriz en PNG.
- Validaciones: tests con predicciones sintéticas de resultado conocido (perfectas, aleatorias, sesgadas en un paso).

**T03 / T04 / T05 · Fuentes de datos** (cada una con las recetas y trampas de §1.2)
- Primero con `make data-sample SOURCE=<id>`. La descarga completa recién cuando el sample pasa.
- Escribir en `sources.lock.json` la entrada de su fuente, en un archivo propio `sources.lock.<id>.json` que T06 une.
- Validaciones:
  - `test_labels_schema` sobre su `labels_<id>.csv`
  - `test_fetch_idempotent`
  - T03: `test_sdb_overlay_removed`
  - T03, preprocesamiento (§2.5):
    - El reporte incluye una tabla **paso del README → función que lo implementa → test que lo cubre → imágenes por clase antes y después**. Cada paso tiene su fila; un paso sin implementar o sin test no pasa la validación.
    - `test_preprocess_steps`: cada paso probado sobre un mini `.rar` de fixture (unas 30 imágenes de 3 clases, armado a partir de un sample real).
    - `test_preprocess_idempotent`: correrlo dos veces da el mismo manifiesto y no reescribe archivos.
    - `preprocess_manifest.json` presente, con los hashes del `.rar` y del README.
    - Hoja de contactos del resultado preprocesado (`eval/contact_sheets/sdb_bcs_pre.jpg`), mirada y descrita.
    - `test_group_id_recoverable`: el `group_id` (`<GRANJA>_<ID>`) se puede reconstruir para el 100% de las imágenes. Si se perdió con el preprocesamiento, **parar y escalar**: sin eso no hay split honesto.
    - `test_no_near_duplicates_across_folds` (hash perceptual): si el balanceo **duplicó o aumentó** imágenes antes de separar, las copias de una misma foto no pueden quedar en folds distintos. El balanceo se rehace **dentro** de cada fold de entrenamiento, nunca antes de separar.
  - T05: el fetcher de `iplab` solo se prueba con HTML de fixture; `make holdout` lo usa después
  - Hoja de contactos `eval/contact_sheets/<id>.jpg`, **mirada y descrita** en el reporte. En T04 hay que confirmar explícitamente qué vaca del frame es la etiquetada y que el recorte la toma.

**T06 · Unión, splits y reporte de datos**
- Validaciones: `test_no_group_leakage`, `test_labels_schema` sobre `labels.csv`, `test_holdout_excluded` (ningún `group_id` de `holdout_groups.txt` en `labels.csv`), `eval/data_report.md` con números y gráficos.
- El orquestador abre **G1** con ese reporte.

**T07 · Web UI**
- `web/index.html` mobile-first con:
  - Campo **ID de la vaca** obligatorio, arriba de todo, con autocompletado desde `/cows`. Recuerda el último ID usado; antes de enviar muestra "Vaca: <ID>" para confirmar.
  - Botón de cámara (`<input type="file" accept="image/*" capture="environment">`) y guía del protocolo de §0.2.
  - El resultado muestra el **ID de la vaca** junto con la miniatura de la foto, y tiene un botón "Siguiente vaca" que limpia la foto y el ID.
  - BCS grande, barra de confianza, intervalo y distribución chica.
  - Estados claros para `low_confidence` y `rejected`.
  - "Rango validado 2.75–5.0" y el aviso "Estimación de apoyo; no reemplaza la evaluación de un técnico".
  - Sin frameworks de front pesados.
- `web/rodeo.html` (dashboard):
  - Tabla o tarjetas, una por vaca: ID, miniatura de la última foto, último BCS con su intervalo y confianza, fecha y estado.
  - Orden por BCS ascendente por defecto (las más flacas primero) y búsqueda por ID.
  - Filtros: "BCS ≤ X", "baja confianza" y "última foto rechazada".
  - Al tocar una vaca: su historial con fotos, BCS a lo largo del tiempo (gráfico simple en SVG) y estados.
- Validaciones: `make browser` contra la API dummy a 390px. Evaluar dos vacas con IDs distintos y verificar que ambas aparecen en el rodeo con su miniatura y que el historial de una muestra solo sus fotos; fixture del perro → ve el rechazo.

**T08 · Filtro de calidad**
- Validaciones (umbrales de `m4`, sobre el filtro solo):
  - Set de rechazo (`tests/fixtures/reject/` + 30 laterales/cenitales de `phone_backview` y `cowdb_beef`): ≥ 85% rechazadas.
  - 72 traseras de `phone_backview`: ≥ 80% aceptadas.
  - Imágenes de val de `sdb_bcs`/`zen_rgbd`: ≤ 10% rechazadas.

**T09 · Baseline**
- Restricciones: todo en CPU (Tenki o local), embeddings cacheados en `$BCS_ARTIFACTS_DIR/T09/`. Las cabezas lineales sobre embeddings cacheados tienen que entrenar en < 15 min.
- Validaciones: `make eval MODEL=baseline STAGE=m2`. El mejor modelo mejora el MAE del baseline tonto en ≥ 25%.

**T10a / T10b · Candidatos**
- Misma CV que T09. Nada de elegir hiperparámetros mirando un solo fold.
- Se lanzan con `make train BACKEND=modal`: primero `SMOKE=1`, después `FOLDS=all`. Si el smoke proyecta más de $12 por candidato, se escala antes de lanzar la corrida completa.
- El balanceo de clases (sampler o pesos en la loss) se aplica **solo** dentro del fold de entrenamiento.
- Validaciones: `make eval MODEL=cand_a|cand_b STAGE=m3`. Umbrales provisorios, que se ajustan en G2:
  - MAE ≤ 0.35
  - Accuracy ±0.25 ≥ 0.60
  - Accuracy ±0.50 ≥ 0.85
  - Leave-one-source-out reportado
- Si no se alcanzan: reporte con curvas e hipótesis, en vez de seguir tocando a ciegas.

**T11 · Calibración y conformal**
- La calibración se ajusta con datos de validación, nunca con los de entrenamiento del mismo fold.
- Validaciones (`m4`) sobre las salidas de T09:
  - ECE ≤ 0.08
  - Cobertura del intervalo al 90% entre 0.87 y 0.95
- En T12 se vuelven a correr sobre el modelo elegido.

**T12 · Integración**
- `inference.predict(image, cow_id)` con el modelo real. Un test verifica que la predicción es **idéntica** para la misma foto con dos `cow_id` distintos (el ID no influye en el modelo).
- Validaciones:
  - `make smoke` y `make browser` con el modelo real, incluido el rodeo
  - `m4` sobre el modelo elegido
  - Latencia p50 de `/predict` < 2 s en el deploy
  - El deploy requiere el OK del humano (`make deploy` está en `ask`)

**T13 · Review adversarial**
- Entregable: `eval/review.md` con cada hallazgo, su severidad, evidencia y estado propuesto (arreglar / aceptar como limitación).

**T14 · README y demo**
- Validaciones: `make check`, `make smoke`, `make browser`. Todas las fuentes usadas tienen su cita y licencia en el README.

### 5.7 Gates (puntos de control humanos)

Cuando la tarea-gate queda lista, el orquestador crea la pregunta con `orca orchestration gate-create --task <task_id del gate> --question "<pregunta>" --json`, con opciones concretas, y avisa al humano. Las tareas que dependen del gate no se despachan hasta que se resuelve; **el resto del DAG sigue andando**.

En cada gate le mostrás al humano:
- qué se terminó;
- la evidencia (métricas, tablas, rutas a gráficos, hojas de contactos);
- qué decisión necesitás y qué recomendás.

| Gate | Pregunta |
|---|---|
| G0 | ¿Apruebo el contrato `Prediction`, las interfaces de §2.0 y el esquema de `labels.csv`? Después de esto quedan congelados. Además: ¿están las credenciales de Modal, Kaggle, Tenki y `gh` (§2.4)? ¿Qué fuentes además de `sdb_bcs` usamos? ¿Está `preprocesamiento/README.md` en el repo y `dataset.rar` en `$BCS_DATA_DIR/raw/sdb_bcs/` (o hay que bajarlo)? |
| GH | Los datos de entrenamiento están descargados. Corré `make holdout` y avisame cuando termine (no necesito ver el resultado). |
| G1 | Con estas distribuciones: ¿qué grilla usamos, agrupamos colas, qué fuentes entran al entrenamiento? |
| G2 | El baseline dio X. ¿Fijo los umbrales de m3/m4 en Y? (El humano edita `eval/thresholds.yaml`.) |
| G3 | Comparación A/B/baseline: ¿cuál va a producción? |
| G4 | Resultado de `make eval-final` (lo corre el humano) y deploy final. |

### 5.8 Cupos y recursos compartidos

- **Máximo 4 workers simultáneos.** El cuello de botella es la revisión, no los agentes.
- **Ancho de banda:** como mucho 2 tareas de descarga a la vez (T03, T04, T05).
- **GPU:** el entrenamiento corre en Modal, que permite varias GPUs simultáneas, así que T10a y T10b van en paralelo. El límite es **el presupuesto** de §2.4, no un lock. Antes de cada corrida completa, el orquestador mira la bitácora; si el gasto acumulado supera el 80% de algún presupuesto, abre un gate.
- **Datos:** después de G1, `$BCS_DATA_DIR` queda de solo lectura.

### 5.9 Reporte de estado al humano

Al abrir cada gate, y cuando se lo pidan, el orquestador muestra:

```
## Estado · <hora>
| ID | Estado | Agente/worktree | Vueltas | Resultado clave |
Bloqueos / esperando al humano:
Próximas tareas que se despachan:
URL pública actual:
```

---

## 6. Rol: Worker

1. **Al arrancar:** leé §0–§4, esta sección y tu `tasks/<ID>.md`. Tomá del preámbulo inyectado tu `task_id`, tu `dispatch_id` y cómo reportar. Para flags exactos, `orca skills get orchestration`; no adivines.
2. **Alcance:** solo tu worktree, solo los archivos de *Owns*. Si necesitás algo fuera (una dependencia, un cambio en una interfaz, una decisión de escala o de umbral), preguntá y esperá: `orca orchestration ask` bloquea hasta que el orquestador conteste. No inventes la respuesta.
3. **Loop:** implementar → correr las Validaciones del brief → iterar. Máximo 5 vueltas.
4. **Datos:** leés de `$BCS_DATA_DIR` y escribís artefactos solo en `$BCS_ARTIFACTS_DIR/<ID>/` y `runs/<ID>/`. Si tu tarea es de datos, escribís solo en `$BCS_DATA_DIR/{raw,images}/<source_id>/` y `labels_<source_id>.csv`.
5. **Mirá lo que producís:** si tu tarea genera imágenes (recortes, hojas de contactos, gráficos), abrí algunas y describí en el reporte lo que viste.
6. **Al terminar:** commit en tu rama con mensaje `<ID>: <resumen>`. Después:
   ```bash
   orca orchestration send --type worker_done --outcome succeeded \
        --files-modified <lista> --subject "<ID> listo" --json      # con el Reporte de §6.1 en el cuerpo
   ```
   Si fallaste a la quinta vuelta, lo mismo con `--outcome failed`.
7. **Después de reportar, esperás.** El orquestador puede mandarte feedback a tu terminal; eso abre una vuelta nueva. No mergees, no pushees, no borres tu worktree.

### 6.1 Reporte de vuelta (cuerpo del `worker_done`)

```
## Reporte · <ID> · vuelta <k>
Objetivo: <una línea>
Qué hice: <3–6 bullets>
Archivos tocados: <lista; todos deberían estar en Owns>
Validaciones:
  | comando | resultado | detalle |
Métricas (si aplica): <tabla, con delta vs. la referencia>
Lo que vi en las imágenes (si aplica):
Decisiones que tomé y por qué:
Dudas / riesgos para el orquestador:
```

---

## 7. Fuera de alcance

- Reconocer la vaca a partir de la foto: el ID siempre lo ingresa el usuario.
- Usuarios, roles y autenticación real (solo hay un token compartido).
- Uso offline o sincronización desde el campo sin señal.
- Combinar varias fotos de la misma vaca en una sola estimación (buen próximo paso: promediar distribuciones de fotos tomadas el mismo día).
- Imágenes de profundidad/3D (salvo que el dataset solo traiga eso).
- Video en tiempo real.
- App nativa.
