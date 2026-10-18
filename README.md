# ARGOS · Condición corporal (BCS) de vacas lecheras desde una foto trasera

Hackathon, Equipo 18. ARGOS estima el **score de condición corporal (BCS, escala lechera 1–5)** de
una vaca a partir de una foto tomada **desde atrás y arriba**, la guarda en el historial de la vaca
y muestra el estado del rodeo.

> **Rango validado: 3.25–4.25.** Es lo que trae el único dataset usado (`sdb_bcs`). El modelo no
> puede ver vacas flacas (< 3.25) ni muy gordas (> 4.25): las asigna al extremo más cercano y la API
> las marca `low_confidence` ("posible condición baja/alta, verificar a mano"). No reemplaza la
> evaluación de un técnico.

## El problema

El BCS es la forma estándar de seguir las reservas de grasa de una vaca lechera: una vaca que pierde
condición después del parto tiene más riesgo de problemas metabólicos y reproductivos. Se puntúa a
ojo (ilíacos, isquiones, hueco de la cola, lomo), lleva tiempo y varía entre evaluadores. ARGOS
busca dar un número consistente y un historial por vaca con una foto.

## Qué hace la app (lo que está en `main`)

- **Evaluar una vaca** (`/`): ID de la vaca → foto (cámara del celular o archivo) → resultado:
  BCS, intervalo, confianza, distribución sobre la grilla y foto guardada. Si la foto no sirve
  (no se encuentra la vaca, muy oscura, sobreexpuesta o borrosa) se rechaza con el motivo.
- **"Qué se observa en la foto"**: una descripción, región por región, generada por un modelo
  multimodal de OpenAI. Corre **en paralelo** con el BCS (no lo demora), es **ciega** (no ve el BCS
  predicho ni el ID) y está etiquetada como texto generado por IA. Requiere `OPENAI_API_KEY`; sin
  la variable la app funciona igual y la sección dice que no está disponible.
- **Rodeo** (`/rodeo.html`): lista de vacas con su última evaluación e historial, y un gráfico de
  **distribución de condición corporal** del rodeo (cada vaca cuenta una vez, con el promedio de
  sus BCS válidos; las que solo tienen fotos rechazadas se informan aparte).
- Diseño **ARGOS** (rediseño T18): una sola tarjeta con pasos Foto → Análisis → Resultado, pensado
  para celular. Las decisiones de diseño están en `docs/rediseno/decisiones.md`.

## API

Todo (API y web) va detrás de un **token compartido**: header `X-App-Token`, `?token=` (para el
link compartido; la web lo pasa a cookie y lo saca de la barra de direcciones) o la cookie
`bcs_token`. No es autenticación de usuarios: es lo mínimo para una demo. `/docs` y
`/openapi.json` están desactivados.

| método y ruta | qué hace |
|---|---|
| `POST /predict` | multipart `image` + `cow_id` → `Prediction` (contrato en `src/bcs/contract.py`) y la guarda |
| `GET /cows` | vacas con su última evaluación (filtros por BCS) |
| `GET /cows/distribution` | conteo de vacas por valor de la grilla (promedio por vaca) |
| `GET /cows/{cow_id}` | historial de una vaca |
| `GET /predictions/{id}` | una evaluación |
| `GET /predictions/{id}/explain` | "Qué se observa en la foto" (OpenAI); espera la que ya está en curso |
| `GET /media/{id}.jpg` | miniatura guardada |
| `GET /health` | estado (sin token; lo usa Railway) |

Campos principales de `Prediction`: `bcs` (argmax de la distribución calibrada), `bcs_expected`
(esperanza), `confidence` (masa de la distribución en `bcs ± 1 paso`), `interval` (conjunto
conformal al 90%), `distribution`, `status` (`ok` · `low_confidence` · `rejected`), `quality`
(motivos de rechazo) y `model_version`. El `cow_id` se guarda pero **nunca llega al modelo**.

## Arquitectura del modelo

```
foto ─► filtro de calidad básico (brillo, sobreexposición, nitidez)
     ─► detector YOLO11n del equipo (models/cow_det.pt, confianza ≥ 0,15) ── sin vaca → rejected
     ─► recorte de la grupa (+10 %)
     ─► EfficientNet-B0 de regresión del equipo (models/bcs_efficientnet_b0.pt, con TTA)
     ─► calibración T11 (models/calibration_efficientnet_b0.json): distribución sobre la grilla
        3.25–4.25 (paso 0,25) + conjunto conformal 90 %
     ─► reglas de status: bcs en un extremo (3.25 o 4.25) → low_confidence
```

- Grilla del modelo `dairy_sdb` (5 clases, `config/scale.yaml`); en producción `BCS_SCALE=dairy_sdb`.
- **Sin CLIP en producción**: con CLIP la app (~1,1 GB de RAM) no entraba en el plan de Railway. El
  "es una vaca" lo decide el detector a 0,15 (el equipo usaba 0,1). `BCS_QUALITY=clip` vuelve a
  CLIP y `BCS_DET_CONF` cambia el umbral.
- `model_version` identifica los tres archivos:
  `efficientnet_b0-5a5b28d6+cal-b69ad635+det-babe59e5` (versión evaluada en G4).
- Código: `src/bcs/inference.py` (pipeline), `src/bcs/api*.py` (API), `src/bcs/explain.py`
  (OpenAI), `src/bcs/store/` (SQLite + miniaturas), `web/` (UI), `eval/` (arnés de evaluación).

## Métricas

Datos: `sdb_bcs`, partido por **vaca/video** con `split.csv` del equipo (seed 42, 70/15/15, sin
vacas compartidas entre splits). **val** se usó para elegir el checkpoint y calibrar; **test** se
evaluó **una sola vez**, al final, con `make eval-final` (G4). "Paso" = 0,25 de BCS.

### Test (G4, 8.043 fotos de vacas que el modelo no vio) vs. val

Fuente: `~/bcs-shared/runs/G4/20261018T072019Z/report.json` (resumen en `tasks/G4.md`).
*Recortes* = el recorte anotado del XML directo al modelo; *punta a punta* = la foto cruda por el
pipeline de producción completo (filtro + detector + modelo + calibración).

| métrica | val (T11) | test recortes | test punta a punta |
|---|---|---|---|
| n | 8.028 | 8.043 | 8.035 aceptadas |
| MAE | 0,157 | 0,164 | **0,169** |
| BCS exacto | 46,9 % | 43,6 % | **42,9 %** |
| ±1 paso (±0,25) | 91,5 % | 91,8 % | **90,7 %** |
| ±2 pasos (±0,50) | 99,0 % | 99,1 % | **99,0 %** |
| sesgo medio | −0,004 | −0,016 | −0,019 |
| ECE | 0,012 | 0,011 | 0,017 |
| cobertura del intervalo 90 % | 90,5 % | 90,2 % | 88,9 % |
| ancho medio del intervalo (pasos) | 1,81 | 1,89 | 1,88 |
| rechazadas | 0,3 % (fotos crudas, T13) | — | 0,1 % (8/8.043) |
| `low_confidence` | — | — | 18,3 % |

- Referencia: un predictor tonto (siempre 3.75) tiene MAE 0,248 en val (T09).
- Umbrales (`eval/thresholds.yaml`): **m3 pasa** en los dos modos (MAE ≤ 0,35, ±1 ≥ 60 %,
  ±2 ≥ 85 %); **m4 pasa** salvo `accept_rear`, que no se pudo medir (no hay fotos traseras
  etiquetadas fuera del dataset; aceptado en G3).
- **La cifra honesta es la de test.** Las de val son optimistas: el equipo eligió el checkpoint
  mirando val y la calibración se ajustó ahí (G2). En test la exacta baja ~4 puntos y la MAE sube
  ~0,01.
- Rechazos en test: 7 "no se encontró la vaca", 1 "muy oscura".

### Por clase real (test punta a punta; sesgo = predicho − real)

| clase | n | MAE | sesgo | ±1 paso (recortes) |
|---|---|---|---|---|
| 3.25 | 1.063 | 0,186 | **+0,186** | 85,0 % |
| 3.50 | 2.017 | 0,148 | +0,066 | 94,8 % |
| 3.75 | 2.009 | 0,146 | −0,013 | 93,9 % |
| 4.00 | 2.064 | 0,176 | −0,123 | 93,9 % |
| 4.25 | 882 | 0,231 | **−0,231** | 83,2 % |

Los extremos **tiran al centro** (en val: +0,19 en 3.25 y −0,22 en 4.25; en test, igual).

### Por origen (granja/cámara del dataset, test punta a punta)

| origen | n | MAE | ±1 paso |
|---|---|---|---|
| GS | 4.420 | 0,190 | 86,9 % |
| YM | 2.259 | 0,160 | 93,0 % |
| L-R | 1.356 | 0,114 | 99,6 % |

L-R es un subconjunto fácil (±1 ≈ 100 % también en val) que **infla el promedio**; no es fuga de
identidad (T13, R11). GS y YM son la referencia más realista.

### Atajo de pintura roja (GS)

Algunas vacas de GS tienen marcas de pintura rojo-naranja, más frecuentes en 3.25 (49 %) que en
4.25 (20 %) en el dataset (T06): el modelo puede aprender "pintada ⇒ flaca".

| GS (recortes) | n | MAE | ±1 paso | MAE en 4.25 | ±1 en 4.25 |
|---|---|---|---|---|---|
| test, pintadas | 1.684 | 0,186 | 87,2 % | **0,365** | **54,2 %** |
| test, no pintadas | 2.738 | 0,180 | 90,1 % | 0,204 | 84,2 % |
| val, 4.25 pintadas vs. no | — | — | — | 0,48 vs. 0,25 | — |

En val, desaturar la pintura cambia la clase predicha en el 16,5 % de las fotos pintadas (T09). El
atajo persiste en test, menos que en val. Reentrenar con la pintura desaturada queda para después.
El texto sobreimpreso del dataset (fecha, "BCS") **no** influye: Δ medio 0,006 de BCS al taparlo (T09).

### Desvío, leave-one-source-out y acuerdo humano

- **Desvío:** no hay desvío entre folds porque, por decisión de G0, se usa un único split
  train/val/test (sin validación cruzada). La dispersión que sí se reporta es por clase y por
  origen (tablas de arriba): MAE de 0,11 (L-R) a 0,19 (GS) y de 0,15 a 0,23 entre clases. No se
  calcularon intervalos de confianza por bootstrap.
- **Leave-one-source-out: no aplica.** Hay una sola fuente (`sdb_bcs`); no se midió generalización
  a otra granja, otra cámara u otro país.
- **Acuerdo humano (`iplab`): no disponible.** `iplab` no se usó (G0), así que no hay techo de
  referencia entre técnicos para comparar.

### Fuga conocida del split

`split.csv` deja algunos casi-duplicados entre splits (cortes de tramos en huecos de numeración y
vacas GS con dos IDs): 109 pares con correlación ≥ 0,9 (75 imágenes), < 0,5 % del total. Se dejó
tal cual (G0, punto 7) y está documentada en el manifiesto de preprocesamiento. Val y test pueden
estar **levemente inflados**; T13 estimó el efecto en val en ~0,0006 de MAE.

## Limitaciones

- **Una sola fuente**: `sdb_bcs`, cámara fija en granja (China), vacas Holando. Nada de razas
  carniceras, otras cámaras ni otras granjas.
- **Sin evaluación en fotos de celular.** Todo lo medido es con la cámara fija del dataset; el
  salto de dominio a una foto de celular no está medido.
- **Rango 3.25–4.25.** No hay vacas flacas etiquetadas: el caso de mayor riesgo (BCS < 3) no se
  detecta, solo se avisa "posible condición baja" cuando la predicción cae en 3.25.
- **Los extremos tiran al centro**: una vaca real de 3.25 o 4.25 suele salir 3.50 o 4.00.
- **Atajo de pintura roja en GS**: en vacas pintadas de 4.25 el modelo subestima (ver arriba).
- **Sin CLIP no se verifica que la foto sea de atrás** (`is_cow_rear` queda en `True`), y algunos
  negativos pasan con `ok`: en el set de rechazo (40 imágenes, dibujos) se rechaza el 85 %; pasan,
  por ejemplo, la cara de una vaca o una figura de círculos concéntricos. También fotos rotadas,
  invertidas o mosaicos de dos vacas (T13, R8).
- **La confianza casi no depende de la imagen** (T13, R1/R7): la calibración tiene σ constante, así
  que `confidence` depende solo del valor predicho y casi nunca baja de 0,8 (en val, entre 0,805 y
  0,987). Está bien calibrada en promedio (ECE 0,017 en test), pero **no** avisa en fotos raras,
  vacas pintadas ni en los extremos reales; las reglas "confianza < 0,5" e "intervalo > 4 pasos"
  no se disparan nunca con esta grilla.
- **La explicación de OpenAI puede equivocarse** (describe rasgos que no siempre están), es
  independiente del BCS, manda la foto a un servicio externo y **tiene costo por predicción**.
- El token compartido no es autenticación de usuarios; no hay límite de uso (T13, R15).

## Cómo correr local

Requisitos: [`uv`](https://docs.astral.sh/uv/) (Python fijado en `.python-version`). Los datos y
artefactos viven fuera del repo, en `~/bcs-shared/` (`$BCS_DATA_DIR`, `$BCS_ARTIFACTS_DIR`).

```bash
uv sync
make check         # ruff format --check, ruff check, pyright, pytest
make smoke         # API con base temporal y predictor dummy: /predict, /cows, token, cow_id
make smoke-model   # lo mismo con el modelo real (BCS_PREDICTOR=model, BCS_SCALE=dairy_sdb)
make eval MODEL=<name> STAGE=<m2|m3|m4>   # métricas sobre val contra eval/thresholds.yaml
```

Levantar la app con el modelo real:

```bash
BCS_APP_TOKEN=<un-token> BCS_PREDICTOR=model BCS_SCALE=dairy_sdb \
  uv run uvicorn bcs.api:app --port 8000
# abrir http://127.0.0.1:8000/?token=<un-token>
```

Sin `BCS_PREDICTOR=model` usa un predictor dummy (para tests). La base SQLite y las miniaturas van a
`BCS_DB_PATH` y `BCS_MEDIA_DIR` (por defecto en `~/bcs-shared/app/`). Con `OPENAI_API_KEY` se
habilita "Qué se observa en la foto" (`OPENAI_MODEL` cambia el modelo).

`make eval-final` evalúa sobre test y **solo lo corre el humano** (G4).

## Deploy (Railway)

El `Dockerfile` arma API + web en un solo contenedor (torch CPU, sin CLIP, sin datos ni imágenes de
terceros) y `railway.json` lo construye y usa `/health` como healthcheck. `/data` es el volumen de
Railway (base SQLite y miniaturas). Variables a configurar en Railway (**nunca** poner valores
reales en el repo):

| variable | para qué |
|---|---|
| `BCS_APP_TOKEN` | token compartido de la app (obligatorio) |
| `BCS_SCALE` | `dairy_sdb` (ya viene en la imagen) |
| `OPENAI_API_KEY` | habilita la explicación; sin ella la app funciona igual |

## Atribución y licencias

- **Dataset `sdb_bcs`**: Huang Xiaoping, Dou Zihao, Huang Fei, Zheng Huanyu, Hou Xiankun,
  Wang Chenyang, Feng Tao, Rao Yuan (Anhui University / Anhui Agricultural University), *Dairy cow
  body condition score target detection data set*, V3, Science Data Bank, 2025.
  DOI [10.57760/sciencedb.16704](https://doi.org/10.57760/sciencedb.16704). Licencia
  **CC BY 4.0** (verificada en los metadatos de DataCite del DOI). Es la única fuente de datos; no
  se redistribuye: ninguna imagen del dataset está en el repo ni en la imagen de Docker.
- **Modelos y scripts del equipo**: `models/bcs_efficientnet_b0.pt`, `models/cow_det.pt`, el
  baseline DINOv2 + LightGBM y los scripts `make_split.py`, `apply_split.py`, `prep_crops.py`,
  `prep_yolo.py`, `train_cnn.py` y `split.csv`, entrenados y escritos por el equipo sobre `sdb_bcs`.
- **Bibliotecas**: [timm](https://github.com/huggingface/pytorch-image-models) (Apache-2.0;
  EfficientNet-B0 preentrenada en ImageNet), PyTorch (BSD-3-Clause),
  [Ultralytics YOLO11](https://github.com/ultralytics/ultralytics) (**AGPL-3.0**: el detector y su
  uso en un servicio quedan bajo esa licencia salvo licencia comercial de Ultralytics), FastAPI,
  scikit-learn y Pillow.
- **OpenAI**: la explicación usa la Responses API de OpenAI; la foto se envía a OpenAI y el texto
  está sujeto a sus términos de uso.
- Guía de rasgos de la explicación: escala de Edmonson et al. (1989, *J Dairy Sci* 72:68–78) y
  diagrama de Ferguson, Galligan y Thomsen (1994, *J Dairy Sci* 77:2695–2703).
- Imágenes de la web (`web/img/`, logo e ilustración de ARGOS): material propio del equipo (T18).

## Reproducir el entrenamiento (scripts del equipo)

Los modelos de `models/` se entrenaron con los scripts de la raíz (Windows/Anaconda, RTX 2050):

```bash
python make_split.py <dataset> split.csv        # split por vaca (sin fuga entre train/val/test)
python apply_split.py <dataset> split.csv dataset_split
python prep_crops.py                             # recortes por bbox -> crops/
python baseline_embed.py                         # baseline DINOv2 + LightGBM
python train_cnn.py --epochs 12                  # EfficientNet-B0
python prep_yolo.py build && yolo detect train data=yolo_data/data.yaml model=yolo11n.pt imgsz=640 epochs=20 batch=16 workers=2 project=runs name=cow
```

Nota Windows/Anaconda: los scripts definen `KMP_DUPLICATE_LIB_OK=TRUE` (conflicto de OpenMP que tira
los workers del DataLoader). No usar `channels_last` en esa GPU: hace el entrenamiento 16× más lento.
La demo original del equipo sigue en `app.py` / `bcs_pipeline.py` y sus métricas propias en
`results/`; la app de producción es `bcs.api` y las métricas de referencia son las de G4.

## Flujo de trabajo del equipo

1. `git pull` antes de empezar.
2. Trabajar en una rama/worktree por tarea (ej: `feature/frontend`).
3. Commit y push de la rama; merge a `main` cuando esté validado.
