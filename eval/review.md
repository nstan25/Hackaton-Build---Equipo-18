# T13 · Review adversarial

Revisión de solo lectura del pipeline de producción de G3 (`BCS_PREDICTOR=model`, `dairy_sdb`,
EfficientNet-B0 del equipo + calibración de T11 + `BasicQualityChecker` + `cow_det.pt` a 0,15),
de la API y de las métricas de val. Commit revisado: `69ba895`. **Nada de test/holdout**: todos los
números salen de val (`splits.csv`), de `tests/fixtures/reject/` o de entradas sintéticas.

Scripts y salidas crudas: `~/bcs-shared/runs/T13/` (`attack.py` contra la API local,
`probe.py` → `probe.csv` con transformaciones y fixtures de rechazo, `e2e.py` → `e2e.csv` con 600
fotos crudas de val de punta a punta).

Severidad: **alta** = rompe la demo o hace que el número mostrado engañe; **media** = falla
reproducible con impacto acotado; **baja** = detalle; **info** = verificación sin defecto.

## Resumen

| # | Sev. | Hallazgo | Estado propuesto |
|---|---|---|---|
| R1 | alta | La confianza no depende de la imagen: auto, gato, flor o dos vacas pegadas reciben `ok` con ~0,9 | arreglar (señal OOD → `low_confidence`) + declarar |
| R2 | alta | La confianza miente en GS pintadas de 4.25 (dice 0,92, acierta 0,35) y en los extremos | arreglar (regla de pintura) + declarar |
| R3 | alta | Una PNG de 250 KB lleva el proceso a 1,35–1,9 GB de RAM (OOM en Railway) | arreglar |
| R4 | media | Archivo que no es imagen (o HEIC) → 500 en vez de 422, por un parche de ultralytics a `Image.open` | arreglar |
| R5 | media | El token en `?token=` queda en los logs de acceso (y la web lo pone en cada miniatura) | arreglar |
| R6 | media | Se aceptan y guardan en disco cuerpos de 300 MB **antes** de chequear el token | arreglar o aceptar |
| R7 | media | Las reglas `confidence < 0,5` e "intervalo > 4 pasos" no se disparan nunca | aceptar + declarar (o gate) |
| R8 | media | Sin CLIP pasan con `ok`: 6/40 negativos, fotos rotadas, invertidas, recortadas y mosaicos | aceptar (G3) + actualizar texto UI |
| R9 | baja | Fuentes de optimismo en val; la fuga conocida pesa 0,0006 de MAE | aceptar |
| R10 | info | Punta a punta en fotos crudas de val reproduce las métricas (MAE 0,151); rechazo 0,3% ≠ 7,7% de G3 | revisar la cifra de G3 |
| R11 | baja | L-R da ±1 = 0,999: no es fuga de identidad, es un subconjunto fácil que infla el promedio | aceptar + reportar por origen |
| R12 | baja | Ruido de etiqueta: casi-duplicados con etiqueta distinta | aceptar |
| R13 | baja | `cow_id` en Unicode se normaliza a otro ID ASCII (`ß1` → `SS1`) | arreglar (archivo caliente) |
| R14 | baja | `taken_at` futuro o sin zona, `bcs_min=nan` | aceptar |
| R15 | baja | Sin límite de uso ni aviso de foto repetida; se guardan miniaturas de selfies rechazadas | aceptar |
| R16 | info | Docstrings que todavía hablan de CLIP; `is_cow_rear=True` en rechazos por "no hay vaca" | arreglar docs |

---

## R1 · alta · La confianza no mira la imagen

**Qué pasa.** La calibración elegida en T11 (`ml_deg0_class`) es de grado 0: la media es
`valor + c` y la σ es **constante**. La distribución sobre la grilla y, por lo tanto, `confidence`
dependen solo del valor continuo de la regresión, no de qué tan "normal" es la foto. Cualquier
cosa que el detector deje pasar recibe la misma confianza que una foto de entrenamiento.

**Evidencia.**
- Val (T11, n = 8.028): `confidence` entre **0,805 y 0,987** (p1 0,827, mediana 0,921); 0 casos < 0,5.
- Fixtures de rechazo por el pipeline de producción (`probe.csv`): `car.png` → 3.75, `ok`, 0,91;
  `cat.png` → 3.75, `ok`, 0,89; `flower.png` → 4.00, `ok`, 0,94; `food_plate.png` → 4.00, `ok`, 0,93;
  `barn_empty.png` → 3.75, `ok`, 0,89; `cow_face.png` → 3.75, `ok`, 0,89.
- 40 fotos de val (8 por clase) transformadas: un mosaico de **dos vacas distintas** lado a lado
  recibe BCS en el 100% de los casos, `ok` en el 85%, confianza media **0,934** (más alta que la
  original, 0,919). Rotada 90°: 37,5% recibe BCS con 0,917. Colores invertidos: 65% con 0,921.

**Propuesta: arreglar.** Usar una señal que sí dependa de la imagen para bajar a `low_confidence`:
confianza del detector (está en el resultado de YOLO y hoy `prep_yolo.detect_cow` la descarta; como `prep_yolo.py` es del equipo y no se edita, habría que leerla desde `CowLocator`), cantidad de cajas
(dos vacas → `low_confidence`/`rejected`), o desacuerdo entre las vistas del TTA. No requiere cambiar
el contrato (solo `decide_status` vía `quality.issues` o un parámetro opcional más, como la regla de
extremos). Mientras tanto, UI/README: "la confianza está calibrada para fotos como las del dataset;
no detecta fotos raras".

## R2 · alta · La confianza miente en subgrupos (pintura roja y extremos)

**Evidencia** (T11 `predictions_val.csv`, `confidence` = masa en `bcs ± 1 paso`, pintura según
`runs/T06/paint_frac.csv` con umbral 0,5%):

| subgrupo (clase real) | n | confianza media | acierto ±1 real | cobertura 90% | brecha |
|---|---|---|---|---|---|
| GS pintada, 4.25 | 100 | 0,921 | **0,350** | **0,550** | +0,57 |
| GS no pintada, 3.25 | 298 | 0,924 | 0,792 | 0,839 | +0,13 |
| GS no pintada, 4.25 | 470 | 0,926 | 0,815 | 0,926 | +0,11 |
| todas, 3.25 | 1.111 | 0,926 | 0,852 | 0,898 | +0,07 |
| todas, 4.25 | 984 | 0,925 | 0,805 | 0,905 | +0,12 |
| GS (todo) | 4.500 | 0,921 | 0,892 | 0,881 | +0,03 |
| L-R (todo) | 1.332 | 0,924 | 0,999 | 0,996 | −0,08 |

Por valor **predicho** la calibración está bien (brecha ≤ 0,05 en todas las clases), que es lo que
mide la ECE; el problema es que una vaca real de 4.25 pintada se predice en 3.75–4.00 con 0,92 de
confianza. La regla de extremos de G3 no la ataja porque la predicción **no** cae en el extremo.
G2 ya declaró el sesgo; lo nuevo es que la **confianza** mostrada no lo refleja.

**Propuesta: arreglar (barato) + declarar.** La pintura se puede medir en inferencia con la misma
definición HSV de T06 sobre el recorte: si `red_frac > 0,5%`, marcar `low_confidence` con un texto
del tipo "marca de pintura: el modelo tiende a subestimar". Si no se arregla, la UI tiene que decir
que la confianza es un promedio y no vale para vacas pintadas ni en los extremos reales.

## R3 · alta · Bomba de memoria con una imagen chica

**Qué pasa.** `MAX_IMAGE_BYTES` (20 MB) limita bytes, no píxeles. Una PNG de un color de
10000×8000 pesa **252 KB**. PIL solo levanta `DecompressionBombError` por encima de ~179 MP; entre
89 y 179 MP es apenas un warning. La foto se decodifica entera (`_to_image`), después otra vez en
`make_thumbnail` (fuera del `_model_lock`), en RGB.

**Evidencia** (API local, `BCS_PREDICTOR=model`, RSS del proceso uvicorn en reposo 0,64 GB):

| entrada | bytes | RSS pico |
|---|---|---|
| PNG 10000×8000 (80 MP) | 252 KB | **1,35 GB** |
| PNG 12000×10000 (120 MP, solo warning) | 375 KB | **1,71 GB** |
| 4 × 80 MP en paralelo | 4 × 252 KB | **1,90 GB** |

G3 dice que la app con CLIP (~1,1 GB) ya superaba el límite de Railway y entraba en loop de
reinicio: **una sola request con token tira el contenedor**. (Las 4 respondieron 200 `rejected`
localmente; en Railway se esperaría OOM.)

**Propuesta: arreglar.** Bajar `Image.MAX_IMAGE_PIXELS` (p. ej. 40 MP; una foto de celular de
48 MP se puede achicar con `draft()` para JPEG) y chequear `img.size` después de `Image.open` y
antes de `load()`/`exif_transpose`; responder 413/422. Generar la miniatura desde la imagen ya
decodificada (o dentro del lock) en vez de decodificar dos veces.

## R4 · media · Archivo que no es imagen → 500

**Qué pasa.** Importar ultralytics reemplaza `PIL.Image.open` por `ultralytics.utils.patches.image_open`.
Ante cualquier archivo no reconocido intenta `check_requirements("pi-heif")` y después
`from pi_heif import ...`, que levanta `ModuleNotFoundError`. `_predict_and_save` solo atrapa
`OSError`/`DecompressionBombError`, así que el cliente recibe **500**.

**Evidencia.** `b"hello world"` → 500; un SVG → 500; una JPEG truncada → 422 (bien). Log:
`requirements: Ultralytics requirement ['pi-heif'] not found, attempting AutoUpdate... AutoUpdate
skipped (offline)` + `ModuleNotFoundError: No module named 'pi_heif'`. En Docker no instala nada
porque `YOLO_OFFLINE=1`; **sin esa variable (local, otro deploy) ultralytics intentaría un
`pip`/`uv install` en runtime** desde dentro de una request. Un HEIC de iPhone subido como archivo
(no por la cámara del navegador, que suele convertir a JPEG) también terminaría en 500.

**Propuesta: arreglar.** En `_to_image`/`_predict_and_save`, mapear cualquier excepción al abrir la
imagen a 422; fijar `YOLO_AUTOINSTALL=false` además de `YOLO_OFFLINE`. Decidir si se agrega
`pi-heif` como dependencia (HEIC) o se declara "solo JPEG/PNG/WebP".

## R5 · media · El token queda en los logs

**Evidencia.** El `CMD` del Dockerfile corre uvicorn con access log por defecto. Local:
`"GET /cows?token=tok&bcs_min=nan HTTP/1.1" 200 OK`. `web/index.html` y `web/rodeo.html` usan
`withToken()` para los `<img src="/media/…jpg?token=…">` y los links entre páginas, así que cada
miniatura del rodeo escribe el token en los logs de Railway (y en el historial del navegador).
SPEC §2.2 permite `?token=` en el link compartido, pero no dice nada de los logs.

**Propuesta: arreglar.** `--no-access-log` o un filtro de logging que borre `token=`; para las
miniaturas, `fetch` con header + `URL.createObjectURL`, o una cookie `HttpOnly` que la web setea
al entrar con `?token=`. Si se acepta, rotar el token después de la demo.

## R6 · media · Cuerpos enormes antes de la autenticación

**Evidencia.** `curl -F image=@300MB.bin` con token válido → 413 en 0,58 s; **con token inválido →
401, pero recién después de recibir y parsear los 300 MB**: FastAPI resuelve el formulario
multipart (Starlette lo vuelca a un temporal en disco) antes de evaluar `Depends(require_token)`.
Cualquiera que conozca la URL puede llenar el disco temporal o el ancho de banda sin token. Que el
proxy de Railway limite el tamaño del cuerpo no está verificado.

**Propuesta: arreglar o aceptar.** Middleware ASGI que rechace `Content-Length` > ~21 MB y chequee
el token antes de leer el cuerpo (requiere tocar `api.py`, archivo caliente). Para una demo de
fin de semana es aceptable si se declara.

## R7 · media · Dos de las tres reglas de `low_confidence` están muertas

**Evidencia.**
- `confidence < 0,5`: mínimo en val 0,805; en fotos fuera de distribución ~0,9 (R1). Nunca dispara.
- Intervalo de **más de 4 pasos**: la grilla `dairy_sdb` tiene 5 clases, el intervalo más ancho
  posible (3.25–4.25) mide exactamente 4 pasos. Imposible por construcción.
- Queda solo la regla de extremos de G3. En la práctica `ok` = "el detector encontró algo y el
  argmax no es 3.25 ni 4.25". En 600 fotos de val de punta a punta: 490 `ok`, 108 `low_confidence`
  (todas por extremo), 2 `rejected`.

**Propuesta: aceptar y declarar**, o llevar a un gate un umbral relativo a la grilla de 5 clases
(p. ej. confianza < 0,85, o intervalo ≥ 3 pasos). No se cambia ningún umbral desde un worker.

## R8 · media · Qué pasa el filtro sin CLIP

**Evidencia** (`probe.csv`, pipeline de producción, detector a 0,15):
- Set de rechazo (40): pasan **6** con `ok` (barn_empty, car, cat, cow_face, flower,
  food_plate) → rechazo 0,85, igual que G3. Pero los que pasan no son los que lista G3:
  `concentric.png` hoy se rechaza ("sobreexpuesta"); auto, gato, flor y plato pasan.
- 40 fotos reales de val transformadas (fracción que recibe BCS / cambia de clase respecto de la
  original):

| transformación | recibe BCS | `ok` | confianza | clase distinta |
|---|---|---|---|---|
| original | 1,00 | 0,78 | 0,919 | — |
| mosaico de dos vacas | 1,00 | 0,85 | 0,934 | 0,55 |
| gris | 1,00 | 0,85 | 0,918 | 0,50 |
| 64×48 px | 0,95 | 0,88 | 0,922 | 0,68 |
| franja superior 40% | 0,83 | 0,73 | 0,920 | 0,52 |
| JPEG calidad 5 | 0,78 | 0,73 | 0,919 | 0,55 |
| rotada 180° | 0,65 | 0,65 | 0,920 | 0,50 |
| colores invertidos | 0,65 | 0,63 | 0,921 | 0,73 |
| franja inferior 40% | 0,53 | 0,53 | 0,929 | 0,57 |
| rotada 90° | 0,38 | 0,38 | 0,917 | 0,67 |

Una foto de 64×48 px pasa el filtro de nitidez y recibe `ok`. Dos vacas en la misma foto no se
detectan como problema (el detector devuelve una caja y se usa esa).

**Propuesta: aceptar** (decisión G3) **y actualizar el texto** de UI/README con ejemplos reales
(animales/objetos sueltos, fotos rotadas o con varias vacas). Mejoras baratas sin CLIP: tamaño
mínimo de imagen y de caja, rechazar si hay > 1 vaca con confianza alta, rechazar si la caja es
mucho más ancha que alta en una foto apaisada rotada.

## R9 · baja · Optimismo de las métricas de val (incluida la fuga conocida)

Val se usó para: elegir el checkpoint del equipo (G2), los umbrales de T08, el umbral 0,15 del
detector (G3) y ajustar la calibración de producción con todo val. El cross-fitting de T11 solo
corrige lo último. La cifra honesta sigue siendo la de test en G4.

Fuga conocida (G0.7) cuantificada sobre la predicción real: 178 imágenes de val tienen un
casi-duplicado en train (dHash ≤ 4, `runs/T03/leak_evidence/dups.json`); su MAE es 0,128 vs.
0,157 del resto, pero sacarlas mueve el MAE global de **0,1566 a 0,1572**. Despreciable. Además,
sin fuga visible de identidad en L-R (R11). Detector y EfficientNet se entrenaron con train/val
del mismo `split.csv` (`prep_yolo.py build`, `train_cnn.py`): no tocaron test.

**Propuesta: aceptar** (ya declarado en G2).

## R10 · info · Punta a punta ≈ métricas publicadas

T09/T11 evalúan sobre los recortes del XML con la franja de texto tapada; producción usa la foto
cruda → detector → recorte +10% sin tapar. Se corrió el pipeline completo sobre 600 fotos crudas
de val (`e2e.csv`): MAE **0,151**, ±1 0,903, cobertura 0,903, confianza media 0,925; diferencia
media del valor continuo vs. el recorte del XML 0,040 (p90 0,085). No hay brecha oculta por el
detector. Por origen: GS ±1 0,864 con confianza 0,926 (sobreconfianza de 6 puntos), YM 0,924,
L-R 0,991.

**A revisar:** rechazo en val de punta a punta **2/600 = 0,3%**, contra el 7,7% de G3 punto 5
("300 fotos de val"). Posiblemente G3 midió sobre los recortes preprocesados; conviene confirmar
qué cifra va al README.

## R11 · baja · L-R infla el promedio, pero no por fuga de identidad

L-R da ±1 = 0,999 y exacta 0,64 en val (GS 0,89/0,43). Chequeo de fuga: 1-NN con los embeddings
DINOv2 de T09, de val contra train del mismo origen: acierto ±1 del vecino 0,786 en L-R vs. 0,746 en
GS y 0,712 en YM; similitud mediana 0,875 / 0,864 / 0,843. No hay señal de que las mismas vacas se
repitan. La explicación es que L-R es fácil: dispersión del valor predicho dentro de cada clase
0,05–0,11 y ningún error de 2 pasos. Como L-R es el 17% de val, el ±1 global (0,916) mezcla un
subconjunto fácil con GS/YM (0,89/0,91).

**Propuesta: aceptar** y reportar las métricas por origen en el README, con GS/YM como referencia
para fotos de otras granjas.

## R12 · baja · Ruido de etiqueta

En los pares val–train con dHash ≤ 4, la etiqueta coincide solo en el **51%**. En L-R, el mismo
número de frame en `L-i` y `R-i` (mismo grupo) tiene etiqueta distinta en el **19%** de los 3.477
pares (¿dos vacas distintas, una por cámara, o ruido de etiqueta?). Pone un techo a la exactitud
y explica parte del error en los extremos. **Aceptar y mencionar.**

## R13 · baja · Normalización de `cow_id` con Unicode

`normalize_cow_id` hace `strip().upper()` y **después** valida la regex ASCII. `str.upper()` de
Python convierte `ß1` → `SS1` y `ı` (i sin punto) → `I`, que pasan; `A1\n` → `A1`. Dos IDs
tipeados distintos terminan en la misma vaca sin aviso. Probado contra la API (200 con
`cow_id: "SS1"`). **Arreglar:** rechazar si `raw.strip()` no es ASCII antes de `upper()`
(`contract.py`, archivo caliente; cuidado con `tests/test_contract.py`, que no se toca).

## R14 · baja · Validaciones blandas

`taken_at=9999-12-31` se acepta (200), igual que fechas sin zona. `GET /cows?bcs_min=nan` devuelve
`[]` sin error (cualquier comparación con NaN es falsa). **Aceptar.**

## R15 · baja · Uso abusivo y privacidad del registro

Sin límite de requests; cada `rejected` también guarda miniatura (una selfie o una persona quedan
en `$BCS_MEDIA_DIR`). `image_sha256` se guarda pero no se usa: la misma foto subida para dos vacas
distintas no genera aviso (`SqliteStore.find_by_image_sha256` no tiene llamadores). **Aceptar**
para la demo; opcional: avisar en la UI si la foto ya existe con otro `cow_id`.

## R16 · info · Documentación desactualizada

- `src/bcs/inference.py`: el docstring del módulo dice "filtro de calidad CLIP (T08)" y `warmup()`
  "Carga CLIP"; en producción no hay CLIP.
- Un rechazo por "no se encontró la vaca" devuelve `quality.is_cow_rear = True` (el
  `BasicQualityChecker` lo fija antes del detector). El contrato lo permite, pero lo que se muestra
  es contradictorio.

**Arreglar** (docstrings; el campo puede quedar).

## Verificado sin hallazgos

- Path traversal: `/media/..%2F..%2Fetc%2Fpasswd.jpg` y `/%2e%2e/pyproject.toml` → 404; `api_web`
  usa `resolve()` + `is_relative_to`; la miniatura sale de la base, no de la URL.
- XSS: `web/*.html` escribe todo con `textContent`; `notes` con HTML se guarda tal cual y no se
  interpreta.
- Token: comparación en bytes con `hmac.compare_digest`; `/docs` y `/openapi.json` apagados.
- Formatos y modos: 1×1, 20000×4, L, CMYK, RGBA, I;16, F (TIFF), GIF, WebP, BMP, ICO y EXIF con
  orientación fuera de rango → 200 (`rejected` o BCS); JPEG truncada → 422; `notes` de 281 → 422;
  `cow_id` `..` / Unicode ancho → 422.
- Concurrencia: 50 `/predict` simultáneos con foto real; `/health` y `/cows` siguieron respondiendo
  en < 3 ms durante la cola.
- `cow_id` no llega al modelo: `run_pipeline` solo lo normaliza y lo copia a la `Prediction`.
