# Datasets de BCS: lo que T23 no verificó · T24

Fecha de consulta de todas las fuentes: **23/10/2026**. Complementa `docs/investigacion/datasets-bcs.md` (T23), que
no se repite acá. Convenciones de T23: ✔ = verificado por mí sobre la página, la API o metadatos chicos guardados en
`~/bcs-shared/artifacts/T24/` (sin bajar imágenes, salvo 4 recortes de 128 px para mirarlos); sin ✔ = sale del
texto del paper; **(no verificado)** = solo snippet de buscador o inferencia.

## Resumen

- **Sigue sin existir un dataset público en escala lechera 1–5 con vacas < 2.75.** Revisé a fondo lo que T23 dejó
  abierto (ScienceDB y su API en chino, la API pública de Kaggle, Roboflow, los statements de Li 2025,
  Halachmi y Embrapa, y papers de 12 países) y no apareció ninguno.
- **Corrección a T23: `sdb_bcs` es CC BY 4.0** ✔ (API de ScienceDB). Hay solo V1, V2 y V3, las tres con el mismo
  archivo de 4,4 GB y las mismas 5 clases (3.25–4.25) ✔. No hay versión con más rango. Con esto se cierra el
  pendiente de SPEC §1.2.
- **Nuevo candidato a pedido para las flacas: Shriramulu, Pushpadass et al. 2024 (India).** 503 fotos RGB
  dorsales de cruzas Holando, escala 1–5 con paso 0,25, **~2/3 de las imágenes por debajo de 2.75** (de 1.5 a
  2.5). Es el segundo mejor candidato para flacas después de Nagy 2023 (T23) y conviene mandarle mail en paralelo.
- **Kaggle no tiene nada nuevo que sirva.** El único dataset llamado "Cattle Body Condition Score" es un **espejo
  de `criollo_dge`** (escala 1–9 de carne, depth) ✔. Lo demás son keypoints sin BCS o un recorte de isquiones con
  3 etiquetas cualitativas sin documentar.
- **Roboflow sigue sin verificarse.** Universe bloquea toda consulta sin navegador (Cloudflare) y la API pide una
  API key, que exige crear cuenta. Hay al menos 5 proyectos con clases de BCS (uno con **BCS-1 a BCS-5 en
  pasos de 0,25**), pero no sé cuántas imágenes tienen por clase (§4).
- **Para ≥ 4.5 no apareció nada nuevo**: `zen_rgbd` sigue siendo la única fuente pública y alcanza (112 animales).

## 1. Tabla de hallazgos nuevos y correcciones

Mismas columnas que T23. "a pedido" = hay que escribir a los autores.

| Dataset | Autores / año | Licencia | Tipo | Vista | Raza | Escala / paso | **BCS ≤ 3.0** | **BCS ≥ 4.5** | Total | Cómo se baja | Compat. con nuestro modelo |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **Corrección** · [`sdb_bcs`](https://www.scidb.cn/detail?dataSetId=16b8bdaf31ee4c8b9891fc7e9df6e41c) V1/V2/V3 (DOI 10.57760/sciencedb.16704) | Huang Xiaoping et al. (Anhui), 2024–2025 | **CC BY 4.0** ✔ (metadatos CC0, acceso `PUBLIC`) | RGB | Desde arriba, zona de la cola; Hikvision a 2,4 m en el pasillo de ordeñe | No informada | 1–5 · 0.25; solo 3.25–4.25 | **0** ✔ | **0** ✔ | 53.566 img: 3.25 = 7.536, 3.5 = 13.256, 3.75 = 14.255, 4.0 = 12.556, 4.25 = 5.963 (según BCS-YOLO, PMC11672832) | Directo; **las 3 versiones son el mismo archivo** (4.407.548.599 bytes) ✔ | Ya es la base. El paper de BCS-YOLO dice que **los autores excluyeron a propósito BCS 1–2 y 4.5–5**: no tiene sentido pedir "la versión completa" |
| **Shriramulu, Pushpadass et al.** — [Indian J. Dairy Sci. 77(3):243–251](https://doi.org/10.33785/IJDS.2024.v77i03.008) | ICAR-NDRI (Bengaluru, India), 2024 | © Indian Dairy Association; **sin data statement** | RGB (Nikon D5100, 4928×3264) | **Dorsal** (ilíacos, isquiones, base de la cola) y lateral | Cruza Holando (5 tambos) | 1–5 · 0.25 (también un modelo de 0.5); 3 veterinarios sobre fotos impresas (Gillund 2001) | **Mayoría.** En el test (~127 img): 1.5: 4 · 1.75: 2 · 2.0: 12 · 2.25: 11 · 2.5: 29 · 2.75: 27 · 3.0: 22. Extrapolado a las 503: **~400 img ≤ 3.0, ~270 < 2.75** (no verificado: solo hay matriz de confusión del test) | **0** (máximo 4.0) | 503 img; animales no reportados | **A pedido**: heartwin1@gmail.com, P.Dhass@icar.gov.in | **Alta para flacas.** Vista dorsal con las mismas estructuras que la nuestra y fotos de cámara de mano (más parecidas a celular que la cámara fija). Riesgo: puntaje sobre foto, no palpado; sin ID de animal |
| [`iananusu/cattle-body-condition-score`](https://www.kaggle.com/datasets/iananusu/cattle-body-condition-score) (Kaggle) | Re-subida de Winkler & Boucheron 2023, 2025 | "Unknown" en Kaggle (el original es CC0) | Depth/DGE | Cenital | Criollo (carne) | 1–9 | = `criollo_dge` (carpetas 2–7: 546 / 536 / 2.212 / 1.962 / 667 / 17) ✔ | — | 5.940 archivos ✔ | API Kaggle sin login ✔ | **Duplicado** de `criollo_dge`; no usar (y no mezclar en CV: serían las mismas vacas) |
| [`leg1rm/cow-pin-classification`](https://www.kaggle.com/datasets/leg1rm/cow-pin-classification) (Kaggle) | "Le G1RM", 2025 | Apache 2.0 | RGB, recortes 128×128 | Isquiones (pin bones) desde atrás | Holando (fotos de Jembere, abajo) | **No es BCS**: 3 etiquetas cualitativas `Sharp` / `Padded` / `Smooth`, sin documentar quién puntuó | `Sharp` (≈ isquión marcado, BCS bajo) = 95 ✔ | `Smooth` = 86 ✔ (no equivale a ≥ 4.5) | 241 recortes + 2 sin etiqueta ✔ | API Kaggle sin login ✔ | **Muy baja.** Miré 4 recortes: son parches borrosos de la grupa, sin escala ni puntuador conocido |
| [`leg1rm/cow-bcs-keypoint-coco`](https://www.kaggle.com/datasets/leg1rm/cow-bcs-keypoint-coco) (Kaggle; export de Roboflow `cattle-erlpe/my-third-project-kbtya`) | "Le G1RM", 2025 | CC BY-NC-SA 4.0 (Kaggle) / CC BY 4.0 (Roboflow) | RGB 1920×1080 (frames de video) | Lateral | — | **Sin BCS**: 25 keypoints (ilíacos, isquiones, base de la cola…) | — | — | 2.123 img, 3.332 vacas anotadas ✔ (sobre el COCO JSON) | API Kaggle sin login ✔ | **Nula como dato de BCS.** Podría servir para un detector de puntos anatómicos |
| [Jembere & Chimonyo 2024](https://www.kaggle.com/datasets/lawrencejembere/cow-images-for-milk-yield-prediction) (Kaggle; paper [SAJAS 54(1):47–58](https://www.sasas.co.za/?p=22137)) | UKZN (Sudáfrica), 2024 | **CC BY-SA 4.0** ✔ | RGB | **Trasera** + lateral (pares) | Holando, 1.ª–2.ª lactancia | **Sin BCS** (la etiqueta es producción a 305 días) | — | — | 1.238 pares, 743 vacas; 14,6 GB ✔ | API Kaggle sin login ✔ | **Pool sin etiquetar** de fotos traseras de Holando. Solo serviría si un técnico las puntúa sobre la foto (menos confiable que palpar) |
| Espasandín, Larracharte, Pérez — [Agrociencia Uruguay 27:e1165](https://doi.org/10.31285/AGRO.27.1165) | Udelar, 2023 | Paper CC BY 4.0; datos **no publicados** ("Download data is not yet available") | RGB | **Trasera-oblicua** (cámara a 175 cm de alto, 125 cm de distancia, 54°) | Hereford, Braford, Angus y cruzas (carne) | **1–8** (Vizcarra), banco de 2 a 6 · 0.25; promedio de 3 observadores | Tiene flacas en 1–8; sin conteo por clase | — | ~1.000 vacas, 1 foto c/u (2013–2019) | **A pedido**: acespasa@fagro.edu.uy | **Media-baja**: otra escala y raza (no se mezcla, SPEC §1.2). Pero es uruguayo, con la misma geometría de foto, y puede servir de **test ordinal de flacas** o como contacto para el relevamiento propio |
| e-Score — [Matos et al. 2017](https://www.alice.cnptia.embrapa.br/alice/bitstream/doc/1079170/1/Validacao-do-software-e-Score-de-analise-de-escore-corporal.pdf) y [Cardoso et al. 2016](https://www.alice.cnptia.embrapa.br/alice/bitstream/doc/1176053/1/Avaliacao-de-fatores-que-interferem-na-avaliacao-computadorizada.pdf) | Embrapa Gado de Leite (Juiz de Fora), 2016–2017 | Sin data statement | RGB | **Trasera (grupa)** + superior (lomo), en el brete | Holando | 1–5 · 0.25 (Wildman) | No informado | No informado | 482 fotos de 116 vacas (2017); 315 vacas en 3 sesiones (2016) | **A pedido**: bruno.carvalho@embrapa.br | **Media**: vista trasera de Holando en escala 1–5, pero no sabemos si hay extremos. Pedir primero la distribución |
| Benicio et al. 2023 — [resumen](https://www.alice.cnptia.embrapa.br/alice/bitstream/doc/1161944/1/2023-cpamt-lbl-automated-body-score-assessment-dairy-cows-using-depth-image-processing-p25.pdf) | Embrapa Agrossilvipastoril + U. Illinois | Sin data statement | Depth (RealSense) | Cenital | Lechera | 1–5; rango **2–4.5** | Sí, sin conteo | Sí, sin conteo | 40 vacas | Pedir a luciano.lopes@embrapa.br | **Baja** (depth, 40 vacas) |
| Li et al. 2025 — [AI in Agriculture 15(2):350–362](https://doi.org/10.1016/j.aiia.2025.01.010) | NWAFU (China) | Paper CC BY 4.0; statement **no leído** (403) | RGB (video) | Trasera, entrada de la sala, vacas en grupo | Holando | 1–5; **3.25–4.25** (no verificado, snippet) | 0 (probable) | 0 (probable) | No verificado | Song Huaibo, songyangfeifei@163.com (no verificado) | **Nula para extremos**: usa la misma escala recortada que `sdb_bcs` |
| Halachmi et al. 2008 ([JDS](https://doi.org/10.3168/jds.2007-0785)) y 2013 ([CEA 99:35](https://doi.org/10.1016/j.compag.2013.08.012)) | ARO Volcani (Israel) | Sin data statement (anteriores a la exigencia) | **Térmica** | Superior | Holando | 1–5; 2008: **BCS medio 2.15–2.23** (rodeo flaco) | Mayoría (2008), sin conteo | No informado | 186 vacas (2008) | Pedir a Ilan Halachmi, halachmi@volcani.agri.gov.il (no verificado) | **Baja**: térmica, datos de hace más de 12 años |
| Zhao Kaixuan et al. — [农业机械学报 2021](https://www.aeeisp.com/nyjxxb/article/id/f472376d-a0bc-4b88-b86a-09d1ba8dd226) | HAUST (China) | Sin statement en el resumen | Depth / nube de puntos | Dorsal | Lechera | 1–5; **2.25–4.0** | Sí, sin conteo | 0 | 5.119 img, 77 vacas | A pedido; contacto no encontrado | **Baja** (depth) |
| Yukun et al. — [JDS 2019](https://doi.org/10.3168/jds.2018-16164) | NEAU (China, grupo de Shen Weizheng) | Sin statement | Depth + gris | Dorsal | Holando | Etiqueta derivada de grasa dorsal por ecografía; "ventaja en vacas flacas" | Sí, sin conteo | No informado | 686 vacas | A pedido; contacto no verificado | **Baja** (depth). Interesante por la etiqueta objetiva (ecografía) |
| Yao et al. — [JDS 2026](https://doi.org/10.3168/jds.2025-27759) | Univ. de Jilin (China) | Statement no leído (403); archivaron código y pesos | RGB | **Lateral** | Holando | 1–5; "los casos extremos son raros" | Pocos | Pocos | 3.208 img, 211 vacas | jiluleilin@126.com | **Baja** (vista lateral, sin extremos) |
| Zin et al. — [Sensors 20(13):3705](https://doi.org/10.3390/s20133705) | Univ. de Miyazaki (Japón), 2020 | CC BY; sin statement | Depth | Cenital, sala rotativa | Holando | 1–5 · 0.25; por vaca: 2.5: 1 · 2.75: 1 · 3.0: 6 · … · 4.0: 1 | 8 vacas | 0 | ~140k frames, 52 vacas | thithi@cc.miyazaki-u.ac.jp | **Baja** (depth, casi sin extremos) |

Revisados y **descartados** (sin BCS o sin extremos, detalle en §5): MmCows (Kaggle, 16 vacas, sin BCS en los
registros de salud ✔), CID (Kaggle `skhalili/cattel`, peso), `twisdu/dairy-cow` (detección de postura),
sciencedb.agriculture.00224 (estéreo, sin BCS), sciencedb.09197 (sangre, sin imágenes), O'Leary et al. 2020
(Teagasc, 3D), Lee et al. 2024 (Corea, nube de puntos), Oblitas et al. 2023 (Perú, sin datos accesibles).

## 2. Fuentes a pedido: a quién escribir y qué pedir

Los mails los manda el humano (los agentes no escriben a terceros). Pedido común para todos: (a) imágenes con
BCS por imagen, (b) **ID de animal o de sesión** (para el `group_id`), (c) quién puntuó y cómo, (d) permiso
escrito para **entrenar y servir un modelo** sin redistribuir las imágenes.

| Prioridad | Fuente | Contacto | Qué pedir en concreto |
|---|---|---|---|
| 1 | Nagy et al. 2023 (T23) | solymosi.norbert@gmail.com | Ya en T23: los 2.275 frames ≤ 3.0 y los 186 ≥ 4.5, con ID de video |
| 2 | **Shriramulu, Pushpadass et al. 2024** | heartwin1@gmail.com · P.Dhass@icar.gov.in | Las 503 fotos dorsales con el puntaje de los 3 veterinarios (no solo el consenso), número de vaca y tambo |
| 3 | Dandıl et al. 2024 (T23) | emre.dandil@bilecik.edu.tr | Ya en T23 |
| 4 | e-Score, Embrapa | bruno.carvalho@embrapa.br | Primero, la **distribución de BCS** de las 482 fotos de 2017 (vista trasera). Si tiene ≤ 2.75 o ≥ 4.5, pedir las fotos traseras |
| 5 | Espasandín et al. 2023 (Udelar) | acespasa@fagro.edu.uy | Las ~1.000 fotos traseras-oblicuas con BCS 1–8, solo como **test ordinal de flacas de carne**. También sirve como contacto para el relevamiento en Uruguay (T23 §5) |
| 6 | NEAU (Shen Weizheng) / HAUST (Zhao Kaixuan) | No verificado; buscar en las páginas de los papers | Solo si algún día hay un modelo depth: imágenes de profundidad con BCS < 2.75 |
| — | Li et al. 2025 (NWAFU) | songyangfeifei@163.com (no verificado) | **No vale la pena**: casi seguro tiene el mismo rango 3.25–4.25 |
| — | Halachmi (Volcani) | halachmi@volcani.agri.gov.il (no verificado) | **No vale la pena**: imágenes térmicas de 2008–2013 |

## 3. Ranking actualizado (T23 + T24)

| # | Dataset | Qué cubre | Esfuerzo | Riesgo principal | Cambio vs. T23 |
|---|---|---|---|---|---|
| 1 | **`zen_rgbd`** | ≥ 4.5 (112 animales) y 3.0 (96 animales) | Bajo–medio (~1 día, brief T04) | Varias vacas por frame; la fuente predice la etiqueta | = |
| 2 | **Nagy et al. 2023** | 1.0–2.5 (1.560 img) + 4.5–5 | Medio, después de que contesten | Un solo puntuador, sin ID de animal | = |
| 3 | **Shriramulu, Pushpadass et al. 2024** | 1.5–3.0 (~400 img, estimado) | Medio: mail + adaptador simple (fotos sueltas con puntaje) | Puntaje sobre foto impresa; cruzas; sin ID de animal; puede no contestar | **Nuevo** |
| 4 | **Datos propios en tambo uruguayo** (T23 §5) | Lo que se busque, **con celular** | Alto en logística, bajo en código | Pocas vacas ≥ 4.5 en pastoreo | = (Espasandín puede ser el contacto en Udelar) |
| 5 | **Dandıl et al. 2024** | 2.0, 2.5 y 4.5 en categorías | Medio; solo como test ordinal | Etiquetas gruesas | Baja de #3 a #5: Shriramulu cubre lo mismo con paso 0,25 |
| 6 | **e-Score (Embrapa)** | Desconocido (trasera, Holando, 1–5) | Mail para preguntar la distribución | Puede no tener extremos | **Nuevo** |
| 7 | Rodríguez Alvarez 2019 / Summerfield 2023 / NEAU / HAUST | 1.75–3.0, pero depth | Alto (modelo aparte) | Otra modalidad | Se suman los chinos de depth |

Roboflow no entra en el ranking hasta verificar sus clases (§4). Si se verifica, el proyecto con BCS-1…BCS-5 en
pasos de 0,25 podría entrar entre el #2 y el #3, porque se bajaría sin pedirle nada a nadie.

## 4. Roboflow Universe

**Todo bloqueado sin cuenta.** `universe.roboflow.com` devuelve 403 (página de desafío de Cloudflare) a `curl` y
al fetch web ✔. `api.roboflow.com` devuelve 401 "This method requires your API key" ✔. Una API key gratuita
exige crear cuenta, cosa que este brief no permite. Lo que sigue sale **solo de snippets de buscador (no
verificado)**:

| Proyecto | Lo que dice el snippet | Clases | Imágenes | Licencia |
|---|---|---|---|---|
| "cattle collection" v2 (workspace no identificado; aparece en la búsqueda `class:deoni`) | Clases `BCS-1`, `BCS-1-25`, `BCS-1-5`, …, `BCS-4-5`, `BCS-5` → **escala 1–5 con paso 0,25** | 16 niveles de BCS + clases de raza | ¿3,49k? (el número puede ser de otro proyecto de la misma página) | ? |
| "cow_side" (aparece junto al anterior) | Clases BCS similares; vista **lateral** | ? | ? | ? |
| [`bcs/bcs-vmw6b`](https://universe.roboflow.com/bcs/bcs-vmw6b) | Detección | `bcs1`…`bcs5` (enteros) | 257 | CC BY 4.0 |
| [`holstein/cowbcs`](https://universe.roboflow.com/holstein/cowbcs) | "CowBCS", 2024 | ? | ? | ? |
| [`fiaw/bcs-beef-cattle-v3-rsrck`](https://universe.roboflow.com/fiaw/bcs-beef-cattle-v3-rsrck) | Detección, **carne** | `SCORE 1`…`SCORE 5` | ? | CC BY 4.0 |
| [`psp-hn811/bcs-ebufl`](https://universe.roboflow.com/psp-hn811/bcs-ebufl) | "BCs", detección | ? | ? | ? |
| [`bcs-g5nxo/bcs-4wm9w`](https://universe.roboflow.com/bcs-g5nxo/bcs-4wm9w) (`rf_bcs` de T23) | Cenital | "clases 5–9" | 1.311 | CC BY 4.0 |
| [`cattle-erlpe/my-third-project-kbtya`](https://universe.roboflow.com/cattle-erlpe/my-third-project-kbtya) | Keypoints, **sin BCS** ✔ (vía su copia en Kaggle) | — | 2.123 ✔ | CC BY 4.0 ✔ |

**Cómo cerrarlo (10 minutos de humano):** crear una cuenta gratuita en Roboflow, abrir en el navegador el
proyecto "cattle collection" v2 (buscar `class:BCS-2-5` en Universe) y anotar workspace, vista, raza, licencia y
el histograma de clases que muestra la pestaña *Health Check*. Con la API key, un agente puede listar los
proyectos con `GET https://api.roboflow.com/<workspace>/<project>?api_key=…`, que devuelve las clases con sus
conteos, sin bajar imágenes.

## 5. Dónde busqué y qué sigue sin verificarse

| Lugar | Cómo | Resultado |
|---|---|---|
| **ScienceDB** (scidb.cn) | API `POST /api/sdb-query-service/query` con 体况评分, 体况, 奶牛, 肉牛, 牛, 荷斯坦, 体型, 背部, 牛 图像, 奶牛 图像, 奶牛 深度, BCS, "body condition (score)", "dairy cow image", "cow depth", Holstein, "cattle image"; API `versionList` del 16704 ✔ | Un solo dataset de BCS (16704). CC BY 4.0 ✔, V1–V3 idénticas ✔, sin V4. El resto son datasets de bovinos sin BCS (09197, agriculture.00224, agriculture.00184, 08745) |
| agridata.cn, AI Studio (Baidu), OpenDataLab, Gitee, Tianchi, HeyWhale | `curl`, endpoints de búsqueda y `site:` en buscador | **No verificado**: agridata.cn da 403 ✔; AI Studio y OpenDataLab renderizan con JS (302/404 en los endpoints); la API de Gitee devuelve `[]` hasta para "cow" (pide autenticación); los `site:` no devolvieron páginas. En los papers chinos revisados no hay links a Baidu Netdisk |
| Papers chinos que usan `sdb_bcs` | Texto completo en Europe PMC (PMC11672832, PMC13508922, PMC12474389, PMC6680808) | Todos con 3.25–4.25; ninguno agrega extremos |
| **Kaggle, datasets** | API pública `GET /api/v1/datasets/list?search=` **sin credenciales** ✔ con 22 términos ("body condition score", "BCS cow", "dairy cow condition", "cattle body condition", "cow back image", "cow rear", "thin cow", "condicion corporal", …); listado de archivos y descarga de metadatos con `/datasets/list/<ref>` y `/datasets/download/<ref>/<archivo>` ✔ | Los 5 de la tabla de §1. Nada con BCS lechero 1–5 |
| **Kaggle, notebooks** | API `/api/v1/kernels/list` → **401 sin credenciales** ✔; búsqueda web | **No verificado**: la búsqueda web no indexa notebooks de BCS. Con un `kaggle.json` (lo crea el humano) se puede repetir con `kaggle kernels list -s "body condition"` |
| **Roboflow** | Ver §4 | **No verificado** (Cloudflare + API key) |
| Embrapa Redape (Dataverse) | API de búsqueda: "condição corporal", "escore corporal", "body condition" | Solo datos tabulares de carne y ovinos; **ninguna imagen** |
| Papers por país | Argentina, Brasil, Uruguay, Chile, México, Perú, España, Italia, Irlanda, Israel, China, Corea, Japón, India, Argelia | Nuevos en la tabla de §1. Sin nada nuevo en Chile, México ni España. El "dataset italiano" es el de Ruchay (`zen_rgbd`, coautor Pezzuolo de Padua), no uno aparte |

**Lo que sigue sin poder verificarse y por qué:**
- **Roboflow (conteos por clase):** exige cuenta (§4).
- **Notebooks de Kaggle:** la API de kernels exige credenciales.
- **Statements de Li et al. 2025, Halachmi 2013, Yao 2026 y O'Leary 2020:** el texto completo dio 403, 429 o
  captcha en ScienceDirect, JDS, MDPI y ResearchGate. Los rangos de BCS de Li 2025 salen de un snippet.
- **Distribución completa de Shriramulu 2024:** el paper solo da la matriz de confusión del test; el total ≤ 3.0
  es una extrapolación.
- **Repositorios chinos fuera de ScienceDB:** necesitan navegador con JS o cuenta.
- **Emails no verificados:** Song (NWAFU), Halachmi y los grupos de NEAU y HAUST.

## Fuentes (consultadas el 23/10/2026)

- `sdb_bcs`: https://www.scidb.cn/detail?dataSetId=16b8bdaf31ee4c8b9891fc7e9df6e41c · API https://www.scidb.cn/api/sdb-dataset-version-service/versionList?dataSetId=16b8bdaf31ee4c8b9891fc7e9df6e41c (copia en `~/bcs-shared/artifacts/T24/scidb_16704_versions.json`) · data paper https://doi.org/10.11922/11-6035.csd.2024.0126.zh
- BCS-YOLO (conteos de `sdb_bcs`): https://pmc.ncbi.nlm.nih.gov/articles/PMC11672832/ · otros que usan `sdb_bcs`: https://pmc.ncbi.nlm.nih.gov/articles/PMC13508922/ · https://pmc.ncbi.nlm.nih.gov/articles/PMC12474389/
- Shriramulu, Pushpadass et al. 2024: https://doi.org/10.33785/IJDS.2024.v77i03.008
- Kaggle: https://www.kaggle.com/datasets/iananusu/cattle-body-condition-score · https://www.kaggle.com/datasets/leg1rm/cow-pin-classification · https://www.kaggle.com/datasets/leg1rm/cow-bcs-keypoint-coco · https://www.kaggle.com/datasets/leg1rm/cow-rear-shot · https://www.kaggle.com/datasets/lawrencejembere/cow-images-for-milk-yield-prediction · https://www.kaggle.com/datasets/hienvuvg/mmcows (listados y metadatos en `~/bcs-shared/artifacts/T24/kaggle_*`)
- Jembere & Chimonyo 2024: https://www.sasas.co.za/?p=22137
- Espasandín et al. 2023: https://doi.org/10.31285/AGRO.27.1165
- Embrapa e-Score: https://www.alice.cnptia.embrapa.br/alice/bitstream/doc/1079170/1/Validacao-do-software-e-Score-de-analise-de-escore-corporal.pdf · https://www.alice.cnptia.embrapa.br/alice/bitstream/doc/1176053/1/Avaliacao-de-fatores-que-interferem-na-avaliacao-computadorizada.pdf · Benicio et al. 2023: https://www.alice.cnptia.embrapa.br/alice/bitstream/doc/1161944/1/2023-cpamt-lbl-automated-body-score-assessment-dairy-cows-using-depth-image-processing-p25.pdf
- Li et al. 2025: https://doi.org/10.1016/j.aiia.2025.01.010
- Halachmi et al. 2008: https://doi.org/10.3168/jds.2007-0785 · https://cris.bgu.ac.il/en/publications/thermally-sensed-automatic-cow-body-condition-scoring-3/ · 2013: https://doi.org/10.1016/j.compag.2013.08.012
- Zhao Kaixuan et al. 2021: https://www.aeeisp.com/nyjxxb/article/id/f472376d-a0bc-4b88-b86a-09d1ba8dd226
- Yukun et al. 2019: https://doi.org/10.3168/jds.2018-16164 · Yao et al. 2026: https://doi.org/10.3168/jds.2025-27759
- Zin et al. 2020: https://doi.org/10.3390/s20133705 · O'Leary et al. 2020: https://doi.org/10.3390/agriculture10060246 · Lee et al. 2024: https://doi.org/10.20944/preprints202401.1156.v1 · Oblitas et al. 2023: https://doi.org/10.1007/978-3-031-31007-2_42
- Ruchay et al. 2026 (paper de `zen_rgbd`): https://doi.org/10.1016/j.compag.2026.112252
- Roboflow: https://universe.roboflow.com/bcs/bcs-vmw6b · https://universe.roboflow.com/holstein/cowbcs · https://universe.roboflow.com/fiaw/bcs-beef-cattle-v3-rsrck · https://universe.roboflow.com/psp-hn811/bcs-ebufl · https://universe.roboflow.com/search?q=class:deoni
