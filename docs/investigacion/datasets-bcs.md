# Datasets de BCS que cubran los extremos (≤ 3.0 y ≥ 4.5) · T23

Fecha de consulta de todas las fuentes: **22/10/2026**. Conteos propios marcados con ✔ (hechos sobre metadatos
chicos guardados en `~/bcs-shared/artifacts/T23/`, sin bajar imágenes). Lo que sale solo del paper va sin ✔,
y lo que no pude verificar está marcado **(no verificado)**.

## Resumen

- **El hueco de vacas gordas ya está cubierto por una fuente pública que tenemos relevada y no usamos:**
  `zen_rgbd` (Zenodo, CC BY 4.0) trae **2.280 frames / 112 animales con BCS ≥ 4.5** y **2.140 frames / 100 animales
  con BCS ≤ 3.0** ✔. El modelo de producción no lo usa porque T04 se canceló en G0. Es la mejora más barata.
- **Debajo de 2.75 no hay ningún dataset público en escala lechera 1–5.** Revisé Zenodo, Dryad, figshare, Mendeley,
  Hugging Face, GitHub, ScienceDB, Kaggle y Roboflow (detalle en §6). Todo lo que tiene vacas flacas
  (BCS 1–2.5) está **a pedido de los autores**.
- El mejor candidato para las flacas es **Nagy et al. 2023 (Hungría)**: RGB de la grupa en sala rotativa
  (la misma geometría que `zen_rgbd`), escala 1–5 y **1.560 imágenes con BCS 1.0–2.5** (el split de test
  incluido). Hay que pedirlo por mail.
- La opción que no depende de terceros es **recolectar en un tambo uruguayo**: unas 150–200 vacas puntuadas
  por un técnico, buscando a propósito las flacas (§5).

## 1. Tabla comparativa

Abreviaturas: img = imágenes o frames; anim = animales distintos; "a pedido" = la *data availability
statement* dice que se entrega a pedido.

| Dataset | Autores / año | Licencia | Tipo | Vista | Raza | Escala / paso | **BCS ≤ 3.0** | **BCS ≥ 4.5** | Total | Descarga | Cómo se baja | Compat. con nuestro modelo |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| [`sdb_bcs`](https://www.scidb.cn/en/detail?dataSetId=16b8bdaf31ee4c8b9891fc7e9df6e41c) (DOI 10.57760/sciencedb.16704) V3 | Huang et al., 2025 (*China Sci. Data*) | Pública; licencia a verificar en la página (SPEC §1.2) | RGB | Trasera-dorsal, cámara fija | Holando | 1–5 · 0.25 | **0** ✔ | **0** ✔ | 45.523 img con etiqueta en `labels_sdb_bcs.csv` ✔ (53.566 según el paper) | 4,4 GB | Directo | Es la base actual (3.25–4.25) |
| [`zen_rgbd`](https://zenodo.org/records/20290988) (DOI 10.5281/zenodo.20290988) | Ruchay, Kolpakov, Riazanov, 2026 | **CC BY 4.0** | RGB + depth | Trasera-dorsal oblicua, sala rotativa, **varias vacas por frame** | Lechera (Rusia); raza no verificada | 1–5 (Ferguson) · 0.25; rango 2.75–5.0 | **2.140 img / 100 anim** ✔ (2.75: 80 / 4; 3.0: 2.060 / 96) | **2.280 img / 112 anim** ✔ (4.5: 1.080 / 54; 4.75: 360 / 18; 5.0: 840 / 42) | 25.700 img, 1.025 anim ✔ | 83,5 GB en 5 zips (se baja por frame con `remotezip`) | Directo (API Zenodo) | **Alta** para ≥ 4.5 y 3.0. Misma vista y escala; hay que recortar la vaca central |
| Nagy et al. — [Animals 13(2):194](https://pmc.ncbi.nlm.nih.gov/articles/PMC9855016/) (DOI 10.3390/ani13020194); pesos en [figshare 21372000](https://doi.org/10.6084/m9.figshare.21372000.v1) | Nagy, Kilim, Csabai, Gábor, Solymosi, 2023 | Paper CC BY 4.0; **datos sin licencia, a pedido**; pesos CC BY 4.0 | RGB (frames de video) | **Grupa desde atrás**, cámara en sala rotativa | Lechera (3 tambos de Hungría); raza no declarada | 1–5, 12 niveles: 1, 1.5, 2, 2.5, 2.75, 3, 3.25, 3.5, 3.75, 4, 4.5, 5 | **2.275 img** (1.0: 225; 1.5: 300; 2.0: 507; 2.5: 528; 2.75: 270; 3.0: 445) · anim **no reportado** | **186 img** (4.5: 64; 5.0: 122) | 4.061 img; tambos de 1.150, 880 y 960 vacas | No informado | **A pedido**: solymosi.norbert@gmail.com | **Alta** en vista y escala. Riesgos: un solo puntuador, cajas a mano y sin `animal_id` publicado (hay que pedirlo para el `group_id`) |
| Dandıl, Çevik, Boğa — [Vet. Sci. 11(9):399](https://pmc.ncbi.nlm.nih.gov/articles/PMC11435893/) | 2024 | Paper CC BY; datos **a pedido** | RGB | **Trasera, región pélvica** | Holando (925) + Simmental (345) | 5 categorías → BCS 2.0 / 2.5 / 3.5 / 4.0 / 4.5 | **330 img** (2.0: 80; 2.5: 250) · anim no reportado (vacas refotografiadas cada mes) | **92 img** (4.5) | 1.270 img, 20 tambos de Niğde (Turquía) | No informado | **A pedido**: emre.dandil@bilecik.edu.tr | **Media**. Buena vista, pero etiquetas gruesas (no hay 3.0, 3.25 ni 3.75) y sin ID de animal |
| Rodríguez Alvarez et al. — [Agronomy 9(2):90](https://doi.org/10.3390/agronomy9020090) (y CEA 155:12–22, 2018) | UNICEN Tandil, 2019 | Paper CC BY; datos **sin statement** (no verificado) | **Depth** (Kinect v2, 512×424) | Cenital, salida de sala | Holando (Argentina) | 1–5 · 0.25; rango 1.75–4.5 | **1.156 img** (1.75: 6; 2.0: 50; 2.25: 123; 2.5: 387; 2.75: 294; 3.0: 296) | **3 img** | 1.661 img; vacas no reportadas | No informado | Pedir a cristian.mateos@isistan.unicen.edu.ar | **Baja** (depth). Útil solo si se arma un modelo depth (§3) |
| Summerfield et al. — [Sensors 23(22):9051](https://pmc.ncbi.nlm.nih.gov/articles/PMC10675635/) | Univ. de Pretoria, 2023 | **"No públicos por privacidad"**; a pedido | **Depth** (3 Kinect v2) | Cenital, **trasera** y oblicua | Holando (Sudáfrica) | 1–5 · 0.25; rango 1.75–3.5 | Mayoría; 6.280 frames crudos en 462 vacas (por clase solo en la Fig. 3, no leída) | **0** | 6.280 frames, 462 vacas | No informado | **A pedido**: gary.summerfield@up.ac.za | **Baja** (depth); nada ≥ 4.5 |
| Tang et al. — [arXiv 2601.22522](https://arxiv.org/abs/2601.22522) | Univ. de Florida, 2026 | Paper CC BY 4.0; datos **sin statement** | **Depth + nube de puntos** (RealSense D455) + RGB | Cenital, brete de salida | Lechera (Georgia, EE. UU.); raza no declarada | 1–5 · 0.25; rango 2.0–4.25; dos puntuadores con acuerdo exacto | **Conteo no verificado** (por clase solo en la Fig. S2). Se sabe que 3.0–3.5 son 601 de 1.020 obs y que 2.0–2.25 y 4.0–4.25 suman 14,8 % | **0** | 1.020 obs / 10.228 archivos depth | No informado | Pedir a haipengyu@ufl.edu | **Baja–media**. La vista cenital no es la nuestra; el RGB existe pero no sé si lo entregan |
| [`iplab`](https://iplab.dmi.unict.it/BCS/index.html) (Corfilac / IPLAB / Penn Vet) | Azzaro et al., 2011 | Uso académico; **sin redistribución ni uso comercial** | RGB | Dorsal desde arriba, salida de robot | Lechera (Italia) | 1–5 · 0.25; 2 técnicos | **48 img** técnico 1 / **47** técnico 2 ✔ (2.5–3.0) | **18 img** ✔ (4.5) | 186 etiquetas ✔ (la página dice 207 img), 29 vacas | ~MB | Directo | **Solo test externo** (es el holdout, SPEC §1.3); nada < 2.5 |
| [`criollo_dge`](https://zenodo.org/records/8108223) (Zenodo 8108223 / Dryad 10.5061/dryad.tqjq2bw4s) | Winkler & Boucheron, 2023 | **CC0** | Depth/gris/bordes (DGE) | Cenital | Criollo (carne) | **1–9** · 1 | Grados 2–3 (1–9): 1.082 frames / 23 vacas ✔ | Grados 6–7: 684 frames / 25 vacas ✔ | 5.940 frames / 148 vacas ✔ | 178 MB (zip DGE); 34,5 GB completo | Directo | **Baja**: otra escala, otra raza, depth |
| Lewis et al. — [AgriEngineering 7(7):241](https://doi.org/10.3390/agriengineering7070241) | NMBU, 2025 | Datos **a pedido** | RGB | Frontal, **trasera** y cenital | Noruega Roja | 1–5 (solo 1.5–3.5) | 1.5 + 2.0: **< 30 img**; 2.5: 521 img (56 vacas en total) | **0** | 1.764 img, 56 vacas | No informado | A pedido (sin contacto específico) | **Baja**: casi sin flacas extremas |
| Siachos et al. — [JDS 107(4):2499](https://livrepository.liverpool.ac.uk/3176832/) | Liverpool + CattleEye, 2024 | Sistema comercial (CattleEye); **datos no públicos** (no verificado) | RGB 2D (video) | **Cenital** a 4 m | Holando (7 tambos de UK) | 1–5 · 0.25 | Incluye una visita dirigida a vacas ≤ 2.5 y ≥ 4.0; conteo no publicado | Ídem | 34.150 puntajes | — | Probablemente no se consigue | **Baja**: vista cenital y dato propietario |
| Zhao et al. — [IJABE 13(4):45](https://ijabe.org/index.php/ijabe/article/view/5655) | Univ. de Kentucky, 2020 | Sin statement | Depth | Cenital | Lechera (EE. UU.) | 2.25–3.75 (> 3.75 colapsado a 3.75) | 2.25–3.0: parte de 4.820 img | **0** (< 1,5 % del rodeo) | 4.820 img | — | Pedir a los autores | **Baja** |
| Bazan et al. — [55 JAIIO 2026](https://55jaiio.sadio.org.ar/wp-content/uploads/2026/07/91.pdf) | UNS + INTA Bordenave, 2026 | No publicado | RGB | No informada | Cría (carne, Argentina) | Intervalos de 0.2, después reagrupados | No informado | No informado | 405 img | — | Contacto INTA (deleo.geronimo@inta.gob.ar) | **Baja**: carne, escala sin documentar |
| [`rf_bcs`](https://universe.roboflow.com/bcs-g5nxo/bcs-4wm9w) (SPEC §1.2) | Roboflow, s/f | CC BY 4.0 | RGB | Cenital | No informada | "Clases 5–9", escala sin documentar | Desconocido | Desconocido | 1.311 img | — | API key | **Baja**: escala desconocida |

**Totales útiles hoy** (públicos y en escala 1–5, sin contar el holdout):
- ≥ 4.5: 2.280 frames / 112 animales (`zen_rgbd`).
- 3.0: 2.060 frames / 96 animales (`zen_rgbd`).
- 2.75: 80 frames / 4 animales (`zen_rgbd`; casi nada).
- **< 2.75: cero.**

## 2. Ranking para cubrir 1–3 y 4.5–5

| # | Dataset | Qué cubre | Esfuerzo de integración | Riesgo principal |
|---|---|---|---|---|
| 1 | **`zen_rgbd`** | ≥ 4.5 (112 anim) y 3.0 (96 anim). Corre el piso validado de 3.25 a 3.0 y el techo de 4.25 a 5.0 | **Bajo–medio, ~1 día.** El brief T04 ya está escrito: `fetch_zen_rgbd.py` con `remotezip` (3 frames por animal y fecha: ~3.800 frames, ~11 GB y ~30 min con 8 hilos en todo el rango; solo las colas ≤ 3.0 y ≥ 4.5 son ~660 frames, ~2 GB) + recorte de la vaca central con YOLO + revisar una hoja de contactos. Sin conversión de escala. Reentrenar y evaluar con leave-one-source-out | Varias vacas por frame (trampa 2 de §1.2). La fuente predice la etiqueta: todos los ≥ 4.5 vienen de `zen_rgbd` |
| 2 | **Nagy et al. 2023** | **1.0–2.5 (1.560 img)** + 2.75–3.0 + 4.5–5.0. El único que llega a BCS 1 | **Medio, ~1–2 días después de que contesten.** Mail a los autores (puede tardar o no llegar). Adaptador nuevo: frames + cajas de grupa → recorte. Los niveles ya caen en la grilla 0.25 (faltan 1.25, 1.75, 2.25 y 4.25). Pedir ID de vaca o de video para el `group_id`; si no hay, agrupar por video | Un solo puntuador. Sin ID de animal, el split puede filtrar la misma vaca entre folds. Licencia a negociar (pedir permiso explícito para entrenar y servir el modelo) |
| 3 | **Dandıl et al. 2024** | 2.0 (80), 2.5 (250) y 4.5 (92) | **Medio.** Mail + adaptador. Las etiquetas son categorías mapeadas a 2.0/2.5/3.5/4.0/4.5, así que conviene usarlo como **test ordinal** ("flaca / normal / gorda") y no como regresión fina | Etiquetas gruesas, cruzas Simmental, sin ID de animal |
| 4 | **Datos propios en tambo uruguayo** (§5) | Lo que se busque a propósito: 1.5–2.75 y 4.5+, **con fotos de celular** (nuestro dominio real) | **Alto en logística** (técnico, tambo, 2–3 visitas) pero **bajo en código**: el formato canónico ya existe | Pocas vacas ≥ 4.5 en rodeos pastoriles. Hay que coordinar con el productor |
| 5 | **Rodríguez Alvarez 2019** (o Summerfield 2023) | 1.75–3.0 con más de 1.000 imágenes, pero **depth** | **Alto.** Haría falta un segundo modelo de profundidad (fuera de alcance, SPEC §7). No sirve para el modelo RGB | Modalidad distinta; no hay conversión depth → RGB |

Recomendación: hacer **1** ya, mandar los mails de **2** y **3** hoy (el que pide es el humano; los agentes no
crean cuentas ni escriben a terceros) y arrancar **4** en paralelo. Con eso se mide si las flacas del dominio
celular se detectan. El 5 queda solo si algún día se suma una cámara de profundidad en el tambo.

## 3. Otras escalas y otras modalidades: ¿sirven?

- **1–9 de carne (`criollo_dge`, NRC) y 1–8 de Uruguay (Hereford):** **no se mezclan** con 1–5 (SPEC §1.2,
  trampa 5). No existe una conversión lineal validada entre escalas lecheras y de carne. La 1–5 lechera mira
  isquiones, ilíacos, ligamentos y base de la cola; la 1–9 de carne mira costillas, lomo y paleta. Además, la
  raza cambia dónde se deposita la grasa. Una regla tipo `bcs_1_5 ≈ 1 + (bcs_1_9 − 1) / 2` metería un error
  sistemático desconocido, justo en las colas que queremos medir. Usos posibles: smoke test del pipeline, o
  preentrenar el backbone con un objetivo ordinal propio y después descartar esa cabeza. Hay que reportarlo
  aparte, nunca como métrica 1–5.
- **Categorías (Dandıl: emaciada / pobre / buena / gorda / obesa):** sirven como **evaluación ordinal** de las
  colas ("¿el modelo pone a las *emaciadas* por debajo de 2.75?"). Como etiqueta de regresión, el error de
  ±0,25–0,5 de la categoría es del mismo orden que lo que queremos medir.
- **12 niveles de Nagy (1, 1.5, 2, 2.5, 2.75, …):** es la escala 1–5 con huecos. Cae directo en la grilla
  `dairy_1_5`; las clases vacías (1.25, 1.75, 2.25, 4.25 en esa fuente) no son problema con la pérdida ordinal.
- **Depth (Rodríguez Alvarez, Summerfield, Zhao, Tang, el canal depth de `zen_rgbd`):** nuestro modelo es RGB
  y la app recibe fotos de celular. El depth no se puede usar como entrada. A lo sumo sirve para un modelo
  aparte o para *distillation* (un modelo depth entrenado con datos flacos que etiquete RGB pareados). Esa
  segunda opción solo funciona si el dataset trae RGB y depth del mismo instante, como `zen_rgbd` o Tang.
  Ninguno de los depth con vacas flacas publica RGB pareado (en Tang no está verificado).
- **Vista cenital pura (Siachos, Tang, `iplab`, Lewis):** vista distinta a nuestra trasera-dorsal. Sirve de
  test externo con advertencia, no para entrenar.

## 4. Licencias y restricciones

| Fuente | Redistribución | Uso comercial | Cita obligatoria | Nota |
|---|---|---|---|---|
| `zen_rgbd` | Sí, con atribución (CC BY 4.0) | Sí | Sí (DOI 10.5281/zenodo.20290988) | Sin restricciones para nuestro caso |
| `sdb_bcs` | **No verificado** | **No verificado** | Sí (DOI 10.57760/sciencedb.16704 + paper *China Sci. Data* 2025) | Pendiente de SPEC §1.2: confirmar la licencia en la página de ScienceDB |
| `iplab` | **No** sin permiso | **Prohibido** | Sí (BCS Database + papers listados) | Si la app es comercial, ni siquiera el test externo es claramente válido. Consultar a caccamo@corfilac.it |
| `criollo_dge` | Sí (CC0) | Sí | No obligatoria (recomendable) | — |
| Nagy 2023 (datos) | A negociar | A negociar | Sí | Los **pesos** en figshare son CC BY 4.0 (15 `.pth` Detectron2 de ~330 MB): sirven como baseline o *teacher*, no como datos |
| Dandıl 2024, Lewis 2025, Rodríguez Alvarez 2019, Summerfield 2023, Tang 2026 | A negociar | A negociar | Sí | Pedir por escrito: entrenar, servir el modelo y si se puede redistribuir. Lo normal es "uso académico, sin redistribución" |
| `rf_bcs` | Sí (CC BY 4.0) | Sí | Sí | Escala desconocida |
| Datos propios | Nuestra decisión | Sí | — | Acuerdo escrito con el productor (fotos de sus vacas, caravanas anonimizadas) |

En todos los casos sigue valiendo SPEC §1.1: **ningún dato de terceros entra al repo**. Todo vive en
`$BCS_DATA_DIR` y cada fuente queda registrada en `sources.lock.json` con su licencia.

## 5. Opción complementaria: datos propios en un tambo uruguayo

**Por qué:** es la única forma de tener vacas flacas **con fotos de celular**, que es lo que la app recibe. Ningún
dataset público tiene esa combinación (SPEC §1.2, "Lo que estos datos NO cubren").

**Cuántas vacas.** Es una estimación propia, no sale de una referencia. Criterios:
- Con CV agrupada de k = 5, cada fold necesita al menos 3–4 animales por clase en test.
- SPEC §0.1 colapsa las clases con menos de 15 imágenes.
- Para estimar el MAE de una clase con un IC razonable (±0,1) hacen falta unos 30 animales.

| Bin (escala 1–5) | Mínimo para **evaluar** | Deseable para **entrenar** |
|---|---|---|
| ≤ 2.0 | 15 vacas | 30 |
| 2.25–2.5 | 20 | 40 |
| 2.75–3.0 | 20 | 40 |
| 3.25–4.25 (control, ya cubierto) | 30 | 50 |
| ≥ 4.5 | 10 (son raras en pastoreo) | 20 |
| **Total** | **~95 vacas** | **~180 vacas** |

- 3–5 fotos por vaca en la misma sesión, siguiendo el protocolo de SPEC §0.2 (celular por encima de la cadera,
  30–60°). El `group_id` es la vaca: nunca se separan sus fotos entre folds.
- **Dónde encontrar las flacas:** vacas en el primer tercio de lactancia (pico de balance energético negativo),
  vaquillonas de primera lactancia y el lote de "refugo". Un tambo de 300–500 Holando probablemente tiene
  10–15 % de vacas ≤ 2.5 en otoño-invierno, pero es una estimación **no verificada**. Las gordas (≥ 4.5) suelen
  estar entre las secas de fin de lactancia.
- Conviene repetir la visita a los ~30 días: vuelven a aparecer flacas y gordas, y se mide la estabilidad
  intra-vaca. Las fotos de la visita nueva quedan en el mismo `group_id`.

**Cómo etiquetar:**
1. Escala **1–5 con paso 0,25** (Ferguson et al. 1994 / Edmonson et al. 1989), la misma que `dairy_1_5`.
   No usar la 1–8 de cría.
2. El técnico puntúa **en el momento**, mirando y palpando, antes de ver la foto. Anota la caravana, el BCS y
   la hora. La foto se saca enseguida, con la vaca en el brete o a la salida de la sala.
3. **Segundo puntuador** en al menos 50 vacas (o el mismo técnico una segunda vez, a ciegas). Así medimos el
   acuerdo humano, como hacemos con `iplab` (`human_agreement.json`).
4. Planilla con `animal_id` = `tambo_uy:<hash de la caravana>`, `scorer`, `date`, `bcs`, `scale=dairy_1_5`,
   `view=rear_phone`. Va a `data/manual/` o a `labels_tambo_uy.csv` en `$BCS_DATA_DIR`, nunca al repo.
5. Las 72 fotos traseras de `phone_backview` (SPEC §1.2) se pueden puntuar en la misma jornada: dan un segundo
   set de dominio celular.
6. Reservar desde el principio un subconjunto de vacas para el holdout humano (SPEC §1.3); los agentes no lo ven.

## 6. Dónde busqué y qué no encontré

| Lugar | Búsqueda | Resultado |
|---|---|---|
| Zenodo (API, `type=dataset`) | "body condition score cow/dairy", "BCS cattle images", "condición corporal" | Solo `zen_rgbd` y `criollo_dge`. **Sin versiones nuevas** de ninguna de las dos (1 versión cada una) ✔ |
| Dryad (API) | "body condition score cattle images" | Solo `criollo_dge` (2 versiones, ambas de 2023) ✔ |
| figshare (API) | "body condition score cow" | Espejo de `criollo_dge` y los pesos de Nagy. Nada más con imágenes ✔ |
| Mendeley Data (API de búsqueda) | 4 variantes | Ningún dataset de BCS con imágenes ✔ |
| Hugging Face (API) | "body condition", "bcs", "cattle", "cow", "dairy" | Solo `Jybbs/cattle-bcs-dge` (= `criollo_dge`) y `ganader-ia-developers/cowdb` (sin BCS) ✔ |
| GitHub (`gh search repos`) | "body condition score (cow)" | Repos de práctica sin datos propios publicados. `LittleForest-Ming/BCS` trae 4 videos de test **sin etiquetas** ✔ |
| ScienceDB | Búsqueda web | Solo `sdb_bcs` V3 (sin V4). **No verifiqué** una búsqueda en chino (奶牛体况评分) dentro de scidb.cn |
| Kaggle | Búsqueda web | Nada (coincide con SPEC §1.2). **No verificado** con la API (pide credenciales) |
| Roboflow Universe | Búsqueda web; la búsqueda del sitio da 403 | Solo `rf_bcs` y `reagan-dmuqj` (ya descartado en SPEC). **No verificado** a fondo |
| PapersWithCode | — | No consultado: el sitio ya no está como catálogo independiente. Cubierto con búsquedas de papers |
| Papers 2015–2026 (statements) | Hungría, Turquía, Noruega, UK, Sudáfrica, EE. UU. (Kentucky, Florida), Argentina (UNICEN, UNS/INTA), Brasil (Embrapa e-Score), Israel (térmicas, Halachmi 2008/2013), China (Li et al. 2025, YOLOv5 en grupo) | Todos a pedido o sin statement. **No pude leer** los statements de Li et al. 2025 (*AI in Agriculture* 15(2):350), Halachmi 2013 ni Embrapa e-Score: quedan como contactos posibles |

## Fuentes (consultadas el 22/10/2026)

- `zen_rgbd`: https://zenodo.org/records/20290988 — conteo ✔ sobre `Dataset.xlsx` (copia en `~/bcs-shared/artifacts/T23/`).
- `sdb_bcs`: https://www.scidb.cn/en/detail?dataSetId=16b8bdaf31ee4c8b9891fc7e9df6e41c · paper https://www.sciengine.com/CSD/doi/10.11922/11-6035.csd.2024.0126.zh
- Nagy et al. 2023: https://pmc.ncbi.nlm.nih.gov/articles/PMC9855016/ · pesos https://doi.org/10.6084/m9.figshare.21372000.v1
- Dandıl et al. 2024: https://pmc.ncbi.nlm.nih.gov/articles/PMC11435893/
- Rodríguez Alvarez et al. 2019: https://doi.org/10.3390/agronomy9020090 (Tabla 1)
- Summerfield et al. 2023: https://pmc.ncbi.nlm.nih.gov/articles/PMC10675635/
- Tang et al. 2026: https://arxiv.org/abs/2601.22522
- Lewis et al. 2025: https://doi.org/10.3390/agriengineering7070241
- Siachos et al. 2024: https://livrepository.liverpool.ac.uk/3176832/
- Zhao et al. 2020: https://ijabe.org/index.php/ijabe/article/view/5655
- Bazan et al. 2026: https://55jaiio.sadio.org.ar/wp-content/uploads/2026/07/91.pdf
- `iplab`: https://iplab.dmi.unict.it/BCS/index.html — conteo ✔ sobre los nombres de `dataset.html` (`iplab_bcs.txt`).
- `criollo_dge`: https://zenodo.org/records/8108223 · https://datadryad.org/dataset/doi:10.5061/dryad.tqjq2bw4s — conteo ✔ sobre el índice del zip (`criollo_dge_namelist.txt`).
- Liu et al. 2025 (usa `sdb_bcs`): https://pmc.ncbi.nlm.nih.gov/articles/PMC12474389/
- Escala 1–8 de cría en Uruguay: https://produccion-animal.com.ar/informacion_tecnica/cria_condicion_corporal/63-cangue_espasandin.pdf
