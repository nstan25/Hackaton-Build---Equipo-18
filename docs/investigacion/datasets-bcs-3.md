# Datasets de BCS 1.0–2.75 (vacas flacas) · T26

Fecha de consulta de todas las fuentes: **23/10/2026**. Complementa `datasets-bcs.md` (T23), `datasets-bcs-2.md`
(T24) y `mails-solicitud-datasets.md`, que no se repiten acá. Convenciones de T23/T24: ✔ = verificado por mí
sobre el PDF, la API o metadatos chicos guardados en `~/bcs-shared/artifacts/T26/` (sin imágenes de vacas);
sin ✔ = sale del texto o resumen del paper; **(no verificado)** = snippet de buscador o inferencia mía.

## Resumen

- **Sigue sin aparecer un dataset público de vacas lecheras con BCS < 2.75 y licencia que permita uso comercial.**
  Revisé, entre otros, unos 500 títulos de OpenAlex, 125 de Europe PMC, Zenodo, figshare, Harvard Dataverse, OSF,
  Hugging Face y Wikimedia Commons, y busqué en 9 idiomas. Todo lo que tiene flacas sigue siendo **a pedido**.
- **Nuevo candidato fuerte (a pedido): Grecheneva et al. 2026 (Universidad Timiriazev, Moscú).** Video de
  **983 vacas lecheras** en vista dorsal y lateral, escala 1–5. Según la Fig. 8 ✔, **52 vacas tienen BCS 1 y 214
  BCS 2**, aunque en clases enteras. Es la fuente con más **animales distintos** flacos de todas las relevadas, pero
  la figura dice "ejemplo de formato" y hay que confirmar que los números sean reales (§5).
- **Nuevo candidato chico pero del dominio exacto: Moreira, Dórea et al. 2022 (UW-Madison).** 276 fotos **traseras
  sacadas con iPhone 13 Pro** a 31 Holando, con dos puntuadores. No se sabe cuántas son flacas. Sirve como **test
  externo de dominio celular**, no para entrenar.
- **Red flag:** el paper "AnimalBCS-1200" (2025) dice que tiene 36 Holando con BCS 1 y 52 con BCS 2 y que el dataset
  es CC BY 4.0, pero no da link y su propio *data statement* dice "a pedido". Además dice que fotografiaron vacas
  sobre un fondo blanco de 1 m². **No es confiable** (§1).
- **Dato útil para el relevamiento propio:** en dos tambos pastoriles de la Pampa argentina (Rearte, Zenodo 8082738,
  CC BY 4.0 ✔), la proporción de vacas en el grupo de "mala condición corporal" va de **3 % a 47 % según el grupo de
  parición** (mediana ≈ 23 %) ✔. La definición exacta del umbral está en el paper y no la verifiqué.
- **Datos sintéticos: no reemplazan a las vacas flacas reales.** La literatura no tiene ningún trabajo de BCS con
  generación de imágenes. El riesgo es que la etiqueta del sintético no sea verificable justo en los detalles
  (isquiones, ilíacos, ligamentos) que definen el puntaje (§4.1).

## 1. Hallazgos nuevos

Mismas columnas que T23/T24. "a pedido" = hay que escribir a los autores. Los conteos por clase ≤ 2.75 están en
imágenes (img) y animales (anim) cuando el paper los da.

| Fuente | Autores / año | Licencia | Tipo | Vista | Raza | Escala | **Clases ≤ 2.75 (img / anim)** | Total | Cómo se baja / pide | Contacto | Compat. |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **Grecheneva et al.** — [Izv. TSKhA 2026(2):105–122](https://doi.org/10.26897/0021-342X-2026-2-105-122) ([PDF](https://elib.timacad.ru/dl/full/iz-08-2026-2.pdf/download)) | RGAU-MSHA Timiriazev (Moscú), 2026 | Paper de acceso abierto; **sin *data statement*** ✔ | RGB (video de pasillo, cámara fija) | **Dorsal (cenital) + lateral**, recorte 512×512 | Lechera (no declarada) | 1–5; la Fig. 8 usa **clases enteras** | **BCS 1: 52 anim · BCS 2: 214 anim** (escala del experto, Fig. 8 ✔). Img por clase: no informado | 983 vacas ✔ | **A pedido** | a.grecheneva@rgau-msha.ru ✔ | **Media**. Muchos animales flacos, pero vista cenital (no la trasera de celular) y clases enteras (¿se redondeó?). Pedir el puntaje original con paso 0,25 |
| **Moreira, Bresolin, Dórea, Rosa** — [J. Anim. Sci. 100(S3):150, 2022](https://doi.org/10.1093/jas/skac247.278) (resumen ASAS) | UW-Madison, 2022 | Sin statement (es un resumen) | **RGB + depth de iPhone 13 Pro** | **Trasera**, distancia fija | Holando | 1–5 (no dice el paso); 2 puntuadores | **No informado** | 276 img, **31 vacas** ✔ | A pedido | Grupo de João Dórea (UW Animal & Dairy Sciences); mail **no verificado** | **Alta en dominio, baja en volumen**. Es lo más parecido a nuestra app que existe; como test externo de celular |
| **Çevik** — [IEEE Access, 2020](https://doi.org/10.1109/ACCESS.2020.3040805) (y Çevik & Boğa, [ASYU 2019](https://doi.org/10.1109/ASYU48272.2019.8946405)) | Univ. Akdeniz (Turquía) | Paper OA; no pude leer el texto completo (IEEE devuelve HTML de login) | RGB | No verificada (probablemente trasera, como Dandıl) | Holando (no verificado) | No verificada | **No verificado** | ~180 img (2019) | A pedido | kkcevik@akdeniz.edu.tr (**no verificado**) | **Baja–media**. Coautor de Dandıl 2024 (T23): casi seguro es un subconjunto de ese dataset. Pedirlo **junto** con Dandıl, no aparte |
| **Wu, Guo, Li, Ma, Zhao, Pezzuolo** — [MetroAgriFor 2021](https://doi.org/10.1109/MetroAgriFor52389.2021.9628437) | China Agricultural Univ. + Padua | Sin statement | RGB | No informada | Lechera | 1–5 · 0.25 | **No informado** | 918 img | A pedido | Andrea Pezzuolo (Padua), mail **no verificado** | **Baja** hasta saber el rango. Pezzuolo también es coautor de `zen_rgbd`, así que conviene preguntarle en el mismo mail |
| **Lai, Huang, Yeh et al.** — [Springer 2024](https://doi.org/10.1007/978-981-97-0376-0_17) | National Ilan Univ. + Council of Agriculture (Taiwán) | Sin acceso al texto | No verificado | No verificada | Lechera | No verificada | **No verificado** | No verificado | A pedido | Fay Huang (NIU), mail **no verificado** | **Desconocida** |
| **Montenegro, García-Santillán et al.** — [Springer 2023](https://doi.org/10.1007/978-3-031-32213-6_7) | Univ. Técnica del Norte (Ecuador) | Sin acceso al texto | RGB, **app Android** | No verificada | Ganado (no verificado si es lechero) | No verificada | **No verificado** | No verificado | A pedido | García-Santillán (UTN), mail **no verificado** | **Desconocida**. Interesa porque es un grupo latinoamericano con app de celular |
| **NARO (Japón)** — [nota técnica 2024](https://www.naro.go.jp/english/topics/laboratory/iam/161469.html) · [JFC 2667](https://www.jfc.go.jp/n/finance/keiei/pdf/2667.pdf) | NARO + consorcio, 2024 | Sin datos publicados | RGB 2D, **apto smartphone** | **Trasera** (ángulo del ilíaco) | Holando | 1–5 · 0.25 | No informado | No informado; 2 instituciones | Formulario de [NARO](https://www.naro.go.jp/english/inquiry/index.html) | — | **Media en método, baja como dato**. Error medio 0,19 con fotos traseras. Es la misma geometría que nuestra app |
| **Bloch** — [Zenodo 5645192](https://zenodo.org/records/5645192) | Luke (Finlandia), 2021 | **CC BY 4.0** ✔ | Depth (3D) + **1 foto color por cámara** | Dorsal en balanza | Lechera | **Sin BCS en los CSV** ✔ (solo medidas 3D por vaca y fecha) | — | 107 vacas el 20/12/2019 ✔; 3 visitas; 22 GB ✔ | Directo | — | **Nula** como etiqueta. Pool sin etiquetar |
| **"AnimalBCS-1200"** — [IJAAP 5(2):108, 2025](https://doi.org/10.55529/ijaap.52.108.121) | R. M. Damle (Kalinga Univ., India), 2025 | Dice "CC BY 4.0, libre", pero el *data statement* dice "a pedido" ✔ | RGB | Lateral / dorsolateral | Holando + 4 especies | 1–5 enteros (AHDB) | Holando **BCS 1: 36 img · BCS 2: 52 img** (Tabla 1 ✔) | 260 img de Holando | Sin link ✔ | drranjanamdamle@gmail.com ✔ | **No usar**. Inconsistencias: vacas "sobre fondo blanco de 1 m²", la foto de ejemplo del gato es la imagen `chelsea` de scikit-image, un solo autor y una revista de baja trazabilidad |
| *Iranian J. Biosystems Eng.* — [IJBSE 2025](https://ijbse.ut.ac.ir/article_103941.html?lang=en) (DOI 10.22059/ijbse.2025.398902.665603) | Irán (University of Tehran Press), 2025; autores no verificados | Paper CC BY-NC 4.0 ✔; datos a pedido ✔ | RGB, cámara y celular | Grupa | Lechera | **No es BCS**: 3 tipos de curvatura de la grupa ✔ | — | 150 img ✔ | A pedido | m.d.vaziri@gmail.com ✔ | **Nula** |
| Ariadi et al. — [TASJ 48(5):402, 2025](https://doi.org/10.5398/tasj.2025.48.5.402) | BRIN + Brawijaya (Indonesia) | Paper OA | RGB (CCTV) | Lateral (no verificado) | **Carne** (12 PO, 2 Simmental, 1 Brahman) ✔ | 1–5 de carne | Sin conteo | 225 img de **15 vacas** ✔ | A pedido | f3.utaminingrum@ub.ac.id ✔ | **Nula** (15 animales, carne) |
| Islam et al. — [Smart Agric. Technol. 2026](https://doi.org/10.1016/j.atech.2026.102628) | Univ. de Tennessee | Paper OA | RGB (video de brete) | Cenital + lateral | **Carne** | **1–9**, flacas desde 4 | — | No verificado | A pedido | — | Solo referencia (§1.1) |
| O'Connor — [tesis de doctorado, Massey 2018](https://mro.massey.ac.nz/bitstreams/32114351-cd37-406e-939f-5fd5c76dab89/download) | Massey (Nueva Zelanda) | Tesis de dominio público (según OpenAlex) | Cámara sobre el pasillo (2D o 3D, no verificado) | Cenital | Lechera pastoril | Escala NZ (1–10, no verificado) | No verificado | 710 vacas | Sin datos publicados | — | **Baja** (otra escala y vista), pero los rodeos pastoriles de NZ son los que más se parecen a los uruguayos |

Revisados y **descartados** (sin BCS, sin imágenes o ya cubiertos):
- Grupo de Huang Xiaoping (Anhui), [YOLO v5s](https://www.aeeisp.com/nyjxxb/article/id/56490a8e-f49e-4ecc-b518-e41fa51aa9d2), [YOLOv7-BDN](https://doi.org/10.13031/ja.16317) y [ICIVC 2019](https://doi.org/10.1109/ICIVC47709.2019.8981055): son los autores de `sdb_bcs`, con el mismo rango (3.25–4.25).
- [Zheng et al. 2024](https://doi.org/10.3390/ani14243668) usa "un dataset público" (`sdb_bcs`).
- [Sumathi et al. 2022](https://zenodo.org/records/6613366) usa 206 imágenes, casi seguro las de `iplab`.
- [Li K. et al. 2024](https://doi.org/10.1109/ACCESS.2023.3349320) es depth, Hebei.
- Nagano (Japón) es depth y no tiene datos: [橋本 2024](https://www.pref.nagano.lg.jp/chikusanshiken/event/documents/22_r6_hashimoto.pdf).
- Datasets de Mendeley sin BCS: [Inner Mongolia, 72 vacas](https://data.mendeley.com/datasets/h2s22wr5py) y [cojera, Wagyu/Angus](https://data.mendeley.com/datasets/f4j83j77ng).
- [Sydney, locomoción](https://ses.library.usyd.edu.au/handle/2123/23647) (CC BY, unos 50 videos, sin BCS).
- [Dataverse KKEWS7](https://doi.org/10.7910/DVN/KKEWS7) (Jersey 3D, solo un PDF suplementario ✔).
- [Tedín et al. 2012](https://doi.org/10.3233/978-1-61499-105-2-2150): A Coruña, fotos de mano, sin datos.
- [Matto et al. 2025](https://doi.org/10.3168/jdsc.2025-0851) (Estonia): no tiene imágenes, pero dice algo útil: **el sistema automático viejo de DeLaval sobreestimaba a las vacas flacas**. Es el mismo sesgo que hay que medir en nuestro modelo.
- [Matos/Cardoso, e-Score](https://www.alice.cnptia.embrapa.br/alice/bitstream/doc/1079170/1/Validacao-do-software-e-Score-de-analise-de-escore-corporal.pdf) ya estaba en T24.

### 1.1 Ganado de carne (1–9 / 1–8): ¿sirve como test ordinal?

Sigue valiendo lo de T23 §3: **no se mezcla con 1–5**. Puede servir como **test ordinal de dirección**: si el modelo
lechero, aplicado a una vaca de carne con 1–9 ≤ 3, da un BCS más bajo que a una con 1–9 ≥ 6, al menos ordena bien.
Sirve solo para detectar errores groseros (por ejemplo, que el modelo dé 3.5 a una vaca emaciada), nunca como MAE.
Islam 2026 suma un dato a favor: **las apófisis espinosas se ven solo en animales flacos** (no aparecen con BCS ≥ 6),
y esa es una señal visual que comparten carne y leche.

### 1.2 Guías y atlas con fotos de referencia

| Fuente | Fotos ≤ 2.75 | Vista | Licencia de las fotos | Uso posible |
|---|---|---|---|---|
| [Penn State Extension, "Examples of cows at various BCS"](https://extension.psu.edu/examples-of-cows-at-various-body-condition-scores) (Jones & Heinrichs, actualizada en 2026) | BCS **1.0** (dibujo), **2.0** y **2.75** (foto lateral + trasera) ✔ | Lateral y **trasera** | © Penn State; **sin licencia de reuso** ✔ | Ejemplo cualitativo. Pedir permiso para usar 4 fotos como test de humo |
| [OMAFRA Ontario](https://www.ontario.ca/page/body-condition-scoring-dairy-cattle) | Esquemas 1–5 | Lateral/trasera (dibujos) | © King's Printer for Ontario (no verificado) | Solo referencia |
| [UNL G1583](https://extensionpubs.unl.edu/publication/g1583/na/pdf/view), [SDSU](https://openprairie.sdstate.edu/extension_extra/137), [UMD](https://extension.umd.edu/resource/body-condition-scoring-important-dairy-cattle-too) | Pocas fotos por puntaje (no verificado) | Varias | Sin licencia abierta (no verificado) | Solo referencia |
| Elanco (Edmonson et al. 1989), DairyNZ, Teagasc, INIA, INTA | **No encontré** versiones con licencia abierta | — | — | — |
| Wikimedia Commons | **Ninguna** imagen etiquetada con BCS ✔ (API: "body condition score cow", "emaciated cow") | — | — | — |

Ningún atlas da más de 1–2 fotos por puntaje y ninguno tiene licencia abierta. **No vale la pena integrarlos**: a
lo sumo, son 5–10 fotos de ejemplo para la documentación, con permiso escrito.

## 2. Ranking combinado para BCS ≤ 2.75 (T23 + T24 + T26)

| # | Fuente | Qué aporta ≤ 2.75 | Dominio vs. nuestra app | Esfuerzo | Riesgo principal | Origen |
|---|---|---|---|---|---|---|
| 1 | **Nagy et al. 2023** | **1.560 img 1.0–2.5 + 270 en 2.75** | Trasera de grupa, sala rotativa | Medio, después de que contesten (mail listo) | Un solo puntuador; sin ID de vaca | T23 |
| 2 | **Shriramulu, Pushpadass 2024** | ~270 img < 2.75 (1.5–2.5), paso 0,25 | Dorsal, cámara de mano | Medio (mail listo) | Puntaje sobre foto; sin ID | T24 |
| 3 | **Relevamiento propio en Uruguay** (§4.2) | Lo que se busque, **con celular** | **Idéntico** | Alto en logística | Pocas vacas ≤ 1.75 | T23 |
| 4 | **Grecheneva et al. 2026** | **266 vacas con BCS 1–2** (clases enteras) | Cenital + lateral, video | Medio: mail + adaptador de video (elegir frames) | Clases enteras; la figura puede ser ilustrativa; vista distinta | **Nuevo** |
| 5 | **Dandıl et al. 2024** (+ Çevik 2020) | 330 img en 2.0 / 2.5 (categorías) | Trasera pélvica | Medio; solo test ordinal | Etiquetas gruesas | T23 (+ T26) |
| 6 | **Moreira, Dórea et al. 2022** | Desconocido (31 vacas) | **Trasera con iPhone**: el más parecido | Bajo (pocas fotos) | Pocas flacas, casi seguro | **Nuevo** |
| 7 | **e-Score Embrapa** | Desconocido | Trasera, Holando | Mail para preguntar la distribución | Puede no tener flacas | T24 |
| 8 | Wu/Pezzuolo 2021 · Lai 2024 · Montenegro 2023 · NARO 2024 | Desconocido | RGB (NARO: trasera con celular) | Un mail corto a cada uno preguntando el rango | Lo más probable es que no tengan extremos | **Nuevo** |
| 9 | Rodríguez Alvarez 2019 / Summerfield 2023 / NEAU / HAUST | 1.75–3.0 en depth | Otra modalidad | Alto | Depth | T23/T24 |
| — | Espasandín 2023 (1–8, carne) / Islam 2026 (1–9) | Flacas de carne | Otra escala | — | No se mezcla | T24 / T26 |
| ✗ | AnimalBCS-1200 | Dice tener 88 img ≤ 2 | — | — | Datos de credibilidad dudosa | **Nuevo, descartado** |

**Cambio respecto de T24:** entra Grecheneva en el #4. Es la fuente con más **vacas distintas** flacas (266 contra
un número desconocido en Nagy y Shriramulu, que no publican ID de animal), pero queda por debajo del relevamiento
propio porque la vista y la escala no coinciden con las nuestras. Moreira/Dórea entra como test de dominio, no como
volumen.

## 3. Fuentes a pedido: contacto y qué pedir

Los mails los manda una persona del equipo (los agentes no escriben a terceros). Usar la plantilla de
`mails-solicitud-datasets.md`: decir explícitamente que **ARGOS es un producto comercial** y pedir permiso escrito
para entrenar y servir el modelo sin redistribuir las imágenes. Para todos, pedir además: (a) BCS por imagen,
(b) **ID de vaca o de sesión** para el `group_id`, (c) quién puntuó y cómo (¿palpando o sobre la foto?).

| Prioridad | Fuente | Contacto | Qué pedir en concreto |
|---|---|---|---|
| 1 | **Grecheneva et al. 2026** | a.grecheneva@rgau-msha.ru ✔ (autora de correspondencia) | (1) Confirmar que la Fig. 8 es la distribución real de las 983 vacas. (2) El **BCS original con paso 0,25** (si lo hay), no la clase entera. (3) Los frames seleccionados por el algoritmo (top-K por vaca), sobre todo de **vacas con BCS ≤ 2.75**, con vaca, fecha y vista. (4) Raza y si puntuaron palpando. El mail puede ir en inglés; agregar un resumen en ruso ayuda |
| 2 | **Moreira, Bresolin, Dórea, Rosa 2022** | Grupo de João Dórea, UW-Madison (Animal & Dairy Sciences); mail **no verificado**, buscarlo en la página del departamento | Las 276 fotos RGB traseras de iPhone con el BCS de **cada** evaluador y el ID de vaca. Preguntar el rango de BCS. Aunque no haya flacas, sirve como **test externo de celular** |
| 3 | **Çevik (Akdeniz)** | En el **mismo mail que Dandıl** (emre.dandil@bilecik.edu.tr, T23), con copia a Çevik (mail **no verificado**) | Si el set de 2019/2020 es parte del de 2024 o es aparte, y su distribución de BCS |
| 4 | **Pezzuolo (Padua)** | Mail institucional de UniPD (**no verificado**) | Rango de BCS de las 918 imágenes de Wu et al. 2021. Aprovechar para preguntar si el grupo de Ruchay (`zen_rgbd`) tiene **vacas < 2.75 fuera del Zenodo** |
| 5 | **NARO** | [Formulario de consultas](https://www.naro.go.jp/english/inquiry/index.html) | Referencia del paper y si comparten fotos traseras con BCS (y el rango). Probablemente no compartan, pero el método (ángulo del ilíaco desde atrás) es un buen *baseline* geométrico |
| 6 | Lai (Taiwán), Montenegro (Ecuador) | Mails **no verificados**; buscarlos en las páginas institucionales | Una línea: "¿qué rango de BCS tiene su dataset y lo comparten para uso comercial?" |
| — | AnimalBCS-1200 | drranjanamdamle@gmail.com ✔ | **No recomendado** (ver §1). Si igual se escribe, pedir fotos crudas con metadatos EXIF antes de confiar |

## 4. Opciones que no dependen de terceros

### 4.1 Datos sintéticos

**Qué dice la literatura.** No encontré **ningún** trabajo que genere imágenes de vacas para BCS. Lo más cercano:
- Aumento con difusión en ganado para **peso** (fondos nuevos para 68 Angus, [arXiv 2510.08955](https://arxiv.org/abs/2510.08955)).
- Clases balanceadas con imágenes generadas en enfermedades de piel ([arXiv 2601.12889](https://arxiv.org/abs/2601.12889)).
- Li K. et al. 2024 usa un **DDPM solo para reconstruir imágenes de profundidad**, no para crear vacas nuevas.

En la literatura general de desbalance:
- Las GAN condicionales colapsan cuando hay pocos datos por clase ([arXiv 2303.04854](https://arxiv.org/abs/2303.04854); [BAGAN](https://arxiv.org/abs/1803.09655)).
- Un benchmark de 2026 en animales ([alphaXiv 2603.16134](https://www.alphaxiv.org/abs/2603.16134), **no verificado**, snippet) encontró que FastGAN **aumentó el sesgo** con menos de ~20–50 imágenes por clase. Stable Diffusion con LoRA funcionó mejor.

**Riesgos específicos de BCS:**
1. **La etiqueta no se puede verificar.** El BCS depende de la prominencia de isquiones, ilíacos y apófisis y de la
   profundidad de la fosa de la cola. Un modelo generativo con el *prompt* "vaca BCS 1.5" no garantiza esa anatomía:
   la etiqueta pasa a ser la intención del prompt, no un puntaje.
2. **Atajos.** Si todas las flacas son sintéticas, el modelo aprende a detectar "imagen generada" (texturas, fondo,
   iluminación) en vez de "vaca flaca". Es la misma trampa que "la fuente predice la etiqueta" (T23 §2).
3. **Edición imagen a imagen** (SDEdit/ControlNet: "afinar" una vaca 3.0) conserva la pose y el fondo, pero el
   cambio de condición sale de lo que el modelo generativo imagina. Pasa lo mismo que en el punto 1.
4. **Mixup entre clases con etiqueta interpolada:** mete ruido justo en la frontera que queremos aprender.

**Recomendación:** no contar con datos sintéticos para cubrir 1.0–2.75. Si se prueba algo, que sea:
- **solo para entrenar**, nunca para evaluar: las métricas de flacas salen solo de vacas reales, agrupadas por `group_id`;
- con **filtrado humano a ciegas**: un técnico puntúa cada imagen sintética sin conocer el prompt y se descarta la que no coincide;
- como experimento acotado: medir el MAE en flacas reales con y sin sintéticos. Si no mejora, se descarta.

Lo que **sí** conviene hacer sin generar imágenes: aumento clásico (color, recorte, perspectiva de celular),
pérdida ordinal (CORAL/CORN) y peso por clase. Así las pocas flacas reales rinden más.

### 4.2 Relevamiento propio en Uruguay enfocado en flacas

**Dónde hay más vacas flacas** (de más a menos probable):
1. **Primer tercio de lactancia (30–90 días posparto)**, sobre todo vacas de **primera lactancia** y partos de
   otoño. Es el nadir del balance energético negativo. En sistemas pastoriles de la Pampa (Rearte, Zenodo 8082738 ✔),
   la proporción de vacas en "mala condición corporal" varía de **3 % a 47 % según el grupo de parición** (mediana
   ≈ 23 %). Conviene ir a los grupos de parición que muestran más vacas flacas.
2. **Lote de refugo / vacas de descarte**, antes de mandarlas a remate o frigorífico. Ahí se juntan las flacas
   crónicas (cojeras, mastitis, paratuberculosis), que son las únicas con BCS ≤ 1.75 en cantidad.
3. **Después de una seca** (como 2022–2023) o en invierno con poco pasto.
4. **Ferias y corrales de frigorífico:** hay vacas emaciadas, pero solo vale la pena con permiso del operador y
   cuidando el bienestar animal (sin demorar el manejo). Además son mayormente de carne o cruzas: anotar la raza.

**Prevalencia esperable de flacas extremas.** En las fuentes que la reportan, **BCS ≤ 1.5 es cerca del 5 %** de lo
fotografiado: Nagy, BCS 1.0 = 225 de 4.061 img (5,5 %); Grecheneva, BCS 1 = 52 de 983 vacas (5,3 %); Shriramulu,
≤ 1.75 = 6 de ~127 en test (~5 %). Es **una estimación propia sobre rodeos de otros países (no verificada para
Uruguay)**, y estos datasets se armaron buscando variedad. En un tambo comercial uruguayo probablemente sea menos.

**Cuántas vacas hacen falta por clase** (estimación propia, con los mismos criterios que T23 §5: CV agrupada k = 5,
SPEC §0.1 colapsa clases con < 15 imágenes y ~30 animales para un IC de ±0,1 en el MAE de la clase):

| Clase | Mínimo para **evaluar** (vacas) | Deseable para **entrenar** (vacas) | Nota |
|---|---|---|---|
| 2.75 | 20 | 40 | Fácil: está en todos los tambos |
| 2.5 | 20 | 40 | Fácil en el primer tercio de lactancia |
| 2.25 | 15 | 30 | |
| 2.0 | 15 | 30 | Hay que buscarla (refugo, primíparas) |
| 1.75 | 10 | 20 | Rara |
| ≤ 1.5 (1.0–1.5 juntas) | 10 | 20 | Muy rara: **colapsar en una sola clase** si no se llega a 15 imágenes (SPEC §0.1) |
| **Total ≤ 2.75** | **~90 vacas** | **~180 vacas** | 3–5 fotos por vaca → 300–900 imágenes |

- **Cuánto hay que recorrer.** Para juntar 20 vacas con BCS ≤ 1.75, si son el 3–5 % del rodeo, hay que **mirar
  400–700 vacas**. El técnico puntúa rápido (~1 min por vaca, palpando) y se fotografía en detalle solo a las flacas,
  más un 20 % de control en 3.0–3.5 para que la flaca no quede asociada a ese día o ese tambo.
- **Varios tambos, siempre.** Con un solo tambo, "flaca" y "tambo X" quedan confundidos. Lo mínimo son 3 tambos, más
  1 lote de refugo o feria.
- **Protocolo:** el de T23 §5 (escala 1–5 con paso 0,25 de Ferguson/Edmonson, puntaje palpando antes de la foto,
  segundo puntuador en ≥ 50 vacas, `animal_id = tambo_uy:<hash de la caravana>`, `view=rear_phone`, holdout humano
  reservado desde el principio). Nada de esto entra al repo.
- **Contactos posibles en Uruguay:** Espasandín (Udelar, T24) y los autores de la revisión sobre la vaca en transición en Uruguay
  ([Agrociencia 26:e1110](https://doi.org/10.31285/agro.26.1110); no leí el texto completo ni verifiqué los autores).
  También INIA La Estanzuela. Ninguno fue contactado.

## 5. Lo que no pude verificar

- **WebSearch quedó bloqueado a mitad del trabajo** por el clasificador de permisos de la sesión. Seguí con APIs
  públicas de solo lectura (Europe PMC, OpenAlex, Zenodo, figshare, Dataverse, OSF, Hugging Face, Wikimedia). Por eso
  la cobertura en persa, francés y alemán se quedó en una búsqueda por idioma y no revisé Google Scholar.
- **Grecheneva 2026, Fig. 8:** el título dice "пример оформления для выборки 983 головы" ("ejemplo de formato para
  una muestra de 983 cabezas"). Puede ser ilustrativa: **hay que confirmar con la autora** que los conteos son reales.
  Tampoco sé la raza ni si el puntaje fue con palpación.
- **Rangos de BCS** de Moreira/Dórea 2022, Çevik 2020, Wu/Pezzuolo 2021, Lai 2024, Montenegro 2023 y NARO 2024:
  los textos completos no estaban accesibles (IEEE y Springer devuelven login; NARO no publica el paper).
- **Mails no verificados:** Dórea, Çevik, Pezzuolo, Lai/Huang y García-Santillán.
- **Rearte (Zenodo 8082738):** el Excel tiene salidas de modelos por grupo de parición (columna `P(E+)`), no BCS
  crudos. La definición de "expuesta" (umbral de BCS) no la verifiqué en el paper.
- **AnimalBCS-1200:** no pude confirmar si el dataset existe; no tiene link.
- **Roboflow** sigue sin verificarse (Cloudflare + API key, igual que en T24).
- **Prevalencia real de vacas ≤ 2.0 en tambos uruguayos:** no encontré un dato publicado. Las cifras de §4.2 son
  extrapolaciones.

## Fuentes (consultadas el 23/10/2026)

- Grecheneva et al. 2026: https://doi.org/10.26897/0021-342X-2026-2-105-122 · PDF https://elib.timacad.ru/dl/full/iz-08-2026-2.pdf/download (Fig. 8 en `~/bcs-shared/artifacts/T26/grecheneva2026_fig8.png`)
- Moreira et al. 2022: https://doi.org/10.1093/jas/skac247.278 · https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9493717/
- Çevik 2020: https://doi.org/10.1109/ACCESS.2020.3040805 · Çevik & Boğa 2019: https://doi.org/10.1109/ASYU48272.2019.8946405
- Wu et al. 2021: https://doi.org/10.1109/MetroAgriFor52389.2021.9628437
- Lai et al. 2024: https://doi.org/10.1007/978-981-97-0376-0_17 · Montenegro et al. 2023: https://doi.org/10.1007/978-3-031-32213-6_7
- NARO 2024: https://www.naro.go.jp/english/topics/laboratory/iam/161469.html · https://www.jfc.go.jp/n/finance/keiei/pdf/2667.pdf
- Bloch 2021: https://zenodo.org/records/5645192 (CSV de una fecha en `~/bcs-shared/artifacts/T26/zenodo/`)
- Rearte 2023: https://zenodo.org/records/8082738 (`data.xlsx` en `~/bcs-shared/artifacts/T26/zenodo/`)
- AnimalBCS-1200: https://doi.org/10.55529/ijaap.52.108.121
- IJBSE 2025 (Irán): https://ijbse.ut.ac.ir/article_103941.html?lang=en (texto extraído en `~/bcs-shared/artifacts/T26/ijbse2025_text.txt`)
- Ariadi et al. 2025: https://doi.org/10.5398/tasj.2025.48.5.402 · Islam et al. 2026: https://doi.org/10.1016/j.atech.2026.102628
- O'Connor 2018 (Massey): https://mro.massey.ac.nz/bitstreams/32114351-cd37-406e-939f-5fd5c76dab89/download
- Matto et al. 2025: https://doi.org/10.3168/jdsc.2025-0851 · Tedín et al. 2012: https://doi.org/10.3233/978-1-61499-105-2-2150
- Penn State Extension: https://extension.psu.edu/examples-of-cows-at-various-body-condition-scores · OMAFRA: https://www.ontario.ca/page/body-condition-scoring-dairy-cattle
- Sintéticos: https://arxiv.org/abs/2510.08955 · https://arxiv.org/abs/2601.12889 · https://arxiv.org/abs/2303.04854 · https://arxiv.org/abs/1803.09655 · https://www.alphaxiv.org/abs/2603.16134
- Búsquedas por API: Europe PMC REST (`search`, 125 resultados), OpenAlex (`works?search=`, 6 consultas, ~535 títulos), Zenodo, figshare, Harvard Dataverse, OSF, Hugging Face y Wikimedia Commons.
