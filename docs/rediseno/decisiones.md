# T18 · Decisiones del rediseño ARGOS

Notas del worker sobre cómo se implementó `rediseno-ux-ui.md` en `web/` (HTML/CSS/JS sin frameworks).

## Estructura
- `web/argos.css`: tokens (paleta, tipografías, radios, sombra) y componentes compartidos (header, botones,
  chips, campos, footer, escala 1–5). Cada página conserva su CSS propio inline.
- El script del token va **primero en `<head>`**: la hoja de estilos y las imágenes también piden token
  (`api_web` las protege), así que la cookie tiene que existir antes de pedirlas. El JS de cada página
  sigue inline (lo exige `test_web_does_not_put_token_in_urls`).
- `web/index.html`: header → hero → **Analizar** (Foto → Análisis → Resultado) → ¿Cómo funciona? →
  Sobre Argos → footer. `web/rodeo.html`: mismo header/footer, lista, filtros e historial con gráfico.
- IDs del flujo sin cambios (`cow-id`, `photo-camera`, `confirm`, `send`, `result`, `res-*`, `cows`,
  `hist`, `chart`…): el recorrido de `tests/browser/run_browser.py` debería pasar sin tocarlo. Se sacó
  `#res-low` (siempre estaba oculto desde G3 punto 7).

## Paleta (extraída de `logo.jpeg`)
| token | hex | origen / uso |
|---|---|---|
| `--forest` | `#2e5134` | "Argos" del logo (medido `#314e35`): botones, títulos, BCS |
| `--forest-2` | `#365e3c` | hoja oscura (`#365e3c`) |
| `--sage` | `#769b56` | hoja clara (`#769b56`): solo decoración (franja del rango validado) |
| `--sun` | `#e3a141` | sol (`#e3a141`): acentos, viñetas, marcas de la guía; nunca texto sobre claro |
| `--ivory` | `#f7f4ec` | papel del logo (`#f2f3ee`) más cálido: fondo |
| `--card` | `#fffdf8` | blanco cálido: tarjetas |
| `--deep` | `#1f3a25` | footer |
| `--ink` / `--muted` | `#1d2a20` / `#4f5c52` | texto |

## Tipografías
- Títulos: **Fraunces** (serif editorial con eje óptico; coherente con la serif del logo).
- UI y texto: **Inter** (sans muy legible en pantallas chicas y al sol).
- Google Fonts con `display=swap` y fallback Georgia / system-ui: si no cargan, la página no se rompe.

## Contraste medido (WCAG 2.x, fórmula de luminancia relativa)
| par | ratio |
|---|---|
| ink sobre ivory | 13.61 |
| ink sobre card | 14.71 |
| muted sobre ivory / card | 6.40 / 6.92 |
| forest sobre ivory | 8.16 |
| blanco sobre botón forest | 8.96 |
| ink sobre sun (paso activo) | 6.72 |
| chip OK `#2e6b3a` / `#e6efdf` | 5.42 |
| chip confianza baja `#7a4f0c` / `#fbefd6` | 6.24 |
| chip rechazada `#9e2a1f` / `#fbe7e3` | 6.29 |
| ivory sobre deep (footer) | 11.30 |
| `#c9d3c4` sobre deep (texto chico del footer) | 8.05 |
| borde de controles `#8a958c` / card (UI, ≥ 3:1) | 3.06 |
| sage / sun sobre ivory | 2.90 / 2.02 → **solo decorativos**, nunca texto |

## Imágenes (`docs/rediseno/build_images.py`)
- `hero-cow-{480,720,832}`: recorte derecho del hero (vaca + líneas de escaneo, **sin el logo** para no
  duplicarlo con el header). WebP + JPEG, `srcset`/`sizes`, `width`/`height` → sin saltos de layout.
  Se muestra entero (sin `object-fit` que recorte) para no cortar la vaca en celulares; las animaciones
  (línea de escaneo, pulsos en la grupa, zoom lento) se ubican en % de este recorte.
- `hero-full-{640,1024,1536}`: la imagen completa, en "Sobre Argos" (ahí sí con su logo, lejos del header).
- `logo-mark-{64,128}`: el símbolo con el fondo de papel vuelto transparente; "Argos" va en Fraunces.
- Total ~700 KB en disco; una visita a 390 px baja ~20 KB de hero + ~3 KB de logo arriba del pliegue.

## UX
- Flujo en una sola tarjeta con indicador de pasos (Foto → Análisis → Resultado) y `aria-current`.
- ID de la vaca obligatorio (contrato), arriba, con autocompletado, último ID recordado y "Vaca: <ID>".
- "Tomar fotografía" (`capture=environment`) y "Subir imagen" son `<button>` reales (foco y teclado);
  arrastrar y soltar en escritorio. Validación local de tipo y de tamaño (20 MB = `MAX_IMAGE_BYTES`).
- Guía de captura: ilustración propia de una vaca **de atrás y desde arriba** (no el hero, que es de
  costado) con ilíacos e isquiones marcados + tres consejos; el protocolo de SPEC §0.2 en un desplegable.
- Análisis: overlay de escaneo sobre la vista previa y "Argos está analizando la imagen…", sin porcentaje
  (el backend no lo da). Error con mensaje claro y reintento con el mismo botón.
- Resultado: BCS grande "/ 5", escala 1–5 con el rango validado 3.25–4.25 y el intervalo marcados,
  lectura relativa al rango validado sin juzgar ("qué valor conviene depende de la etapa productiva"),
  intervalo y confianza (sin los textos que sacó G3 punto 7), distribución en un desplegable, foto
  original, nota de "estimación automatizada de apoyo". Rechazo con los `issues` y cómo reintentar.
- T17 integrado (vuelta 2): "Ver qué se observa en la foto" y el panel "Qué se observa en la foto" van en
  `.res-slot[data-slot="explicacion"]` del resultado y al final de cada `.hist-item > div` del rodeo
  (no en las rechazadas), con la lógica JS de T17 sin cambios de comportamiento.
- CTA fija "Analizar una vaca" en celulares cuando no se ve ni el CTA del hero ni la sección de análisis.
- Animaciones solo con `prefers-reduced-motion: no-preference`; el scroll suave solo en links internos.

## Pendientes / requiere información
- Sin modo demo ni foto de ejemplo: no hay una imagen propia con licencia clara para incluir en `web/`.
- Footer sin nombre de la hackatón (decisión del humano, vuelta 2).
- Sin capturas: no se corrió Firefox (cuelga la VM); `make browser` lo tiene que correr el orquestador.
- El aviso "Sin evaluación en fotos de celular" se sacó de todas las páginas (decisión del humano,
  vuelta 2); `tests/browser/run_browser.py` ahora verifica que no aparezca.
- G3 punto 5: el filtro ya no verifica que la foto sea de atrás; la UI no lo promete (dice "si no
  detecta una vaca o la imagen está oscura o borrosa").
