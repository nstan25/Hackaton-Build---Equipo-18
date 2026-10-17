# Set de rechazo del filtro de calidad (T08)

Todo lo que hay acá es **generado** con PIL/numpy por `src/bcs/quality/reject_set.py`, con semillas
fijas: no hay imágenes ni datos de terceros (SPEC §4.1, G0 §5). Cada imagen es algo que el filtro
tiene que **rechazar**: no es una vaca, no es una vaca de atrás, no es una foto, o la foto no sirve.

Regenerar: `uv run python -m bcs.quality.reject_set` (mismos bytes; lo verifica
`tests/test_quality_reject_set.py`). Medir: `uv run python -m bcs.quality.measure`.

Limitación: son dibujos y texturas, no fotos reales. Para negativos con fotos reales sin meter
imágenes de terceros, `measure` además degrada imágenes de val (oscura, quemada, borrosa) en
memoria y reporta su tasa de rechazo en `quality_metrics.json` (`val_degraded`).

| Archivo | Motivo de rechazo | Qué es |
|---|---|---|
| `side_cow.png` | no es de atrás | dibujo de una vaca de costado |
| `cow_face.png` | no es de atrás | dibujo de la cara de una vaca de frente |
| `dog.png` | no es vaca | perro de costado |
| `cat.png` | no es vaca | gato sentado |
| `bird.png` | no es vaca | pájaro en una rama |
| `fish.png` | no es vaca | pez en agua |
| `horse_side.png` | no es vaca | caballo de costado |
| `sheep_flock.png` | no es vaca | ovejas en el campo, de lejos |
| `person.png` | no es vaca | persona parada |
| `face_selfie.png` | no es vaca | cara de una persona (tipo selfie) |
| `car.png` | no es vaca | auto en la ruta |
| `tractor.png` | no es vaca | tractor en el campo |
| `house.png` | no es vaca | casa |
| `barn_empty.png` | no es vaca | galpón y alambrado, sin animales |
| `tree_landscape.png` | no es vaca | paisaje con árbol, sol y cerro |
| `flower.png` | no es vaca | flor |
| `mug.png` | no es vaca | taza sobre una mesa |
| `chair.png` | no es vaca | silla |
| `food_plate.png` | no es vaca | plato de comida |
| `text_document.png` | no es foto | página de texto |
| `spreadsheet.png` | no es foto | planilla con números |
| `phone_screenshot.png` | no es foto | captura de un chat de celular |
| `qr_like.png` | no es foto | código tipo QR |
| `color_noise.png` | no es foto | ruido de color |
| `gray_noise_blur.png` | no es foto | ruido gris suavizado |
| `gradient.png` | no es foto | degradé de color |
| `checkerboard.png` | no es foto | tablero de ajedrez |
| `stripes.png` | no es foto | franjas |
| `solid_color.png` | no es foto | color liso |
| `concentric.png` | no es foto | círculos concéntricos |
| `sky_only.png` | no es vaca | cielo con nubes |
| `floor_concrete.png` | no es vaca | piso de hormigón |
| `black_frame.png` | muy oscura | cuadro casi negro (tapa puesta, de noche) |
| `very_dark_cow.png` | muy oscura | dibujo de vaca de atrás al 7% de brillo |
| `night_noise.png` | muy oscura | vaca de atrás al 5% de brillo con ruido de sensor |
| `overexposed.png` | sobreexpuesta | vaca de atrás quemada (casi blanca) |
| `white_frame.png` | sobreexpuesta | cuadro blanco |
| `blurred_cow.png` | borrosa | vaca de atrás con desenfoque gaussiano σ=12 |
| `motion_blur_cow.png` | borrosa | vaca de atrás con barrido horizontal (movimiento) |
| `lens_cap_finger.png` | borrosa | dedo tapando la lente, desenfocado |
