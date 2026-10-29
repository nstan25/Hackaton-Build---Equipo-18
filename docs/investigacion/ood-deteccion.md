# Detección de fotos fuera de distribución (OOD): Mahalanobis vs. kNN · T27

Fecha: **26/10/2026**. Reporte completo, con todas las tablas y gráficos:
`~/bcs-shared/runs/T27/REPORT.md`. Código: `src/bcs/ood/` y `scripts/ood/`; parámetros:
`config/ood.yaml`. Es un experimento: la app no cambió. Todo corrió en CPU, sin test ni holdout.
Ninguna imagen de terceros está en el repo.

## Pregunta

Hoy la app solo tiene el filtro básico (brillo, sobreexposición, nitidez) y el detector YOLO a
conf 0,15. Una foto rara que pase el detector recibe igual un BCS. La pregunta es si un puntaje
"qué tan distinta es esta foto de las de entrenamiento", calculado sobre el embedding del modelo,
detecta esas fotos, y cuánto cuesta en memoria y latencia.

## Qué hice

- **Embedding:** el pooling global (1.280-d) de la EfficientNet-B0 de producción, antes de la
  cabeza, con la transformación de `inference.py`, sobre el recorte. Sin TTA.
- **Banco de referencia:** `sdb_bcs` train, ≤ 5 frames por grupo (12.514). ID de prueba:
  `sdb_bcs` val, partido por grupo en calibración y evaluación.
- **Métodos:** kNN coseno (k ∈ {1, 5, 10, 50}; bancos de 2 k, 5 k y todo) y Mahalanobis
  Ledoit-Wolf, con una gaussiana, por clase, relativo y sobre features L2. También la combinación
  por z-scores (kNN + Mahalanobis), la norma de la feature y el pipeline actual como baselines.
- **Conjuntos OOD**, pasados por el pipeline real (filtro de calidad + detector + recorte):
  - los 40 dibujos de `tests/fixtures/reject/`;
  - 10 degradaciones de fotos de val (oscura, quemada, borrosa, ruido, JPEG, rotada, cabeza
    abajo, parcial, muy de cerca, descentrada);
  - recortes y frames crudos de `zen_rgbd` val, otra granja y otra cámara;
  - 400 fotos de COCO 2017 con licencia CC BY, BY-SA o BY-ND: vacas, otros animales y otras cosas.

## Resultados

**Hoy, el 27 % de las fotos de COCO (de cualquier cosa) pasa el detector y recibe un BCS.**

AUROC / FPR@95 % TPR sobre lo que llega al modelo (más detalle en el reporte):

| conjunto | kNN k=1 (todo) | kNN k=10 | **Mahalanobis** | combinación |
|---|---|---|---|---|
| COCO | 0,80 / 50 % | 0,76 / 58 % | **0,99 / 3,7 %** | 0,95 / 16 % |
| COCO, solo vacas (de costado, de frente) | 0,74 / 55 % | 0,68 / 70 % | **0,99 / 2,3 %** | 0,94 / 23 % |
| dibujos de T08 | 1,00 / 0 % | 1,00 / 0 % | **1,00 / 0 %** | 1,00 / 0 % |
| borrosa | 0,79 / 55 % | 0,77 / 56 % | **0,98 / 11 %** | 0,94 / 25 % |
| muy de cerca | 0,87 / 42 % | 0,84 / 48 % | **0,99 / 3 %** | 0,98 / 7 % |
| rotada 90° | 0,76 / 83 % | 0,74 / 83 % | **0,89 / 58 %** | 0,87 / 69 % |
| zen val (vacas reales, otro dominio) | 0,985 / 6 % | 0,98 / 9 % | 0,98 / 11 % | 0,995 / 1,8 % |

- **Mahalanobis (una gaussiana) le gana a kNN** en todo lo que no es una vaca de atrás o es una
  foto mala, y ocupa 6,3 MB contra 61 MB. La versión por clase da lo mismo. La relativa sale
  invertida (descartada). La combinación no le gana a Mahalanobis solo, salvo en zen.
- **Pero las vacas reales de otra granja también puntúan "raras"**: con el umbral p99 de val se
  marcaría el 42 % de los recortes de zen val. El puntaje mide "se parece a la cámara de
  `sdb_bcs`", no "es una vaca de atrás". Las fotos de COCO menos raras (una vaca de costado, una
  pizza, una nena) puntúan por debajo de muchas vacas de zen.
- **Predicción selectiva:** dentro de un dominio casi no ayuda. En sdb val, rechazar el 10 %
  baja el MAE de 0,173 a 0,168; en zen no cambia. Mezclando sdb y zen sí ayuda (0,211 → 0,185
  al 10 %), porque saca primero las de zen.
- **Costo de Mahalanobis:** 6,3 MB y 0,2 ms por imagen sobre la feature, que sale de la misma
  pasada del modelo. Despreciable para Railway (1 GB).
- **Lo que nadie detecta:** la foto levemente quemada que pasa el filtro (su error es normal) y
  la descentrada (el detector recorta bien la vaca: está bien que pase).

## Recomendación

1. Usar **Mahalanobis con una gaussiana** (Ledoit-Wolf) sobre el embedding sin flip, ajustado con
   `sdb_bcs` train (≤ 5 frames por grupo).
2. **Dos niveles**, no un rechazo único:
   - **≥ p99 de val (≈ 3.780) → `low_confidence`**, con el motivo *"la foto no se parece a las
     del entrenamiento; verificá encuadre y luz"*. Rechaza el 0,4–0,7 % de las fotos buenas y el
     70 % de lo de COCO que hoy pasa el detector. Lo de COCO que recibe un BCS sin aviso baja del
     27 % al 8 %.
   - **≥ p99 de vacas reales de otro dominio (zen; ≈ 5.900) → `rejected`**, con *"no parece una
     vaca de atrás"*. Agarra todos los dibujos de T08 y el 19 % de COCO, y ninguna vaca real.
     Hay que calibrarlo con zen **train**.
3. **El banco tiene que cubrir todas las fuentes de entrenamiento** y recalibrarse cuando cambie
   el modelo (p. ej., si entra zen por T25). Antes de confiar en los umbrales en el campo, hay
   que medir la tasa de aviso con fotos de celular (`phone_backview`).
4. **Integración** (otra tarea, con gate humano):
   - Va en `inference._run_decoded`, después del detector y del forward, y antes de
     `decide_status`.
   - Archivos a tocar: `models/efficientnet.py`, para devolver el pooling de la vista sin flip;
     `inference.py` y `contract.py` (archivos calientes), para el puntaje, el status y el
     motivo; `config/ood.yaml`; un artefacto nuevo `models/ood_mahalanobis.npz` (6,3 MB, en
     `model_version`); y el texto del motivo en la web.
