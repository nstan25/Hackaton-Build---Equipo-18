# Formatos compartidos (definidos por tareas mergeadas)

## Predicciones para `make eval` (T02)
CSV en `$BCS_ARTIFACTS_DIR/<MODEL>/predictions_val.csv` (o `PREDICTIONS=`):
`image_id, group_id, source, bcs, p_<clave grilla>…` (una columna por clase de la escala activa, p. ej. `p_3.25`), `fold` opcional (default `val`; `test`/`holdout` se rechaza), `interval_lo`/`interval_hi` opcionales. Helper: `bcs.eval.report.synthetic_predictions()`.
Filtro de calidad para m4: `$BCS_ARTIFACTS_DIR/<MODEL>/quality_metrics.json` (`reject_negatives`, `accept_rear`, `reject_val`) o `QUALITY=`.

## `report.json` (T02, schema_version 1)
`mean` (contra esto se comparan umbrales), `pooled`, `folds`, `loso`, `dummy`, `mae_improvement_vs_dummy`, `thresholds{passed, checks}`, `plots`. `make eval` sale 0 si pasa, 1 si no, 2 si la entrada es inválida o viola una pared.
