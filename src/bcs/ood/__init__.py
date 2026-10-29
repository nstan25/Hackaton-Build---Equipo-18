"""Detección de fotos fuera de distribución (OOD) sobre los embeddings del modelo (T27).

`methods`: puntajes kNN, Mahalanobis (una gaussiana, por clase, relativo), norma de la feature y
combinación por z-scores, más las métricas (AUROC, FPR@95% TPR, predicción selectiva).
`embed`: el vector de 1.280-d de la EfficientNet-B0 de producción antes de la cabeza.

Convención: en todos los puntajes, **más alto = más raro** (más OOD).
"""
