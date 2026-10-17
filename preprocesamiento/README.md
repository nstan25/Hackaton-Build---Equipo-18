# Preprocesamiento

Scripts para preparar el dataset de BCS (condición corporal) antes de entrenar. Correr todo desde esta carpeta.

| Paso | Script | Qué hace | Salida |
|---|---|---|---|
| 1 | `make_split.py <dataset>` | Split train/val/test (70/15/15) agrupado por vaca/video para evitar fuga entre splits | `split.csv` |
| 2 | `apply_split.py <dataset>` | Copia imágenes + XML a `{train,val,test}/<clase>/` según `split.csv` | `dataset_split/` |
| 3 | `prep_crops.py` | Recorta cada imagen por su bbox (+10% margen, lado máx. 320px) | `crops/` + `crops/manifest.csv` |
| 4 | `prep_yolo.py build` | Arma un dataset YOLO de 1 clase (vaca) desde los bbox | `yolo_data/` |
|   | `prep_yolo.py eval [modelo.pt] [n]` | IoU del detector vs. bbox del XML en val | — |

`split.csv` está versionado para que todo el equipo use el mismo split (seed 42).
Las imágenes, recortes y pesos no se suben (ver `.gitignore`); se regeneran con los scripts.

Dependencias: `pandas`, `pillow`, `ultralytics` (solo para `prep_yolo.py`).
