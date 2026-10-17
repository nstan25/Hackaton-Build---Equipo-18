# BCS de vacas — Hackathon

Score de condición corporal (BCS 3.25–4.25) a partir de la foto de la cámara superior trasera.

```
foto ──► detector YOLO11n (models/cow_det.pt) ──► recorte de la grupa (+10 %) ──► EfficientNet-B0 (models/bcs_efficientnet_b0.pt) ──► BCS
```

## Resultados (test, 8.043 imágenes de vacas que el modelo nunca vio)

| Modelo | MAE | BCS exacto | ±0.25 |
|---|---|---|---|
| Baseline: DINOv2 congelado + LightGBM | 0.187 | 39.9 % | 89.4 % |
| **EfficientNet-B0 fine-tune (regresión)** | **0.170** | **43.4 %** | **92.6 %** |
| EfficientNet-B0, promedio por video | 0.168 | 44.7 % | 93.0 % |

- Detector: encuentra la grupa en el 100 % de las imágenes de val, IoU media 0.815 con la caja anotada.
- Pipeline completo (detector en lugar de la caja del XML): +0.01 de MAE, 62 ms por imagen en la RTX 2050.
- Por cámara (CNN): GS 90.1 % · YM 93.5 % · L-i 100 % · R-i 99.1 % dentro de ±0.25.
- Debilidad: los extremos se tiran al centro (4.25 suele salir 4.0; 3.25 suele salir 3.5).

Métricas completas y matrices de confusión en `results/`.

## Cómo correr

```bash
python app.py                 # demo en http://127.0.0.1:7860
python bcs_pipeline.py foto.jpg
python monitor_server.py      # panel de entrenamiento en http://127.0.0.1:8765
```

Reentrenar desde cero:

```bash
python make_split.py <dataset> split.csv        # split por vaca (sin fuga entre train/val/test)
python apply_split.py <dataset> split.csv dataset_split
python prep_crops.py                             # recortes por bbox -> crops/
python baseline_embed.py                         # baseline A
python train_cnn.py --epochs 12                  # modelo B
python prep_yolo.py build && yolo detect train data=yolo_data/data.yaml model=yolo11n.pt imgsz=640 epochs=20 batch=16 workers=2 project=runs name=cow
```

Nota Windows/Anaconda: los scripts definen `KMP_DUPLICATE_LIB_OK=TRUE` (conflicto de OpenMP que tira los workers del DataLoader). No usar `channels_last` en esta GPU: hace el entrenamiento 16× más lento.

## Flujo de trabajo del equipo

1. `git pull` antes de empezar.
2. Trabajar en una rama/worktree por tarea (ej: `feature/frontend`).
3. Commit y push de la rama; merge a `main` cuando esté validado.
