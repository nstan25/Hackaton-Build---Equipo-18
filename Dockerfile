# API + web en un solo contenedor (SPEC §2.1). Railway lo construye con railway.json.
# Solo se copia lo que la app necesita: nada de datos ni imágenes de terceros (SPEC §4.1.6).
FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.8.22 /uv /usr/local/bin/uv

# BCS_PREDICTOR=model: el pipeline de G3 (T12). T28 (rama dev): el modelo de T25 (BCS_MODEL=zen)
# sobre la grilla dairy_zen (3.0–5.0) con detección OOD; BCS_MODEL=sdb BCS_SCALE=dairy_sdb vuelve al
# de main (sus archivos también van en la imagen). T30: el default es BCS_MODEL=zen_grupa (el de T25
# ajustado con recortes de grupa de zen); BCS_MODEL=zen sigue disponible.
# En runtime no se consulta el Hub.
ENV UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:$PATH" \
    BCS_DB_PATH=/data/bcs.sqlite \
    BCS_MEDIA_DIR=/data/media \
    BCS_PREDICTOR=model \
    BCS_MODEL=zen_grupa \
    BCS_SCALE=dairy_zen \
    BCS_ARTIFACTS_DIR=/app/artifacts \
    YOLO_CONFIG_DIR=/tmp/Ultralytics \
    YOLO_OFFLINE=1

WORKDIR /app

# opencv-python-headless igual pide libGL/glib en slim
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Capa de dependencias (cambia poco): torch CPU desde el lock.
COPY pyproject.toml uv.lock .python-version README.md ./
RUN uv sync --frozen --no-dev --no-install-project --no-cache \
    # Solo entrenamiento/datos: la app no los importa (~400 MB menos).
    && uv pip uninstall --python /app/.venv/bin/python datasets pyarrow polars polars-runtime-32 onnx onnxruntime remotezip openpyxl \
    && find /app/.venv -name '__pycache__' -type d -prune -exec rm -rf {} +

# Sin CLIP en producción (18/10): no entraba en la RAM de Railway. Filtro = brillo/nitidez + detector.
ENV HF_HUB_OFFLINE=1

# Código y recursos. `bcs.paths` resuelve config/, models/ y tests/fixtures/ relativos a /app.
# El código del equipo que usa el pipeline (transformación, detector, recorte) va tal cual.
COPY train_cnn.py metrics.py prep_yolo.py prep_crops.py ./
COPY models/bcs_efficientnet_b0.pt models/cow_det.pt models/calibration_efficientnet_b0.json models/
# T28: modelo de T25 (16 MB), su calibración y el banco OOD (6,3 MB). Sin el .json de metadatos.
COPY models/bcs_efficientnet_b0_zen.pt models/calibration_efficientnet_b0_zen.json models/ood_mahalanobis_zen.npz models/
# T30: el de grupa (default), su calibración y su banco OOD (+22 MB; se carga uno solo).
COPY models/bcs_efficientnet_b0_zen_grupa.pt models/calibration_efficientnet_b0_zen_grupa.json models/ood_mahalanobis_zen_grupa.npz models/
COPY src/ src/
COPY config/ config/
COPY web/ web/
COPY tests/fixtures/prediction_example.json tests/fixtures/prediction_example.json
RUN uv sync --frozen --no-dev --no-cache --no-install-package datasets --no-install-package pyarrow --no-install-package polars --no-install-package polars-runtime-32 --no-install-package onnx --no-install-package onnxruntime --no-install-package remotezip --no-install-package openpyxl

# /data es el volumen de Railway (base SQLite + miniaturas). BCS_APP_TOKEN va en sus variables.
# Al importar la app arranca la precarga de los modelos en un thread (`inference.start_warmup`).
EXPOSE 8000
CMD ["sh", "-c", "mkdir -p \"$(dirname \"$BCS_DB_PATH\")\" \"$BCS_MEDIA_DIR\" && exec uvicorn bcs.api:app --host 0.0.0.0 --port \"${PORT:-8000}\" --proxy-headers --forwarded-allow-ips='*'"]
