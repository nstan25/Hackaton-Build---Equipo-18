# API + web en un solo contenedor (SPEC §2.1). Railway lo construye con railway.json.
# Solo se copia lo que la app necesita: nada de datos ni imágenes de terceros (SPEC §4.1.6).
FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.8.22 /uv /usr/local/bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:$PATH" \
    BCS_DB_PATH=/data/bcs.sqlite \
    BCS_MEDIA_DIR=/data/media

WORKDIR /app

# opencv-python-headless igual pide libGL/glib en slim
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Capa de dependencias (cambia poco): torch CPU desde el lock.
COPY pyproject.toml uv.lock .python-version README.md ./
RUN uv sync --frozen --no-dev --no-install-project --no-cache

# Código y recursos. `bcs.paths` resuelve config/ y tests/fixtures/ relativos a /app.
COPY src/ src/
COPY config/ config/
COPY web/ web/
COPY tests/fixtures/prediction_example.json tests/fixtures/prediction_example.json
RUN uv sync --frozen --no-dev --no-cache

# /data es el volumen de Railway (base SQLite + miniaturas). BCS_APP_TOKEN va en sus variables.
EXPOSE 8000
CMD ["sh", "-c", "mkdir -p \"$(dirname \"$BCS_DB_PATH\")\" \"$BCS_MEDIA_DIR\" && exec uvicorn bcs.api:app --host 0.0.0.0 --port \"${PORT:-8000}\" --proxy-headers --forwarded-allow-ips='*'"]
