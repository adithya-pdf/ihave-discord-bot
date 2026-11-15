FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# A fixed non-root UID keeps bind-mounted data directories predictable.
RUN groupadd --gid 1000 appuser \
    && useradd --uid 1000 --gid 1000 --create-home appuser

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Only the runtime modules: tests, dev requirements and .env never enter the image.
COPY bot.py predictor.py db.py cf_data.py ./

# Pre-create the persistent data directory with the right ownership.
RUN mkdir -p /app/data && chown -R appuser:appuser /app

ENV BOT_DATA_DIR=/app/data

USER appuser

CMD ["python", "bot.py"]
