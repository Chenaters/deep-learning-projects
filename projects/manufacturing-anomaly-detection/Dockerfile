# syntax=docker/dockerfile:1

FROM python:3.12-slim AS builder

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src

COPY requirements.txt ./
RUN python -m pip install --no-cache-dir --prefix=/install -r requirements.txt

COPY src ./src
RUN python -m crowe_anomaly.train \
    --data-dir /app/data \
    --output-dir /app/artifacts \
    --rows 6000 \
    --seed 42

FROM python:3.12-slim AS runtime

RUN groupadd --system app && useradd --system --gid app --home /app app
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src

COPY --from=builder /install /usr/local
COPY --from=builder /app/src ./src
COPY --from=builder /app/artifacts ./artifacts

USER app
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2)" || exit 1

CMD ["python", "-m", "crowe_anomaly.service", "--host", "0.0.0.0", "--port", "8000", "--model", "artifacts/model.joblib"]

