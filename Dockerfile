# ── Stage 1: Build stage (compiles pyswisseph C extension) ────────────────────
FROM python:3.11-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential gcc g++ libpq-dev curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt


# ── Stage 2: Minimal runtime image ────────────────────────────────────────────
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        libpq5 curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=builder /install /usr/local
COPY . .

RUN chmod +x start_railway.sh

# Ephemeris files and logs are mounted as Railway volumes at runtime
RUN mkdir -p /app/logs /app/ephe

EXPOSE 8000

ENV BOT_DIR=/app \
    PYTHONUNBUFFERED=1

CMD ["bash", "start_railway.sh"]
