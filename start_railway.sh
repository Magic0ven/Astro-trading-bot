#!/bin/bash
# Railway startup script for the Astro-Bot trading engine.
# The dashboard API runs as a separate Railway service (astro-bot-frontend/backend).
set -e

echo "=== Astro-Bot Railway Startup ==="
echo "Python: $(python3 --version)"
echo "Storage: ${DATABASE_URL:+postgres}${DATABASE_URL:-local files}"

# ── Ephemeris setup ────────────────────────────────────────────────────────────
mkdir -p /app/ephe /app/logs

if [ ! -f "/app/ephe/seas_18.se1" ]; then
  echo "[ephe] Downloading Swiss Ephemeris files..."
  BASE="https://www.astro.com/ftp/swisseph/ephe"
  for f in seas_18.se1 semo_18.se1 sepl_18.se1; do
    curl -sSL "${BASE}/${f}" -o "/app/ephe/${f}" && echo "  Downloaded ${f}"
  done
  echo "[ephe] Done."
else
  echo "[ephe] Files already present — skipping download."
fi

# ── Trading bot ────────────────────────────────────────────────────────────────
echo "[bot] Starting trading engine..."
exec python3 main.py
