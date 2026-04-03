# ═══════════════════════════════════════════════
# VisionC — Single-service Dockerfile for Railway
# Stage 1: Build Next.js static frontend
# Stage 2: Python backend serves API + static files
# ═══════════════════════════════════════════════

# ── Stage 1: Build frontend ──
FROM node:20-slim AS frontend
WORKDIR /build
COPY frontend/package*.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

# ── Stage 2: Python production ──
FROM python:3.11-slim-bookworm

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libglib2.0-0 libsm6 libxext6 libxrender1 && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip install --no-cache-dir \
    fastapi>=0.110.0 \
    "uvicorn[standard]>=0.27.0" \
    websockets>=12.0 \
    opencv-python-headless>=4.9.0 \
    numpy>=1.26.0 \
    scipy>=1.12.0

# Copy backend
COPY backend/ /app/backend/

# Copy built frontend static files
COPY --from=frontend /build/out /app/static

ENV PYTHONPATH=/app/backend:/app

CMD python -m uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1 --ws-max-size 10485760 --timeout-keep-alive 120
