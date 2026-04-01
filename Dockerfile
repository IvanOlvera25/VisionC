# ═══════════════════════════════════════════════
# VisionC Backend — Dockerfile for Railway
# Python + OpenCV + FastAPI (WebSocket server)
# ═══════════════════════════════════════════════
FROM python:3.11-slim

# System deps for OpenCV headless
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Python deps (no torch/YOLO — industrial mode only needs OpenCV + scipy)
RUN pip install --no-cache-dir \
    fastapi>=0.110.0 \
    "uvicorn[standard]>=0.27.0" \
    websockets>=12.0 \
    opencv-python-headless>=4.9.0 \
    numpy>=1.26.0 \
    scipy>=1.12.0

# Copy backend code
COPY backend/ /app/backend/

ENV PYTHONPATH=/app/backend:/app

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "backend.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1", \
     "--ws-max-size", "10485760", \
     "--timeout-keep-alive", "120"]
