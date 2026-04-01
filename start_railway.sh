#!/bin/bash
# ═══════════════════════════════════════════
# VisionC — Railway Startup
# Single-port: Backend serves API + static frontend
# ═══════════════════════════════════════════

echo "🚀 Starting VisionC on Railway (port ${PORT:-8000})..."

cd /app/backend
PYTHONPATH="/app/backend:/app" python -m uvicorn main:app \
    --host 0.0.0.0 \
    --port ${PORT:-8000} \
    --workers 1 \
    --ws-max-size 10485760 \
    --timeout-keep-alive 120
