#!/bin/bash
# ═══════════════════════════════════════════
# VisionC — Start Script
# Launches backend (FastAPI) and frontend (Next.js)
# ═══════════════════════════════════════════

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
echo "🏭 VisionC — Control de Calidad Industrial"
echo "============================================"
echo ""

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

# ── Start Backend ──
echo -e "${BLUE}🔧 Starting Backend (FastAPI on :8000)...${NC}"
cd "$SCRIPT_DIR"

# Activate venv if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
    echo -e "${GREEN}   ✅ Virtual environment activated${NC}"
fi

# Install backend deps if needed
if ! python -c "import fastapi" 2>/dev/null; then
    echo -e "${YELLOW}   ⏳ Installing backend dependencies...${NC}"
    pip install -r backend/requirements.txt -q
fi

# Launch backend in background
cd "$SCRIPT_DIR"
python -c "
import os, sys
os.chdir('$SCRIPT_DIR')
sys.path.insert(0, '$SCRIPT_DIR/backend')
" 2>/dev/null

cd "$SCRIPT_DIR/backend"
PYTHONPATH="$SCRIPT_DIR/backend:$SCRIPT_DIR" python -m uvicorn main:app --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!
echo -e "${GREEN}   ✅ Backend started (PID: $BACKEND_PID)${NC}"

# ── Start Frontend ──
echo ""
echo -e "${BLUE}🎨 Starting Frontend (Next.js on :3000)...${NC}"
cd "$SCRIPT_DIR/frontend"

# Install deps if needed
if [ ! -d "node_modules" ]; then
    echo -e "${YELLOW}   ⏳ Installing frontend dependencies...${NC}"
    npm install --silent
fi

npm run dev &
FRONTEND_PID=$!
echo -e "${GREEN}   ✅ Frontend started (PID: $FRONTEND_PID)${NC}"

# ── Info ──
echo ""
echo "============================================"
echo -e "${GREEN}🚀 VisionC is running!${NC}"
echo ""
echo -e "   Frontend:  ${BLUE}http://localhost:3000${NC}"
echo -e "   Backend:   ${BLUE}http://localhost:8000${NC}"
echo -e "   API Docs:  ${BLUE}http://localhost:8000/docs${NC}"
echo ""
echo "   Press Ctrl+C to stop both servers"
echo "============================================"

# ── Cleanup on exit ──
cleanup() {
    echo ""
    echo -e "${YELLOW}🛑 Shutting down VisionC...${NC}"
    kill $BACKEND_PID 2>/dev/null
    kill $FRONTEND_PID 2>/dev/null
    wait $BACKEND_PID 2>/dev/null
    wait $FRONTEND_PID 2>/dev/null
    echo -e "${GREEN}✅ VisionC stopped.${NC}"
}

trap cleanup EXIT INT TERM

# Wait for both processes
wait
