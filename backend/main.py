"""
🔥 VisionC — FastAPI Backend
WebSocket server for real-time YOLO processing.
"""

import asyncio
import base64
import json
import sys
import os

import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Add parent dir so we can find model weights
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qc_engine import QCEngine, MODELS, COCO_CLASSES, TASK_SUFFIXES, INDUSTRIAL_PRESETS

app = FastAPI(title="VisionC API", version="2.0")

# CORS for local dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Shared engine instance
engine = QCEngine()


# ──────────────────────────────────────────────
# REST endpoints
# ──────────────────────────────────────────────
@app.get("/api/models")
def list_models():
    return {
        "models": list(MODELS.keys()),
        "tasks": list(TASK_SUFFIXES.keys()),
        "classes": COCO_CLASSES,
    }


@app.get("/api/presets")
def list_presets():
    return {"presets": INDUSTRIAL_PRESETS}


@app.get("/api/state")
def get_state():
    return engine.get_state()


class ConfigUpdate(BaseModel):
    config: dict


@app.post("/api/config")
def update_config(body: ConfigUpdate):
    engine.update_config(body.config)
    return {"status": "ok", "config": engine.config}


@app.post("/api/reset")
def reset_counters():
    engine.reset()
    return {"status": "ok", "message": "Counters reset"}


# ──────────────────────────────────────────────
# WebSocket — QC mode
# ──────────────────────────────────────────────
@app.websocket("/ws/qc")
async def ws_qc(websocket: WebSocket):
    await websocket.accept()
    print("🔌 QC WebSocket connected")

    try:
        while True:
            data = await websocket.receive_text()
            msg = json.loads(data)

            if msg.get("type") == "config":
                engine.update_config(msg.get("config", {}))
                await websocket.send_text(json.dumps({"type": "config_ack"}))
                continue

            if msg.get("type") == "reset":
                engine.reset()
                await websocket.send_text(json.dumps({
                    "type": "reset_ack",
                    "state": engine.get_state(),
                }))
                continue

            if msg.get("type") == "frame":
                # Decode base64 JPEG frame
                img_data = base64.b64decode(msg["data"])
                nparr = np.frombuffer(img_data, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                if frame is None:
                    continue

                # Process
                annotated, state_data = engine.process_qc(frame)

                # Encode result as JPEG
                _, buffer = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])
                result_b64 = base64.b64encode(buffer).decode("utf-8")

                await websocket.send_text(json.dumps({
                    "type": "result",
                    "frame": result_b64,
                    "state": state_data,
                }))

    except WebSocketDisconnect:
        print("🔌 QC WebSocket disconnected")
    except Exception as e:
        print(f"❌ QC WebSocket error: {e}")


# ──────────────────────────────────────────────
# WebSocket — General mode
# ──────────────────────────────────────────────
@app.websocket("/ws/general")
async def ws_general(websocket: WebSocket):
    await websocket.accept()
    print("🔌 General WebSocket connected")

    try:
        while True:
            data = await websocket.receive_text()
            msg = json.loads(data)

            if msg.get("type") == "config":
                engine.update_config(msg.get("config", {}))
                await websocket.send_text(json.dumps({"type": "config_ack"}))
                continue

            if msg.get("type") == "frame":
                img_data = base64.b64decode(msg["data"])
                nparr = np.frombuffer(img_data, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                if frame is None:
                    continue

                annotated, info = engine.process_general(frame)

                _, buffer = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])
                result_b64 = base64.b64encode(buffer).decode("utf-8")

                await websocket.send_text(json.dumps({
                    "type": "result",
                    "frame": result_b64,
                    "info": info,
                }))

    except WebSocketDisconnect:
        print("🔌 General WebSocket disconnected")
    except Exception as e:
        print(f"❌ General WebSocket error: {e}")


# ──────────────────────────────────────────────
# WebSocket — Industrial mode (Gear detection)
# ──────────────────────────────────────────────
@app.websocket("/ws/industrial")
async def ws_industrial(websocket: WebSocket):
    await websocket.accept()
    print("🔌 Industrial WebSocket connected")

    try:
        while True:
            data = await websocket.receive_text()
            msg = json.loads(data)

            if msg.get("type") == "config":
                engine.update_config(msg.get("config", {}))
                await websocket.send_text(json.dumps({"type": "config_ack"}))
                continue

            if msg.get("type") == "reset":
                engine.reset()
                await websocket.send_text(json.dumps({
                    "type": "reset_ack",
                    "state": engine.get_state(),
                }))
                continue

            if msg.get("type") == "frame":
                img_data = base64.b64decode(msg["data"])
                nparr = np.frombuffer(img_data, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                if frame is None:
                    continue

                # Use the industrial processing pipeline
                annotated, state_data = engine.process_industrial(frame)

                _, buffer = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])
                result_b64 = base64.b64encode(buffer).decode("utf-8")

                await websocket.send_text(json.dumps({
                    "type": "result",
                    "frame": result_b64,
                    "state": state_data,
                }))

    except WebSocketDisconnect:
        print("🔌 Industrial WebSocket disconnected")
    except Exception as e:
        print(f"❌ Industrial WebSocket error: {e}")


if __name__ == "__main__":
    import uvicorn
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)

