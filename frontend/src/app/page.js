"use client";

import { useState, useCallback, useEffect, useRef } from "react";
import Header from "@/components/Header";
import CameraFeed from "@/components/CameraFeed";
import StatusIndicator from "@/components/StatusIndicator";
import StatsCards from "@/components/StatsCards";
import InspectionLog from "@/components/InspectionLog";
import ControlPanel from "@/components/ControlPanel";
import GeneralInfo from "@/components/GeneralInfo";
import { useWebSocket } from "@/hooks/useWebSocket";
import { WS_BASE } from "@/lib/constants";

const DEFAULT_CONFIG = {
  model: "YOLO26-Nano",
  use_seg: true,
  confidence: 0.4,
  iou: 0.5,
  imgsz: 416,
  px_per_cm: 30,
  min_w: 3.0,
  max_w: 15.0,
  min_h: 3.0,
  max_h: 15.0,
  target_class: "Todas",
  show_all: true,
  // General mode
  task: "detection",
  show_labels: true,
  show_conf: true,
  show_boxes: true,
  line_width: 2,
  // Industrial mode (HSV Color segmentation + teeth analysis)
  ind_gear_color: "blue",
  ind_hue_low: 90,
  ind_hue_high: 130,
  ind_sat_low: 80,
  ind_sat_high: 255,
  ind_val_low: 50,
  ind_val_high: 255,
  ind_morph_kernel: 5,
  ind_min_area: 5000,
  ind_expected_teeth: 12,
  ind_tooth_tolerance: 0.35,
  ind_expected_diameter_cm: 16.0,
  ind_diameter_tolerance_cm: 15.0,
  ind_px_per_cm: 14,
};

const WS_URLS = {
  qc: `${WS_BASE}/ws/qc`,
  general: `${WS_BASE}/ws/general`,
  industrial: `${WS_BASE}/ws/industrial`,
};

export default function Home() {
  const [mode, setMode] = useState("industrial"); // Start on industrial mode
  const [config, setConfig] = useState(DEFAULT_CONFIG);
  const [streaming, setStreaming] = useState(false);

  const wsUrl = WS_URLS[mode] || WS_URLS.qc;
  const { connected, resultFrame, stateData, connect, disconnect, sendFrame, sendConfig, sendReset } = useWebSocket(wsUrl);

  // Connect WebSocket when streaming starts
  useEffect(() => {
    if (streaming) {
      connect();
    } else {
      disconnect();
    }
  }, [streaming, connect, disconnect]);

  // Send config updates to backend
  const configDebounce = useRef(null);
  const handleConfigChange = useCallback(
    (newConfig) => {
      setConfig(newConfig);
      if (configDebounce.current) clearTimeout(configDebounce.current);
      configDebounce.current = setTimeout(() => {
        sendConfig(newConfig);
      }, 200);
    },
    [sendConfig]
  );

  const handleFrameCapture = useCallback(
    (base64) => {
      sendFrame(base64);
    },
    [sendFrame]
  );

  const handleReset = useCallback(() => {
    sendReset();
  }, [sendReset]);

  // Reconnect when mode changes
  const handleModeChange = useCallback(
    (newMode) => {
      if (newMode === mode) return;
      disconnect();
      setMode(newMode);
      if (streaming) {
        setTimeout(() => connect(), 300);
      }
    },
    [mode, streaming, connect, disconnect]
  );

  const fps = stateData?.fps || 0;

  return (
    <div className="app">
      <Header connected={connected} fps={fps} />

      {/* Mode Tabs */}
      <div className="mode-tabs">
        <button
          className={`mode-tab ${mode === "industrial" ? "mode-tab-active mode-tab-industrial" : ""}`}
          onClick={() => handleModeChange("industrial")}
        >
          <span>🔩</span>
          <span>Pieza Industrial</span>
        </button>
        <button
          className={`mode-tab ${mode === "qc" ? "mode-tab-active" : ""}`}
          onClick={() => handleModeChange("qc")}
        >
          <span>🏭</span>
          <span>QC Industrial</span>
        </button>
        <button
          className={`mode-tab ${mode === "general" ? "mode-tab-active" : ""}`}
          onClick={() => handleModeChange("general")}
        >
          <span>🔬</span>
          <span>Modo General</span>
        </button>
      </div>

      {/* Main Layout */}
      <div className="main-layout">
        {/* Left — Controls */}
        <ControlPanel
          config={config}
          onConfigChange={handleConfigChange}
          onReset={handleReset}
          mode={mode}
        />

        {/* Center — Camera */}
        <main className="main-content">
          <CameraFeed
            resultFrame={resultFrame}
            onFrameCapture={handleFrameCapture}
            streaming={streaming}
            setStreaming={setStreaming}
          />
        </main>

        {/* Right — Dashboard */}
        <aside className="dashboard-panel">
          {mode === "qc" || mode === "industrial" ? (
            <>
              <StatusIndicator status={stateData?.last_status || "idle"} />
              <StatsCards stateData={stateData} mode={mode} />
              <InspectionLog log={stateData?.log} mode={mode} />
            </>
          ) : (
            <GeneralInfo info={stateData} />
          )}
        </aside>
      </div>

      {/* Footer */}
      <footer className="footer">
        <span>
          <strong>VisionC</strong> — Control de Calidad Industrial
        </span>
        <span className="footer-sep">·</span>
        <span>YOLO26 · YOLOv12 · YOLO11 · OpenCV</span>
        <span className="footer-sep">·</span>
        <a href="https://ultralytics.com" target="_blank" rel="noopener noreferrer">
          Ultralytics
        </a>
      </footer>
    </div>
  );
}
