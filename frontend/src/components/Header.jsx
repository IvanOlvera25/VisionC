"use client";

import {
  Factory,
  Wifi,
  WifiOff,
  Cpu,
  Zap,
} from "lucide-react";

export default function Header({ connected, fps }) {
  return (
    <header className="header">
      <div className="header-left">
        <div className="header-logo">
          <Factory size={28} />
        </div>
        <div className="header-text">
          <h1>VisionC</h1>
          <p>Control de Calidad Industrial — Inspección Visual en Tiempo Real</p>
        </div>
      </div>

      <div className="header-badges">
        <div className={`badge ${connected ? "badge-ok" : "badge-off"}`}>
          {connected ? <Wifi size={14} /> : <WifiOff size={14} />}
          <span>{connected ? "Conectado" : "Desconectado"}</span>
        </div>

        {fps > 0 && (
          <div className="badge badge-fps">
            <Zap size={14} />
            <span>{fps.toFixed(0)} FPS</span>
          </div>
        )}

        <div className="badge badge-device">
          <Cpu size={14} />
          <span>YOLO26</span>
        </div>
      </div>
    </header>
  );
}
