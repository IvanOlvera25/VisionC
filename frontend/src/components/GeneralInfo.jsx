"use client";

import { Package, Crosshair, User } from "lucide-react";

export default function GeneralInfo({ info }) {
  if (!info) {
    return (
      <div className="panel">
        <div className="panel-header">
          <span>📊</span>
          <h3>Estadísticas</h3>
        </div>
        <div className="log-empty">
          <p>Activa la cámara para comenzar</p>
        </div>
      </div>
    );
  }

  return (
    <div className="panel">
      <div className="panel-header">
        <span>📊</span>
        <h3>Estadísticas</h3>
      </div>

      <div className="general-stats">
        <div className="general-stat-row">
          <span className="general-stat-label">⚡ FPS</span>
          <span className="general-stat-value">{info.fps?.toFixed(0) || 0}</span>
        </div>
        <div className="general-stat-row">
          <span className="general-stat-label">⏱️ Latencia</span>
          <span className="general-stat-value">{info.latency_ms?.toFixed(0) || 0}ms</span>
        </div>
        <div className="general-stat-row">
          <span className="general-stat-label">🤖 Modelo</span>
          <span className="general-stat-value mono">{info.model || "--"}</span>
        </div>

        {info.task === "classification" && info.classifications && (
          <div className="general-section">
            <h4>🏷️ Clasificaciones</h4>
            <table className="log-table">
              <thead>
                <tr>
                  <th>#</th>
                  <th>Clase</th>
                  <th>Conf</th>
                </tr>
              </thead>
              <tbody>
                {info.classifications.map((c, i) => (
                  <tr key={i}>
                    <td>{i + 1}</td>
                    <td>{c.class}</td>
                    <td>{(c.conf * 100).toFixed(0)}%</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}

        {info.task === "pose" && (
          <div className="general-section">
            <div className="general-stat-row big">
              <User size={20} />
              <span>Personas detectadas</span>
              <span className="general-stat-value big">{info.persons || 0}</span>
            </div>
          </div>
        )}

        {(info.task === "detection" || info.task === "segmentation") && (
          <div className="general-section">
            <div className="general-stat-row">
              <span className="general-stat-label">📦 Objetos</span>
              <span className="general-stat-value">{info.objects || 0}</span>
            </div>
            {info.class_counts && Object.keys(info.class_counts).length > 0 && (
              <table className="log-table compact">
                <thead>
                  <tr>
                    <th>Clase</th>
                    <th>Qty</th>
                  </tr>
                </thead>
                <tbody>
                  {Object.entries(info.class_counts).map(([cls, count]) => (
                    <tr key={cls}>
                      <td>{cls}</td>
                      <td>{count}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
