"use client";

import { useState } from "react";
import {
  Settings,
  Ruler,
  Target,
  RotateCcw,
  SlidersHorizontal,
  Eye,
  ChevronDown,
  ChevronUp,
  Cog,
  Palette,
} from "lucide-react";
import { MODELS, TASKS } from "@/lib/constants";

function Slider({ label, value, min, max, step, onChange, unit }) {
  const pct = ((value - min) / (max - min)) * 100;
  return (
    <div className="slider-group">
      <div className="slider-header">
        <label>{label}</label>
        <span className="slider-value">
          {typeof value === "number" && value % 1 !== 0 ? value.toFixed(2) : value}
          {unit && <span className="slider-unit">{unit}</span>}
        </span>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(parseFloat(e.target.value))}
        className="range-input"
        style={{ "--pct": `${pct}%` }}
      />
    </div>
  );
}

const GEAR_COLORS = [
  { id: "blue", label: "🔵 Azul", desc: "Engrane azul plástico" },
  { id: "black", label: "⚫ Negro", desc: "Engrane negro plástico" },
];

export default function ControlPanel({ config, onConfigChange, onReset, mode }) {
  const [sections, setSections] = useState({
    model: true,
    calibration: true,
    filter: false,
    visualization: false,
    color: true,
    tolerances: true,
    hsv: false,
  });

  const toggle = (key) =>
    setSections((prev) => ({ ...prev, [key]: !prev[key] }));

  const update = (key, val) => {
    onConfigChange({ ...config, [key]: val });
  };

  return (
    <aside className="control-panel">
      <div className="panel-header">
        <Settings size={16} />
        <h3>Controles</h3>
      </div>

      {/* ══════════════════════════════════════
         INDUSTRIAL MODE — Color HSV Segmentation
         ══════════════════════════════════════ */}
      {mode === "industrial" && (
        <>
          {/* Gear Color Selection */}
          <div className="ctrl-section">
            <button className="ctrl-section-toggle" onClick={() => toggle("color")}>
              <Palette size={14} />
              <span>Color del Engrane</span>
              {sections.color ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
            </button>
            {sections.color && (
              <div className="ctrl-section-body">
                <div className="gear-color-grid">
                  {GEAR_COLORS.map((c) => (
                    <button
                      key={c.id}
                      className={`gear-color-btn ${config.ind_gear_color === c.id ? "gear-color-active" : ""}`}
                      onClick={() => update("ind_gear_color", c.id)}
                    >
                      <span className="gear-color-label">{c.label}</span>
                      <span className="gear-color-desc">{c.desc}</span>
                    </button>
                  ))}
                </div>

                <div className="ind-model-info">
                  🎯 Segmenta por color HSV — ignora mano/cuerpo automáticamente
                </div>

                <Slider
                  label="Área mínima"
                  value={config.ind_min_area}
                  min={1000}
                  max={30000}
                  step={500}
                  unit="px"
                  onChange={(v) => update("ind_min_area", v)}
                />
                <Slider
                  label="Kernel morfológico"
                  value={config.ind_morph_kernel}
                  min={3}
                  max={15}
                  step={2}
                  onChange={(v) => update("ind_morph_kernel", v)}
                />
              </div>
            )}
          </div>

          {/* Gear Tolerances */}
          <div className="ctrl-section">
            <button className="ctrl-section-toggle" onClick={() => toggle("tolerances")}>
              <Ruler size={14} />
              <span>Tolerancias del Engrane</span>
              {sections.tolerances ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
            </button>
            {sections.tolerances && (
              <div className="ctrl-section-body">
                <Slider
                  label="Dientes esperados"
                  value={config.ind_expected_teeth}
                  min={4}
                  max={30}
                  step={1}
                  unit="T"
                  onChange={(v) => update("ind_expected_teeth", v)}
                />
                <Slider
                  label="Diámetro esperado"
                  value={config.ind_expected_diameter_cm}
                  min={1}
                  max={30}
                  step={0.5}
                  unit="cm"
                  onChange={(v) => update("ind_expected_diameter_cm", v)}
                />
                <Slider
                  label="Tolerancia diámetro"
                  value={config.ind_diameter_tolerance_cm}
                  min={0.5}
                  max={5.0}
                  step={0.5}
                  unit="±cm"
                  onChange={(v) => update("ind_diameter_tolerance_cm", v)}
                />
                <Slider
                  label="Tolerancia dientes"
                  value={config.ind_tooth_tolerance}
                  min={0.05}
                  max={0.50}
                  step={0.05}
                  onChange={(v) => update("ind_tooth_tolerance", v)}
                />
                <Slider
                  label="Píxeles por cm"
                  value={config.ind_px_per_cm}
                  min={5}
                  max={50}
                  step={1}
                  unit="px/cm"
                  onChange={(v) => update("ind_px_per_cm", v)}
                />
              </div>
            )}
          </div>

          {/* HSV Fine-tuning */}
          <div className="ctrl-section">
            <button className="ctrl-section-toggle" onClick={() => toggle("hsv")}>
              <SlidersHorizontal size={14} />
              <span>Ajuste HSV Fino</span>
              {sections.hsv ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
            </button>
            {sections.hsv && (
              <div className="ctrl-section-body">
                <div className="hsv-info">
                  🎨 Ajusta si el color por defecto no detecta bien tu pieza
                </div>
                <Slider label="Hue mín" value={config.ind_hue_low} min={0} max={180} step={1} onChange={(v) => update("ind_hue_low", v)} />
                <Slider label="Hue máx" value={config.ind_hue_high} min={0} max={180} step={1} onChange={(v) => update("ind_hue_high", v)} />
                <Slider label="Sat mín" value={config.ind_sat_low} min={0} max={255} step={5} onChange={(v) => update("ind_sat_low", v)} />
                <Slider label="Sat máx" value={config.ind_sat_high} min={0} max={255} step={5} onChange={(v) => update("ind_sat_high", v)} />
                <Slider label="Val mín" value={config.ind_val_low} min={0} max={255} step={5} onChange={(v) => update("ind_val_low", v)} />
                <Slider label="Val máx" value={config.ind_val_high} min={0} max={255} step={5} onChange={(v) => update("ind_val_high", v)} />
              </div>
            )}
          </div>
        </>
      )}

      {/* ══════════════════════════════════════
         QC & GENERAL MODES (unchanged)
         ══════════════════════════════════════ */}

      {/* Model & Detection (QC and General modes) */}
      {mode !== "industrial" && (
        <div className="ctrl-section">
          <button className="ctrl-section-toggle" onClick={() => toggle("model")}>
            <SlidersHorizontal size={14} />
            <span>Modelo & Detección</span>
            {sections.model ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
          </button>
          {sections.model && (
            <div className="ctrl-section-body">
              <div className="select-group">
                <label>Modelo</label>
                <select
                  value={config.model}
                  onChange={(e) => update("model", e.target.value)}
                  className="select-input"
                >
                  {MODELS.map((m) => (
                    <option key={m.id} value={m.id}>
                      {m.tier === "star" ? "⭐ " : ""}
                      {m.label}
                      {m.badge ? ` (${m.badge})` : ""}
                    </option>
                  ))}
                </select>
              </div>

              {mode === "general" && (
                <div className="select-group">
                  <label>Tarea</label>
                  <div className="task-grid">
                    {TASKS.map((t) => (
                      <button
                        key={t.id}
                        className={`task-btn ${config.task === t.id ? "task-active" : ""}`}
                        onClick={() => update("task", t.id)}
                      >
                        <span>{t.icon}</span>
                        <span>{t.label}</span>
                      </button>
                    ))}
                  </div>
                </div>
              )}

              {mode === "qc" && (
                <div className="switch-group">
                  <label>
                    <span>🎭 Segmentación</span>
                    <input type="checkbox" checked={config.use_seg} onChange={(e) => update("use_seg", e.target.checked)} className="toggle-input" />
                    <div className="toggle-track"><div className="toggle-thumb" /></div>
                  </label>
                </div>
              )}

              <Slider label="Confianza" value={config.confidence} min={0.15} max={0.95} step={0.05} onChange={(v) => update("confidence", v)} />
              <Slider label="IoU (NMS)" value={config.iou} min={0.1} max={1.0} step={0.05} onChange={(v) => update("iou", v)} />
              <Slider label="Resolución" value={config.imgsz} min={192} max={640} step={64} unit="px" onChange={(v) => update("imgsz", v)} />
            </div>
          )}
        </div>
      )}

      {/* Calibration (QC only) */}
      {mode === "qc" && (
        <div className="ctrl-section">
          <button className="ctrl-section-toggle" onClick={() => toggle("calibration")}>
            <Ruler size={14} />
            <span>Calibración & Tolerancias</span>
            {sections.calibration ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
          </button>
          {sections.calibration && (
            <div className="ctrl-section-body">
              <Slider label="Píxeles por cm" value={config.px_per_cm} min={5} max={100} step={1} unit="px/cm" onChange={(v) => update("px_per_cm", v)} />
              <div className="tolerance-group">
                <span className="tolerance-title">📏 Ancho permitido (cm)</span>
                <div className="tolerance-row">
                  <Slider label="Mín" value={config.min_w} min={0.5} max={50} step={0.5} unit="cm" onChange={(v) => update("min_w", v)} />
                  <Slider label="Máx" value={config.max_w} min={0.5} max={100} step={0.5} unit="cm" onChange={(v) => update("max_w", v)} />
                </div>
              </div>
              <div className="tolerance-group">
                <span className="tolerance-title">📐 Alto permitido (cm)</span>
                <div className="tolerance-row">
                  <Slider label="Mín" value={config.min_h} min={0.5} max={50} step={0.5} unit="cm" onChange={(v) => update("min_h", v)} />
                  <Slider label="Máx" value={config.max_h} min={0.5} max={100} step={0.5} unit="cm" onChange={(v) => update("max_h", v)} />
                </div>
              </div>
            </div>
          )}
        </div>
      )}

      {/* Class Filter (QC only) */}
      {mode === "qc" && (
        <div className="ctrl-section">
          <button className="ctrl-section-toggle" onClick={() => toggle("filter")}>
            <Target size={14} />
            <span>Filtro de Clase</span>
            {sections.filter ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
          </button>
          {sections.filter && (
            <div className="ctrl-section-body">
              <div className="select-group">
                <label>Clase a inspeccionar</label>
                <select value={config.target_class} onChange={(e) => update("target_class", e.target.value)} className="select-input">
                  <option value="Todas">Todas las clases</option>
                  <option value="person">person</option>
                  <option value="bottle">bottle</option>
                  <option value="cup">cup</option>
                  <option value="cell phone">cell phone</option>
                  <option value="book">book</option>
                  <option value="laptop">laptop</option>
                </select>
              </div>
              <div className="switch-group">
                <label>
                  <span>Mostrar otros objetos</span>
                  <input type="checkbox" checked={config.show_all} onChange={(e) => update("show_all", e.target.checked)} className="toggle-input" />
                  <div className="toggle-track"><div className="toggle-thumb" /></div>
                </label>
              </div>
            </div>
          )}
        </div>
      )}

      {/* Visualization (General only) */}
      {mode === "general" && (
        <div className="ctrl-section">
          <button className="ctrl-section-toggle" onClick={() => toggle("visualization")}>
            <Eye size={14} />
            <span>Visualización</span>
            {sections.visualization ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
          </button>
          {sections.visualization && (
            <div className="ctrl-section-body">
              <div className="switch-group">
                <label><span>Etiquetas</span><input type="checkbox" checked={config.show_labels} onChange={(e) => update("show_labels", e.target.checked)} className="toggle-input" /><div className="toggle-track"><div className="toggle-thumb" /></div></label>
              </div>
              <div className="switch-group">
                <label><span>Confianza</span><input type="checkbox" checked={config.show_conf} onChange={(e) => update("show_conf", e.target.checked)} className="toggle-input" /><div className="toggle-track"><div className="toggle-thumb" /></div></label>
              </div>
              <div className="switch-group">
                <label><span>Bounding Boxes</span><input type="checkbox" checked={config.show_boxes} onChange={(e) => update("show_boxes", e.target.checked)} className="toggle-input" /><div className="toggle-track"><div className="toggle-thumb" /></div></label>
              </div>
              <Slider label="Grosor línea" value={config.line_width} min={1} max={5} step={1} unit="px" onChange={(v) => update("line_width", v)} />
            </div>
          )}
        </div>
      )}

      {/* Reset */}
      {(mode === "qc" || mode === "industrial") && (
        <button className="btn btn-secondary btn-full" onClick={onReset}>
          <RotateCcw size={16} />
          <span>Reiniciar Contadores</span>
        </button>
      )}
    </aside>
  );
}
