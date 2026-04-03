// Auto-detect backend URL from browser location
// Dev: localhost:3000 → backend at localhost:8000
// Production: same origin (FastAPI serves both API and static files)
const isBrowser = typeof window !== "undefined";
const loc = isBrowser ? window.location : { hostname: "localhost", protocol: "http:", port: "3000" };

const isDev = loc.hostname === "localhost" && loc.port === "3000";
const API_HOST = isDev ? "http://localhost:8000" : `${loc.protocol}//${loc.host}`;
const WS_PROTOCOL = loc.protocol === "https:" ? "wss" : "ws";
const WS_HOST = isDev ? "localhost:8000" : loc.host;

export const API_BASE = API_HOST;
export const WS_BASE = `${WS_PROTOCOL}://${WS_HOST}`;

export const MODELS = [
  { id: "YOLO26-Nano", label: "YOLO26 Nano", badge: "Rápido", tier: "star" },
  { id: "YOLO26-Small", label: "YOLO26 Small", badge: null, tier: "star" },
  { id: "YOLO26-Medium", label: "YOLO26 Medium", badge: null, tier: "star" },
  { id: "YOLO26-Large", label: "YOLO26 Large", badge: "Preciso", tier: "star" },
  { id: "YOLO11-Nano", label: "YOLO11 Nano", badge: null, tier: "standard" },
  { id: "YOLO11-Small", label: "YOLO11 Small", badge: null, tier: "standard" },
  { id: "YOLOv12-Nano", label: "YOLOv12 Nano", badge: null, tier: "standard" },
  { id: "YOLOv12-Small", label: "YOLOv12 Small", badge: null, tier: "standard" },
];

export const TASKS = [
  { id: "detection", label: "Detección", icon: "🎯" },
  { id: "segmentation", label: "Segmentación", icon: "🎭" },
  { id: "pose", label: "Pose", icon: "🤸" },
  { id: "classification", label: "Clasificación", icon: "🏷️" },
];

export const INDUSTRIAL_PRESETS = [
  {
    id: "gear_large",
    label: "🔩 Engrane Grande",
    description: "Tamaño de mano · 8–25 cm",
    icon: "🔩",
    defaults: {
      ind_min_diameter: 8.0,
      ind_max_diameter: 25.0,
      ind_min_circularity: 0.30,
      ind_min_area_px: 3000,
      ind_detect_teeth: true,
      ind_piece_type: "engrane",
    },
  },
  {
    id: "gear_small",
    label: "⚙️ Engrane Pequeño",
    description: "Precisión · 2–8 cm",
    icon: "⚙️",
    defaults: {
      ind_min_diameter: 2.0,
      ind_max_diameter: 8.0,
      ind_min_circularity: 0.35,
      ind_min_area_px: 800,
      ind_detect_teeth: true,
      ind_piece_type: "engrane",
    },
  },
  {
    id: "rectangular",
    label: "📐 Pieza Rectangular",
    description: "Placas, brackets",
    icon: "📐",
    defaults: {
      ind_min_diameter: 2.0,
      ind_max_diameter: 30.0,
      ind_min_circularity: 0.0,
      ind_min_area_px: 1500,
      ind_detect_teeth: false,
      ind_piece_type: "rectangular",
    },
  },
  {
    id: "auto",
    label: "🔍 Auto-detectar",
    description: "Detecta forma automática",
    icon: "🔍",
    defaults: {
      ind_min_diameter: 1.0,
      ind_max_diameter: 40.0,
      ind_min_circularity: 0.0,
      ind_min_area_px: 1000,
      ind_detect_teeth: true,
      ind_piece_type: "auto",
    },
  },
];

export const BG_METHODS = [
  { id: "auto", label: "Auto-detectar" },
  { id: "light", label: "Fondo claro" },
  { id: "dark", label: "Fondo oscuro" },
];

export const STATUS_MAP = {
  idle: {
    icon: "⏳",
    label: "ESPERANDO",
    sub: "Sin piezas detectadas",
    cssClass: "idle",
  },
  ok: {
    icon: "✅",
    label: "APROBADA",
    sub: "Pieza dentro de especificación",
    cssClass: "ok",
  },
  rejected: {
    icon: "🚨",
    label: "RECHAZADA",
    sub: "Pieza fuera de tolerancia",
    cssClass: "fail",
  },
};
