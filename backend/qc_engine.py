"""
🔥 VisionC — QC Engine
Encapsulates all YOLO processing and QC logic for the FastAPI backend.
Includes industrial gear detection via OpenCV contour analysis.
"""

import cv2
import numpy as np
import time
from collections import defaultdict, deque
import math

# Optional: torch + ultralytics (only needed for QC/General modes, not Industrial)
try:
    import torch
    from ultralytics import YOLO
    HAS_YOLO = True
except ImportError:
    HAS_YOLO = False
    torch = None
    YOLO = None

# ──────────────────────────────────────────────
# Device detection
# ──────────────────────────────────────────────
if HAS_YOLO:
    if torch.backends.mps.is_available():
        DEVICE = "mps"
        USE_HALF = False
    elif torch.cuda.is_available():
        DEVICE = "cuda"
        USE_HALF = True
    else:
        DEVICE = "cpu"
        USE_HALF = False
    print(f"🖥️ Dispositivo: {DEVICE} | FP16: {USE_HALF}")
else:
    DEVICE = "cpu"
    USE_HALF = False
    print("🖥️ Modo ligero (sin YOLO) — solo Industrial disponible")

# ──────────────────────────────────────────────
# Available models
# ──────────────────────────────────────────────
MODELS = {
    "YOLO26-Nano": "yolo26n",
    "YOLO26-Small": "yolo26s",
    "YOLO26-Medium": "yolo26m",
    "YOLO26-Large": "yolo26l",
    "YOLO11-Nano": "yolo11n",
    "YOLO11-Small": "yolo11s",
    "YOLOv12-Nano": "yolov12n",
    "YOLOv12-Small": "yolov12s",
}

TASK_SUFFIXES = {
    "detection": "",
    "segmentation": "-seg",
    "pose": "-pose",
    "classification": "-cls",
}

# COCO classes
COCO_CLASSES = [
    "Todas", "person", "bicycle", "car", "motorcycle", "airplane", "bus",
    "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag",
    "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana",
    "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza",
    "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table",
    "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock",
    "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
]

# ──────────────────────────────────────────────
# Industrial Presets
# ──────────────────────────────────────────────
INDUSTRIAL_PRESETS = {
    "gear_large": {
        "label": "🔩 Engrane Grande",
        "description": "Engrane industrial tamaño mano (8–25 cm)",
        "min_diameter": 8.0,
        "max_diameter": 25.0,
        "min_circularity": 0.30,
        "min_area_px": 3000,
        "detect_teeth": True,
        "piece_type": "engrane",
    },
    "gear_small": {
        "label": "⚙️ Engrane Pequeño",
        "description": "Engrane de precisión (2–8 cm)",
        "min_diameter": 2.0,
        "max_diameter": 8.0,
        "min_circularity": 0.35,
        "min_area_px": 800,
        "detect_teeth": True,
        "piece_type": "engrane",
    },
    "rectangular": {
        "label": "📐 Pieza Rectangular",
        "description": "Placas, brackets, piezas planas",
        "min_diameter": 2.0,
        "max_diameter": 30.0,
        "min_circularity": 0.0,
        "min_area_px": 1500,
        "detect_teeth": False,
        "piece_type": "rectangular",
    },
    "auto": {
        "label": "🔍 Auto-detectar",
        "description": "Detecta forma automáticamente",
        "min_diameter": 1.0,
        "max_diameter": 40.0,
        "min_circularity": 0.0,
        "min_area_px": 1000,
        "detect_teeth": True,
        "piece_type": "auto",
    },
}

# Colors (BGR for OpenCV)
CLR_OK = (72, 199, 142)
CLR_FAIL = (80, 80, 255)
CLR_WHITE = (255, 255, 255)
CLR_CYAN = (255, 200, 0)
CLR_YELLOW = (0, 220, 255)
CLR_ORANGE = (0, 165, 255)
CLR_GEAR_RING = (255, 180, 50)
CLR_CENTER = (100, 255, 255)
CLR_TEETH = (180, 130, 255)


class QCEngine:
    """Encapsulates QC processing state and model management."""

    def __init__(self):
        self.model_cache = {}
        self.state = {
            "total": 0,
            "passed": 0,
            "rejected": 0,
            "log": deque(maxlen=20),
            "last_status": "idle",
            "fps_buf": deque(maxlen=10),
        }
        self.config = {
            "model": "YOLO26-Nano",
            "use_seg": True,
            "confidence": 0.40,
            "iou": 0.50,
            "imgsz": 416,
            "px_per_cm": 30,
            "min_w": 3.0,
            "max_w": 15.0,
            "min_h": 3.0,
            "max_h": 15.0,
            "target_class": "Todas",
            "show_all": True,
            # General mode
            "task": "detection",
            "show_labels": True,
            "show_conf": True,
            "show_boxes": True,
            "line_width": 2,
            # Industrial mode (Color HSV segmentation + teeth analysis)
            "ind_gear_color": "blue",         # "blue" or "black"
            # HSV ranges for BLUE gear
            "ind_hue_low": 90,
            "ind_hue_high": 130,
            "ind_sat_low": 80,
            "ind_sat_high": 255,
            "ind_val_low": 50,
            "ind_val_high": 255,
            # Morphological cleanup
            "ind_morph_kernel": 5,
            "ind_min_area": 5000,             # min contour area in px
            # Gear tolerances
            "ind_expected_teeth": 12,
            "ind_tooth_tolerance": 0.35,      # 35% deviation = defect
            "ind_expected_diameter_cm": 16.0,
            "ind_diameter_tolerance_cm": 5.0,
            "ind_px_per_cm": 14,              # ~14px/cm at 40cm
        }

    def _get_model(self, key: str, seg: bool) -> YOLO:
        suffix = "-seg" if seg else ""
        base = MODELS.get(key, "yolo26n")
        name = f"{base}{suffix}.pt"
        if name not in self.model_cache:
            print(f"⏳ Cargando {name}...")
            self.model_cache[name] = YOLO(name)
            print(f"✅ {name} listo")
        return self.model_cache[name]

    def _get_model_general(self, key: str, task: str) -> YOLO:
        suffix = TASK_SUFFIXES.get(task, "")
        base = MODELS.get(key, "yolo26n")
        name = f"{base}{suffix}.pt"
        if name not in self.model_cache:
            print(f"⏳ Cargando {name}...")
            self.model_cache[name] = YOLO(name)
            print(f"✅ {name} listo")
        return self.model_cache[name]

    def update_config(self, new_config: dict):
        self.config.update(new_config)

    def reset(self):
        self.state["total"] = 0
        self.state["passed"] = 0
        self.state["rejected"] = 0
        self.state["log"].clear()
        self.state["last_status"] = "idle"

    def get_state(self) -> dict:
        return {
            "total": self.state["total"],
            "passed": self.state["passed"],
            "rejected": self.state["rejected"],
            "rate": (self.state["rejected"] / self.state["total"] * 100)
            if self.state["total"] > 0 else 0,
            "last_status": self.state["last_status"],
            "log": list(self.state["log"]),
            "fps": float(np.mean(self.state["fps_buf"])) if self.state["fps_buf"] else 0,
        }

    # ──────────────────────────────────────────
    # Contour-Based Polar Teeth Analysis
    # ──────────────────────────────────────────
    def _analyze_teeth_from_contour(self, contour, px_per_cm, tolerance=0.25):
        """Analyze gear teeth directly from the external contour.
        
        Converts contour points to polar coordinates (angle, radius),
        then uses scipy.signal.find_peaks for robust peak detection.
        Much faster and more accurate than ray-marching.
        """
        from scipy.signal import find_peaks
        
        area = cv2.contourArea(contour)
        if area < 500 or len(contour) < 50:
            return None

        # Centroid
        M = cv2.moments(contour)
        if M["m00"] == 0:
            return None
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

        _, enc_radius = cv2.minEnclosingCircle(contour)
        enc_radius = float(enc_radius)
        bx, by, bw, bh = cv2.boundingRect(contour)

        # ── Convert contour points to polar ──
        pts = contour.reshape(-1, 2).astype(np.float64)
        dx = pts[:, 0] - cx
        dy = -(pts[:, 1] - cy)  # flip Y
        radii_raw = np.sqrt(dx * dx + dy * dy)
        angles_raw = np.arctan2(dy, dx) % (2 * np.pi)

        # Sort by angle
        order = np.argsort(angles_raw)
        angles_sorted = angles_raw[order]
        radii_sorted = radii_raw[order]

        # Resample to uniform angular spacing (720 samples = 0.5° each)
        n_samples = 720
        uniform_angles = np.linspace(0, 2 * np.pi, n_samples, endpoint=False)
        radii_uniform = np.interp(
            uniform_angles, angles_sorted, radii_sorted, period=2 * np.pi
        )

        # Smooth to reduce noise but preserve tooth shape
        from scipy.ndimage import gaussian_filter1d
        radii_smooth = gaussian_filter1d(radii_uniform, sigma=4, mode='wrap')

        mean_r = np.mean(radii_smooth)
        if mean_r < 10:
            return None

        # ── Find peaks (tooth tips) ──
        # Tuned for 6 external teeth at ~60° spacing
        min_prominence = mean_r * 0.02
        min_distance = n_samples // 10  # ~36° → gives 6 peaks

        peaks, properties = find_peaks(
            radii_smooth,
            prominence=min_prominence,
            distance=min_distance,
            height=mean_r * 0.5,
        )

        # Find valleys (internal teeth / notches between external teeth)
        valleys, _ = find_peaks(
            -radii_smooth,
            prominence=min_prominence * 0.5,
            distance=min_distance,
        )

        # ── Build teeth list ──
        teeth = []
        for i, pk in enumerate(peaks):
            angle = uniform_angles[pk]
            r = radii_smooth[pk]
            tip_x = int(cx + r * math.cos(angle))
            tip_y = int(cy - r * math.sin(angle))

            # Tooth angular width: distance to nearest valleys
            left_valleys = valleys[valleys < pk]
            right_valleys = valleys[valleys > pk]
            
            if len(left_valleys) > 0 and len(right_valleys) > 0:
                span = (right_valleys[0] - left_valleys[-1]) / n_samples * 360
            else:
                span = 360.0 / max(len(peaks), 1)

            teeth.append({
                "peak_r": float(r),
                "peak_angle": float(angle),
                "tip_x": tip_x,
                "tip_y": tip_y,
                "height": float(r - mean_r),
                "angular_span": float(span),
                "peak_idx": int(pk),
            })

        # ── Detect irregularities (robust: median + IQR) ──
        defects = []
        if len(teeth) >= 3:
            heights = np.array([t["height"] for t in teeth])
            spans = np.array([t["angular_span"] for t in teeth])
            
            # Use median + IQR for robustness (outlier-resistant)
            med_h = float(np.median(heights))
            q1_h, q3_h = float(np.percentile(heights, 25)), float(np.percentile(heights, 75))
            iqr_h = q3_h - q1_h
            
            med_span = float(np.median(spans))
            q1_s, q3_s = float(np.percentile(spans, 25)), float(np.percentile(spans, 75))
            iqr_s = q3_s - q1_s

            for i, tooth in enumerate(teeth):
                issues = []
                
                # Height outlier: outside median ± tolerance * IQR 
                if iqr_h > 0:
                    h_score = abs(tooth["height"] - med_h) / iqr_h
                    if h_score > 2.0:  # >2 IQR = strong outlier
                        issues.append(f"altura {'↑' if tooth['height'] > med_h else '↓'}")
                elif med_h != 0:
                    h_dev = abs(tooth["height"] - med_h) / abs(med_h)
                    if h_dev > tolerance:
                        issues.append(f"altura {'↑' if tooth['height'] > med_h else '↓'} {h_dev:.0%}")

                # Span outlier
                if iqr_s > 0:
                    s_score = abs(tooth["angular_span"] - med_span) / iqr_s
                    if s_score > 2.0:
                        issues.append(f"ancho {'↑' if tooth['angular_span'] > med_span else '↓'}")
                elif med_span > 0:
                    w_dev = abs(tooth["angular_span"] - med_span) / med_span
                    if w_dev > tolerance * 1.5:
                        issues.append(f"ancho {'↑' if tooth['angular_span'] > med_span else '↓'} {w_dev:.0%}")

                if issues:
                    defects.append({
                        "tooth_idx": i,
                        "tip_x": tooth["tip_x"],
                        "tip_y": tooth["tip_y"],
                        "issues": issues,
                    })
                    teeth[i]["ok"] = False
                else:
                    teeth[i]["ok"] = True

            # Safety: if >50% teeth are "defective", the baseline is wrong — reset
            if len(defects) > len(teeth) * 0.5:
                defects = []
                for i in range(len(teeth)):
                    teeth[i]["ok"] = True

        return {
            "contour": contour,
            "center": (cx, cy),
            "enc_radius": enc_radius,
            "bbox": (bx, by, bw, bh),
            "diameter_cm": round(enc_radius * 2 / px_per_cm, 1),
            "teeth": teeth,
            "teeth_count": len(teeth),
            "defects": defects,
            "mean_radius": float(mean_r),
            "radii_smooth": radii_smooth,
            "uniform_angles": uniform_angles,
            "peaks": peaks,
            "valleys": valleys,
        }

    # ──────────────────────────────────────────
    # Industrial Processing Pipeline (Color HSV)
    # ──────────────────────────────────────────
    def process_industrial(self, frame: np.ndarray) -> tuple:
        """Process frame with HSV color segmentation + contour polar teeth analysis.
        
        Supports: Blue gear, Black gear.
        Uses contour-based polar profiling with scipy find_peaks.
        """
        cfg = self.config
        t0 = time.perf_counter()

        H, W = frame.shape[:2]
        annotated = frame.copy()
        px_per_cm = cfg.get("ind_px_per_cm", 14)
        tolerance = cfg.get("ind_tooth_tolerance", 0.35)

        # ── Step 1: Pre-blur + HSV Color Segmentation ──
        blurred = cv2.bilateralFilter(frame, d=7, sigmaColor=60, sigmaSpace=60)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gear_color = cfg.get("ind_gear_color", "blue")

        if gear_color == "blue":
            lower = np.array([
                int(cfg.get("ind_hue_low", 90)),
                int(cfg.get("ind_sat_low", 80)),
                int(cfg.get("ind_val_low", 50)),
            ], dtype=np.uint8)
            upper = np.array([
                int(cfg.get("ind_hue_high", 130)),
                int(cfg.get("ind_sat_high", 255)),
                int(cfg.get("ind_val_high", 255)),
            ], dtype=np.uint8)
            color_mask = cv2.inRange(hsv, lower, upper)

            # Gentle skin subtraction — only obvious skin, minimal dilation
            skin_lower1 = np.array([0, 50, 80], dtype=np.uint8)
            skin_upper1 = np.array([20, 180, 255], dtype=np.uint8)
            skin_lower2 = np.array([165, 50, 80], dtype=np.uint8)
            skin_upper2 = np.array([180, 180, 255], dtype=np.uint8)
            skin_mask = cv2.bitwise_or(
                cv2.inRange(hsv, skin_lower1, skin_upper1),
                cv2.inRange(hsv, skin_lower2, skin_upper2)
            )
            # Minimal dilation — just 1 iter with small kernel
            skin_mask = cv2.dilate(skin_mask, np.ones((3, 3), np.uint8), iterations=1)
            color_mask = cv2.bitwise_and(color_mask, cv2.bitwise_not(skin_mask))

        elif gear_color == "black":
            # Black = very low value, any hue
            # Use adaptive approach: dark pixels that aren't just shadow
            val_ch = hsv[:, :, 2]
            sat_ch = hsv[:, :, 1]
            # Dark pixels (V < 75) with some texture
            dark_mask = (val_ch < 75).astype(np.uint8) * 255
            # Also include very dark saturated pixels
            dark_sat = ((val_ch < 100) & (sat_ch < 100)).astype(np.uint8) * 255
            color_mask = cv2.bitwise_or(dark_mask, dark_sat)
            # Exclude skin tones (high hue, warm colors)
            skin_lower = np.array([5, 40, 80], dtype=np.uint8)
            skin_upper = np.array([25, 200, 255], dtype=np.uint8)
            skin_mask = cv2.inRange(hsv, skin_lower, skin_upper)
            color_mask = cv2.bitwise_and(color_mask, cv2.bitwise_not(skin_mask))
            # Use edges to refine (black gear has strong edges against hand/background)
            edges = cv2.Canny(gray, 40, 120)
            edge_dilated = cv2.dilate(edges, np.ones((5, 5), np.uint8), iterations=2)
            # Combine: dark pixels near strong edges
            color_mask = cv2.bitwise_and(color_mask, edge_dilated)
            # Re-fill the interior
            color_mask = cv2.dilate(color_mask, np.ones((7, 7), np.uint8), iterations=3)

        else:
            # Auto: try blue
            lower_b = np.array([90, 80, 50], dtype=np.uint8)
            upper_b = np.array([130, 255, 255], dtype=np.uint8)
            color_mask = cv2.inRange(hsv, lower_b, upper_b)

        # ── Step 2: Morphological Cleanup + Smooth ──
        morph_k = max(3, int(cfg.get("ind_morph_kernel", 5)))
        if morph_k % 2 == 0:
            morph_k += 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_k, morph_k))
        small_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

        # Close small gaps within the gear body (2 iters to preserve tooth detail)
        color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        # Remove tiny noise
        color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, small_kernel, iterations=1)

        # Fill small internal holes (screw holes)
        cnts_holes, hier_h = cv2.findContours(color_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        if hier_h is not None:
            for i in range(len(cnts_holes)):
                if hier_h[0][i][3] >= 0 and cv2.contourArea(cnts_holes[i]) < 2000:
                    cv2.drawContours(color_mask, cnts_holes, i, 255, -1)

        # Smooth mask for clean contour boundaries
        mask_blurred = cv2.GaussianBlur(color_mask, (5, 5), 2.0)
        _, color_mask = cv2.threshold(mask_blurred, 127, 255, cv2.THRESH_BINARY)

        # ── Step 3: Find Gear Contour ──
        contours, _ = cv2.findContours(
            color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )

        min_area = int(cfg.get("ind_min_area", 5000))
        valid = [c for c in contours if cv2.contourArea(c) >= min_area]
        valid.sort(key=cv2.contourArea, reverse=True)

        frame_has_reject = False
        pieces_this_frame = []

        if valid:
            gear_contour = valid[0]
            gear_area = cv2.contourArea(gear_contour)

            # Ultra-smooth contour for drawing
            epsilon = 0.0008 * cv2.arcLength(gear_contour, True)
            smooth_contour = cv2.approxPolyDP(gear_contour, epsilon, True)

            # Create filled mask for overlay
            gear_mask = np.zeros((H, W), dtype=np.uint8)
            cv2.drawContours(gear_mask, [gear_contour], -1, 255, -1)
            gear_mask_bool = gear_mask > 0

            # ── Step 4: Subtle mask overlay ──
            mask_overlay = annotated.copy()
            # Very subtle tint so the gear is still visible
            if gear_color == "blue":
                tint = np.array([255, 220, 80], dtype=np.float32)
            elif gear_color == "black":
                tint = np.array([180, 255, 200], dtype=np.float32)
            else:
                tint = np.array([255, 200, 100], dtype=np.float32)

            mask_overlay[gear_mask_bool] = (
                annotated[gear_mask_bool].astype(np.float32) * 0.85 +
                tint * 0.15
            ).astype(np.uint8)
            annotated = mask_overlay

            # Smooth contour outline (anti-aliased, thin)
            cv2.drawContours(annotated, [smooth_contour], -1, (0, 255, 200), 2, cv2.LINE_AA)

            # ── Step 5: Teeth Analysis (contour-based) ──
            analysis = self._analyze_teeth_from_contour(gear_contour, px_per_cm, tolerance)

            if analysis:
                cx, cy = analysis["center"]
                enc_radius = analysis["enc_radius"]
                diameter_cm = analysis["diameter_cm"]
                teeth = analysis["teeth"]
                ext_count = len(analysis.get("peaks", []))
                int_count = len(analysis.get("valleys", []))
                teeth_count = ext_count + int_count  # 6 ext + 6 int = 12
                defects = analysis["defects"]
                mean_r = analysis["mean_radius"]

                # ═══════ PRO HUD OVERLAY ═══════

                # ── Subtle radial profile polyline (shows actual tooth contour shape) ──
                if "radii_smooth" in analysis:
                    rs = analysis["radii_smooth"]
                    ua = analysis["uniform_angles"]
                    pts_arr = []
                    for k in range(0, len(ua), 2):  # every 2° for perf
                        px_p = int(cx + rs[k] * math.cos(ua[k]))
                        py_p = int(cy - rs[k] * math.sin(ua[k]))
                        pts_arr.append([px_p, py_p])
                    if pts_arr:
                        pts_np = np.array(pts_arr, np.int32).reshape((-1, 1, 2))
                        cv2.polylines(annotated, [pts_np], True, (160, 120, 255), 1, cv2.LINE_AA)

                # ── Center dot (minimal) ──
                cv2.circle(annotated, (cx, cy), 3, CLR_CENTER, -1, cv2.LINE_AA)

                # ── Mean radius ring (dashed look via partial arcs) ──
                if mean_r > 20:
                    cv2.circle(annotated, (cx, cy), int(mean_r), (70, 70, 100), 1, cv2.LINE_AA)

                # ── Tooth markers (clean dots + optional lines) ──
                defect_indices = {d["tooth_idx"] for d in defects}
                for i, tooth in enumerate(teeth):
                    is_defective = i in defect_indices
                    tip_x, tip_y = tooth["tip_x"], tooth["tip_y"]

                    if is_defective:
                        # Defective: red dot + thin red ring
                        cv2.circle(annotated, (tip_x, tip_y), 6, (60, 60, 255), -1, cv2.LINE_AA)
                        cv2.circle(annotated, (tip_x, tip_y), 8, (80, 80, 255), 1, cv2.LINE_AA)
                    else:
                        # OK: small green dot
                        cv2.circle(annotated, (tip_x, tip_y), 3, (80, 220, 120), -1, cv2.LINE_AA)
                        cv2.circle(annotated, (tip_x, tip_y), 5, (80, 220, 120), 1, cv2.LINE_AA)

                    # Tooth number (only if < 20 teeth to avoid clutter)
                    if teeth_count <= 20:
                        lx = int(16 * math.cos(tooth["peak_angle"]))
                        ly = int(-16 * math.sin(tooth["peak_angle"]))
                        color_n = (120, 120, 255) if is_defective else (120, 220, 160)
                        cv2.putText(annotated, str(i + 1),
                                   (tip_x + lx - 4, tip_y + ly + 4),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.30, color_n, 1, cv2.LINE_AA)

                # ── Pass/Fail: ONLY teeth count + defects (no diameter rules) ──
                expected_teeth = cfg.get("ind_expected_teeth", 12)
                teeth_count_ok = abs(teeth_count - expected_teeth) <= 3
                no_defects = len(defects) == 0
                is_ok = teeth_count_ok and no_defects
                if not is_ok:
                    frame_has_reject = True

                # ── Measurements (informational only) ──
                bx, by, bbw, bbh = analysis["bbox"]
                perimeter = cv2.arcLength(gear_contour, True)
                circularity = round(4 * math.pi * gear_area / (perimeter ** 2), 3) if perimeter > 0 else 0
                perimeter_cm = round(perimeter / px_per_cm, 1)

                # ── Glass info panel ──
                panel_lines = [
                    f"  {ext_count}E + {int_count}I = {teeth_count}T",
                    f"  D {diameter_cm:.1f}cm | P {perimeter_cm:.0f}cm",
                    f"  Circ {circularity:.3f} | {'OK' if no_defects else f'{len(defects)} def'}",
                ]
                line_h = 20
                panel_w = max(len(l) * 8 for l in panel_lines) + 16
                panel_h = len(panel_lines) * line_h + 12
                panel_x = max(bx, 4)
                panel_y = max(by - panel_h - 8, 36)

                # Glassmorphism background
                py1, py2 = panel_y, min(panel_y + panel_h, H)
                px1, px2 = panel_x, min(panel_x + panel_w, W)
                panel_bg = annotated[py1:py2, px1:px2].copy()
                if panel_bg.size > 0:
                    overlay_color = np.array([30, 35, 45], dtype=np.float32)
                    blended = (panel_bg.astype(np.float32) * 0.30 + overlay_color * 0.70).astype(np.uint8)
                    annotated[py1:py2, px1:px2] = blended
                    accent = (80, 220, 120) if is_ok else (100, 100, 255)
                    cv2.line(annotated, (px1, py1), (px2, py1), accent, 2, cv2.LINE_AA)

                for j, line in enumerate(panel_lines):
                    ly = panel_y + 16 + j * line_h
                    cv2.putText(annotated, line, (panel_x + 2, ly),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.35, (215, 220, 230), 1, cv2.LINE_AA)

                # ── Status pill ──
                if is_ok:
                    status_text = "APROBADA"
                    pill_color = (55, 165, 85)
                else:
                    reasons = []
                    if not teeth_count_ok:
                        reasons.append(f"{teeth_count}T")
                    if not no_defects:
                        reasons.append(f"{len(defects)}def")
                    status_text = f"RECHAZADA ({', '.join(reasons)})" if reasons else "RECHAZADA"
                    pill_color = (50, 55, 200)

                (stw, sth), _ = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_SIMPLEX, 0.40, 1)
                pill_cx = cx
                pill_y = min(by + bbh + 18, H - 26)
                pill_x1 = max(0, pill_cx - stw // 2 - 10)
                pill_x2 = min(W, pill_cx + stw // 2 + 10)
                pill_y1 = max(0, pill_y - sth - 5)
                pill_y2 = min(H, pill_y + 5)

                pill_region = annotated[pill_y1:pill_y2, pill_x1:pill_x2].copy()
                if pill_region.size > 0:
                    pill_bg = (pill_region.astype(np.float32) * 0.20 +
                              np.array(pill_color, dtype=np.float32) * 0.80).astype(np.uint8)
                    annotated[pill_y1:pill_y2, pill_x1:pill_x2] = pill_bg
                cv2.putText(annotated, status_text,
                           (pill_cx - stw // 2, pill_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.40, CLR_WHITE, 1, cv2.LINE_AA)

                # Counters
                self.state["total"] += 1
                if is_ok:
                    self.state["passed"] += 1
                else:
                    self.state["rejected"] += 1

                defect_details = [{"tooth": d["tooth_idx"] + 1, "issues": d["issues"]} for d in defects]

                pieces_this_frame.append({
                    "cls": "engrane",
                    "w": diameter_cm,
                    "h": diameter_cm,
                    "diameter": diameter_cm,
                    "circularity": circularity,
                    "perimeter_cm": perimeter_cm,
                    "teeth": teeth_count,
                    "ext_teeth": ext_count,
                    "int_teeth": int_count,
                    "ok": is_ok,
                    "conf": round(circularity, 2),
                    "defects": defect_details,
                    "teeth_ok": teeth_count_ok,
                    "yolo_class": f"gear_{gear_color}",
                })

        # Log
        for p in pieces_this_frame:
            self.state["log"].appendleft(p)
        if frame_has_reject:
            self.state["last_status"] = "rejected"
        elif pieces_this_frame:
            self.state["last_status"] = "ok"

        # ── HUD: Top bar ──
        elapsed = time.perf_counter() - t0
        fps = 1.0 / elapsed if elapsed > 0 else 0
        self.state["fps_buf"].append(fps)
        avg_fps = float(np.mean(self.state["fps_buf"]))

        bar_h = 30
        bar_overlay = np.zeros((bar_h, W, 3), dtype=np.uint8)
        if pieces_this_frame:
            p = pieces_this_frame[0]
            if p["ok"]:
                bar_overlay[:] = (45, 135, 65)
                bar_text = f"APROBADA  |  {p['ext_teeth']}E+{p['int_teeth']}I={p['teeth']}T  |  D:{p['diameter']}cm  |  Circ:{p['circularity']}  |  {avg_fps:.0f}fps"
            else:
                bar_overlay[:] = (40, 45, 160)
                n_def = len(p.get("defects", []))
                bar_text = f"RECHAZADA  |  {p['ext_teeth']}E+{p['int_teeth']}I={p['teeth']}T  |  {n_def}def  |  {avg_fps:.0f}fps"
        else:
            bar_overlay[:] = (30, 30, 40)
            bar_text = f"Buscando engrane...  |  {avg_fps:.0f}fps"

        annotated[:bar_h, :] = cv2.addWeighted(
            annotated[:bar_h, :], 0.20, bar_overlay, 0.80, 0
        )
        cv2.putText(annotated, bar_text, (8, 21),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (225, 230, 240), 1, cv2.LINE_AA)

        # ── HUD: Bottom-left mode pill ──
        mode_text = f"HSV | {gear_color.upper()}"
        (mw, mh), _ = cv2.getTextSize(mode_text, cv2.FONT_HERSHEY_SIMPLEX, 0.30, 1)
        bl_x, bl_y = 4, H - mh - 10
        bl_region = annotated[bl_y:bl_y + mh + 8, bl_x:bl_x + mw + 12].copy()
        if bl_region.size > 0:
            annotated[bl_y:bl_y + mh + 8, bl_x:bl_x + mw + 12] = (
                bl_region.astype(np.float32) * 0.3 + np.array([30, 30, 40], dtype=np.float32) * 0.7
            ).astype(np.uint8)
        cv2.putText(annotated, mode_text, (bl_x + 6, H - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.30, (180, 200, 220), 1, cv2.LINE_AA)

        state_data = self.get_state()
        state_data["latency_ms"] = round(elapsed * 1000, 1)
        state_data["pieces_this_frame"] = pieces_this_frame
        state_data["mode"] = "industrial"

        return annotated, state_data

    def process_qc(self, frame: np.ndarray) -> tuple:
        """Process a frame with QC logic. Returns (annotated_frame, state_dict)."""
        cfg = self.config
        t0 = time.perf_counter()

        try:
            model = self._get_model(cfg["model"], cfg["use_seg"])
        except Exception as e:
            return frame, {"error": str(e)}

        try:
            results = model(
                frame,
                conf=cfg["confidence"],
                iou=cfg["iou"],
                imgsz=int(cfg["imgsz"]),
                half=USE_HALF,
                device=DEVICE,
                verbose=False,
                retina_masks=cfg["use_seg"],
                agnostic_nms=True,
            )
        except Exception as e:
            return frame, {"error": str(e)}

        elapsed = time.perf_counter() - t0
        fps = 1.0 / elapsed if elapsed > 0 else 0
        self.state["fps_buf"].append(fps)
        avg_fps = float(np.mean(self.state["fps_buf"]))

        result = results[0]
        H, W = frame.shape[:2]
        annotated = frame.copy()

        frame_has_reject = False
        pieces_this_frame = []

        if result.boxes is not None and len(result.boxes) > 0:
            for i, box in enumerate(result.boxes):
                cls_id = int(box.cls[0])
                cls_name = result.names[cls_id]
                conf_val = float(box.conf[0])

                if cfg["target_class"] != "Todas" and cls_name != cfg["target_class"]:
                    if cfg["show_all"]:
                        x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].cpu().numpy()]
                        cv2.rectangle(annotated, (x1, y1), (x2, y2), (80, 80, 80), 1)
                        cv2.putText(annotated, cls_name, (x1, y1 - 4),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (120, 120, 120), 1)
                    continue

                x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].cpu().numpy()]
                px_w = x2 - x1
                px_h = y2 - y1
                cm_w = px_w / cfg["px_per_cm"]
                cm_h = px_h / cfg["px_per_cm"]

                w_ok = cfg["min_w"] <= cm_w <= cfg["max_w"]
                h_ok = cfg["min_h"] <= cm_h <= cfg["max_h"]
                is_ok = w_ok and h_ok

                if is_ok:
                    color = CLR_OK
                    status_text = "OK"
                else:
                    color = CLR_FAIL
                    frame_has_reject = True
                    status_text = "RECHAZADA"

                # Draw bbox
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

                # Mask overlay
                if cfg["use_seg"] and result.masks is not None and i < len(result.masks.data):
                    mask = result.masks.data[i].cpu().numpy()
                    mask_resized = cv2.resize(mask, (W, H))
                    mask_bool = mask_resized > 0.5
                    overlay = annotated.copy()
                    overlay[mask_bool] = (
                        annotated[mask_bool].astype(np.float32) * 0.5 +
                        np.array(color, dtype=np.float32) * 0.5
                    ).astype(np.uint8)
                    annotated = overlay

                # Labels
                size_label = f"{cm_w:.1f}x{cm_h:.1f}cm"
                label = f"{cls_name} {conf_val:.0%} | {size_label}"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
                label_y = max(y1 - 8, th + 6)
                cv2.rectangle(annotated, (x1, label_y - th - 6), (x1 + tw + 10, label_y + 4), color, -1)
                cv2.putText(annotated, label, (x1 + 5, label_y - 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, CLR_WHITE, 1, cv2.LINE_AA)

                # Status badge
                badge = f" {status_text} "
                (bw, bh), _ = cv2.getTextSize(badge, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                badge_y = min(y2 + bh + 12, H - 4)
                cx = (x1 + x2) // 2
                cv2.rectangle(annotated, (cx - bw // 2 - 6, badge_y - bh - 4),
                              (cx + bw // 2 + 6, badge_y + 4), color, -1)
                cv2.putText(annotated, badge, (cx - bw // 2, badge_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, CLR_WHITE, 2, cv2.LINE_AA)

                # Counters
                self.state["total"] += 1
                if is_ok:
                    self.state["passed"] += 1
                else:
                    self.state["rejected"] += 1

                pieces_this_frame.append({
                    "cls": cls_name,
                    "w": round(cm_w, 1),
                    "h": round(cm_h, 1),
                    "ok": is_ok,
                    "conf": round(conf_val, 2),
                })

        for p in pieces_this_frame:
            self.state["log"].appendleft(p)

        if frame_has_reject:
            self.state["last_status"] = "rejected"
        elif len(pieces_this_frame) > 0:
            self.state["last_status"] = "ok"

        # Top bar overlay
        bar_h = 40
        bar_overlay = np.zeros((bar_h, W, 3), dtype=np.uint8)
        if frame_has_reject:
            bar_overlay[:] = (40, 40, 200)
            bar_text = "PIEZA RECHAZADA — FUERA DE TOLERANCIA"
        elif len(pieces_this_frame) > 0:
            bar_overlay[:] = (60, 160, 80)
            bar_text = f"LINEA OK — {len(pieces_this_frame)} pieza(s) en spec"
        else:
            bar_overlay[:] = (40, 40, 50)
            bar_text = f"Escaneando... | {avg_fps:.0f} FPS"

        annotated[:bar_h, :] = cv2.addWeighted(annotated[:bar_h, :], 0.3, bar_overlay, 0.7, 0)
        cv2.putText(annotated, bar_text, (12, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, CLR_WHITE, 2, cv2.LINE_AA)

        # FPS badge
        fps_text = f"{avg_fps:.0f} FPS"
        (fw, fh), _ = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.rectangle(annotated, (W - fw - 16, 4), (W - 4, fh + 12), (0, 0, 0), -1)
        cv2.putText(annotated, fps_text, (W - fw - 10, fh + 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 160), 2, cv2.LINE_AA)

        state_data = self.get_state()
        state_data["latency_ms"] = round(elapsed * 1000, 1)
        state_data["pieces_this_frame"] = pieces_this_frame

        return annotated, state_data

    def process_general(self, frame: np.ndarray) -> tuple:
        """Process a frame in general mode. Returns (annotated_frame, info_dict)."""
        cfg = self.config
        t0 = time.perf_counter()

        try:
            model = self._get_model_general(cfg["model"], cfg["task"])
        except Exception as e:
            return frame, {"error": str(e)}

        is_seg = cfg["task"] == "segmentation"
        try:
            results = model(
                frame,
                conf=cfg["confidence"],
                iou=cfg["iou"],
                imgsz=int(cfg["imgsz"]),
                half=USE_HALF,
                device=DEVICE,
                verbose=False,
                retina_masks=is_seg,
                agnostic_nms=True,
            )
        except Exception as e:
            return frame, {"error": str(e)}

        elapsed = time.perf_counter() - t0
        fps = 1.0 / elapsed if elapsed > 0 else 0
        self.state["fps_buf"].append(fps)
        avg_fps = float(np.mean(self.state["fps_buf"]))

        result = results[0]
        annotated = result.plot(
            labels=cfg["show_labels"],
            conf=cfg["show_conf"],
            boxes=cfg["show_boxes"],
            line_width=cfg["line_width"],
        )

        info = {
            "fps": round(avg_fps, 1),
            "latency_ms": round(elapsed * 1000, 1),
            "model": f"{MODELS.get(cfg['model'], 'yolo26n')}{TASK_SUFFIXES.get(cfg['task'], '')}",
            "task": cfg["task"],
        }

        if cfg["task"] == "classification":
            if result.probs is not None:
                top5_i = result.probs.top5
                top5_c = result.probs.top5conf.cpu().numpy()
                info["classifications"] = [
                    {"class": result.names[idx], "conf": round(float(c), 3)}
                    for idx, c in zip(top5_i, top5_c)
                ]
        elif cfg["task"] == "pose":
            n = len(result.boxes) if result.boxes is not None else 0
            info["persons"] = n
        else:
            n = len(result.boxes) if result.boxes is not None else 0
            info["objects"] = n
            if n > 0:
                counts = defaultdict(int)
                for box in result.boxes:
                    counts[result.names[int(box.cls[0])]] += 1
                info["class_counts"] = dict(sorted(counts.items(), key=lambda x: -x[1])[:10])

        return annotated, info
