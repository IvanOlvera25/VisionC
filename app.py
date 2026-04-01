"""
🔥 VisionC — Control de Calidad Industrial
Sistema de inspección visual en tiempo real para líneas de producción.
Detecta piezas, mide dimensiones, y alerta cuando están fuera de tolerancia.

Powered by YOLO26 · YOLOv12 · YOLO11 | Ultralytics 2026
"""

import gradio as gr
import cv2
import numpy as np
import time
import torch
from ultralytics import YOLO
from collections import defaultdict, deque

# ──────────────────────────────────────────────
# Dispositivo
# ──────────────────────────────────────────────
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

# ──────────────────────────────────────────────
# Models
# ──────────────────────────────────────────────
MODELS = {
    "⭐ YOLO26-Nano (rápido)": "yolo26n",
    "⭐ YOLO26-Small": "yolo26s",
    "⭐ YOLO26-Medium": "yolo26m",
    "⭐ YOLO26-Large (preciso)": "yolo26l",
    "YOLO11-Nano": "yolo11n",
    "YOLO11-Small": "yolo11s",
    "YOLOv12-Nano": "yolov12n",
    "YOLOv12-Small": "yolov12s",
}

model_cache = {}


def get_model(key: str, seg: bool) -> YOLO:
    suffix = "-seg" if seg else ""
    name = f"{MODELS[key]}{suffix}.pt"
    if name not in model_cache:
        print(f"⏳ Cargando {name}...")
        model_cache[name] = YOLO(name)
        print(f"✅ {name} listo")
    return model_cache[name]


# ──────────────────────────────────────────────
# QC State (global)
# ──────────────────────────────────────────────
qc_state = {
    "total": 0,
    "passed": 0,
    "rejected": 0,
    "log": deque(maxlen=12),       # últimas N inspecciones
    "last_status": "idle",         # ok | rejected | idle
    "fps_buf": deque(maxlen=10),
}


def reset_counters():
    qc_state["total"] = 0
    qc_state["passed"] = 0
    qc_state["rejected"] = 0
    qc_state["log"].clear()
    qc_state["last_status"] = "idle"
    return "✅ Contadores reiniciados"


# ──────────────────────────────────────────────
# Industrial Processing
# ──────────────────────────────────────────────
# Colors (BGR for OpenCV)
CLR_OK = (72, 199, 142)       # green
CLR_FAIL = (80, 80, 255)      # red
CLR_WARN = (0, 200, 255)      # yellow
CLR_TEXT_BG = (20, 20, 30)    # dark overlay
CLR_WHITE = (255, 255, 255)


def draw_rounded_rect(img, pt1, pt2, color, radius=8, thickness=-1):
    """Draws a rounded rectangle."""
    x1, y1 = pt1
    x2, y2 = pt2
    cv2.rectangle(img, (x1 + radius, y1), (x2 - radius, y2), color, thickness)
    cv2.rectangle(img, (x1, y1 + radius), (x2, y2 - radius), color, thickness)
    cv2.circle(img, (x1 + radius, y1 + radius), radius, color, thickness)
    cv2.circle(img, (x2 - radius, y1 + radius), radius, color, thickness)
    cv2.circle(img, (x1 + radius, y2 - radius), radius, color, thickness)
    cv2.circle(img, (x2 - radius, y2 - radius), radius, color, thickness)


def process_qc(
    frame,
    model_key,
    use_seg,
    confidence,
    iou_thresh,
    imgsz,
    px_per_cm,
    min_w, max_w,
    min_h, max_h,
    target_class,
    show_all,
):
    """Procesa frame con lógica QC industrial."""
    if frame is None:
        return None, build_dashboard("idle"), build_status_html("idle")

    t0 = time.perf_counter()

    try:
        model = get_model(model_key, use_seg)
    except Exception as e:
        return frame, f"❌ Error: {e}", build_status_html("error")

    try:
        results = model(
            frame,
            conf=confidence,
            iou=iou_thresh,
            imgsz=int(imgsz),
            half=USE_HALF,
            device=DEVICE,
            verbose=False,
            retina_masks=use_seg,
            agnostic_nms=True,
        )
    except Exception as e:
        return frame, f"❌ Inferencia: {e}", build_status_html("error")

    elapsed = time.perf_counter() - t0
    fps = 1.0 / elapsed if elapsed > 0 else 0
    qc_state["fps_buf"].append(fps)
    avg_fps = np.mean(qc_state["fps_buf"])

    result = results[0]
    H, W = frame.shape[:2]
    annotated = frame.copy()

    # ── Draw masks if segmentation ──
    if use_seg and result.masks is not None:
        for i, mask_tensor in enumerate(result.masks.data):
            mask = mask_tensor.cpu().numpy()
            mask_resized = cv2.resize(mask, (W, H))
            # Determine color later, first just get the mask overlay ready
            # We'll color after classification

    # ── Process each detection ──
    frame_has_reject = False
    pieces_this_frame = []

    if result.boxes is not None and len(result.boxes) > 0:
        for i, box in enumerate(result.boxes):
            cls_id = int(box.cls[0])
            cls_name = result.names[cls_id]
            conf_val = float(box.conf[0])

            # Filter by target class if set
            if target_class != "Todas" and cls_name != target_class:
                if show_all:
                    # Draw dimmed
                    x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].cpu().numpy()]
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), (80, 80, 80), 1)
                    cv2.putText(annotated, cls_name, (x1, y1 - 4),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (120, 120, 120), 1)
                continue

            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].cpu().numpy()]
            px_w = x2 - x1
            px_h = y2 - y1

            # Convert to cm
            cm_w = px_w / px_per_cm
            cm_h = px_h / px_per_cm

            # Classify
            w_ok = min_w <= cm_w <= max_w
            h_ok = min_h <= cm_h <= max_h
            is_ok = w_ok and h_ok

            if is_ok:
                color = CLR_OK
                status_text = "OK"
                status_emoji = "✅"
            else:
                color = CLR_FAIL
                frame_has_reject = True
                reasons = []
                if cm_w < min_w:
                    reasons.append("ancho↓")
                elif cm_w > max_w:
                    reasons.append("ancho↑")
                if cm_h < min_h:
                    reasons.append("alto↓")
                elif cm_h > max_h:
                    reasons.append("alto↑")
                status_text = "RECHAZADA"
                status_emoji = "🔴"

            # ── Draw on frame ──
            # Bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            # Mask overlay (colored)
            if use_seg and result.masks is not None and i < len(result.masks.data):
                mask = result.masks.data[i].cpu().numpy()
                mask_resized = cv2.resize(mask, (W, H))
                mask_bool = mask_resized > 0.5
                overlay_color = np.array(color[::-1], dtype=np.float32) / 255.0  # RGB
                overlay = annotated.copy()
                overlay[mask_bool] = (
                    annotated[mask_bool].astype(np.float32) * 0.5 +
                    np.array(color, dtype=np.float32) * 0.5
                ).astype(np.uint8)
                annotated = overlay

            # Size label
            size_label = f"{cm_w:.1f}x{cm_h:.1f}cm"
            label = f"{cls_name} {conf_val:.0%} | {size_label}"

            # Label background
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            label_y = max(y1 - 8, th + 6)
            cv2.rectangle(annotated,
                          (x1, label_y - th - 6), (x1 + tw + 10, label_y + 4),
                          color, -1)
            cv2.putText(annotated, label, (x1 + 5, label_y - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, CLR_WHITE, 1, cv2.LINE_AA)

            # Status badge (below box)
            badge = f" {status_text} "
            (bw, bh), _ = cv2.getTextSize(badge, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            badge_y = min(y2 + bh + 12, H - 4)
            cx = (x1 + x2) // 2
            cv2.rectangle(annotated,
                          (cx - bw // 2 - 6, badge_y - bh - 4),
                          (cx + bw // 2 + 6, badge_y + 4),
                          color, -1)
            cv2.putText(annotated, badge, (cx - bw // 2, badge_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, CLR_WHITE, 2, cv2.LINE_AA)

            # Update QC counters
            qc_state["total"] += 1
            if is_ok:
                qc_state["passed"] += 1
            else:
                qc_state["rejected"] += 1

            pieces_this_frame.append({
                "cls": cls_name,
                "w": cm_w,
                "h": cm_h,
                "ok": is_ok,
                "emoji": status_emoji,
            })

    # Update log
    for p in pieces_this_frame:
        qc_state["log"].appendleft(p)

    # Update global status
    if frame_has_reject:
        qc_state["last_status"] = "rejected"
    elif len(pieces_this_frame) > 0:
        qc_state["last_status"] = "ok"

    # ── Top status bar overlay ──
    bar_h = 40
    overlay_bar = annotated[:bar_h, :].copy()
    bar_overlay = np.zeros((bar_h, W, 3), dtype=np.uint8)

    if frame_has_reject:
        bar_overlay[:] = (40, 40, 200)
        bar_text = "🚨 PIEZA RECHAZADA — FUERA DE TOLERANCIA"
    elif len(pieces_this_frame) > 0:
        bar_overlay[:] = (60, 160, 80)
        bar_text = f"✅ LÍNEA OK — {len(pieces_this_frame)} pieza(s) en spec"
    else:
        bar_overlay[:] = (40, 40, 50)
        bar_text = f"⏳ Escaneando... | {avg_fps:.0f} FPS"

    annotated[:bar_h, :] = cv2.addWeighted(annotated[:bar_h, :], 0.3, bar_overlay, 0.7, 0)
    cv2.putText(annotated, bar_text, (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, CLR_WHITE, 2, cv2.LINE_AA)

    # FPS badge (top right)
    fps_text = f"{avg_fps:.0f} FPS"
    (fw, fh), _ = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
    cv2.rectangle(annotated, (W - fw - 16, 4), (W - 4, fh + 12), (0, 0, 0), -1)
    cv2.putText(annotated, fps_text, (W - fw - 10, fh + 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 160), 2, cv2.LINE_AA)

    # Build dashboard + status
    dashboard = build_dashboard(qc_state["last_status"], avg_fps, elapsed)
    status_html = build_status_html(qc_state["last_status"])

    return annotated, dashboard, status_html


def build_status_html(status):
    """Genera el indicador gigante de estado."""
    if status == "rejected":
        return """
        <div class="status-indicator status-fail">
            <div class="status-icon">🚨</div>
            <div class="status-label">RECHAZADA</div>
            <div class="status-sub">Pieza fuera de tolerancia</div>
        </div>
        """
    elif status == "ok":
        return """
        <div class="status-indicator status-ok">
            <div class="status-icon">✅</div>
            <div class="status-label">APROBADA</div>
            <div class="status-sub">Pieza dentro de especificación</div>
        </div>
        """
    else:
        return """
        <div class="status-indicator status-idle">
            <div class="status-icon">⏳</div>
            <div class="status-label">ESPERANDO</div>
            <div class="status-sub">Sin piezas detectadas</div>
        </div>
        """


def build_dashboard(status, fps=0, latency=0):
    """Genera markdown del dashboard QC."""
    total = qc_state["total"]
    passed = qc_state["passed"]
    rejected = qc_state["rejected"]
    rate = (rejected / total * 100) if total > 0 else 0

    lines = [
        f"### ⚡ {fps:.0f} FPS · {latency*1000:.0f}ms",
        "",
        "| Métrica | Valor |",
        "|---------|-------|",
        f"| 📦 Total inspeccionadas | **{total}** |",
        f"| ✅ Aprobadas | **{passed}** |",
        f"| 🔴 Rechazadas | **{rejected}** |",
        f"| 📊 Tasa de rechazo | **{rate:.1f}%** |",
    ]

    # Log últimas piezas
    if qc_state["log"]:
        lines.append("")
        lines.append("### 📋 Últimas inspecciones")
        lines.append("| # | Clase | Tamaño | Estado |")
        lines.append("|---|-------|--------|--------|")
        for i, p in enumerate(list(qc_state["log"])[:8]):
            st = "✅ OK" if p["ok"] else "🔴 RECHAZADA"
            lines.append(f"| {i+1} | {p['cls']} | {p['w']:.1f}×{p['h']:.1f}cm | {st} |")

    return "\n".join(lines)


# ──────────────────────────────────────────────
# General Mode (legacy)
# ──────────────────────────────────────────────
TASK_SUFFIXES = {
    "🎯 Detección": "",
    "🎭 Segmentación": "-seg",
    "🤸 Pose": "-pose",
    "🏷️ Clasificación": "-cls",
}


def process_general(
    frame, model_key, task, confidence, iou_thresh, imgsz,
    show_labels, show_conf, show_boxes, line_width,
):
    if frame is None:
        return None, "⏳ Esperando cámara...", build_status_html("idle")

    t0 = time.perf_counter()
    try:
        suffix = TASK_SUFFIXES[task]
        base = MODELS[model_key]
        mn = f"{base}{suffix}.pt"
        if mn not in model_cache:
            model_cache[mn] = YOLO(mn)
        model = model_cache[mn]
    except Exception as e:
        return frame, f"❌ {e}", build_status_html("error")

    is_seg = task == "🎭 Segmentación"
    try:
        results = model(
            frame, conf=confidence, iou=iou_thresh, imgsz=int(imgsz),
            half=USE_HALF, device=DEVICE, verbose=False,
            retina_masks=is_seg, agnostic_nms=True,
        )
    except Exception as e:
        return frame, f"❌ {e}", build_status_html("error")

    elapsed = time.perf_counter() - t0
    fps = 1.0 / elapsed if elapsed > 0 else 0
    qc_state["fps_buf"].append(fps)
    avg_fps = np.mean(qc_state["fps_buf"])

    result = results[0]
    annotated = result.plot(
        labels=show_labels, conf=show_conf,
        boxes=show_boxes, line_width=line_width,
    )

    model_label = f"{base}{suffix}"
    lines = [f"### ⚡ {avg_fps:.0f} FPS · {elapsed*1000:.0f}ms · `{model_label}`", ""]

    if task == "🏷️ Clasificación":
        if result.probs is not None:
            top5_i = result.probs.top5
            top5_c = result.probs.top5conf.cpu().numpy()
            lines.append("| # | Clase | Conf |")
            lines.append("|---|-------|------|")
            for i, (idx, c) in enumerate(zip(top5_i, top5_c)):
                lines.append(f"| {i+1} | {result.names[idx]} | {c:.0%} |")
    elif task == "🤸 Pose":
        n = len(result.boxes) if result.boxes is not None else 0
        lines.append(f"**👤 Personas:** {n}")
    else:
        n = len(result.boxes) if result.boxes is not None else 0
        lines.append(f"**📦 Objetos:** {n}")
        if n > 0:
            counts = defaultdict(int)
            for box in result.boxes:
                counts[result.names[int(box.cls[0])]] += 1
            lines += ["", "| Clase | Qty |", "|-------|-----|"]
            for nm, c in sorted(counts.items(), key=lambda x: -x[1])[:6]:
                lines.append(f"| {nm} | {c} |")

    return annotated, "\n".join(lines), build_status_html("idle")


# ──────────────────────────────────────────────
# Industrial Mode (Gear detection via OpenCV)
# ──────────────────────────────────────────────
import math


def process_industrial(
    frame,
    px_per_cm,
    min_diameter, max_diameter,
    min_circularity,
    min_area_px,
    detect_teeth,
    filter_skin,
    bg_method,
    blur_kernel,
    canny_low, canny_high,
):
    """Detect industrial parts (gears, circular pieces) using OpenCV contours."""
    if frame is None:
        return None, build_dashboard("idle"), build_status_html("idle")

    t0 = time.perf_counter()
    H, W = frame.shape[:2]
    annotated = frame.copy()

    # Preprocessing
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    bk = max(3, int(blur_kernel))
    if bk % 2 == 0:
        bk += 1
    blurred = cv2.GaussianBlur(gray, (bk, bk), 0)

    # Skin mask
    object_mask = np.ones((H, W), dtype=np.uint8) * 255
    if filter_skin:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        m1 = cv2.inRange(hsv, np.array([0, 30, 60]), np.array([20, 180, 255]))
        m2 = cv2.inRange(hsv, np.array([160, 30, 60]), np.array([180, 180, 255]))
        skin = cv2.bitwise_or(m1, m2)
        kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        skin = cv2.dilate(skin, kern, iterations=2)
        object_mask = cv2.bitwise_not(skin)
        object_mask = cv2.erode(object_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=1)

    # Thresholding
    if bg_method == "Auto":
        mean_val = np.mean(blurred)
        if mean_val > 127:
            _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        else:
            _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    elif bg_method == "Fondo claro":
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    else:
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    thresh = cv2.bitwise_and(thresh, object_mask)
    edges = cv2.Canny(blurred, int(canny_low), int(canny_high))
    edges = cv2.bitwise_and(edges, object_mask)
    combined = cv2.bitwise_or(thresh, edges)
    kern_c = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kern_c, iterations=2)
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kern_c, iterations=1)

    contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Analyze contours
    frame_has_reject = False
    pieces = []
    valid = sorted([c for c in contours if cv2.contourArea(c) >= min_area_px], key=cv2.contourArea, reverse=True)

    for contour in valid[:5]:
        area = cv2.contourArea(contour)
        perim = cv2.arcLength(contour, True)
        if perim == 0:
            continue

        circ = (4 * math.pi * area) / (perim * perim)
        (cx, cy), rad = cv2.minEnclosingCircle(contour)
        cx, cy, rad = int(cx), int(cy), int(rad)
        x, y, bw, bh = cv2.boundingRect(contour)
        diam_cm = (rad * 2) / px_per_cm

        is_circular = circ >= 0.25
        dim_ok = min_diameter <= diam_cm <= max_diameter if is_circular else min_diameter <= max(bw, bh) / px_per_cm <= max_diameter
        circ_ok = circ >= min_circularity if is_circular else True
        is_ok = dim_ok and circ_ok

        if not is_ok:
            frame_has_reject = True

        color = CLR_OK if is_ok else CLR_FAIL

        # Count teeth
        teeth = 0
        if detect_teeth and is_circular and len(contour) >= 5:
            hull = cv2.convexHull(contour, returnPoints=False)
            if hull is not None and len(hull) >= 4:
                try:
                    defects = cv2.convexityDefects(contour, hull)
                    if defects is not None:
                        for i in range(defects.shape[0]):
                            s, e, f, d = defects[i, 0]
                            depth = d / 256.0
                            if depth > rad * 0.08:
                                far = tuple(contour[f][0])
                                dist = math.sqrt((far[0] - cx)**2 + (far[1] - cy)**2)
                                if dist > rad * 0.3:
                                    teeth += 1
                except cv2.error:
                    pass

        # Draw
        cv2.drawContours(annotated, [contour], -1, color, 3)
        ov = annotated.copy()
        cv2.drawContours(ov, [contour], -1, color, -1)
        annotated = cv2.addWeighted(annotated, 0.8, ov, 0.2, 0)

        if is_circular:
            cv2.circle(annotated, (cx, cy), rad, (255, 180, 50), 2, cv2.LINE_AA)
            cv2.line(annotated, (cx - 12, cy), (cx + 12, cy), (100, 255, 255), 2, cv2.LINE_AA)
            cv2.line(annotated, (cx, cy - 12), (cx, cy + 12), (100, 255, 255), 2, cv2.LINE_AA)
            cv2.circle(annotated, (cx, cy), 4, (100, 255, 255), -1, cv2.LINE_AA)

        det_type = "ENGRANE" if is_circular and circ >= min_circularity else ("CIRCULAR" if is_circular else "RECTANGULAR")
        label = f"{det_type} | D:{diam_cm:.1f}cm" if is_circular else f"{det_type} | {bw/px_per_cm:.1f}x{bh/px_per_cm:.1f}cm"
        if teeth > 0:
            label += f" | {teeth}T"

        (tw, th_), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        ly = max(y - 12, th_ + 8)
        cv2.rectangle(annotated, (x, ly - th_ - 8), (x + tw + 14, ly + 4), color, -1)
        cv2.putText(annotated, label, (x + 7, ly - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, CLR_WHITE, 2, cv2.LINE_AA)

        badge = " OK " if is_ok else " RECHAZADA "
        (bwt, bht), _ = cv2.getTextSize(badge, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        by_ = min(y + bh + bht + 20, H - 6)
        bcx = x + bw // 2
        cv2.rectangle(annotated, (bcx - bwt // 2 - 8, by_ - bht - 6), (bcx + bwt // 2 + 8, by_ + 6), color, -1)
        cv2.putText(annotated, badge, (bcx - bwt // 2, by_), cv2.FONT_HERSHEY_SIMPLEX, 0.6, CLR_WHITE, 2, cv2.LINE_AA)

        # Update QC counters
        qc_state["total"] += 1
        if is_ok:
            qc_state["passed"] += 1
        else:
            qc_state["rejected"] += 1

        pieces.append({
            "cls": det_type.lower(),
            "w": round(diam_cm if is_circular else bw / px_per_cm, 1),
            "h": round(diam_cm if is_circular else bh / px_per_cm, 1),
            "ok": is_ok,
            "emoji": "✅" if is_ok else "🔴",
            "teeth": teeth,
            "circularity": round(circ, 3),
        })

    for p in pieces:
        qc_state["log"].appendleft(p)

    if frame_has_reject:
        qc_state["last_status"] = "rejected"
    elif len(pieces) > 0:
        qc_state["last_status"] = "ok"

    elapsed = time.perf_counter() - t0
    fps = 1.0 / elapsed if elapsed > 0 else 0
    qc_state["fps_buf"].append(fps)
    avg_fps = np.mean(qc_state["fps_buf"])

    # Status bar
    bar_h = 44
    bar_overlay = np.zeros((bar_h, W, 3), dtype=np.uint8)
    if frame_has_reject:
        bar_overlay[:] = (40, 40, 200)
        bar_text = "PIEZA RECHAZADA — FUERA DE TOLERANCIA"
    elif len(pieces) > 0:
        bar_overlay[:] = (60, 160, 80)
        p = pieces[0]
        t_str = f" — {p['teeth']}T" if p['teeth'] > 0 else ""
        bar_text = f"PIEZA OK — {p['cls'].upper()} D:{p['w']}cm{t_str}"
    else:
        bar_overlay[:] = (40, 40, 50)
        bar_text = f"Escaneando pieza... | {avg_fps:.0f} FPS"

    annotated[:bar_h, :] = cv2.addWeighted(annotated[:bar_h, :], 0.3, bar_overlay, 0.7, 0)
    cv2.putText(annotated, bar_text, (14, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, CLR_WHITE, 2, cv2.LINE_AA)

    # FPS
    fps_t = f"{avg_fps:.0f} FPS"
    (fw, fh), _ = cv2.getTextSize(fps_t, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
    cv2.rectangle(annotated, (W - fw - 18, 6), (W - 4, fh + 16), (0, 0, 0), -1)
    cv2.putText(annotated, fps_t, (W - fw - 12, fh + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 160), 2, cv2.LINE_AA)

    dashboard = build_dashboard(qc_state["last_status"], avg_fps, elapsed)
    status_html = build_status_html(qc_state["last_status"])

    return annotated, dashboard, status_html


# ──────────────────────────────────────────────
# CSS — Industrial Dark Theme
# ──────────────────────────────────────────────
CUSTOM_CSS = """
:root {
    --bg-primary: #080810;
    --bg-card: rgba(14, 16, 24, 0.92);
    --bg-glass: rgba(255,255,255,0.025);
    --border-glass: rgba(255,255,255,0.07);
    --accent: #8b5cf6;
    --green: #22c55e;
    --red: #ef4444;
    --yellow: #eab308;
    --text-1: #f1f5f9;
    --text-2: #94a3b8;
    --text-3: #64748b;
}

.gradio-container {
    max-width: 1600px !important;
    margin: auto;
    background: var(--bg-primary) !important;
}

/* ── Header ── */
#header-block {
    text-align: center;
    padding: 20px 16px 14px;
    background: linear-gradient(135deg, rgba(139,92,246,0.08), rgba(34,197,94,0.05), rgba(59,130,246,0.04));
    border-radius: 14px;
    border: 1px solid var(--border-glass);
    backdrop-filter: blur(10px);
    margin-bottom: 12px;
}

#header-block h1 {
    background: linear-gradient(135deg, #a78bfa, #34d399);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    font-size: 1.8rem !important;
    font-weight: 800 !important;
    margin: 0 0 2px !important;
    letter-spacing: -0.5px;
}

#header-block p {
    color: var(--text-2);
    font-size: 0.82rem;
    margin: 0 !important;
}

/* ── Panels ── */
.control-panel, .info-panel-card, .qc-panel {
    background: var(--bg-card) !important;
    border: 1px solid var(--border-glass) !important;
    border-radius: 12px !important;
    padding: 14px !important;
    backdrop-filter: blur(16px) !important;
}

/* ── Status Indicator ── */
.status-indicator {
    text-align: center;
    padding: 20px 12px;
    border-radius: 14px;
    border: 2px solid;
    transition: all 0.3s ease;
    margin-bottom: 10px;
}

.status-icon {
    font-size: 2.4rem;
    margin-bottom: 4px;
}

.status-label {
    font-size: 1.3rem;
    font-weight: 800;
    letter-spacing: 2px;
}

.status-sub {
    font-size: 0.75rem;
    margin-top: 2px;
    opacity: 0.7;
}

.status-ok {
    background: rgba(34,197,94,0.1);
    border-color: rgba(34,197,94,0.4);
    color: #4ade80;
    box-shadow: 0 0 30px rgba(34,197,94,0.15), inset 0 0 20px rgba(34,197,94,0.05);
}

.status-fail {
    background: rgba(239,68,68,0.12);
    border-color: rgba(239,68,68,0.5);
    color: #f87171;
    box-shadow: 0 0 30px rgba(239,68,68,0.2), inset 0 0 20px rgba(239,68,68,0.05);
    animation: pulse-red 1s ease-in-out infinite alternate;
}

.status-idle {
    background: rgba(148,163,184,0.06);
    border-color: rgba(148,163,184,0.15);
    color: #94a3b8;
}

@keyframes pulse-red {
    from { box-shadow: 0 0 20px rgba(239,68,68,0.15); }
    to { box-shadow: 0 0 40px rgba(239,68,68,0.35), 0 0 60px rgba(239,68,68,0.1); }
}

/* ── Video ── */
.video-output {
    border-radius: 12px !important;
    overflow: hidden;
    border: 1px solid var(--border-glass) !important;
    box-shadow: 0 4px 20px rgba(0,0,0,0.5);
}

/* ── Info panel ── */
.info-panel-card h3 {
    background: linear-gradient(90deg, #a78bfa, #34d399);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    font-weight: 700 !important;
    font-size: 0.95rem !important;
}

.info-panel-card table {
    width: 100%;
    border-collapse: collapse;
    margin: 6px 0;
}

.info-panel-card th, .info-panel-card td {
    padding: 5px 8px;
    border-bottom: 1px solid var(--border-glass);
    font-size: 0.82rem;
}

.info-panel-card code {
    background: rgba(139,92,246,0.12);
    padding: 1px 6px;
    border-radius: 4px;
    color: #c084fc;
    font-size: 0.82rem;
}

/* ── QC tolerances section ── */
.tolerance-label {
    font-size: 0.75rem !important;
    color: var(--text-3) !important;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-top: 8px !important;
}

/* ── Footer ── */
#footer-block {
    text-align: center;
    padding: 10px;
    color: var(--text-3);
    font-size: 0.72rem;
    border-top: 1px solid var(--border-glass);
    margin-top: 12px;
}

#footer-block a {
    color: var(--accent) !important;
    text-decoration: none;
}

/* ── Responsive ── */
@media (max-width: 768px) {
    #header-block h1 { font-size: 1.3rem !important; }
}

/* ── Transitions ── */
.gr-button { transition: all 0.2s ease; }
.gr-button:hover {
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(139,92,246,0.25);
}

/* ── Mode tabs ── */
.mode-tabs button {
    font-weight: 600 !important;
    letter-spacing: 0.5px;
}
"""

# ──────────────────────────────────────────────
# Theme
# ──────────────────────────────────────────────
THEME = gr.themes.Soft(
    primary_hue="violet",
    secondary_hue="emerald",
    neutral_hue="slate",
    font=gr.themes.GoogleFont("Inter"),
).set(
    body_background_fill="#080810",
    body_background_fill_dark="#080810",
    block_background_fill="rgba(14, 16, 24, 0.92)",
    block_background_fill_dark="rgba(14, 16, 24, 0.92)",
    block_border_color="rgba(255,255,255,0.07)",
    block_border_color_dark="rgba(255,255,255,0.07)",
    block_label_text_color="#94a3b8",
    block_label_text_color_dark="#94a3b8",
    block_title_text_color="#f1f5f9",
    block_title_text_color_dark="#f1f5f9",
    input_background_fill="rgba(255,255,255,0.04)",
    input_background_fill_dark="rgba(255,255,255,0.04)",
    input_border_color="rgba(255,255,255,0.1)",
    input_border_color_dark="rgba(255,255,255,0.1)",
    button_primary_background_fill="linear-gradient(135deg, #8b5cf6, #6366f1)",
    button_primary_background_fill_dark="linear-gradient(135deg, #8b5cf6, #6366f1)",
    button_primary_text_color="#ffffff",
    button_primary_text_color_dark="#ffffff",
)

# ──────────────────────────────────────────────
# COCO classes for filter dropdown
# ──────────────────────────────────────────────
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
# Gradio App
# ──────────────────────────────────────────────
with gr.Blocks(
    title="VisionC — Control de Calidad Industrial",
) as demo:

    # Header
    gr.HTML("""
    <div id="header-block">
        <h1>🏭 VisionC — Control de Calidad Industrial</h1>
        <p>Inspección visual en tiempo real · Detección de dimensiones · Alertas de tolerancia</p>
        <p style="margin-top:4px !important;font-size:0.72rem;opacity:0.5;">
            YOLO26 · YOLOv12 · YOLO11 — Ultralytics 2026
        </p>
    </div>
    """)

    with gr.Tabs(elem_classes="mode-tabs") as mode_tabs:

        # ═══════════════════════════════════════
        # TAB 1: QC INDUSTRIAL
        # ═══════════════════════════════════════
        with gr.Tab("🏭 QC Industrial", id="qc"):
            with gr.Row(equal_height=False):

                # ── Left: Controls ──
                with gr.Column(scale=1, min_width=260):

                    with gr.Group(elem_classes="control-panel"):
                        gr.Markdown("#### ⚙️ Modelo & Detección")

                        qc_model = gr.Dropdown(
                            choices=list(MODELS.keys()),
                            value="⭐ YOLO26-Nano (rápido)",
                            label="Modelo",
                        )
                        qc_use_seg = gr.Checkbox(
                            value=True, label="🎭 Usar segmentación",
                            info="Máscaras pixel-level para mejor medición"
                        )
                        qc_conf = gr.Slider(
                            0.15, 0.95, value=0.40, step=0.05,
                            label="Confianza mínima"
                        )
                        qc_iou = gr.Slider(
                            0.1, 1.0, value=0.50, step=0.05,
                            label="IoU (NMS)"
                        )
                        qc_imgsz = gr.Slider(
                            192, 640, value=416, step=64,
                            label="Resolución inferencia"
                        )

                    with gr.Group(elem_classes="qc-panel"):
                        gr.Markdown("#### 📐 Calibración & Tolerancias")

                        qc_px_cm = gr.Slider(
                            5, 100, value=30, step=1,
                            label="Píxeles por cm",
                            info="Ajusta según distancia de cámara al objeto"
                        )

                        gr.Markdown("**Ancho permitido (cm)**", elem_classes="tolerance-label")
                        qc_min_w = gr.Slider(0.5, 50, value=3.0, step=0.5, label="Mínimo ancho")
                        qc_max_w = gr.Slider(0.5, 100, value=15.0, step=0.5, label="Máximo ancho")

                        gr.Markdown("**Alto permitido (cm)**", elem_classes="tolerance-label")
                        qc_min_h = gr.Slider(0.5, 50, value=3.0, step=0.5, label="Mínimo alto")
                        qc_max_h = gr.Slider(0.5, 100, value=15.0, step=0.5, label="Máximo alto")

                    with gr.Group(elem_classes="control-panel"):
                        gr.Markdown("#### 🎯 Filtro de Clase")
                        qc_target = gr.Dropdown(
                            choices=COCO_CLASSES,
                            value="Todas",
                            label="Clase a inspeccionar",
                            info="Filtra por tipo de objeto"
                        )
                        qc_show_all = gr.Checkbox(
                            value=True,
                            label="Mostrar objetos no filtrados (dimmed)"
                        )

                    qc_reset_btn = gr.Button(
                        "🔄 Reiniciar Contadores",
                        variant="secondary",
                        size="sm",
                    )
                    qc_reset_output = gr.Markdown("")

                # ── Center: Video ──
                with gr.Column(scale=3):
                    qc_webcam = gr.Image(
                        sources=["webcam"],
                        streaming=True,
                        label="📹 Cámara de Inspección",
                    )
                    qc_output = gr.Image(
                        label="🔍 Vista de Inspección",
                        interactive=False,
                        elem_classes="video-output",
                    )

                # ── Right: Dashboard ──
                with gr.Column(scale=1, min_width=280):
                    qc_status_html = gr.HTML(
                        value=build_status_html("idle"),
                    )
                    with gr.Group(elem_classes="info-panel-card"):
                        gr.Markdown("#### 📊 Dashboard de Producción")
                        qc_dashboard = gr.Markdown(
                            value=build_dashboard("idle"),
                        )

            # ── Events ──
            qc_webcam.stream(
                fn=process_qc,
                inputs=[
                    qc_webcam, qc_model, qc_use_seg, qc_conf, qc_iou,
                    qc_imgsz, qc_px_cm,
                    qc_min_w, qc_max_w, qc_min_h, qc_max_h,
                    qc_target, qc_show_all,
                ],
                outputs=[qc_output, qc_dashboard, qc_status_html],
                stream_every=0.12,
            )

            qc_reset_btn.click(
                fn=reset_counters,
                outputs=qc_reset_output,
            )

        # ═══════════════════════════════════════
        # TAB 2: MODO GENERAL
        # ═══════════════════════════════════════
        with gr.Tab("🔬 Modo General", id="general"):
            with gr.Row(equal_height=False):

                with gr.Column(scale=1, min_width=260):
                    with gr.Group(elem_classes="control-panel"):
                        gr.Markdown("#### ⚙️ Controles")
                        gen_task = gr.Radio(
                            choices=list(TASK_SUFFIXES.keys()),
                            value="🎯 Detección",
                            label="Tarea",
                        )
                        gen_model = gr.Dropdown(
                            choices=list(MODELS.keys()),
                            value="⭐ YOLO26-Nano (rápido)",
                            label="Modelo",
                        )
                        gen_conf = gr.Slider(0.10, 0.95, value=0.35, step=0.05, label="Confianza")
                        gen_iou = gr.Slider(0.1, 1.0, value=0.50, step=0.05, label="IoU")
                        gen_imgsz = gr.Slider(192, 640, value=384, step=64, label="Resolución")

                    with gr.Accordion("🎨 Visualización", open=False):
                        gen_labels = gr.Checkbox(value=True, label="Etiquetas")
                        gen_show_conf = gr.Checkbox(value=True, label="Confianza")
                        gen_boxes = gr.Checkbox(value=True, label="Bounding boxes")
                        gen_lw = gr.Slider(1, 5, value=2, step=1, label="Grosor")

                with gr.Column(scale=3):
                    gen_webcam = gr.Image(
                        sources=["webcam"], streaming=True, label="📹 Cámara",
                    )
                    gen_output = gr.Image(
                        label="🖼️ Resultado", interactive=False,
                        elem_classes="video-output",
                    )

                with gr.Column(scale=1, min_width=280):
                    gen_status_html = gr.HTML(value=build_status_html("idle"))
                    with gr.Group(elem_classes="info-panel-card"):
                        gr.Markdown("#### 📊 Estadísticas")
                        gen_info = gr.Markdown("⏳ Activa la cámara...")

            gen_webcam.stream(
                fn=process_general,
                inputs=[
                    gen_webcam, gen_model, gen_task, gen_conf, gen_iou,
                    gen_imgsz, gen_labels, gen_show_conf, gen_boxes, gen_lw,
                ],
                outputs=[gen_output, gen_info, gen_status_html],
                stream_every=0.15,
            )

        # ═══════════════════════════════════════
        # TAB 3: PIEZA INDUSTRIAL (Gear Detection)
        # ═══════════════════════════════════════
        with gr.Tab("🔩 Pieza Industrial", id="industrial"):
            with gr.Row(equal_height=False):

                # ── Left: Controls ──
                with gr.Column(scale=1, min_width=260):

                    with gr.Group(elem_classes="qc-panel"):
                        gr.Markdown("#### 🔩 Detección de Engranes")
                        gr.Markdown("*Detección por contornos OpenCV — sin modelo YOLO*", elem_classes="tolerance-label")

                        ind_px_cm = gr.Slider(
                            5, 100, value=30, step=1,
                            label="Píxeles por cm",
                            info="Ajusta según distancia de cámara"
                        )

                    with gr.Group(elem_classes="qc-panel"):
                        gr.Markdown("#### 📐 Tolerancias de Diámetro")

                        ind_min_d = gr.Slider(0.5, 40, value=8.0, step=0.5, label="Diámetro mínimo (cm)")
                        ind_max_d = gr.Slider(1, 50, value=25.0, step=0.5, label="Diámetro máximo (cm)")
                        ind_min_circ = gr.Slider(0, 1.0, value=0.30, step=0.05, label="Circularidad mínima")
                        ind_min_area = gr.Slider(200, 10000, value=3000, step=100, label="Área mínima (px²)")

                    with gr.Group(elem_classes="control-panel"):
                        gr.Markdown("#### ⚙️ Opciones")
                        ind_teeth = gr.Checkbox(value=True, label="🦷 Detectar dientes del engrane")
                        ind_skin = gr.Checkbox(value=True, label="🖐️ Filtrar piel (para mano sosteniendo pieza)")

                    with gr.Accordion("🔧 Avanzado", open=False):
                        ind_bg = gr.Radio(
                            choices=["Auto", "Fondo claro", "Fondo oscuro"],
                            value="Auto",
                            label="Fondo"
                        )
                        ind_blur = gr.Slider(3, 21, value=7, step=2, label="Blur kernel")
                        ind_canny_lo = gr.Slider(10, 150, value=30, step=5, label="Canny bajo")
                        ind_canny_hi = gr.Slider(30, 300, value=100, step=5, label="Canny alto")

                    ind_reset_btn = gr.Button(
                        "🔄 Reiniciar Contadores",
                        variant="secondary",
                        size="sm",
                    )
                    ind_reset_output = gr.Markdown("")

                # ── Center: Video ──
                with gr.Column(scale=3):
                    ind_webcam = gr.Image(
                        sources=["webcam"],
                        streaming=True,
                        label="📹 Cámara — Coloca la pieza frente a la cámara",
                    )
                    ind_output = gr.Image(
                        label="🔍 Detección de Pieza Industrial",
                        interactive=False,
                        elem_classes="video-output",
                    )

                # ── Right: Dashboard ──
                with gr.Column(scale=1, min_width=280):
                    ind_status_html = gr.HTML(
                        value=build_status_html("idle"),
                    )
                    with gr.Group(elem_classes="info-panel-card"):
                        gr.Markdown("#### 📊 Dashboard Industrial")
                        ind_dashboard = gr.Markdown(
                            value=build_dashboard("idle"),
                        )

            # ── Events ──
            ind_webcam.stream(
                fn=process_industrial,
                inputs=[
                    ind_webcam, ind_px_cm,
                    ind_min_d, ind_max_d,
                    ind_min_circ, ind_min_area,
                    ind_teeth, ind_skin,
                    ind_bg, ind_blur,
                    ind_canny_lo, ind_canny_hi,
                ],
                outputs=[ind_output, ind_dashboard, ind_status_html],
                stream_every=0.12,
            )

            ind_reset_btn.click(
                fn=reset_counters,
                outputs=ind_reset_output,
            )

    # Footer
    gr.HTML("""
    <div id="footer-block">
        <b>VisionC</b> — Control de Calidad Industrial ·
        <a href="https://ultralytics.com">Ultralytics</a> ·
        <a href="https://gradio.app">Gradio</a> ·
        OpenCV
    </div>
    """)


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        css=CUSTOM_CSS,
        theme=THEME,
    )
