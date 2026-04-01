"use client";

import { useRef, useEffect, useCallback, useState } from "react";
import { Camera, CameraOff, Maximize2, Minimize2 } from "lucide-react";

export default function CameraFeed({
  resultFrame,
  onFrameCapture,
  streaming,
  setStreaming,
}) {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const streamRef = useRef(null);
  const intervalRef = useRef(null);
  const [expanded, setExpanded] = useState(false);
  const [cameraReady, setCameraReady] = useState(false);

  const startCamera = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: { ideal: 640 },
          height: { ideal: 480 },
          facingMode: "environment",
        },
      });
      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        videoRef.current.onloadeddata = () => {
          setCameraReady(true);
        };
      }
      setStreaming(true);
    } catch (err) {
      console.error("Camera error:", err);
      alert("No se pudo acceder a la cámara. Verifica los permisos del navegador.");
      setStreaming(false);
    }
  }, [setStreaming]);

  const stopCamera = useCallback(() => {
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
    }
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((t) => t.stop());
      streamRef.current = null;
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
    setCameraReady(false);
    setStreaming(false);
  }, [setStreaming]);

  // Capture frames at ~8 FPS and send to backend
  useEffect(() => {
    if (!streaming || !cameraReady) return;

    intervalRef.current = setInterval(() => {
      const video = videoRef.current;
      const canvas = canvasRef.current;
      if (!video || !canvas || video.readyState < 2) return;

      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      const ctx = canvas.getContext("2d");
      ctx.drawImage(video, 0, 0);

      canvas.toBlob(
        (blob) => {
          if (!blob) return;
          const reader = new FileReader();
          reader.onload = () => {
            const base64 = reader.result.split(",")[1];
            onFrameCapture(base64);
          };
          reader.readAsDataURL(blob);
        },
        "image/jpeg",
        0.75
      );
    }, 120); // ~8 fps capture

    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current);
    };
  }, [streaming, cameraReady, onFrameCapture]);

  useEffect(() => {
    return () => stopCamera();
  }, [stopCamera]);

  return (
    <div className={`camera-container ${expanded ? "camera-expanded" : ""}`}>
      <div className="camera-header">
        <div className="camera-title">
          <Camera size={16} />
          <span>Cámara de Inspección</span>
          {streaming && cameraReady && <div className="live-dot" />}
          {streaming && cameraReady && <span className="live-label">LIVE</span>}
        </div>
        <div className="camera-actions">
          <button
            className="btn-icon"
            onClick={() => setExpanded(!expanded)}
            title={expanded ? "Minimizar" : "Expandir"}
          >
            {expanded ? <Minimize2 size={16} /> : <Maximize2 size={16} />}
          </button>
        </div>
      </div>

      <div className="camera-viewport">
        {/* Hidden canvas for frame capture */}
        <canvas ref={canvasRef} style={{ display: "none" }} />

        {/* Show result frame from backend when available */}
        {resultFrame && streaming && cameraReady ? (
          <img
            src={resultFrame}
            alt="Inspection result"
            className="camera-result"
          />
        ) : null}

        {/* Show raw camera feed as main view when no results, or as PiP when results arrive */}
        <video
          ref={videoRef}
          autoPlay
          playsInline
          muted
          className={
            resultFrame && streaming && cameraReady
              ? "camera-video-pip"
              : "camera-video-main"
          }
        />

        {/* Placeholder when camera is off */}
        {!streaming && !resultFrame && (
          <div className="camera-placeholder">
            <CameraOff size={48} />
            <p>Cámara desactivada</p>
            <p className="camera-hint">Presiona el botón para iniciar</p>
          </div>
        )}

        {/* Loading state */}
        {streaming && !cameraReady && (
          <div className="camera-placeholder">
            <div className="spinner" />
            <p>Iniciando cámara...</p>
          </div>
        )}
      </div>

      <div className="camera-controls">
        <button
          className={`btn ${streaming ? "btn-danger" : "btn-primary"}`}
          onClick={streaming ? stopCamera : startCamera}
        >
          {streaming ? (
            <>
              <CameraOff size={16} />
              <span>Detener Cámara</span>
            </>
          ) : (
            <>
              <Camera size={16} />
              <span>Iniciar Cámara</span>
            </>
          )}
        </button>
      </div>
    </div>
  );
}
