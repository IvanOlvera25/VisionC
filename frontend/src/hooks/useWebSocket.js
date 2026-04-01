"use client";

import { useRef, useState, useCallback, useEffect } from "react";

/**
 * Custom hook for WebSocket communication with the VisionC backend.
 * Handles connection lifecycle, frame sending, and result receiving.
 */
export function useWebSocket(url) {
  const wsRef = useRef(null);
  const [connected, setConnected] = useState(false);
  const [resultFrame, setResultFrame] = useState(null);
  const [stateData, setStateData] = useState(null);
  const reconnectTimer = useRef(null);
  const isMounted = useRef(true);

  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) return;

    try {
      const ws = new WebSocket(url);

      ws.onopen = () => {
        if (isMounted.current) {
          setConnected(true);
          console.log("🔌 WebSocket connected:", url);
        }
      };

      ws.onmessage = (event) => {
        try {
          const msg = JSON.parse(event.data);
          if (msg.type === "result") {
            if (msg.frame) {
              setResultFrame(`data:image/jpeg;base64,${msg.frame}`);
            }
            if (msg.state) setStateData(msg.state);
            if (msg.info) setStateData(msg.info);
          } else if (msg.type === "reset_ack" && msg.state) {
            setStateData(msg.state);
          }
        } catch (e) {
          console.error("Parse error:", e);
        }
      };

      ws.onclose = () => {
        if (isMounted.current) {
          setConnected(false);
          console.log("🔌 WebSocket disconnected, reconnecting...");
          reconnectTimer.current = setTimeout(() => {
            if (isMounted.current) connect();
          }, 2000);
        }
      };

      ws.onerror = (err) => {
        console.error("WebSocket error:", err);
        ws.close();
      };

      wsRef.current = ws;
    } catch (e) {
      console.error("WebSocket connection error:", e);
    }
  }, [url]);

  const disconnect = useCallback(() => {
    if (reconnectTimer.current) clearTimeout(reconnectTimer.current);
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }
    setConnected(false);
  }, []);

  const sendFrame = useCallback((base64Data) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({
        type: "frame",
        data: base64Data,
      }));
    }
  }, []);

  const sendConfig = useCallback((config) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({
        type: "config",
        config,
      }));
    }
  }, []);

  const sendReset = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ type: "reset" }));
    }
  }, []);

  useEffect(() => {
    isMounted.current = true;
    return () => {
      isMounted.current = false;
      disconnect();
    };
  }, [disconnect]);

  return {
    connected,
    resultFrame,
    stateData,
    connect,
    disconnect,
    sendFrame,
    sendConfig,
    sendReset,
  };
}
