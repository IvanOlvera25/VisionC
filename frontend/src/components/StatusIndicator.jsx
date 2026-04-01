"use client";

import { STATUS_MAP } from "@/lib/constants";

export default function StatusIndicator({ status }) {
  const s = STATUS_MAP[status] || STATUS_MAP.idle;

  return (
    <div className={`status-indicator status-${s.cssClass}`}>
      <div className="status-glow" />
      <div className="status-icon">{s.icon}</div>
      <div className="status-label">{s.label}</div>
      <div className="status-sub">{s.sub}</div>
    </div>
  );
}
