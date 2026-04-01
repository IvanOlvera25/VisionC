"use client";

import {
  Package,
  CheckCircle2,
  XCircle,
  TrendingDown,
  Circle,
  Cog,
} from "lucide-react";

export default function StatsCards({ stateData, mode }) {
  const total = stateData?.total || 0;
  const passed = stateData?.passed || 0;
  const rejected = stateData?.rejected || 0;
  const rate = stateData?.rate || 0;

  // Get latest piece data for industrial mode
  const latestPiece = stateData?.pieces_this_frame?.[0] || stateData?.log?.[0] || null;

  const baseCards = [
    {
      label: "Total Inspeccionadas",
      value: total,
      icon: <Package size={20} />,
      color: "purple",
    },
    {
      label: "Aprobadas",
      value: passed,
      icon: <CheckCircle2 size={20} />,
      color: "green",
    },
    {
      label: "Rechazadas",
      value: rejected,
      icon: <XCircle size={20} />,
      color: "red",
    },
    {
      label: "Tasa de Rechazo",
      value: `${rate.toFixed(1)}%`,
      icon: <TrendingDown size={20} />,
      color: rate > 10 ? "red" : rate > 5 ? "yellow" : "green",
    },
  ];

  // Add industrial-specific cards
  const industrialCards = mode === "industrial" && latestPiece ? [
    {
      label: "Diámetro",
      value: latestPiece.diameter ? `Ø${latestPiece.diameter}` : `${latestPiece.w}×${latestPiece.h}`,
      icon: <Circle size={20} />,
      color: "blue",
      unit: "cm",
    },
    {
      label: "Dientes",
      value: latestPiece.teeth > 0 ? latestPiece.teeth : "—",
      icon: <Cog size={20} />,
      color: latestPiece.teeth > 0 ? "purple" : "yellow",
    },
  ] : [];

  const cards = [...baseCards, ...industrialCards];

  return (
    <div className="stats-grid">
      {cards.map((card) => (
        <div key={card.label} className={`stat-card stat-${card.color}`}>
          <div className="stat-icon">{card.icon}</div>
          <div className="stat-content">
            <div className="stat-value">
              {card.value}
              {card.unit && <span className="stat-unit">{card.unit}</span>}
            </div>
            <div className="stat-label">{card.label}</div>
          </div>
        </div>
      ))}
    </div>
  );
}
