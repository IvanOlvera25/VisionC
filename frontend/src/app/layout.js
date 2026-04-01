import { Inter, JetBrains_Mono } from "next/font/google";
import "./globals.css";

const inter = Inter({
  subsets: ["latin"],
  variable: "--font-inter",
  display: "swap",
});

const jetbrains = JetBrains_Mono({
  subsets: ["latin"],
  variable: "--font-mono",
  display: "swap",
});

export const metadata = {
  title: "VisionC — Control de Calidad Industrial",
  description:
    "Sistema de inspección visual en tiempo real para líneas de producción. Detecta piezas, mide dimensiones, y alerta cuando están fuera de tolerancia. Powered by YOLO26.",
  keywords: [
    "visión artificial",
    "control de calidad",
    "YOLO",
    "inspección industrial",
    "detección de objetos",
  ],
};

export default function RootLayout({ children }) {
  return (
    <html lang="es" className={`${inter.variable} ${jetbrains.variable}`}>
      <body>{children}</body>
    </html>
  );
}
