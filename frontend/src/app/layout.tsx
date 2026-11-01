import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "AgriGuard-ML | Agentic Apple Leaf Disease Detection",
  description:
    "Next-generation precision agriculture diagnostic platform utilizing 8-agent LangGraph orchestration, 848 handcrafted classical ML visual features, and authoritative university extension advisory.",
  keywords: [
    "Apple Leaf Disease",
    "AgriGuard",
    "Machine Learning",
    "Precision Agriculture",
    "AppleLeaf9",
    "LangGraph",
    "LightGBM",
  ],
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html
      lang="en"
      className={`${geistSans.variable} ${geistMono.variable} h-full antialiased`}
      suppressHydrationWarning
    >
      <body className="min-h-full flex flex-col font-sans">{children}</body>
    </html>
  );
}
