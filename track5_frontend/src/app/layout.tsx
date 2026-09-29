import type { Metadata } from "next";
import { Inter, JetBrains_Mono } from "next/font/google";
import { Analytics } from '@vercel/analytics/next';
import "./globals.css";

const inter = Inter({ subsets: ["latin"], variable: "--font-inter" });
const mono = JetBrains_Mono({ subsets: ["latin"], variable: "--font-jetbrains" });

export const metadata: Metadata = {
  title: "OceanEmbed · INCOIS NIO reconstruction",
  description: "SIH26066 MoES/INCOIS satellite-to-subsurface ocean intelligence console",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className={`${inter.variable} ${mono.variable} bg-[#0a0f1e] font-sans text-base-800 antialiased`}>
        {children}
        <Analytics />
      </body>
    </html>
  );
}
