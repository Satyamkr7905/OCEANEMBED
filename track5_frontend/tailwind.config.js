/** @type {import('tailwindcss').Config} */
module.exports = {
  content: ["./src/**/*.{js,ts,jsx,tsx}"],
  theme: {
    extend: {
      colors: {
        base: {
          50: "#061422",
          100: "#0a1e33",
          200: "#12324d",
          300: "#1c4a6e",
          400: "#7eb6d4",
          500: "#a8d4ea",
          600: "#c5e4f3",
          700: "#dceef7",
          800: "#eef7fb",
          900: "#f7fcff",
          950: "#030b14",
        },
        brand: {
          50: "#ecfeff",
          100: "#cffafe",
          200: "#a5f3fc",
          400: "#67e8f9",
          500: "#22d3ee",
          600: "#06b6d4",
          700: "#0891b2",
          800: "#155e75",
          900: "#164e63",
        },
        temp: {
          light: "rgba(251,113,133,0.15)",
          main: "#fb7185",
          dark: "#e11d48",
        },
        salt: {
          light: "rgba(45,212,191,0.15)",
          main: "#2dd4bf",
          dark: "#0f766e",
        },
        heat: {
          light: "rgba(251,146,60,0.15)",
          main: "#fb923c",
          dark: "#c2410c",
        },
        depth: {
          light: "rgba(34,211,238,0.15)",
          main: "#22d3ee",
          dark: "#0e7490",
        },
      },
      fontFamily: {
        sans: ["var(--font-inter)", "system-ui", "sans-serif"],
        mono: ["var(--font-jetbrains)", "ui-monospace", "monospace"],
      },
      boxShadow: {
        soft: "0 8px 28px rgba(0,8,24,0.28)",
        card: "0 12px 40px rgba(0,10,30,0.35)",
        lifted: "0 18px 50px rgba(0,12,36,0.45)",
        glow: "0 0 0 3px rgba(34,211,238,0.22)",
        glass: "0 10px 40px rgba(0,10,30,0.35)",
      },
      borderRadius: {
        "2xl": "1.1rem",
        "3xl": "1.5rem",
      },
    },
  },
  plugins: [],
};
