import type { Config } from "tailwindcss";
import animate from "tailwindcss-animate";

const config: Config = {
  content: ["./src/**/*.{js,ts,jsx,tsx,mdx}"],
  theme: {
    extend: {
      colors: {
        primary: "#E4E4E7",
        accent: "#FF3B00",
        success: "#00C851",
        danger: "#FF3B30",
        background: "#080808",
        card: "#141414",
        electric: "#1A5FFF",
        ink: "#0a0a0a",
      },
      fontFamily: {
        sans: ["var(--font-sans)", "Plus Jakarta Sans", "system-ui", "sans-serif"],
        display: ["var(--font-display)", "Outfit", "var(--font-sans)", "system-ui", "sans-serif"],
      },
      borderRadius: {
        card: "12px",
        button: "8px",
      },
      keyframes: {
        "hv-grain": {
          "0%, 100%": { transform: "translate(0, 0)" },
          "33%": { transform: "translate(1.5%, -1%)" },
          "66%": { transform: "translate(-1%, 1.2%)" },
        },
        "hv-shimmer": {
          "0%": { backgroundPosition: "200% 0" },
          "100%": { backgroundPosition: "-200% 0" },
        },
        "hv-aurora": {
          "0%, 100%": { backgroundPosition: "0% 50%" },
          "50%":      { backgroundPosition: "100% 50%" },
        },
        "hv-marquee": {
          "0%":   { transform: "translateX(0)" },
          "100%": { transform: "translateX(-50%)" },
        },
        "hv-float": {
          "0%, 100%": { transform: "translateY(0) rotate(0)" },
          "50%":      { transform: "translateY(-14px) rotate(0.5deg)" },
        },
        "hv-pulse-glow": {
          "0%, 100%": { boxShadow: "0 0 0 0 rgba(255,59,0,0.4)" },
          "50%":      { boxShadow: "0 0 40px 10px rgba(255,59,0,0.15)" },
        },
        "hv-spin-slow": {
          "0%":   { transform: "rotate(0deg)" },
          "100%": { transform: "rotate(360deg)" },
        },
        "hv-scroll-down": {
          "0%":   { transform: "translateY(-6px)", opacity: "0" },
          "40%":  { opacity: "1" },
          "100%": { transform: "translateY(14px)", opacity: "0" },
        },
        "hv-ticker": {
          "0%":   { opacity: "0", transform: "translateY(16px)" },
          "100%": { opacity: "1", transform: "translateY(0)" },
        },
      },
      animation: {
        "hv-grain":      "hv-grain 9s ease-in-out infinite",
        "hv-shimmer":    "hv-shimmer 3.5s ease-in-out infinite",
        "hv-aurora":     "hv-aurora 18s ease-in-out infinite",
        "hv-marquee":    "hv-marquee 40s linear infinite",
        "hv-marquee-slow": "hv-marquee 60s linear infinite",
        "hv-float":      "hv-float 7s ease-in-out infinite",
        "hv-pulse-glow": "hv-pulse-glow 2.4s ease-in-out infinite",
        "hv-spin-slow":  "hv-spin-slow 22s linear infinite",
        "hv-scroll-down": "hv-scroll-down 1.8s ease-in-out infinite",
        "hv-ticker":     "hv-ticker 0.6s cubic-bezier(0.22, 1, 0.36, 1) both",
      },
    },
  },
  plugins: [animate],
};

export default config;
