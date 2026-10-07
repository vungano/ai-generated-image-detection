/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./app/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      fontFamily: {
        sans: ["var(--font-jakarta-sans)", "ui-sans-serif", "system-ui", "sans-serif"],
      },
      colors: {
        surface: "#ffffff",
        panel: "#f9fafb",
        border: "#e5e7eb",
        "border-strong": "#d1d5db",
        muted: "#6b7280",
        dim: "#9ca3af",
        foreground: "#111827",
        accent: "#2563eb",
        "accent-light": "#3b82f6",
        real: "#16a34a",
        "real-bg": "#dcfce7",
        fake: "#dc2626",
        "fake-bg": "#fee2e2",
      },
    },
  },
  plugins: [],
};
