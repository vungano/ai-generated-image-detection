import { Plus_Jakarta_Sans } from "next/font/google";
import "./globals.css";

const jakartaSans = Plus_Jakarta_Sans({
  subsets: ["latin"],
  display: "swap",
  variable: "--font-jakarta-sans",
});

export const metadata = {
  title: "AI Image Detector",
  description: "CIFAKE-trained fusion model demo",
};

export default function RootLayout({ children }) {
  return (
    <html lang="en" className={jakartaSans.variable}>
      <body className="min-h-screen bg-surface font-sans text-foreground antialiased">
        {children}
      </body>
    </html>
  );
}
