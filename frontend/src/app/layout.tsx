import type { Metadata } from "next";
import { Big_Shoulders_Display, Josefin_Sans, Plus_Jakarta_Sans } from "next/font/google";
import "./globals.css";

import { MobileNav } from "@/components/MobileNav";
import { PageEnter } from "@/components/PageEnter";
import { Providers } from "@/components/Providers";
import { ScrollProgress } from "@/components/ScrollProgress";
import { SmoothScrollProvider } from "@/components/SmoothScrollProvider";
import { SiteFooter } from "@/components/layout/SiteFooter";
import { TopNav } from "@/components/layout/TopNav";

const jakarta = Plus_Jakarta_Sans({
  subsets: ["latin"],
  variable: "--font-sans",
  display: "swap",
});

// Condensed Art Deco display, for titles like a GTA VI card heading.
const bigShoulders = Big_Shoulders_Display({
  subsets: ["latin"],
  weight: ["600", "700", "800", "900"],
  variable: "--font-display",
  display: "swap",
});

// Geometric deco sans, for wide-tracked caps and the salmon headlines.
const josefin = Josefin_Sans({
  subsets: ["latin"],
  weight: ["400", "600", "700"],
  variable: "--font-wide",
  display: "swap",
});

export const metadata: Metadata = {
  title: {
    default: "HypeVault — Watch market intelligence",
    template: "%s · HypeVault",
  },
  description:
    "Research prototype for in-scope watch-brand screening. Not a certificate and not a real-world authenticity guarantee.",
  openGraph: {
    title: "HypeVault",
    description:
      "Research prototype for in-scope watch-brand screening. Not a real-world authenticity guarantee.",
    type: "website",
  },
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en" className={`${jakarta.variable} ${bigShoulders.variable} ${josefin.variable}`}>
      <body className="min-h-dvh bg-background pb-24 font-sans text-primary antialiased md:pb-0">
        <Providers>
          <SmoothScrollProvider>
            <ScrollProgress />
            <TopNav />
            <main className="relative z-10">
              <PageEnter>{children}</PageEnter>
              <MobileNav />
            </main>
            <SiteFooter />
          </SmoothScrollProvider>
        </Providers>
      </body>
    </html>
  );
}
