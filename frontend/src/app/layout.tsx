import type { Metadata } from "next";
import { Outfit, Plus_Jakarta_Sans } from "next/font/google";
import "./globals.css";

import { MobileNav } from "@/components/MobileNav";
import { PageEnter } from "@/components/PageEnter";
import { Providers } from "@/components/Providers";
import { ScrollProgress } from "@/components/ScrollProgress";
import { SmoothScrollProvider } from "@/components/SmoothScrollProvider";
import { SiteFooter } from "@/components/layout/SiteFooter";
import { TopNav } from "@/components/layout/TopNav";

const plusJakarta = Plus_Jakarta_Sans({
  subsets: ["latin"],
  variable: "--font-sans",
  display: "swap",
});

const outfit = Outfit({
  subsets: ["latin"],
  variable: "--font-display",
  display: "swap",
});

export const metadata: Metadata = {
  title: {
    default: "HypeVault — Five-brand research prototype",
    template: "%s · HypeVault",
  },
  description:
    "Research prototype for authenticity classification within five evaluated luxury-watch brands. Not a universal authenticity guarantee.",
  openGraph: {
    title: "HypeVault",
    description:
      "Research prototype for authenticity classification within five evaluated luxury-watch brands.",
    type: "website",
  },
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en" className={`${plusJakarta.variable} ${outfit.variable}`}>
      <body className="min-h-dvh bg-[#0B0118] pb-24 font-sans text-[#FFEDF6] antialiased md:pb-0">
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
