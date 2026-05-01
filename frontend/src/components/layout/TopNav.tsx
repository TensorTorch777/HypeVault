"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import { motion } from "framer-motion";
import { useQuery } from "@tanstack/react-query";
import { Search, ShoppingBag, User } from "lucide-react";

import { WobbleText } from "@/components/motion/BouncyText";
import { fetchMe, logoutRemote } from "@/lib/api";
import { cn } from "@/lib/utils";

const NAV_LINKS = [
  { href: "/product/Sneakers",         label: "Sneakers" },
  { href: "/product/Luxury%20watches", label: "Watches" },
  { href: "/#how",                     label: "Method" },
];

export function TopNav() {
  const [scrolled, setScrolled] = useState(false);

  const meQuery = useQuery({
    queryKey: ["auth-me-nav"],
    queryFn: fetchMe,
    retry: false,
  });

  const me = meQuery.data;
  const isSeller = me?.role === "seller";

  useEffect(() => {
    const read = () => {
      const y = window.__hvScroll ?? window.scrollY;
      setScrolled(y > 12);
    };
    read();
    window.addEventListener("scroll", read, { passive: true });
    window.addEventListener("hv:scroll", read);
    return () => {
      window.removeEventListener("scroll", read);
      window.removeEventListener("hv:scroll", read);
    };
  }, []);

  return (
    <motion.header
      initial={{ y: -32, opacity: 0 }}
      animate={{ y: 0, opacity: 1 }}
      transition={{ duration: 0.5, ease: [0.22, 1, 0.36, 1] }}
      className={cn(
        "fixed inset-x-0 top-0 z-[90] transition-[background,backdrop-filter,border] duration-300",
        scrolled
          ? "border-b border-[rgba(255,31,164,0.18)] bg-[#0B0118]/72 backdrop-blur-xl backdrop-saturate-150"
          : "border-b border-transparent bg-transparent"
      )}
    >
      <div className="mx-auto flex h-[48px] max-w-[980px] items-center justify-between px-5">
        <Link
          href="/"
          aria-label="HypeVault"
          className="flex items-center gap-1.5 font-[family-name:var(--font-display)] text-[15px] font-extrabold italic tracking-tight"
          style={{
            background: "linear-gradient(95deg, #FFD24A, #FF7A1A 30%, #FF1FA4 60%, #9B5BFF)",
            WebkitBackgroundClip: "text",
            backgroundClip: "text",
            color: "transparent",
            filter: "drop-shadow(0 0 12px rgba(255,31,164,0.5))",
          }}
        >
          <WobbleText text="HYPE/VAULT" />
        </Link>

        <nav className="hidden items-center gap-7 md:flex">
          {NAV_LINKS.map((l) => (
            <Link
              key={l.href}
              href={l.href}
              className="text-[12px] font-bold uppercase tracking-[0.14em] text-[#FFEDF6]/80 transition-colors hover:text-[#FFEDF6]"
            >
              {l.label}
            </Link>
          ))}
          {isSeller ? (
            <>
              <Link
                href="/seller/upload"
                className="text-[12px] font-bold uppercase tracking-[0.14em] text-[#FFEDF6]/80 transition-colors hover:text-[#FFEDF6]"
              >
                Authenticate
              </Link>
              <Link
                href="/seller/dashboard"
                className="text-[12px] font-bold uppercase tracking-[0.14em] text-[#FFEDF6]/80 transition-colors hover:text-[#FFEDF6]"
              >
                Sellers
              </Link>
            </>
          ) : null}
        </nav>

        <div className="flex items-center gap-4">
          <Link href="/product/Sneakers" aria-label="Search" className="text-[#FFEDF6]/80 transition-colors hover:text-[#00E1FF]">
            <Search className="h-[15px] w-[15px]" strokeWidth={1.6} />
          </Link>
          <Link
            href={me?.role === "seller" ? "/seller/dashboard" : me ? "/" : "/login"}
            aria-label="Account"
            className="text-[#FFEDF6]/80 transition-colors hover:text-[#00E1FF]"
          >
            <User className="h-[15px] w-[15px]" strokeWidth={1.6} />
          </Link>
          {isSeller ? (
            <Link href="/seller/upload" aria-label="Bag" className="text-[#FFEDF6]/80 transition-colors hover:text-[#00E1FF]">
              <ShoppingBag className="h-[15px] w-[15px]" strokeWidth={1.6} />
            </Link>
          ) : null}
          {me ? (
            <button
              type="button"
              className="hidden text-[11px] font-bold uppercase tracking-[0.12em] text-[#FFEDF6]/75 transition-colors hover:text-[#FFEDF6] md:inline-flex"
              onClick={() => {
                void logoutRemote().then(() => {
                  window.location.href = "/login";
                });
              }}
            >
              Log out
            </button>
          ) : null}
        </div>
      </div>
    </motion.header>
  );
}
