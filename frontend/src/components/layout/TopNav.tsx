"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useEffect, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { Search, User } from "lucide-react";

import { ME_QUERY_KEY, fetchMe, logoutRemote } from "@/lib/api";
import { Monogram } from "@/components/brand/Lockup";
import { canSell } from "@/lib/roles";
import { cn } from "@/lib/utils";

const NAV_LINKS = [
  { href: "/market", label: "Market" },
  { href: "/benchmark", label: "Benchmark" },
  { href: "/#method", label: "Method" },
  { href: "/research", label: "Research" },
];

export function Wordmark({ className }: { className?: string }) {
  return (
    <span className={cn("font-display text-[22px] font-extrabold uppercase leading-none tracking-[0.02em] text-primary", className)}>
      Hype<span className="hv-gradient-text">vault</span>
    </span>
  );
}

export function TopNav() {
  const pathname = usePathname();
  const [scrolled, setScrolled] = useState(false);
  const meQuery = useQuery({ queryKey: ME_QUERY_KEY, queryFn: fetchMe, retry: false });
  const me = meQuery.data;
  const isSeller = canSell(me?.role);

  useEffect(() => {
    const read = () => setScrolled((window.__hvScroll ?? window.scrollY) > 8);
    read();
    window.addEventListener("scroll", read, { passive: true });
    window.addEventListener("hv:scroll", read);
    return () => {
      window.removeEventListener("scroll", read);
      window.removeEventListener("hv:scroll", read);
    };
  }, []);

  const linkClass = (href: string) =>
    cn(
      "hv-link-underline font-display text-[14px] font-bold uppercase tracking-[0.06em] transition-colors",
      pathname === href ? "text-accent" : "text-primary/75 hover:text-primary"
    );

  return (
    <header
      className={cn(
        "fixed inset-x-0 top-0 z-[90] border-b transition-colors duration-300",
        scrolled ? "hv-glass border-primary/10" : "border-transparent bg-transparent"
      )}
    >
      <div className="mx-auto flex h-16 w-full max-w-[1240px] items-center justify-between gap-4 px-4 sm:px-6">
        <Link href="/" aria-label="HypeVault home" className="shrink-0">
          <Monogram />
        </Link>

        <nav aria-label="Primary" className="hidden items-center gap-8 md:flex">
          {NAV_LINKS.map((l) => (
            <Link key={l.href} href={l.href} className={linkClass(l.href)}>
              {l.label}
            </Link>
          ))}
          {isSeller ? (
            <Link href="/seller/dashboard" className={linkClass("/seller/dashboard")}>
              Seller desk
            </Link>
          ) : null}
        </nav>

        <div className="flex shrink-0 items-center gap-1">
          <Link
            href="/market"
            aria-label="Search the market"
            className="grid h-11 w-11 place-items-center rounded-full text-primary/80 transition-colors hover:text-accent"
          >
            <Search className="h-[18px] w-[18px]" strokeWidth={1.5} aria-hidden="true" />
          </Link>
          <Link
            href={isSeller ? "/seller/dashboard" : me ? "/" : "/login"}
            aria-label={me ? "Your account" : "Sign in"}
            className="grid h-11 w-11 place-items-center rounded-full text-primary/80 transition-colors hover:text-accent"
          >
            <User className="h-[18px] w-[18px]" strokeWidth={1.5} aria-hidden="true" />
          </Link>
          {me ? (
            <button
              type="button"
              className="ml-2 hidden min-h-[44px] cursor-pointer font-display text-[14px] font-bold uppercase tracking-[0.06em] text-primary/70 transition-colors hover:text-primary lg:inline-flex lg:items-center"
              onClick={() => void logoutRemote().then(() => (window.location.href = "/login"))}
            >
              Sign out
            </button>
          ) : (
            <span className="ml-3 hidden sm:inline-flex">
              <Link
                href="/seller/upload"
                className="hv-btn-pill hv-btn-primary !min-h-[36px] !px-4 !py-1.5 !text-[13px]"
              >
                Sell a watch
              </Link>
            </span>
          )}
        </div>
      </div>
    </header>
  );
}
