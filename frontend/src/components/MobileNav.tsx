"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useQuery } from "@tanstack/react-query";
import { BarChart3, Home, LineChart, UploadCloud, User } from "lucide-react";

import { ME_QUERY_KEY, fetchMe } from "@/lib/api";
import { canSell } from "@/lib/roles";
import { cn } from "@/lib/utils";

export function MobileNav() {
  const pathname = usePathname();
  const meQuery = useQuery({ queryKey: ME_QUERY_KEY, queryFn: fetchMe, retry: false });
  const me = meQuery.data;

  const items = [
    { href: "/", label: "Home", icon: Home },
    { href: "/market", label: "Market", icon: LineChart },
    { href: "/benchmark", label: "Benchmark", icon: BarChart3 },
    canSell(me?.role)
      ? { href: "/seller/upload", label: "Sell", icon: UploadCloud }
      : { href: me ? "/" : "/login", label: "Account", icon: User },
  ];

  return (
    <nav
      aria-label="Mobile"
      className="hv-glass fixed inset-x-0 bottom-0 z-40 border-t border-primary/10 pb-[env(safe-area-inset-bottom)] md:hidden"
    >
      <div className="mx-auto flex max-w-lg items-stretch justify-between px-2 py-1.5">
        {items.map((it) => {
          const active = pathname === it.href;
          const Icon = it.icon;
          return (
            <Link
              key={it.label}
              href={it.href}
              aria-current={active ? "page" : undefined}
              className={cn(
                "hv-mono flex min-h-[52px] flex-1 flex-col items-center justify-center gap-1 text-[10px] uppercase tracking-[0.1em] transition-colors",
                active ? "text-accent" : "text-primary/60 hover:text-primary"
              )}
            >
              <Icon className="h-5 w-5" strokeWidth={1.5} aria-hidden="true" />
              {it.label}
            </Link>
          );
        })}
      </div>
    </nav>
  );
}
