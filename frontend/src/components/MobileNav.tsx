"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useMemo } from "react";
import { useQuery } from "@tanstack/react-query";
import { Home, Search, UploadCloud, User } from "lucide-react";

import { fetchMe } from "@/lib/api";
import { cn } from "@/lib/utils";

export function MobileNav() {
  const pathname = usePathname();

  const meQuery = useQuery({
    queryKey: ["auth-me-mobile-nav"],
    queryFn: fetchMe,
    retry: false,
  });
  const me = meQuery.data;
  const role = me?.role;

  const items = useMemo(() => {
    if (role === "seller") {
      return [
        { href: "/", label: "Home", icon: Home },
        { href: "/seller/dashboard", label: "Account", icon: User },
        { href: "/seller/upload", label: "Sell", icon: UploadCloud },
        { href: "/product/Sneakers", label: "Search", icon: Search },
      ] as const;
    }
    return [
      { href: "/", label: "Home", icon: Home },
        { href: me ? "/" : "/login", label: "Account", icon: User },
      { href: "/product/Sneakers", label: "Search", icon: Search },
    ] as const;
  }, [me, role]);

  return (
    <nav className="fixed inset-x-0 bottom-0 z-40 border-t border-white/10 bg-[#0c0c0c]/92 backdrop-blur-xl md:hidden">
      <div className="mx-auto flex max-w-lg items-stretch justify-between px-2 py-2">
        {items.map((it) => {
          const active = pathname === it.href;
          const Icon = it.icon;
          return (
            <Link
              key={it.label}
              href={it.href}
              className={cn(
                "flex min-h-[48px] min-w-[64px] flex-1 flex-col items-center justify-center gap-1 text-[11px] font-semibold",
                active ? "text-accent" : "text-primary/55"
              )}
            >
              <Icon className="h-5 w-5" />
              {it.label}
            </Link>
          );
        })}
      </div>
    </nav>
  );
}
