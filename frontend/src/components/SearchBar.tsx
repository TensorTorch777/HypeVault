"use client";

import { Search } from "lucide-react";
import { useRouter } from "next/navigation";
import { useCallback, useEffect, useMemo, useState } from "react";

import { Input } from "@/components/ui/input";
import { cn } from "@/lib/utils";

const LS_KEY = "hypevault_recent_searches";

function loadRecent(): string[] {
  try {
    const raw = window.localStorage.getItem(LS_KEY);
    if (!raw) return [];
    return JSON.parse(raw) as string[];
  } catch {
    return [];
  }
}

function saveRecent(q: string) {
  const prev = loadRecent().filter((x) => x.toLowerCase() !== q.toLowerCase());
  const next = [q, ...prev].slice(0, 8);
  window.localStorage.setItem(LS_KEY, JSON.stringify(next));
}

export function SearchBar({
  className,
  placeholder = "Search sneakers or watches...",
  initialQuery = "",
  variant = "light",
}: {
  className?: string;
  placeholder?: string;
  initialQuery?: string;
  /** `dark` = hero on black (StockX-style band) */
  variant?: "light" | "dark";
}) {
  const router = useRouter();
  const [value, setValue] = useState(initialQuery);
  const [open, setOpen] = useState(false);
  const [recent, setRecent] = useState<string[]>([]);

  useEffect(() => {
    setRecent(loadRecent());
  }, []);

  const suggestions = useMemo(() => {
    const q = value.trim().toLowerCase();
    if (!q) return recent.slice(0, 6);
    return recent.filter((r) => r.toLowerCase().includes(q)).slice(0, 6);
  }, [recent, value]);

  const go = useCallback(
    (q: string) => {
      const trimmed = q.trim();
      if (trimmed.length < 2) return;
      saveRecent(trimmed);
      setOpen(false);
      router.push(`/product/${encodeURIComponent(trimmed)}`);
    },
    [router]
  );

  useEffect(() => {
    const t = setTimeout(() => {
      /* debounce hook for instant feel: no remote fetch yet */
    }, 300);
    return () => clearTimeout(t);
  }, [value]);

  return (
    <div className={cn("relative w-full max-w-3xl", className)}>
      <div className="relative">
        <Search
          className={cn(
            "pointer-events-none absolute left-4 top-1/2 h-5 w-5 -translate-y-1/2",
            variant === "dark" ? "text-white/45" : "text-primary/35"
          )}
        />
        <Input
          value={value}
          onChange={(e) => {
            setValue(e.target.value);
            setOpen(true);
          }}
          onFocus={() => setOpen(true)}
          onKeyDown={(e) => {
            if (e.key === "Enter") go(value);
          }}
          placeholder={placeholder}
          className={cn(
            "h-14 rounded-full pl-12 pr-5 text-[15px]",
            variant === "dark"
              ? "border-white/15 bg-white/[0.06] text-white shadow-[0_12px_50px_rgba(0,0,0,0.35)] placeholder:text-white/40 focus:border-accent focus:ring-accent/30"
              : "border-black/[0.08] bg-white text-[#1D1D1F] placeholder:text-[#86868B] shadow-[0_8px_30px_rgba(0,0,0,0.05)] focus-visible:border-[#0066CC]/60 focus-visible:ring-[#0066CC]/15"
          )}
        />
      </div>
      {open && suggestions.length > 0 && (
        <div
          className={cn(
            "absolute z-50 mt-2 w-full overflow-hidden rounded-2xl border shadow-[0_18px_60px_rgba(0,0,0,0.12)]",
            variant === "dark"
              ? "border-white/15 bg-[#141414]/95 text-white backdrop-blur-xl"
              : "border-black/[0.06] bg-white"
          )}
        >
          {suggestions.map((s) => (
            <button
              key={s}
              type="button"
              className={cn(
                "flex w-full min-h-[44px] items-center px-4 py-3 text-left text-[15px]",
                variant === "dark" ? "text-white/90 hover:bg-white/10" : "text-[#1D1D1F] hover:bg-[#F5F5F7]"
              )}
              onMouseDown={(e) => e.preventDefault()}
              onClick={() => {
                setValue(s);
                go(s);
              }}
            >
              {s}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}
