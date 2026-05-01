"use client";

import { useEffect, useRef } from "react";
import { usePathname } from "next/navigation";
import Lenis from "@studio-freight/lenis";

/** Fixed header (~48px) + small gap so section titles are not clipped */
const HASH_SCROLL_OFFSET = -64;
const HASH_SCROLL_DURATION = 1.25;

const defaultEasing = (t: number) => Math.min(1, 1.001 - Math.pow(2, -10 * t));

/**
 * Silky smooth page scrolling.
 * Mirrors Lenis' internal scroll position into `window.__hvScroll` / `window.__hvScrollMax`
 * and exposes `window.__hvLenis` for programmatic scroll.
 *
 * Same-page `#hash` links are handled with Lenis (native jumps fight Lenis).
 * After client navigation to a URL with a hash, we scroll once the target exists in the DOM.
 */
declare global {
  interface Window {
    __hvScroll?: number;
    __hvScrollMax?: number;
    __hvLenis?: Lenis;
  }
}

function scrollToHash(lenis: Lenis, hash: string) {
  const id = hash.replace(/^#/, "").trim();
  if (!id) {
    lenis.scrollTo(0, {
      duration: HASH_SCROLL_DURATION,
      easing: defaultEasing,
    });
    return;
  }
  const el = document.getElementById(id);
  if (!el) return;
  lenis.scrollTo(el, {
    offset: HASH_SCROLL_OFFSET,
    duration: HASH_SCROLL_DURATION,
    easing: defaultEasing,
  });
}

/** Retry until RSC/streaming has mounted the section (cross-route hash links). */
function scrollToHashWhenReady(lenis: Lenis, hash: string, attempt = 0) {
  const id = hash.replace(/^#/, "").trim();
  if (!id) {
    scrollToHash(lenis, "");
    return;
  }
  const el = document.getElementById(id);
  if (el) {
    scrollToHash(lenis, hash);
    return;
  }
  if (attempt < 45) {
    requestAnimationFrame(() => scrollToHashWhenReady(lenis, hash, attempt + 1));
  }
}

export function SmoothScrollProvider({ children }: { children: React.ReactNode }) {
  const pathname = usePathname();
  const lenisRef = useRef<Lenis | null>(null);

  useEffect(() => {
    const prefersReduced =
      typeof window !== "undefined" && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    if (prefersReduced) return;

    const lenis = new Lenis({
      duration: 1.15,
      easing: defaultEasing,
      wheelMultiplier: 1.0,
      touchMultiplier: 1.2,
      smoothWheel: true,
    });

    lenisRef.current = lenis;
    window.__hvLenis = lenis;

    lenis.on("scroll", ({ scroll, limit }: { scroll: number; limit: number }) => {
      window.__hvScroll = scroll;
      window.__hvScrollMax = limit;
      window.dispatchEvent(new CustomEvent("hv:scroll", { detail: { scroll } }));
    });

    const onClickCapture = (e: MouseEvent) => {
      if (e.defaultPrevented || e.button !== 0) return;
      const target = (e.target as HTMLElement | null)?.closest?.("a[href]");
      if (!target || !(target instanceof HTMLAnchorElement)) return;
      const hrefAttr = target.getAttribute("href");
      if (!hrefAttr || !hrefAttr.includes("#")) return;

      let url: URL;
      try {
        url = new URL(hrefAttr, window.location.href);
      } catch {
        return;
      }
      if (url.origin !== window.location.origin) return;
      if (url.pathname !== window.location.pathname) return;

      const hash = url.hash;
      if (!hash) return;

      e.preventDefault();
      scrollToHashWhenReady(lenis, hash, 0);
      const next = `${url.pathname}${url.search}${hash}`;
      window.history.pushState(null, "", next);
    };

    document.addEventListener("click", onClickCapture, true);

    let rafId = 0;
    const raf = (time: number) => {
      lenis.raf(time);
      rafId = requestAnimationFrame(raf);
    };
    rafId = requestAnimationFrame(raf);

    return () => {
      document.removeEventListener("click", onClickCapture, true);
      cancelAnimationFrame(rafId);
      lenis.destroy();
      lenisRef.current = null;
      delete window.__hvLenis;
      delete window.__hvScroll;
      delete window.__hvScrollMax;
    };
  }, []);

  // Client navigations to e.g. `/#how` or `/#recent` — scroll after the new page paints.
  useEffect(() => {
    const prefersReduced =
      typeof window !== "undefined" && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    if (prefersReduced) return;

    let scrollT = 0;
    const runForCurrentHash = () => {
      const hash = window.location.hash;
      if (!hash) return;
      const lenis = lenisRef.current;
      if (!lenis) return;
      window.clearTimeout(scrollT);
      scrollT = window.setTimeout(() => scrollToHashWhenReady(lenis, hash, 0), 0);
    };

    runForCurrentHash();

    const onHashChange = () => runForCurrentHash();
    window.addEventListener("hashchange", onHashChange);
    return () => {
      window.removeEventListener("hashchange", onHashChange);
      window.clearTimeout(scrollT);
    };
  }, [pathname]);

  return <>{children}</>;
}
