"use client";

import Link from "next/link";
import { useMemo } from "react";
import { motion } from "framer-motion";
import { useQuery } from "@tanstack/react-query";

import { BouncyText, WobbleText } from "@/components/motion/BouncyText";
import { fetchMe } from "@/lib/api";

export function ClosingCTA() {
  const meQuery = useQuery({
    queryKey: ["auth-me-closing-cta"],
    queryFn: fetchMe,
    retry: false,
  });
  const authed = useMemo(() => Boolean(meQuery.data), [meQuery.data]);
  const isSeller = meQuery.data?.role === "seller";

  return (
    <section className="relative bg-[#0B0118] py-32 md:py-40">
      {/* Sunset orb — clip glow here so the headline row is not clipped horizontally */}
      <div
        aria-hidden
        className="pointer-events-none absolute inset-x-0 -bottom-32 h-[60vh] overflow-hidden"
        style={{
          background:
            "radial-gradient(ellipse 80% 70% at 50% 100%, #FF7A1A 0%, #FF1FA4 35%, transparent 70%)",
          filter: "blur(20px)",
          opacity: 0.6,
        }}
      />

      <div className="relative z-[2] mx-auto max-w-3xl px-5 text-center sm:px-6">
        <p className="hv-eyebrow">Ready when you are</p>

        <motion.h2
          initial={{ opacity: 0, y: 18 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true, margin: "-60px" }}
          transition={{ duration: 0.7 }}
          className="mt-4 font-[family-name:var(--font-display)] hv-display-xl"
        >
          <span className="hv-neon-title block">
            <WobbleText text="AUTHENTICITY" />
          </span>
          <span className="mt-1 block min-w-0">
            <BouncyText
              className="hv-gradient-text"
              text="IS A STANDARD."
              delay={0.4}
              stagger={0.05}
              bounce={0.6}
            />
          </span>
        </motion.h2>

        <motion.p
          initial={{ opacity: 0, y: 14 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true, margin: "-60px" }}
          transition={{ delay: 0.1, duration: 0.7 }}
          className="mx-auto mt-7 max-w-xl text-[18px] leading-[1.45] text-[#FFEDF6]/70 md:text-[20px]"
        >
          Join the buyers who refuse to gamble on luxury. Run your first
          AI-verified listing today.
        </motion.p>

        <motion.div
          initial={{ opacity: 0, y: 14 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true, margin: "-60px" }}
          transition={{ delay: 0.2, duration: 0.6 }}
          className="mt-10 flex flex-wrap items-center justify-center gap-3"
        >
          {!authed ? (
            <>
              <Link href="/register" className="hv-btn-pill hv-btn-primary">
                Enter the vault
              </Link>
              <Link href="/login" className="inline-flex min-h-[44px] items-center hv-chevron">
                Log in
              </Link>
            </>
          ) : (
            <>
              <Link href="/" className="hv-btn-pill hv-btn-primary">
                Continue
              </Link>
              {isSeller ? (
                <Link href="/seller/dashboard" className="inline-flex min-h-[44px] items-center hv-chevron">
                  Seller dashboard
                </Link>
              ) : (
                <Link href="/product/Sneakers" className="inline-flex min-h-[44px] items-center hv-chevron">
                  Explore market
                </Link>
              )}
            </>
          )}
        </motion.div>

        <p className="mt-10 text-[11px] font-bold uppercase tracking-[0.28em] text-[#FFEDF6]/40">
          No credit card · Free verifications while in beta
        </p>
      </div>
    </section>
  );
}
