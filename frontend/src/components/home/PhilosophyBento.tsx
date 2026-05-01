"use client";

import { motion } from "framer-motion";
import { BadgeCheck, Clock, Eye, Scale } from "lucide-react";

const fadeUp = {
  initial: { opacity: 0, y: 24 },
  whileInView: { opacity: 1, y: 0 },
  viewport: { once: true, margin: "-60px" },
  transition: { duration: 0.7, ease: [0.22, 1, 0.36, 1] },
};

export function PhilosophyBento() {
  return (
    <section className="bg-white py-24 md:py-32">
      <div className="mx-auto max-w-[min(100%,1120px)] px-5">
        <div className="mb-14 flex flex-col gap-5 md:mb-20 md:flex-row md:items-end md:justify-between">
          <div>
            <p className="hv-eyebrow">Philosophy</p>
            <motion.h2
              {...fadeUp}
              className="mt-3 max-w-2xl font-[family-name:var(--font-display)] hv-display-lg text-[#1D1D1F]"
            >
              A marketplace built the way{" "}
              <span className="hv-gradient-text">luxury should feel</span>.
            </motion.h2>
          </div>
          <motion.p
            {...fadeUp}
            transition={{ ...fadeUp.transition, delay: 0.1 }}
            className="max-w-sm text-[17px] leading-[1.45] text-[#6E6E73]"
          >
            Four principles that separate a trustworthy resale platform from a
            risky one.
          </motion.p>
        </div>

        <div className="grid gap-4 md:grid-cols-6 md:gap-5">
          {/* Hero bento */}
          <motion.div
            {...fadeUp}
            className="hv-card relative overflow-hidden md:col-span-4 md:row-span-2"
          >
            <div className="relative flex h-full flex-col justify-between p-8 md:p-12">
              <div>
                <BadgeCheck className="h-10 w-10 text-[#FF3B00]" strokeWidth={1.65} />
                <h3 className="mt-10 font-[family-name:var(--font-display)] text-[32px] font-bold leading-[1.05] text-[#1D1D1F] md:text-[40px]">
                  Verified or it doesn&apos;t exist.
                </h3>
                <p className="mt-5 max-w-md text-[16px] leading-[1.5] text-[#6E6E73]">
                  Every listing passes a DINOv2-Giant classifier before it&apos;s
                  visible to buyers. Below 95% confidence? It never goes live.
                  There is no &ldquo;maybe&rdquo; on HypeVault.
                </p>
              </div>
              <div className="mt-10 inline-flex w-fit items-center gap-3 rounded-full bg-[#F5F5F7] px-4 py-2">
                <span className="relative flex h-2 w-2">
                  <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-[#00A652]/70" />
                  <span className="relative inline-flex h-2 w-2 rounded-full bg-[#00A652]" />
                </span>
                <span className="text-[13px] font-semibold text-[#1D1D1F]">
                  99.8% Precision
                </span>
              </div>
            </div>
          </motion.div>

          {/* Scale */}
          <motion.div
            {...fadeUp}
            transition={{ ...fadeUp.transition, delay: 0.1 }}
            className="hv-card md:col-span-2"
          >
            <div className="p-8">
              <Scale className="h-8 w-8 text-[#1D1D1F]" strokeWidth={1.65} />
              <h3 className="mt-6 font-[family-name:var(--font-display)] text-[22px] font-semibold text-[#1D1D1F]">
                One surface, three markets.
              </h3>
              <p className="mt-3 text-[15px] leading-[1.5] text-[#6E6E73]">
                StockX, Chrono24, eBay — compared side by side. No tab-hopping.
              </p>
            </div>
          </motion.div>

          {/* Eye */}
          <motion.div
            {...fadeUp}
            transition={{ ...fadeUp.transition, delay: 0.18 }}
            className="hv-card md:col-span-2"
          >
            <div className="p-8">
              <Eye className="h-8 w-8 text-[#1D1D1F]" strokeWidth={1.65} />
              <h3 className="mt-6 font-[family-name:var(--font-display)] text-[22px] font-semibold text-[#1D1D1F]">
                Geometry, not guesswork.
              </h3>
              <p className="mt-3 text-[15px] leading-[1.5] text-[#6E6E73]">
                The model inspects 3D structure — not colours a counterfeiter can
                match in Photoshop.
              </p>
            </div>
          </motion.div>

          {/* Wide bottom — Speed */}
          <motion.div
            {...fadeUp}
            transition={{ ...fadeUp.transition, delay: 0.24 }}
            className="hv-card md:col-span-6"
          >
            <div className="flex flex-col gap-8 p-8 md:flex-row md:items-center md:justify-between md:p-10">
              <div className="flex items-start gap-5">
                <Clock className="h-8 w-8 shrink-0 text-[#1D1D1F]" strokeWidth={1.65} />
                <div>
                  <h3 className="font-[family-name:var(--font-display)] text-[22px] font-semibold text-[#1D1D1F]">
                    Inference + pricing in under 5 seconds.
                  </h3>
                  <p className="mt-2 max-w-xl text-[15px] leading-[1.5] text-[#6E6E73]">
                    TensorRT FP16 on NVIDIA Triton with dynamic batching. The
                    tech stack the biggest marketplaces use — with none of the
                    opacity.
                  </p>
                </div>
              </div>
              <div className="flex gap-10 md:gap-14">
                <Stat label="Inference" value="<200ms" />
                <Stat label="Full verdict" value="<5s" />
                <Stat label="Uptime" value="99.9%" />
              </div>
            </div>
          </motion.div>
        </div>
      </div>
    </section>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <p className="text-[12px] font-medium text-[#86868B]">{label}</p>
      <p className="mt-1 font-[family-name:var(--font-display)] text-[24px] font-bold text-[#1D1D1F] tabular-nums">
        {value}
      </p>
    </div>
  );
}
