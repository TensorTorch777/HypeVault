"use client";

import { useId, useMemo, useState } from "react";
import {
  Area,
  AreaChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

type Range = "30D" | "90D" | "1Y";

function makeSeries(days: number, seed: string) {
  let h = 0;
  for (let i = 0; i < seed.length; i++) h = (h * 31 + seed.charCodeAt(i)) >>> 0;
  const base = 450;
  const start = new Date();
  start.setDate(start.getDate() - days);

  return Array.from({ length: days }, (_, i) => {
    const wave = Math.sin(i / 4 + h * 0.01) * 45 + Math.cos(i / 7) * 28;
    const noise = (((h >> (i % 12)) & 31) - 15) * 2.5;
    const drift = (i / Math.max(days - 1, 1)) * 24 * (h % 2 === 0 ? 1 : -1);
    const raw = base + wave + noise + drift;
    const price = Math.round(Math.min(base + 80, Math.max(base - 80, raw)));
    const d = new Date(start);
    d.setDate(d.getDate() + i);
    const dateLabel = d.toLocaleDateString(undefined, { month: "short", day: "numeric" });
    return {
      day: i + 1,
      dateLabel,
      ts: d.getTime(),
      price,
    };
  });
}

export function PriceTrendChart({ titleSeed }: { titleSeed: string }) {
  const gid = useId().replace(/:/g, "");
  const gradId = `priceFill-${gid}`;
  const [range, setRange] = useState<Range>("30D");
  const days = range === "30D" ? 30 : range === "90D" ? 90 : 365;
  const data = useMemo(() => makeSeries(days, titleSeed), [days, titleSeed]);
  const prices = data.map((d) => d.price);
  const minP = Math.min(...prices);
  const maxP = Math.max(...prices);

  return (
    <div className="w-full">
      <div className="mb-3 flex gap-6 border-b border-white/[0.06] pb-2">
        {(["30D", "90D", "1Y"] as const).map((r) => (
          <button
            key={r}
            type="button"
            onClick={() => setRange(r)}
            className={`text-xs font-medium uppercase tracking-wider transition-all duration-150 ${
              range === r ? "text-white underline decoration-[#FF3B00] decoration-2 underline-offset-8" : "text-[#888888]"
            }`}
          >
            {r}
          </button>
        ))}
      </div>
      <div className="h-52 w-full md:h-56">
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart data={data} margin={{ top: 8, right: 8, left: 0, bottom: 0 }}>
            <defs>
              <linearGradient id={gradId} x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor="#FF3B00" stopOpacity={0.4} />
                <stop offset="100%" stopColor="#FF3B00" stopOpacity={0} />
              </linearGradient>
            </defs>
            <CartesianGrid strokeDasharray="3 6" stroke="rgba(255,255,255,0.06)" horizontal vertical={false} />
            <XAxis dataKey="day" hide />
            <YAxis
              domain={[minP - 20, maxP + 20]}
              orientation="right"
              axisLine={false}
              tickLine={false}
              tick={{ fill: "#888888", fontSize: 11 }}
              ticks={[minP, maxP]}
              tickFormatter={(v) => `$${v}`}
            />
            <Tooltip
              contentStyle={{
                background: "#141414",
                border: "1px solid rgba(255,255,255,0.1)",
                borderRadius: 8,
              }}
              labelFormatter={(_, payload) => {
                const p = payload?.[0]?.payload as { dateLabel?: string } | undefined;
                return p?.dateLabel ?? "";
              }}
              formatter={(value: number) => [`$${value}`, "Price"]}
            />
            <Area
              type="monotone"
              dataKey="price"
              stroke="#FF3B00"
              strokeWidth={2}
              fill={`url(#${gradId})`}
              dot={false}
            />
          </AreaChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
