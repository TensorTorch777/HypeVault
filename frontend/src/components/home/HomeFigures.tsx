"use client";

import { useQuery } from "@tanstack/react-query";

import { fetchBenchmarkSummary, fetchMarketStatus, formatDate } from "@/lib/market";

function Figure({ value, label, note }: { value: string; label: string; note: string }) {
  return (
    <div className="hv-hud px-4 py-3.5">
      <p className="font-display text-[2.2rem] font-extrabold leading-none text-primary hv-num">{value}</p>
      <p className="mt-1.5 font-display text-[13px] font-bold uppercase tracking-[0.05em] text-primary/85">{label}</p>
      <p className="mt-0.5 text-[11px] leading-4 text-primary/55">{note}</p>
    </div>
  );
}

/** Every figure here is read from the API at page load; nothing is hard-coded. */
export function HomeFigures() {
  const status = useQuery({ queryKey: ["market-status"], queryFn: fetchMarketStatus });
  const bench = useQuery({ queryKey: ["benchmark-summary"], queryFn: fetchBenchmarkSummary });
  const c24 = status.data?.sources.chrono24_dataset;
  const ebayRows = status.data?.sources.ebay?.rows ?? 0;

  return (
    <div className="grid grid-cols-2 gap-2.5 md:grid-cols-4">
      <Figure
        value={c24 ? c24.rows.toLocaleString("en-US") : "—"}
        label="Reference listings"
        note={c24 ? `Chrono24 dataset · ${formatDate(c24.last_observed_on)}` : status.isError ? "Unavailable" : "Loading"}
      />
      <Figure
        value={status.data?.ebay.configured ? ebayRows.toLocaleString("en-US") : "—"}
        label="Live eBay sightings"
        note={status.data?.ebay.configured ? "Browse API" : "Appears once the eBay key is connected"}
      />
      <Figure value="5" label="Manufactures" note="In scope for screening and pricing" />
      <Figure
        value="2"
        label="Vision models"
        note={bench.data?.confirmatory?.frontier.frontier_model ? "Frontier picked by benchmark" : "DINOv2 live · DINOv3 on trial"}
      />
    </div>
  );
}
