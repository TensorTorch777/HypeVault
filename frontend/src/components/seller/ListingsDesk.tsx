"use client";

import Link from "next/link";
import { useQuery } from "@tanstack/react-query";
import { useMemo, useState, type ReactNode } from "react";
import { ArrowRight } from "lucide-react";

import { PageHero } from "@/components/layout/PageHero";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import { api, type Listing } from "@/lib/api";
import { customerStatusCopy } from "@/lib/customerLabel";
import { formatDate } from "@/lib/market";
import { cn } from "@/lib/utils";

type Filter = "all" | "live" | "rejected" | "pending";

const FILTERS: { id: Filter; label: string }[] = [
  { id: "all", label: "All" },
  { id: "live", label: "Listed" },
  { id: "rejected", label: "Not listed" },
  { id: "pending", label: "Held" },
];

const TONE: Record<string, string> = {
  live: "bg-success/15 text-success",
  rejected: "bg-danger/15 text-danger",
  pending: "bg-review/15 text-review",
};

/** Stats, filters and listing cards shared by the seller desk (own listings) and the admin desk (every listing). */
export function ListingsDesk({
  eyebrow,
  title,
  subtitle,
  actions,
  endpoint,
  queryKey,
  totalNote,
  showSeller = false,
  beforeListings,
}: {
  eyebrow: string;
  title: string;
  subtitle: ReactNode;
  actions: ReactNode;
  endpoint: string;
  queryKey: readonly unknown[];
  totalNote: string;
  /** Show whose listing each card is. Only the admin desk loads listings from other sellers. */
  showSeller?: boolean;
  /** Rendered between the stats and the listing cards (the admin review queue). */
  beforeListings?: ReactNode;
}) {
  const [filter, setFilter] = useState<Filter>("all");
  const q = useQuery({
    queryKey,
    queryFn: async () => (await api.get<Listing[]>(endpoint)).data,
  });

  const listings = useMemo(() => q.data ?? [], [q.data]);
  const counts = useMemo(
    () => ({
      all: listings.length,
      live: listings.filter((l) => l.status === "live").length,
      rejected: listings.filter((l) => l.status === "rejected").length,
      pending: listings.filter((l) => l.status === "pending").length,
    }),
    [listings]
  );
  const shown = filter === "all" ? listings : listings.filter((l) => l.status === filter);

  return (
    <div className="pb-24">
      <PageHero eyebrow={eyebrow} title={title} subtitle={subtitle} scene="penthouse" render="/renders/lange_one/hero.webp" actions={actions} />

      <div className="mx-auto max-w-[73.75rem] px-4 sm:px-6">
        {/* Stats */}
        <div className="-mt-2 grid grid-cols-2 gap-3 md:grid-cols-4">
          {(
            [
              ["Listings", counts.all, totalNote, "text-primary"],
              ["Listed", counts.live, "Visible to buyers", "text-success"],
              ["Not listed", counts.rejected, "Stopped by screening, with a reason", "text-danger"],
              ["Held", counts.pending, "Not live yet: waiting for a check", "text-review"],
            ] as const
          ).map(([label, value, note, tone]) => (
            <div key={label} className="hv-hud px-5 py-4">
              <p className={cn("font-display text-[2.6rem] font-extrabold leading-none hv-num", tone)}>{q.isLoading ? "—" : value}</p>
              <p className="mt-1 font-display text-[0.875rem] font-bold uppercase tracking-[0.05em] text-primary/85">{label}</p>
              <p className="text-[0.75rem] text-primary/50">{note}</p>
            </div>
          ))}
        </div>

      </div>
      {beforeListings}
      <div className="mx-auto max-w-[73.75rem] px-4 sm:px-6">
        {/* Filters */}
        <div className="mt-12 flex flex-wrap items-center justify-between gap-4">
          <h2 className="font-display text-[2rem] font-extrabold uppercase leading-none text-primary">Listings</h2>
          <div className="flex flex-wrap gap-2" role="tablist" aria-label="Filter listings">
            {FILTERS.map((f) => (
              <button
                key={f.id}
                type="button"
                role="tab"
                aria-selected={filter === f.id}
                onClick={() => setFilter(f.id)}
                className={cn(
                  "min-h-[2.25rem] cursor-pointer rounded-full border px-4 text-[0.8125rem] font-bold transition-colors",
                  filter === f.id ? "border-accent bg-accent text-ink" : "border-white/15 text-primary/70 hover:border-white/40 hover:text-primary"
                )}
              >
                {f.label} <span className="opacity-60">{counts[f.id]}</span>
              </button>
            ))}
          </div>
        </div>

        <div className="mt-6 grid gap-4 md:grid-cols-2">
          {q.isLoading ? (
            <>
              <Skeleton className="h-44 rounded-[1.25rem]" />
              <Skeleton className="h-44 rounded-[1.25rem]" />
            </>
          ) : null}
          {q.isError ? (
            <div className="hv-hud p-6 md:col-span-2" role="alert">
              <p className="font-display text-[1.4rem] font-bold uppercase text-danger">Could not load listings</p>
              <p className="mt-1 text-sm text-primary/70">The request failed. This is not an empty catalog.</p>
              <Button className="mt-4" variant="outline" onClick={() => void q.refetch()}>
                Retry
              </Button>
            </div>
          ) : null}
          {!q.isLoading && !q.isError && shown.length === 0 ? (
            <div className="rounded-[1.25rem] border border-dashed border-white/15 px-6 py-14 text-center md:col-span-2">
              <p className="font-display text-[1.8rem] font-extrabold uppercase text-primary">Nothing here yet</p>
              {showSeller ? (
                <p className="mx-auto mt-2 max-w-md text-sm text-primary/60">No seller has listed a watch yet.</p>
              ) : (
                <>
                  <p className="mx-auto mt-2 max-w-md text-sm text-primary/60">Upload a photo of a watch from one of the five houses. Screening takes a few seconds.</p>
                  <Link href="/seller/upload" className="hv-btn-pill hv-btn-primary mt-6">
                    New listing
                  </Link>
                </>
              )}
            </div>
          ) : null}
          {shown.map((listing) => {
            const status = customerStatusCopy(listing);
            return (
              <Link
                key={listing.id}
                href={`/product/${listing.id}`}
                className="group grid grid-cols-[7rem_1fr] overflow-hidden rounded-[1.25rem] border border-white/10 bg-card transition-[border-color,transform] duration-300 hover:-translate-y-0.5 hover:border-accent/40 sm:grid-cols-[8.75rem_1fr]"
              >
                <div className="hv-sky relative">
                  {listing.s3_url ? (
                    // eslint-disable-next-line @next/next/no-img-element
                    <img
                      src={listing.s3_url}
                      alt=""
                      className="absolute inset-0 h-full w-full object-cover"
                      loading="lazy"
                      onError={(e) => (e.currentTarget.style.display = "none")}
                    />
                  ) : null}
                </div>
                <div className="min-w-0 p-5">
                  <div className="flex items-start justify-between gap-3">
                    <span className={cn("rounded-full px-2.5 py-0.5 text-[0.6875rem] font-bold", TONE[listing.status] ?? "bg-white/10 text-primary")}>{status.title}</span>
                    <span className="text-[0.6875rem] text-primary/45">{formatDate(listing.created_at)}</span>
                  </div>
                  <p className="mt-3 truncate font-display text-[1.45rem] font-extrabold uppercase leading-none text-primary">{listing.product_name}</p>
                  <p className="mt-1 text-[0.75rem] text-primary/55">{listing.brand ?? "No brand declared"}</p>
                  {showSeller ? <p className="mt-1 truncate text-[0.75rem] text-primary/55">Seller: {listing.seller_email ?? "unknown"}</p> : null}
                  <p className="mt-3 line-clamp-2 text-[0.75rem] leading-5 text-primary/60">{status.detail}</p>
                  <span className="mt-3 inline-flex items-center gap-1 text-[0.75rem] font-bold text-accent transition-[gap] group-hover:gap-2">
                    View <ArrowRight className="h-3.5 w-3.5" aria-hidden="true" />
                  </span>
                </div>
              </Link>
            );
          })}
        </div>
      </div>
    </div>
  );
}
