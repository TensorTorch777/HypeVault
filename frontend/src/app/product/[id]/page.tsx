"use client";

import Link from "next/link";
import { useEffect } from "react";
import { useRouter } from "next/navigation";
import { useQuery } from "@tanstack/react-query";
import axios from "axios";

import { Chrono24Results } from "@/components/market/Chrono24Results";
import { MarketPanel } from "@/components/market/MarketPanel";
import { ProductVisualPanel } from "@/components/product/ProductVisualPanel";
import { Skeleton } from "@/components/ui/skeleton";
import { fetchListing } from "@/lib/api";
import { customerStatusCopy } from "@/lib/customerLabel";
import { fetchListingMarket, formatDate } from "@/lib/market";

const UUID_RE = /^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i;

export default function ProductPage({ params }: { params: { id: string } }) {
  const router = useRouter();
  const raw = decodeURIComponent(params.id);
  const isUuid = UUID_RE.test(raw);

  // Older links used /product/<search text>; searches now live on /market.
  useEffect(() => {
    if (!isUuid) router.replace(`/market?q=${encodeURIComponent(raw)}`);
  }, [isUuid, raw, router]);

  const listingQ = useQuery({
    queryKey: ["listing", raw],
    queryFn: () => fetchListing(raw),
    enabled: isUuid,
    // A 404 is an answer (missing, or not listed and not yours), not a glitch to retry.
    retry: (count, err) => !(axios.isAxiosError(err) && err.response?.status === 404) && count < 2,
  });
  const marketQ = useQuery({
    queryKey: ["listing-market", raw],
    queryFn: () => fetchListingMarket(raw),
    enabled: isUuid && listingQ.data?.status === "live",
  });

  if (!isUuid) return <div className="min-h-screen" />;
  const listing = listingQ.data;
  const status = listing ? customerStatusCopy(listing) : null;
  const unlisted = listing != null && listing.status !== "live";

  return (
    <div className="mx-auto max-w-[77.5rem] px-4 pb-28 pt-28 sm:px-6">
      <div className="grid gap-12 lg:grid-cols-2">
        <ProductVisualPanel
          title={listing?.product_name ?? "Listing"}
          category={listing?.category ?? "watch"}
          listing={listing}
          loading={listingQ.isLoading}
          isUuid
        />

        <div className="flex flex-col">
          {listingQ.isLoading ? <Skeleton className="h-64 rounded-2xl" /> : null}
          {listingQ.isError ? (
            <div role="alert" className="hv-hud p-7">
              <p className="hv-serif text-[1.9rem] uppercase leading-tight text-primary">Listing not found</p>
              <p className="mt-3 text-[0.875rem] leading-7 text-primary/75">
                This listing could not be found. It may have been removed, or it did not pass screening and is visible only to its seller.
              </p>
              <Link href="/market" className="hv-btn-pill hv-btn-secondary mt-6">Browse the market</Link>
            </div>
          ) : null}
          {listing && status ? (
            <>
              <p className="hv-eyebrow">{listing.brand ?? "Watch"}</p>
              <h1 className="hv-display-lg mt-4 text-primary">{listing.product_name}</h1>

              <div className={`hv-hud mt-10 p-7 ${unlisted ? "!border-danger/50" : ""}`}>
                <p className="hv-mono text-[0.6875rem] uppercase tracking-[0.18em] text-primary/55">
                  {unlisted ? "Only you can see this listing" : "Listing status"}
                </p>
                <p className={`hv-serif mt-3 text-[1.9rem] uppercase leading-tight ${unlisted ? "text-danger" : "text-primary"}`}>{status.title}</p>
                <p className="mt-3 text-[0.875rem] leading-7 text-primary/80">{status.detail}</p>
                {unlisted ? (
                  <Link href="/seller/upload" className="hv-btn-pill hv-btn-primary mt-6">
                    List with a new photo
                  </Link>
                ) : null}
              </div>

              <dl className="mt-8 divide-y divide-primary/10 border-y border-primary/10 text-[0.8125rem]">
                {[
                  ["Manufacture", listing.brand ? `${listing.brand} (declared by the seller)` : "Not supplied"],
                  ["Condition", listing.condition ?? "—"],
                  ["Size", listing.size ?? "—"],
                  ["Listed", formatDate(listing.created_at)],
                ].map(([k, v]) => (
                  <div key={k} className="flex justify-between gap-6 py-4">
                    <dt className="text-primary/55">{k}</dt>
                    <dd className="text-right text-primary">{v}</dd>
                  </div>
                ))}
              </dl>
            </>
          ) : null}
        </div>
      </div>

      {listing?.status !== "live" ? null : (
      <section aria-labelledby="market-title" className="mt-24 border-t border-primary/10 pt-16">
        <p className="hv-eyebrow">The market for this watch</p>
        <h2 id="market-title" className="hv-display-md mt-4 text-primary">What comparable pieces are asking</h2>
        <p className="mt-4 max-w-2xl text-[0.875rem] leading-7 text-primary/65">
          Matched on the listing title and declared manufacture. Prices are asking prices, not sale prices, and they say
          nothing about whether this particular watch is genuine.
        </p>
        <div className="mt-10">
          <MarketPanel data={marketQ.data} loading={listingQ.isLoading || marketQ.isLoading} error={marketQ.isError} />
        </div>
        {listing?.id ? (
          <div className="mt-16">
            <Chrono24Results listingId={String(listing.id)} />
          </div>
        ) : null}
      </section>
      )}
    </div>
  );
}
