"use client";

import { useMemo } from "react";
import { useQuery } from "@tanstack/react-query";

import { AuthBadge } from "@/components/AuthBadge";
import { ComparisonTable } from "@/components/ComparisonTable";
import { PriceTrendChart } from "@/components/PriceTrendChart";
import { ProductVisualPanel } from "@/components/product/ProductVisualPanel";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import { fetchComparison, fetchComparisonByQuery, fetchListing } from "@/lib/api";

const UUID_RE =
  /^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i;

function inferCategory(title: string, listingCategory?: string): string {
  if (listingCategory === "sneaker" || listingCategory === "watch") return listingCategory;
  const t = title.toLowerCase();
  if (t.includes("watch")) return "watch";
  return "sneaker";
}

export default function ProductPage({ params }: { params: { id: string } }) {
  const raw = decodeURIComponent(params.id);
  const isUuid = UUID_RE.test(raw);

  const listingQ = useQuery({
    queryKey: ["listing", raw],
    queryFn: () => fetchListing(raw),
    enabled: isUuid,
  });

  const title = isUuid ? listingQ.data?.product_name ?? "Product" : raw;
  const category = inferCategory(title, listingQ.data?.category);

  const comparisonKey = isUuid
    ? ["comparison", "listing", listingQ.data?.id ?? raw]
    : ["comparison", "q", title];

  const compQ = useQuery({
    queryKey: comparisonKey,
    queryFn: async () => {
      if (isUuid) {
        return await fetchComparison(raw);
      }
      return await fetchComparisonByQuery(title);
    },
    enabled: !isUuid || listingQ.isFetched,
  });

  const mockSeed = useMemo(() => (isUuid ? listingQ.data?.id ?? raw : title), [isUuid, listingQ.data?.id, raw, title]);

  return (
    <div className="mx-auto max-w-6xl px-4 py-10 pb-28 md:pb-16">
      <div>
        <div className="flex flex-col gap-8 lg:grid lg:grid-cols-12 lg:items-stretch">
          <div className="flex flex-col lg:col-span-6">
            <ProductVisualPanel
              title={title}
              category={category}
              listing={listingQ.data}
              loading={Boolean(isUuid && listingQ.isLoading)}
              isUuid={isUuid}
            />
            <div className="mt-4 flex flex-wrap gap-2">
              <Button
                className="min-h-[44px]"
                variant="outline"
                onClick={() => {
                  const url = typeof window !== "undefined" ? window.location.href : "";
                  void navigator.clipboard.writeText(url);
                }}
              >
                Share
              </Button>
            </div>
          </div>

          <div className="flex flex-col lg:col-span-6">
            <p className="hv-section-label">Product</p>
            <h1 className="mt-0 font-[family-name:var(--font-display)] text-3xl font-extrabold tracking-tight md:text-4xl">
              {title}
            </h1>

            <div className="mt-6">
              {isUuid && listingQ.isLoading ? <Skeleton className="h-40 w-full rounded-2xl" /> : null}
              {isUuid && listingQ.data ? (
                <AuthBadge
                  verdict={(listingQ.data.verdict as "AUTHENTIC" | "FAKE" | null) ?? null}
                  confidence={listingQ.data.confidence}
                />
              ) : null}
              {!isUuid ? (
                <div className="rounded-2xl border border-white/[0.08] bg-[#111111] p-6 text-sm text-primary/65">
                  You’re viewing a quick market comparison for{" "}
                  <span className="font-semibold text-primary">{title}</span>. Create a listing to run AI verification.
                </div>
              ) : null}
            </div>

            <div className="mt-8">
              <p className="hv-section-label">Price trend</p>
              <div className="mt-0 rounded-2xl border border-white/[0.08] bg-transparent p-3 md:p-4">
                <PriceTrendChart titleSeed={title} />
              </div>
            </div>

            <div className="mt-6 grid gap-3 text-sm text-primary/65">
              {isUuid && listingQ.data ? (
                <>
                  <div className="flex justify-between border-b border-white/[0.06] py-2">
                    <span className="text-primary/50">Brand</span>
                    <span className="font-semibold text-primary">{listingQ.data.brand ?? "—"}</span>
                  </div>
                  <div className="flex justify-between border-b border-white/[0.06] py-2">
                    <span className="text-primary/50">Condition</span>
                    <span className="font-semibold text-primary">{listingQ.data.condition ?? "—"}</span>
                  </div>
                  <div className="flex justify-between border-b border-white/[0.06] py-2">
                    <span className="text-primary/50">Size</span>
                    <span className="font-semibold text-primary">{listingQ.data.size ?? "—"}</span>
                  </div>
                </>
              ) : null}
            </div>
          </div>
        </div>

        <div className="mt-12 border-t border-white/[0.06] pt-12 md:mt-16 md:pt-16">
          <p className="hv-section-label">Compare market pricing</p>
          <h2 className="font-[family-name:var(--font-display)] text-xl font-extrabold tracking-tight md:text-2xl">
            Live market snapshot
          </h2>
          <p className="mt-2 text-sm text-primary/60">
            Side-by-side reference pricing. HypeVault highlights strong offers — this is not an auction flow.
          </p>
          <div className="mt-6">
            <ComparisonTable
              data={compQ.data}
              isLoading={compQ.isLoading}
              isError={compQ.isError}
              listingId={isUuid ? raw : undefined}
              queryKey={comparisonKey}
              mockSeed={mockSeed}
            />
          </div>
        </div>
      </div>
    </div>
  );
}
