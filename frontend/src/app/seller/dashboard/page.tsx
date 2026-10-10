"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { useQuery } from "@tanstack/react-query";
import { useEffect, type ReactNode } from "react";
import { LayoutGrid, LogOut, Plus, ShieldAlert, Eye } from "lucide-react";

import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { api, fetchMe, logoutRemote, type Listing } from "@/lib/api";
import { customerStatusCopy } from "@/lib/customerLabel";

export default function SellerDashboardPage() {
  const router = useRouter();
  const meQuery = useQuery({
    queryKey: ["auth-me-seller-dashboard"],
    queryFn: fetchMe,
    retry: false,
  });
  const me = meQuery.data;

  useEffect(() => {
    if (meQuery.isError) router.replace("/login");
    else if (me && me.role !== "seller") router.replace("/");
  }, [me, meQuery.isError, router]);

  const q = useQuery({
    queryKey: ["my-listings"],
    enabled: me?.role === "seller",
    queryFn: async () => {
      const { data } = await api.get<Listing[]>("/listings/");
      return data;
    },
  });

  if (meQuery.isError || meQuery.isLoading || !me) {
    return (
      <div className="mx-auto max-w-2xl px-4 py-12">
        <Card className="hover:translate-y-0">
          <CardHeader>
            <h1 className="text-2xl font-semibold not-italic">Seller dashboard</h1>
          </CardHeader>
          <CardContent className="space-y-4">
            <p className="text-sm text-primary/65">Log in as a seller to manage listings and research classifications.</p>
            <div className="flex flex-wrap gap-3">
              <Link href="/login">
                <Button className="min-h-[44px]">Log in</Button>
              </Link>
              <Link href="/register">
                <Button variant="outline" className="min-h-[44px]">
                  Register
                </Button>
              </Link>
            </div>
          </CardContent>
        </Card>
      </div>
    );
  }

  const listings = q.data ?? [];
  const liveCount = listings.filter((listing) => listing.status === "live").length;
  const rejectedCount = listings.filter((listing) => listing.status === "rejected").length;

  return (
    <div className="mx-auto w-full max-w-6xl px-4 py-10 md:px-6 md:py-14">
      <header className="flex flex-col justify-between gap-6 md:flex-row md:items-end">
        <div className="max-w-2xl">
          <p className="text-[11px] font-semibold uppercase tracking-[0.22em] text-[#FFB089]">Seller</p>
          <h1 className="mt-2 font-[family-name:var(--font-display)] text-3xl font-semibold not-italic tracking-tight text-white md:text-4xl">
            Dashboard
          </h1>
          <p className="mt-3 text-sm leading-6 text-primary/65">
            Listings and their publication labels. A live listing is visible, not verified.
          </p>
        </div>
        <div className="flex flex-wrap items-center gap-3">
          <Link href="/seller/upload">
            <Button className="min-h-[44px] gap-2">
              <Plus className="h-4 w-4" aria-hidden />
              New listing
            </Button>
          </Link>
          <Button
            type="button"
            variant="outline"
            className="min-h-[44px] gap-2"
            onClick={() => void logoutRemote().then(() => router.push("/login"))}
          >
            <LogOut className="h-4 w-4" aria-hidden />
            Log out
          </Button>
        </div>
      </header>

      <div className="mt-8 grid gap-4 md:grid-cols-3">
        <SummaryCard icon={<LayoutGrid className="h-4 w-4" />} label="Listings" value={q.isLoading ? "—" : listings.length} detail="Rows returned for this account." />
        <SummaryCard icon={<Eye className="h-4 w-4" />} label="Live" value={q.isLoading ? "—" : liveCount} detail="Visible in the marketplace. Not verified authentic." />
        <SummaryCard icon={<ShieldAlert className="h-4 w-4" />} label="Rejected" value={q.isLoading ? "—" : rejectedCount} detail="Rejected by the screening workflow. Not proof of counterfeit." />
      </div>

      <section className="mt-12">
        <h2 className="text-lg font-semibold not-italic text-white">Your listings</h2>
        <div className="mt-4 grid gap-3">
          {q.isLoading ? (
            <>
              <Skeleton className="h-28 w-full rounded-2xl" />
              <Skeleton className="h-28 w-full rounded-2xl" />
            </>
          ) : null}
          {q.isError ? (
            <div className="rounded-2xl border border-[#ffb4ab]/30 bg-[#2a1216] p-6" role="alert">
              <p className="font-semibold text-[#ffd7d2]">Could not load your listings</p>
              <p className="mt-2 text-sm text-white/70">The request failed. This is not an empty catalog.</p>
              <Button className="mt-4 min-h-[44px]" variant="outline" onClick={() => void q.refetch()}>
                Retry
              </Button>
            </div>
          ) : null}
          {!q.isLoading && !q.isError && listings.length === 0 ? (
            <div className="rounded-2xl border border-dashed border-white/15 bg-white/[0.03] px-6 py-12 text-center">
              <p className="text-lg font-semibold text-white">No listings yet</p>
              <p className="mx-auto mt-2 max-w-md text-sm text-white/60">
                When you create one, it stays pending until legacy screening finishes. A result is not a verified authentic certificate.
              </p>
              <Link href="/seller/upload" className="mt-6 inline-flex">
                <Button className="min-h-[44px]">New listing</Button>
              </Link>
            </div>
          ) : null}
          {q.data?.map((listing) => {
            const status = customerStatusCopy(listing);
            return (
              <article key={listing.id} className="rounded-2xl border border-white/10 bg-[#140a22] p-5 transition-colors duration-200 hover:border-white/20">
                <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
                  <div className="min-w-0">
                    <p className="truncate text-base font-semibold text-white">{listing.product_name}</p>
                    <p className="mt-1 text-sm text-white/55">
                      {listing.category}
                      {listing.brand ? ` · Declared brand: ${listing.brand}` : ""}
                    </p>
                    <p className="mt-3 inline-flex max-w-full rounded-full border border-white/10 bg-white/[0.04] px-3 py-1 text-xs font-medium text-[#FFEDF6]">
                      {status.title}
                    </p>
                    <p className="mt-2 text-xs leading-5 text-white/45">{status.detail}</p>
                  </div>
                  <Link href={`/product/${listing.id}`} className="shrink-0">
                    <Button variant="outline" className="min-h-[44px] w-full sm:w-auto">
                      View
                    </Button>
                  </Link>
                </div>
              </article>
            );
          })}
        </div>
      </section>
    </div>
  );
}

function SummaryCard({
  icon,
  label,
  value,
  detail,
}: {
  icon: ReactNode;
  label: string;
  value: number | string;
  detail: string;
}) {
  return (
    <div className="rounded-2xl border border-white/10 bg-[#140a22] p-5 shadow-[inset_0_1px_0_rgba(255,255,255,0.04)]">
      <div className="flex items-center justify-between text-[#FFB089]">
        <p className="text-[11px] font-semibold uppercase tracking-[0.18em]">{label}</p>
        {icon}
      </div>
      <p className="mt-4 font-[family-name:var(--font-display)] text-4xl font-semibold not-italic tracking-tight text-white">{value}</p>
      <p className="mt-3 text-xs leading-5 text-white/50">{detail}</p>
    </div>
  );
}
