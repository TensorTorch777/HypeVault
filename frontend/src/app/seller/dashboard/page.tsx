"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { useQuery } from "@tanstack/react-query";
import { useEffect } from "react";

import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { api, fetchMe, logoutRemote, type Listing } from "@/lib/api";

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
        <Card>
          <CardHeader>
            <h1 className="text-2xl font-semibold">Seller dashboard</h1>
          </CardHeader>
          <CardContent className="space-y-4">
            <p className="text-sm text-primary/65">Log in as a seller to manage listings and track verification.</p>
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

  return (
    <div className="mx-auto max-w-6xl px-4 py-12">
      <div className="flex flex-col justify-between gap-4 md:flex-row md:items-end">
        <div>
          <p className="text-xs font-semibold uppercase tracking-wider text-primary/45">Seller</p>
          <h1 className="mt-2 text-3xl font-semibold tracking-tight">Dashboard</h1>
          <p className="mt-2 text-sm text-primary/60">Listings, verification outcomes, and quick actions.</p>
        </div>
        <div className="flex flex-wrap items-center gap-3">
          <Link href="/seller/upload">
            <Button className="min-h-[44px]">New listing</Button>
          </Link>
          <Button
            type="button"
            variant="outline"
            className="min-h-[44px]"
            onClick={() => void logoutRemote().then(() => router.push("/login"))}
          >
            Log out
          </Button>
        </div>
      </div>

      <div className="mt-8 grid gap-4 md:grid-cols-3">
        <Card>
          <CardContent className="p-5">
            <p className="text-xs font-semibold uppercase tracking-wider text-primary/45">Listings</p>
            <p className="mt-3 text-3xl font-semibold">{q.data?.length ?? (q.isLoading ? "—" : 0)}</p>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="p-5">
            <p className="text-xs font-semibold uppercase tracking-wider text-primary/45">Live</p>
            <p className="mt-3 text-3xl font-semibold">
              {q.data ? q.data.filter((l) => l.status === "live").length : "—"}
            </p>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="p-5">
            <p className="text-xs font-semibold uppercase tracking-wider text-primary/45">Rejected</p>
            <p className="mt-3 text-3xl font-semibold">
              {q.data ? q.data.filter((l) => l.status === "rejected").length : "—"}
            </p>
          </CardContent>
        </Card>
      </div>

      <div className="mt-10">
        <h2 className="text-lg font-semibold">Your listings</h2>
        <div className="mt-4 grid gap-3">
          {q.isLoading ? (
            <>
              <Skeleton className="h-24 w-full" />
              <Skeleton className="h-24 w-full" />
            </>
          ) : null}
          {q.data?.map((l) => (
            <Link key={l.id} href={`/product/${l.id}`} prefetch>
              <Card>
                <CardContent className="flex items-center justify-between gap-4 p-5">
                  <div className="min-w-0">
                    <p className="truncate text-sm font-semibold">{l.product_name}</p>
                    <p className="mt-1 text-xs text-primary/55">
                      {l.category} • <span className="font-semibold text-primary">{l.status}</span>
                    </p>
                  </div>
                  <Button variant="outline" className="min-h-[44px] shrink-0">
                    View
                  </Button>
                </CardContent>
              </Card>
            </Link>
          ))}
        </div>
      </div>
    </div>
  );
}
