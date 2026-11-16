"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { useQuery } from "@tanstack/react-query";
import { useEffect } from "react";
import { ArrowRight, Plus, ShieldCheck } from "lucide-react";

import { PageHero } from "@/components/layout/PageHero";
import { ListingsDesk } from "@/components/seller/ListingsDesk";
import { ME_QUERY_KEY, fetchMe, logoutRemote } from "@/lib/api";
import { canSell, isAdmin } from "@/lib/roles";

export default function SellerDashboardPage() {
  const router = useRouter();
  const meQuery = useQuery({ queryKey: ME_QUERY_KEY, queryFn: fetchMe, retry: false });
  const me = meQuery.data;

  useEffect(() => {
    if (meQuery.isError) router.replace("/login");
  }, [meQuery.isError, router]);

  if (!me) {
    return <PageHero eyebrow="Seller desk" title="Loading…" scene="penthouse" compact />;
  }

  if (!canSell(me.role)) {
    return (
      <PageHero
        eyebrow="Seller desk"
        title="Not a seller account"
        subtitle={
          <>
            Signed in as <span className="font-bold text-white">{me.email}</span>, which cannot list watches. Sign in with your
            Google account to sell.
          </>
        }
        scene="penthouse"
        compact
        actions={
          <Link href="/market" className="hv-btn-pill hv-btn-secondary">
            Browse the market <ArrowRight className="h-4 w-4" />
          </Link>
        }
      />
    );
  }

  // Every seller, the admin included, sees only their own listings here. The admin reviews everyone's at /admin.
  return (
    <ListingsDesk
      eyebrow="Seller desk"
      title="Your listings"
      subtitle={
        <>
          Signed in as <span className="font-bold text-white">{me.email}</span>. Listed watches passed photo screening; screening is not a
          certificate of authenticity.
        </>
      }
      endpoint="/listings/"
      queryKey={["my-listings", me.id]}
      totalNote="On this account"
      actions={
        <>
          <Link href="/seller/upload" className="hv-btn-pill hv-btn-primary">
            <Plus className="h-4 w-4" aria-hidden="true" /> New listing
          </Link>
          {isAdmin(me.role) ? (
            <Link href="/admin" className="hv-btn-pill hv-btn-secondary">
              <ShieldCheck className="h-4 w-4" aria-hidden="true" /> Admin desk
            </Link>
          ) : null}
          <button type="button" className="hv-btn-pill hv-btn-secondary" onClick={() => void logoutRemote().then(() => router.push("/login"))}>
            Sign out
          </button>
        </>
      }
    />
  );
}
