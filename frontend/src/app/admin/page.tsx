"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { useQuery } from "@tanstack/react-query";
import { useEffect } from "react";
import { ArrowRight } from "lucide-react";

import { PageHero } from "@/components/layout/PageHero";
import { ListingsDesk } from "@/components/seller/ListingsDesk";
import { ME_QUERY_KEY, fetchMe, logoutRemote } from "@/lib/api";
import { isAdmin } from "@/lib/roles";

/** Every seller's listings. The server serves /listings/all to the admin account only; this page just hides the desk from others. */
export default function AdminDeskPage() {
  const router = useRouter();
  const meQuery = useQuery({ queryKey: ME_QUERY_KEY, queryFn: fetchMe, retry: false });
  const me = meQuery.data;

  useEffect(() => {
    if (meQuery.isError) router.replace("/login");
  }, [meQuery.isError, router]);

  if (!me) {
    return <PageHero eyebrow="Admin desk" title="Loading…" scene="penthouse" compact />;
  }

  if (!isAdmin(me.role)) {
    return (
      <PageHero
        eyebrow="Admin desk"
        title="Admin only"
        subtitle={
          <>
            Signed in as <span className="font-bold text-white">{me.email}</span>, which is not the admin account. Your own listings are on
            the seller desk.
          </>
        }
        scene="penthouse"
        compact
        actions={
          <Link href="/seller/dashboard" className="hv-btn-pill hv-btn-secondary">
            Seller desk <ArrowRight className="h-4 w-4" />
          </Link>
        }
      />
    );
  }

  return (
    <ListingsDesk
      eyebrow="Admin desk · every seller"
      title="All listings"
      subtitle={
        <>
          Signed in as <span className="font-bold text-white">{me.email}</span> (admin). Every seller&apos;s listings, including the ones
          screening stopped.
        </>
      }
      endpoint="/listings/all"
      queryKey={["all-listings", me.id]}
      totalNote="Across every seller"
      showSeller
      actions={
        <>
          <Link href="/seller/dashboard" className="hv-btn-pill hv-btn-primary">
            Your listings
          </Link>
          <button type="button" className="hv-btn-pill hv-btn-secondary" onClick={() => void logoutRemote().then(() => router.push("/login"))}>
            Sign out
          </button>
        </>
      }
    />
  );
}
