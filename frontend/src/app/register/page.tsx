import Link from "next/link";

import { PageHero } from "@/components/layout/PageHero";

/** Accounts come from single sign-on; there is nothing to fill in. */
export default function RegisterPage() {
  return (
    <PageHero
      eyebrow="Accounts"
      title={
        <>
          No sign-up form.
          <br />
          Just your Somaiya account.
        </>
      }
      subtitle="HypeVault creates your account the first time you sign in with your @somaiya.edu Google account. Every Somaiya account can list and browse watches."
      scene="ocean_drive"
      render="/renders/vc_overseas/hero.webp"
      actions={
        <>
          <Link href="/login" className="hv-btn-pill hv-btn-primary">
            Sign in with Somaiya <span aria-hidden="true">→</span>
          </Link>
          <Link href="/market" className="hv-btn-pill hv-btn-secondary">
            Browse the market
          </Link>
        </>
      }
    />
  );
}
