"use client";

import Image from "next/image";
import { useRouter } from "next/navigation";
import { useCallback, useEffect, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { ShieldCheck } from "lucide-react";

import { Lockup } from "@/components/brand/Lockup";
import { GoogleSignInButton } from "@/components/auth/GoogleSignInButton";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { api, fetchMe, getApiErrorMessage, googleLogin } from "@/lib/api";
import { getLastSignedInEmail, setLastSignedInEmail } from "@/lib/auth";
import { canSell } from "@/lib/roles";

type AuthConfig = {
  sso: { provider: "google"; configured: boolean; domains: string[] };
  password_login: boolean;
};

export default function LoginPage() {
  const router = useRouter();
  const [err, setErr] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [lastEmail, setLastEmail] = useState<string | null>(null);
  const [showDev, setShowDev] = useState(false);
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");

  const config = useQuery({
    queryKey: ["auth-config"],
    queryFn: async () => (await api.get<AuthConfig>("/auth/config")).data,
    staleTime: Infinity,
  });
  const domain = config.data?.sso.domains[0] ?? "somaiya.edu";
  // "*" on the server means any verified Google account may sign in.
  const anyAccount = Boolean(config.data?.sso.domains.includes("*"));
  const ssoReady = Boolean(config.data?.sso.configured && process.env.NEXT_PUBLIC_GOOGLE_CLIENT_ID?.trim());
  // The password form exists only for local development, and only until SSO is connected.
  const devLogin = Boolean(config.data?.password_login) && !ssoReady;

  useEffect(() => {
    setLastEmail(getLastSignedInEmail());
  }, []);

  useEffect(() => {
    let cancelled = false;
    void (async () => {
      try {
        const me = await fetchMe();
        if (!cancelled) router.replace(canSell(me.role) ? "/seller/dashboard" : "/");
      } catch {
        /* not signed in */
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [router]);

  const routePostLogin = useCallback(async () => {
    const me = await fetchMe();
    setLastSignedInEmail(me.email);
    router.push(canSell(me.role) ? "/seller/dashboard" : "/");
  }, [router]);

  const onGoogleCredential = useCallback(
    (idToken: string) => {
      setErr(null);
      setLoading(true);
      void (async () => {
        try {
          await googleLogin(idToken);
          await routePostLogin();
        } catch (e) {
          setErr(getApiErrorMessage(e, "Sign-in failed."));
        } finally {
          setLoading(false);
        }
      })();
    },
    [routePostLogin]
  );

  async function onDevSubmit(e: React.FormEvent) {
    e.preventDefault();
    setErr(null);
    setLoading(true);
    try {
      await api.post("/auth/login", { email, password });
      await routePostLogin();
    } catch (e) {
      setErr(getApiErrorMessage(e, "Invalid credentials."));
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="relative isolate -mt-16 grid min-h-[100svh] lg:grid-cols-[1.15fr_1fr]">
      {/* Cinematic side */}
      <section className="relative hidden overflow-hidden lg:block" aria-hidden="true">
        <div className="hv-sky absolute inset-0" />
        <Image src="/renders/backdrops/rooftop_pool_0.webp" alt="" fill priority sizes="58vw" className="hv-kenburns object-cover" />
        <div className="absolute inset-0 bg-gradient-to-r from-transparent via-background/10 to-background" />
        <div className="absolute inset-0 bg-gradient-to-t from-background/80 via-transparent to-background/40" />
        <div className="absolute bottom-[-6%] left-[8%] aspect-square w-[min(40vw,35rem)]">
          <Image src="/renders/patek_nautilus/hero.webp" alt="" fill sizes="40vw" className="hv-fade-y object-contain drop-shadow-[0_40px_60px_rgba(10,11,23,0.7)]" />
        </div>
        <div className="absolute left-10 top-[22%]">
          <Lockup compact />
        </div>
      </section>

      {/* Sign-in side */}
      <section className="relative flex items-center justify-center px-4 pb-16 pt-28 sm:px-8">
        <div className="hv-sky absolute inset-0 -z-10 opacity-25 lg:hidden" aria-hidden="true" />
        <div className="w-full min-w-0 max-w-[26.25rem]">
          <p className="hv-eyebrow">{anyAccount ? "Google sign-in" : "Somaiya single sign-on"}</p>
          <h1 className="mt-1 font-display text-[clamp(3rem,6vw,4.4rem)] font-extrabold uppercase leading-[0.88] text-white">Sign in</h1>
          <p className="mt-4 text-[0.9375rem] leading-7 text-primary/75">
            {anyAccount ? (
              "Use your Google account."
            ) : (
              <>
                Use your <span className="font-bold text-primary">@{domain}</span> Google account.
              </>
            )}{" "}
            Your HypeVault account is created the first time you sign in. No separate password.
          </p>

          <div className="mt-8 rounded-[1.25rem] border border-white/10 bg-card/80 p-6 backdrop-blur">
            {config.isLoading ? (
              <div className="h-[2.75rem] animate-pulse rounded-full bg-white/10" />
            ) : ssoReady ? (
              <div className="flex w-full justify-center">
                <GoogleSignInButton
                  role="seller"
                  hostedDomain={anyAccount ? undefined : domain}
                  disabled={loading}
                  onCredential={(token) => onGoogleCredential(token)}
                  onError={(message) => setErr(message)}
                />
              </div>
            ) : (
              <div className="rounded-2xl bg-white/[0.04] p-4">
                <p className="font-display text-[1.2rem] font-bold uppercase text-neon-yellow">SSO not connected yet</p>
                <p className="mt-1 text-[0.8125rem] leading-6 text-primary/70">
                  An administrator needs to add the Google OAuth client ID for @{domain} (GOOGLE_CLIENT_ID and NEXT_PUBLIC_GOOGLE_CLIENT_ID).
                </p>
              </div>
            )}
            <div className="mt-4 flex items-start gap-2 text-[0.75rem] leading-5 text-primary/55">
              <ShieldCheck className="mt-0.5 h-4 w-4 shrink-0 text-success" aria-hidden="true" />
              {anyAccount
                ? "Admin access is limited to the project owner; the server checks this on every request."
                : `Only accounts managed by the ${domain} Workspace are accepted; the server checks this on every sign-in.`}
            </div>
            {lastEmail ? <p className="mt-3 text-[0.75rem] text-primary/45">Last signed in as {lastEmail}</p> : null}
            <div className="mt-3 min-h-[1.25rem]" aria-live="polite">
              {err ? (
                <p className="text-sm font-semibold text-danger" role="alert">
                  {err}
                </p>
              ) : null}
            </div>
          </div>

          {devLogin ? (
            <div className="mt-6">
              <button type="button" className="text-[0.8125rem] font-semibold text-primary/50 underline-offset-4 hover:text-primary hover:underline" onClick={() => setShowDev((v) => !v)}>
                {showDev ? "Hide" : "Use a"} development account
              </button>
              {showDev ? (
                <form onSubmit={(e) => void onDevSubmit(e)} className="mt-4 space-y-3 rounded-[1.25rem] border border-dashed border-white/15 p-5">
                  <p className="text-[0.75rem] text-primary/55">Local development only. Disappears once SSO is connected.</p>
                  <Input aria-label="Email" type="email" autoComplete="email" placeholder="Email" value={email} onChange={(e) => setEmail(e.target.value)} required />
                  <Input aria-label="Password" type="password" autoComplete="current-password" placeholder="Password" value={password} onChange={(e) => setPassword(e.target.value)} required />
                  <Button type="submit" variant="outline" className="w-full" disabled={loading}>
                    {loading ? "Signing in…" : "Sign in"}
                  </Button>
                </form>
              ) : null}
            </div>
          ) : null}
        </div>
      </section>
    </div>
  );
}
