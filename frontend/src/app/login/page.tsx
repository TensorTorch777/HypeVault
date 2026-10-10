"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { useCallback, useEffect, useState } from "react";

import { GoogleSignInButton } from "@/components/auth/GoogleSignInButton";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { api, fetchMe, getApiErrorMessage, googleLogin } from "@/lib/api";
import { getLastSignedInEmail, setLastSignedInEmail } from "@/lib/auth";

export default function LoginPage() {
  const router = useRouter();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [err, setErr] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [lastEmail, setLastEmail] = useState<string | null>(null);

  useEffect(() => {
    setLastEmail(getLastSignedInEmail());
  }, []);

  useEffect(() => {
    let cancelled = false;
    void (async () => {
      try {
        const me = await fetchMe();
        if (cancelled) return;
        if (me.role === "seller") router.replace("/seller/dashboard");
        else router.replace("/");
      } catch {
        /* stale token is cleared by interceptor */
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [router]);

  const routePostLogin = useCallback(async () => {
    const me = await fetchMe();
    setLastSignedInEmail(me.email);
    if (me.role === "seller") router.push("/seller/dashboard");
    else router.push("/");
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
          setErr(getApiErrorMessage(e, "Google sign-in failed."));
        } finally {
          setLoading(false);
        }
      })();
    },
    [routePostLogin]
  );

  async function onSubmit(e: React.FormEvent) {
    e.preventDefault();
    setErr(null);
    setLoading(true);
    try {
      await api.post<{ access_token: string; refresh_token: string }>("/auth/login", {
        email,
        password,
      });
      await routePostLogin();
    } catch (e) {
      setErr(getApiErrorMessage(e, "Invalid credentials."));
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="mx-auto grid min-h-[calc(100dvh-12rem)] w-full max-w-6xl items-center gap-10 px-4 py-10 md:px-6 lg:grid-cols-2 lg:gap-16">
      <section className="hidden lg:block">
        <p className="text-[11px] font-semibold uppercase tracking-[0.22em] text-[#FFB089]">HypeVault</p>
        <h1 className="mt-4 max-w-md font-[family-name:var(--font-display)] text-5xl font-semibold not-italic leading-[1.05] tracking-tight text-white">
          A research prototype for in-scope watch screening.
        </h1>
        <p className="mt-6 max-w-md text-base leading-7 text-white/65">
          Declared brands are chosen by you. A model classification is not a certificate, and a visible listing is not verified authentic.
        </p>
      </section>

      <section className="mx-auto w-full max-w-md rounded-2xl border border-white/10 bg-[#140a22]/95 p-6 shadow-[0_24px_70px_rgba(0,0,0,0.4)] sm:p-8">
        <p className="text-[11px] font-semibold uppercase tracking-[0.22em] text-[#FFB089] lg:hidden">HypeVault</p>
        <h1 className="mt-2 text-3xl font-semibold not-italic tracking-tight text-white">Log in</h1>
        <p className="mt-2 min-h-[1.25rem] text-xs text-white/50">{lastEmail ? `Last signed in: ${lastEmail}` : "Welcome back"}</p>
        <form onSubmit={(e) => void onSubmit(e)} className="mt-6 space-y-5">
          <div>
            <label className="text-xs font-semibold text-white/70" htmlFor="email">
              Email
            </label>
            <Input
              id="email"
              name="email"
              type="email"
              inputMode="email"
              autoComplete="email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              className="mt-2"
              required
            />
          </div>
          <div>
            <label className="text-xs font-semibold text-white/70" htmlFor="pw">
              Password
            </label>
            <Input
              id="pw"
              name="password"
              type="password"
              autoComplete="current-password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              className="mt-2"
              required
            />
          </div>
          <div className="min-h-[1.25rem]" aria-live="polite">
            {err ? <p className="text-sm font-semibold text-danger" role="alert">{err}</p> : null}
          </div>
          <Button className="w-full min-h-[48px]" type="submit" disabled={loading}>
            {loading ? "Signing in…" : "Continue"}
          </Button>
          <div className="flex items-center gap-3 text-xs text-white/40">
            <span className="h-px flex-1 bg-white/10" />
            <span>or</span>
            <span className="h-px flex-1 bg-white/10" />
          </div>
          <div className="flex justify-center">
            <GoogleSignInButton
              role="buyer"
              disabled={loading}
              onCredential={(token) => onGoogleCredential(token)}
              onError={(message) => setErr(message)}
            />
          </div>
          <p className="text-center text-sm text-white/60">
            New here?{" "}
            <Link href="/register" className="font-semibold text-[#FFB089] hover:underline">
              Create an account
            </Link>
          </p>
        </form>
      </section>
    </div>
  );
}
