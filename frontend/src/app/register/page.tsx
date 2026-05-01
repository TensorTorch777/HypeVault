"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { useCallback, useEffect, useState } from "react";

import { GoogleSignInButton } from "@/components/auth/GoogleSignInButton";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { api, fetchMe, getApiErrorMessage, googleLogin } from "@/lib/api";
import { setLastSignedInEmail } from "@/lib/auth";

export default function RegisterPage() {
  const router = useRouter();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [role, setRole] = useState<"buyer" | "seller">("buyer");
  const [err, setErr] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

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
          await googleLogin(idToken, "buyer");
          await routePostLogin();
        } catch (e) {
          setErr(getApiErrorMessage(e, "Google sign-up failed."));
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
      await api.post("/auth/register", { email, password, role });
      router.push("/login");
    } catch (e) {
      setErr(
        getApiErrorMessage(
          e,
          "Could not register. Use 8+ characters with letters and numbers."
        )
      );
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="mx-auto max-w-lg px-4 py-12">
      <Card>
        <CardHeader>
          <p className="text-xs font-semibold uppercase tracking-wider text-primary/45">Join HypeVault</p>
          <h1 className="mt-2 text-3xl font-semibold tracking-tight">Create account</h1>
        </CardHeader>
        <CardContent>
          <form onSubmit={(e) => void onSubmit(e)} className="space-y-4">
            <div>
              <label className="text-xs font-semibold text-primary/55" htmlFor="email">
                Email
              </label>
              <Input id="email" value={email} onChange={(e) => setEmail(e.target.value)} className="mt-2" />
            </div>
            <div>
              <label className="text-xs font-semibold text-primary/55" htmlFor="pw">
                Password
              </label>
              <Input id="pw" type="password" value={password} onChange={(e) => setPassword(e.target.value)} className="mt-2" />
              <p className="mt-2 text-xs text-primary/50">Use at least 8 characters with letters and numbers.</p>
            </div>
            <div>
              <p className="text-xs font-semibold text-primary/55">Role</p>
              <div className="mt-2 grid grid-cols-2 gap-2">
                <Button type="button" variant={role === "buyer" ? "primary" : "outline"} onClick={() => setRole("buyer")}>
                  Buyer
                </Button>
                <Button type="button" variant={role === "seller" ? "primary" : "outline"} onClick={() => setRole("seller")}>
                  Seller
                </Button>
              </div>
            </div>
            {err ? <p className="text-sm font-semibold text-danger">{err}</p> : null}
            <Button className="w-full min-h-[48px]" type="submit" disabled={loading}>
              {loading ? "Creating…" : "Create account"}
            </Button>
            <div className="flex items-center gap-2 py-1 text-xs text-primary/45">
              <span className="h-px flex-1 bg-white/[0.1]" />
              <span>or</span>
              <span className="h-px flex-1 bg-white/[0.1]" />
            </div>
            <div className="flex justify-center">
              <GoogleSignInButton
                role="buyer"
                disabled={loading}
                onCredential={(token) => onGoogleCredential(token)}
                onError={(message) => setErr(message)}
              />
            </div>
            <p className="text-center text-xs text-primary/50">
              Google sign-in creates buyer access only. Seller/employee accounts must use email and password.
            </p>
            <p className="text-center text-sm text-primary/60">
              Already have an account?{" "}
              <Link href="/login" className="font-semibold text-accent hover:underline">
                Log in
              </Link>
            </p>
          </form>
        </CardContent>
      </Card>
    </div>
  );
}
