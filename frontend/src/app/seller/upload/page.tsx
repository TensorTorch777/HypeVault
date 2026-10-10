"use client";

import { AnimatePresence, motion, useReducedMotion } from "framer-motion";
import { useRouter } from "next/navigation";
import { useEffect, useState } from "react";
import { useQuery } from "@tanstack/react-query";

import { DeclaredBrandSelect } from "@/components/DeclaredBrandSelect";
import { ImageUploader } from "@/components/ImageUploader";
import { AuthBadge } from "@/components/AuthBadge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import axios from "axios";

import { api, fetchMe } from "@/lib/api";
import { DECLARED_BRAND_NOTE, classifyHttpFailure, researchFailureMessage } from "@/lib/semanticCopy";

const steps = ["Images", "Details", "Legacy check", "Result"] as const;

export default function SellerUploadPage() {
  const router = useRouter();
  const meQuery = useQuery({
    queryKey: ["auth-me-seller-upload"],
    queryFn: fetchMe,
    retry: false,
  });
  const authed = Boolean(meQuery.data);
  const isSeller = meQuery.data?.role === "seller";

  useEffect(() => {
    if (meQuery.isError) router.replace("/login");
    else if (authed && !isSeller) router.replace("/");
  }, [authed, isSeller, meQuery.isError, router]);
  const [step, setStep] = useState(0);
  const [files, setFiles] = useState<File[]>([]);
  const [productName, setProductName] = useState("");
  const category = "watch" as const;
  const [brand, setBrand] = useState("");
  const [condition, setCondition] = useState("");
  const [size, setSize] = useState("");
  const [listingId, setListingId] = useState<string | null>(null);
  const [listingStatus, setListingStatus] = useState<string | null>(null);
  const [stage, setStage] = useState<"idle" | "creating" | "screening" | "screened" | "screening_failed" | "create_failed">("idle");
  const [verdict, setVerdict] = useState<"AUTHENTIC" | "FAKE" | null>(null);
  const [confidence, setConfidence] = useState<number | null>(null);
  const [err, setErr] = useState<string | null>(null);

  const reduceMotion = useReducedMotion();
  const stepMotion = reduceMotion ? { duration: 0 } : { duration: 0.28, ease: [0.22, 1, 0.36, 1] as const };

  async function createListing() {
    setErr(null);
    const { data } = await api.post("/listings/", {
      product_name: productName || "Untitled listing",
      category,
      brand: brand || null,
      condition: condition || null,
      size: size || null,
    });
    setListingId(data.id as string);
    return data.id as string;
  }

  async function runVerify(id: string) {
    const first = files[0];
    if (!first) throw new Error("No image");
    const fd = new FormData();
    fd.append("image", first);
    fd.append("product_name", productName || "Untitled listing");
    fd.append("category", category);
    fd.append("listing_id", id);
    if (brand.trim()) fd.append("brand", brand.trim());
    const { data } = await api.post<{
      verdict: "AUTHENTIC" | "FAKE";
      confidence: number;
      s3_url: string;
      listing_id: string;
      listing_status: "live" | "rejected" | "pending";
    }>("/verify/authenticate", fd);
    setVerdict(data.verdict);
    setConfidence(data.confidence);
    setListingStatus(data.listing_status);
    return data;
  }

  async function runLegacyCheck() {
    setErr(null);
    let id = listingId;
    try {
      if (!id) {
        setStage("creating");
        id = await createListing();
        setListingId(id);
      }
      setStage("screening");
      await runVerify(id);
      setStage("screened");
      setStep(3);
    } catch (e) {
      if (id) {
        setListingId(id);
        setStage("screening_failed");
        const status = axios.isAxiosError(e) ? e.response?.status : undefined;
        const bodyStatus = axios.isAxiosError(e) ? (e.response?.data as { status?: string } | undefined)?.status : undefined;
        const kind = classifyHttpFailure(status, bodyStatus);
        const specific = kind === "scope" || kind === "invalid" || kind === "access" || kind === "unavailable";
        setErr(
          specific
            ? `${researchFailureMessage(kind)} Listing ${id} already exists and was not created again.`
            : `Listing ${id} was created, but screening did not complete. It may still be pending and is not verified. Retry screening without creating another listing.`,
        );
        return;
      }
      setStage("create_failed");
      setErr("The listing was not created. Nothing was screened.");
    }
  }

  const busy = stage === "creating" || stage === "screening";

  return (
    <div className="mx-auto w-full max-w-6xl px-4 py-10 md:px-6 md:py-14">
      {!authed || !isSeller ? (
        <Card className="mx-auto max-w-lg hover:translate-y-0">
          <CardHeader>
            <h1 className="text-2xl font-semibold not-italic">Upload</h1>
          </CardHeader>
          <CardContent className="space-y-3">
            <p className="text-sm text-primary/65">Log in to upload a listing. The listing check uses legacy DINOv2, not the DINOv3 research demo.</p>
            <Button className="min-h-[44px]" onClick={() => router.push("/login")}>
              Log in
            </Button>
          </CardContent>
        </Card>
      ) : (
        <div className="grid items-start gap-8 lg:grid-cols-[minmax(0,1fr)_280px]">
          <section className="min-w-0 rounded-2xl border border-white/10 bg-[#140a22]/90 p-5 shadow-[0_24px_60px_rgba(0,0,0,0.35)] md:p-8">
            <p className="text-[11px] font-semibold uppercase tracking-[0.22em] text-[#FFB089]">Seller flow</p>
            <h1 className="mt-2 font-[family-name:var(--font-display)] text-3xl font-semibold not-italic tracking-tight text-[#FFEDF6] md:text-4xl">
              List an item
            </h1>
            <ol className="mt-8 grid grid-cols-2 gap-3 sm:grid-cols-4" aria-label="Listing steps">
              {steps.map((label, index) => {
                const complete = index < step || (index === 3 && stage === "screened");
                const current = index === step;
                const failed = current && stage === "screening_failed";
                const running = current && busy;
                return (
                  <li key={label} className="min-w-0">
                    <div className="flex items-center gap-2">
                      <span
                        className={`flex h-7 w-7 shrink-0 items-center justify-center rounded-full text-[11px] font-semibold ${
                          failed
                            ? "bg-[#3a1218] text-[#ffb4ab]"
                            : complete
                              ? "bg-[#FF7A1A] text-[#1a0b12]"
                              : current
                                ? "bg-white text-[#1a0b12]"
                                : "bg-white/10 text-white/55"
                        }`}
                      >
                        {complete ? "✓" : index + 1}
                      </span>
                      <span className={`truncate text-xs font-medium ${current ? "text-white" : "text-white/45"}`}>{label}</span>
                    </div>
                    <div className={`mt-2 h-0.5 rounded-full ${complete || current ? "bg-[#FF7A1A]" : "bg-white/10"}`} />
                    {running ? <p className="mt-1 text-[10px] text-[#FFB089]">In progress</p> : null}
                    {failed ? <p className="mt-1 text-[10px] text-[#ffb4ab]">Needs retry</p> : null}
                  </li>
                );
              })}
            </ol>

            <AnimatePresence mode="wait">
              {step === 0 ? (
                <motion.div key="s0" initial={{ opacity: 0, y: reduceMotion ? 0 : 8 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: reduceMotion ? 0 : -8 }} transition={stepMotion} className="mt-8">
                  <ImageUploader minImages={1} onChange={setFiles} />
                  <div className="mt-6 flex justify-end">
                    <Button className="min-h-[44px]" disabled={files.length < 1} onClick={() => setStep(1)}>
                      Continue
                    </Button>
                  </div>
                </motion.div>
              ) : null}

              {step === 1 ? (
                <motion.div key="s1" initial={{ opacity: 0, y: reduceMotion ? 0 : 8 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: reduceMotion ? 0 : -8 }} transition={stepMotion} className="mt-8 space-y-5">
                  <div>
                    <label className="text-xs font-semibold text-primary/70" htmlFor="product-name">Product name</label>
                    <Input id="product-name" className="mt-2" value={productName} onChange={(e) => setProductName(e.target.value)} placeholder="Reference name for this listing" autoComplete="off" />
                  </div>
                  <div>
                    <p className="text-xs font-semibold text-primary/70">Category</p>
                    <p className="mt-2 rounded-xl border border-white/10 bg-white/[0.03] px-4 py-3 text-sm text-primary/80">Luxury watch</p>
                  </div>
                  <div>
                    <label className="text-xs font-semibold text-primary/70" htmlFor="seller-brand">Declared brand</label>
                    <DeclaredBrandSelect id="seller-brand" value={brand} onChange={setBrand} />
                    <p className="mt-2 text-xs leading-5 text-primary/55">
                      {DECLARED_BRAND_NOTE} Legacy DINOv2 listing check. This is not the DINOv3 research prototype. An out-of-scope brand produces no authenticity result.
                    </p>
                  </div>
                  <div className="grid gap-5 sm:grid-cols-2">
                    <div>
                      <label className="text-xs font-semibold text-primary/70" htmlFor="condition">Condition</label>
                      <Input id="condition" className="mt-2" value={condition} onChange={(e) => setCondition(e.target.value)} placeholder="Optional" autoComplete="off" />
                    </div>
                    <div>
                      <label className="text-xs font-semibold text-primary/70" htmlFor="size">Size</label>
                      <Input id="size" className="mt-2" value={size} onChange={(e) => setSize(e.target.value)} placeholder="Optional" autoComplete="off" />
                    </div>
                  </div>
                  <div className="flex justify-between gap-3 pt-2">
                    <Button variant="outline" className="min-h-[44px]" onClick={() => setStep(0)}>
                      Back
                    </Button>
                    <Button className="min-h-[44px]" disabled={!brand} onClick={() => setStep(2)}>
                      Continue
                    </Button>
                  </div>
                </motion.div>
              ) : null}

              {step === 2 ? (
                <motion.div key="s2" initial={{ opacity: 0, y: reduceMotion ? 0 : 8 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: reduceMotion ? 0 : -8 }} transition={stepMotion} className="mt-8 space-y-4">
                  <p className="text-sm leading-6 text-primary/70">
                    This runs the legacy DINOv2 classifier. It is not the frozen DINOv3 research prototype, and it is not a production authenticity guarantee. An unsupported brand returns no authentic or fake verdict.
                  </p>
                  <div className="min-h-[4.5rem] rounded-xl border border-white/10 bg-white/[0.03] px-4 py-3" aria-live="polite">
                    <p className="text-sm text-primary/75" role="status">
                      {stage === "creating" ? "Creating listing." : null}
                      {stage === "screening" ? "Legacy screening in progress. The listing is not verified yet." : null}
                      {stage === "screening_failed" && listingId ? `Listing ${listingId} exists. Screening did not complete.` : null}
                      {stage === "idle" || stage === "create_failed" ? "Listing creation and screening are separate steps." : null}
                    </p>
                    {busy ? <div className="mt-3 h-1 overflow-hidden rounded-full bg-white/10"><div className="h-full w-1/3 animate-pulse rounded-full bg-[#FF7A1A]" /></div> : null}
                  </div>
                  {err ? <p className="text-sm font-semibold text-danger" role="alert">{err}</p> : null}
                  <div className="flex justify-between gap-3">
                    <Button variant="outline" className="min-h-[44px]" onClick={() => setStep(1)} disabled={busy}>
                      Back
                    </Button>
                    <Button className="min-h-[44px]" disabled={busy} onClick={() => void runLegacyCheck()}>
                      {listingId ? "Retry screening" : "Run legacy check"}
                    </Button>
                  </div>
                </motion.div>
              ) : null}

              {step === 3 ? (
                <motion.div key="s3" initial={{ opacity: 0, y: reduceMotion ? 0 : 8 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: reduceMotion ? 0 : -8 }} transition={stepMotion} className="mt-8 space-y-5">
                  <AuthBadge verdict={verdict} confidence={confidence} declaredBrand={brand} lane="legacy" />
                  {listingId ? <p className="text-sm text-primary/70">Listing {listingId}. Workflow status: {listingStatus ?? "pending"} — not a verified authentic result.</p> : null}
                  <div className="flex flex-wrap gap-3">
                    {listingId ? (
                      <Button className="min-h-[44px]" onClick={() => router.push(`/product/${listingId}`)}>
                        View listing
                      </Button>
                    ) : null}
                    <Button variant="outline" className="min-h-[44px]" onClick={() => router.push("/seller/dashboard")}>
                      Dashboard
                    </Button>
                  </div>
                </motion.div>
              ) : null}
            </AnimatePresence>
          </section>

          <aside className="rounded-2xl border border-white/10 bg-[#100818] p-5 text-sm text-primary/70 lg:sticky lg:top-6">
            <p className="text-[11px] font-semibold uppercase tracking-[0.18em] text-white/45">This listing</p>
            <p className="mt-3 text-base font-semibold text-white">{productName.trim() || "Untitled listing"}</p>
            <p className="mt-1">{brand || "Declared brand not selected"}</p>
            <p className="mt-4 text-xs leading-5 text-white/50">
              {files.length} image{files.length === 1 ? "" : "s"} selected. A model classification is not a certificate, and a visible listing is not verified authentic.
            </p>
            {listingId ? <p className="mt-4 break-all text-xs text-white/60">Listing {listingId}</p> : null}
          </aside>
        </div>
      )}
    </div>
  );
}
