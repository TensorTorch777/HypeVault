"use client";

import { AnimatePresence, motion, useReducedMotion } from "framer-motion";
import { useRouter } from "next/navigation";
import { useEffect, useState } from "react";
import { useQuery } from "@tanstack/react-query";

import { DeclaredBrandSelect } from "@/components/DeclaredBrandSelect";
import { ImageUploader } from "@/components/ImageUploader";
import { ScreeningResult, type ScreeningOutcome } from "@/components/ScreeningResult";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import axios from "axios";

import { PageHero } from "@/components/layout/PageHero";
import { ME_QUERY_KEY, api, fetchMe } from "@/lib/api";
import { canSell } from "@/lib/roles";
import { DECLARED_BRAND_NOTE, classifyHttpFailure, researchFailureMessage } from "@/lib/semanticCopy";

const steps = ["Images", "Details", "Screening", "Result"] as const;

export default function SellerUploadPage() {
  const router = useRouter();
  const meQuery = useQuery({
    queryKey: ME_QUERY_KEY,
    queryFn: fetchMe,
    retry: false,
  });
  const authed = Boolean(meQuery.data);
  const isSeller = canSell(meQuery.data?.role);

  useEffect(() => {
    if (meQuery.isError) router.replace("/login");
  }, [meQuery.isError, router]);
  const [step, setStep] = useState(0);
  const [files, setFiles] = useState<File[]>([]);
  const [productName, setProductName] = useState("");
  const category = "watch" as const;
  const [brand, setBrand] = useState("");
  const [condition, setCondition] = useState("");
  const [size, setSize] = useState("");
  const [listingId, setListingId] = useState<string | null>(null);
  const [outcome, setOutcome] = useState<ScreeningOutcome | null>(null);
  const [stage, setStage] = useState<"idle" | "creating" | "screening" | "screened" | "screening_failed" | "create_failed">("idle");
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
    const { data } = await api.post<
      ScreeningOutcome & { verdict: "AUTHENTIC" | "FAKE" | null; confidence: number | null; listing_id: string }
    >("/verify/authenticate", fd);
    setOutcome({
      listing_status: data.listing_status,
      screening_code: data.screening_code ?? null,
      screening_reason: data.screening_reason ?? null,
      seller_hint: data.seller_hint ?? null,
    });
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
            : `Listing ${id} was created, but screening did not complete. It is held and not visible to buyers. Retry screening without creating another listing.`,
        );
        return;
      }
      setStage("create_failed");
      setErr("The listing was not created. Nothing was screened.");
    }
  }

  const busy = stage === "creating" || stage === "screening";

  return (
    <div className="pb-24">
    <PageHero
      eyebrow="Seller desk · new listing"
      title="List a watch"
      subtitle="One clear photo of the watch, the house that made it, and a few seconds of screening. Not a watch, too small or AI-generated? You get told exactly why."
      scene="convertible"
      render="/renders/ap_royal_oak/hero.webp"
      compact
    />
    <div className="mx-auto w-full max-w-6xl px-4 md:px-6">
      {!authed || !isSeller ? (
        <Card className="mx-auto max-w-lg hover:translate-y-0">
          <CardHeader>
            <h2 className="font-display text-[1.8rem] font-extrabold uppercase leading-none">
              {authed ? "Not a seller account" : "Sign in to list"}
            </h2>
          </CardHeader>
          <CardContent className="space-y-3">
            <p className="text-sm text-primary/65">
              {authed
                ? `${meQuery.data?.email} cannot list watches. Sign in with your Somaiya Google account to sell.`
                : "Sign in with your Somaiya account to list a watch."}
            </p>
            <Button className="min-h-[2.75rem]" onClick={() => router.push(authed ? "/market" : "/login")}>
              {authed ? "Browse the market" : "Log in"}
            </Button>
          </CardContent>
        </Card>
      ) : (
        <div className="grid items-start gap-8 lg:grid-cols-[minmax(0,1fr)_17.5rem]">
          <section className="min-w-0 rounded-[1.375rem] border border-white/10 bg-card/90 p-5 shadow-[0_40px_80px_-40px_rgba(0,0,0,0.9)] backdrop-blur md:p-8">
            <ol className="grid grid-cols-2 gap-3 sm:grid-cols-4" aria-label="Listing steps">
              {steps.map((label, index) => {
                const complete = index < step || (index === 3 && stage === "screened");
                const current = index === step;
                const failed = current && stage === "screening_failed";
                const running = current && busy;
                return (
                  <li key={label} className="min-w-0">
                    <div className="flex items-center gap-2">
                      <span
                        className={`flex h-7 w-7 shrink-0 items-center justify-center rounded-full text-[0.6875rem] font-semibold ${
                          failed
                            ? "bg-[#3A1A2C] text-[#FF8A8A]"
                            : complete
                              ? "bg-[#FFB8D2] text-[#14112A]"
                              : current
                                ? "bg-white text-[#14112A]"
                                : "bg-white/10 text-white/55"
                        }`}
                      >
                        {complete ? "✓" : index + 1}
                      </span>
                      <span className={`truncate font-display text-[0.875rem] font-bold uppercase tracking-[0.04em] ${current ? "text-white" : "text-white/45"}`}>{label}</span>
                    </div>
                    <div className={`mt-2 h-0.5 rounded-full ${complete || current ? "bg-[#FFB8D2]" : "bg-white/10"}`} />
                    {running ? <p className="mt-1 text-[0.625rem] text-[#F08AA8]">In progress</p> : null}
                    {failed ? <p className="mt-1 text-[0.625rem] text-[#FF8A8A]">Needs retry</p> : null}
                  </li>
                );
              })}
            </ol>

            <AnimatePresence mode="wait">
              {step === 0 ? (
                <motion.div key="s0" initial={{ opacity: 0, y: reduceMotion ? 0 : 8 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: reduceMotion ? 0 : -8 }} transition={stepMotion} className="mt-8">
                  <ImageUploader minImages={1} onChange={setFiles} />
                  <div className="mt-6 flex justify-end">
                    <Button className="min-h-[2.75rem]" disabled={files.length < 1} onClick={() => setStep(1)}>
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
                    <Button variant="outline" className="min-h-[2.75rem]" onClick={() => setStep(0)}>
                      Back
                    </Button>
                    <Button className="min-h-[2.75rem]" disabled={!brand} onClick={() => setStep(2)}>
                      Continue
                    </Button>
                  </div>
                </motion.div>
              ) : null}

              {step === 2 ? (
                <motion.div key="s2" initial={{ opacity: 0, y: reduceMotion ? 0 : 8 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: reduceMotion ? 0 : -8 }} transition={stepMotion} className="mt-8 space-y-4">
                  <p className="text-sm leading-6 text-primary/70">
                    Three checks run on your photo: it must be at least 400 px on its shorter side, show a wristwatch, and look like an
                    original photograph rather than an AI-generated or heavily edited image (legacy DINOv2 model). Pass all three and the
                    watch is listed; otherwise you get the reason. Screening checks the photo, not the watch — it is not an authenticity
                    guarantee.
                  </p>
                  <div className="min-h-[4.5rem] rounded-xl border border-white/10 bg-white/[0.03] px-4 py-3" aria-live="polite">
                    <p className="text-sm text-primary/75" role="status">
                      {stage === "creating" ? "Creating listing." : null}
                      {stage === "screening" ? "Screening in progress. The listing is not visible to buyers yet." : null}
                      {stage === "screening_failed" && listingId ? `Listing ${listingId} exists. Screening did not complete.` : null}
                      {stage === "idle" || stage === "create_failed" ? "Listing creation and screening are separate steps." : null}
                    </p>
                    {busy ? <div className="mt-3 h-1 overflow-hidden rounded-full bg-white/10"><div className="h-full w-1/3 animate-pulse rounded-full bg-[#FFB8D2]" /></div> : null}
                  </div>
                  {err ? <p className="text-sm font-semibold text-danger" role="alert">{err}</p> : null}
                  <div className="flex justify-between gap-3">
                    <Button variant="outline" className="min-h-[2.75rem]" onClick={() => setStep(1)} disabled={busy}>
                      Back
                    </Button>
                    <Button className="min-h-[2.75rem]" disabled={busy} onClick={() => void runLegacyCheck()}>
                      {busy ? "Screening…" : listingId ? "Retry screening" : "Run screening"}
                    </Button>
                  </div>
                </motion.div>
              ) : null}

              {step === 3 ? (
                <motion.div key="s3" initial={{ opacity: 0, y: reduceMotion ? 0 : 8 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: reduceMotion ? 0 : -8 }} transition={stepMotion} className="mt-8 space-y-5">
                  {outcome ? <ScreeningResult outcome={outcome} /> : null}
                  <div className="flex flex-wrap gap-3">
                    {outcome?.listing_status === "live" && listingId ? (
                      <Button className="min-h-[2.75rem]" onClick={() => router.push(`/product/${listingId}`)}>
                        View listing
                      </Button>
                    ) : null}
                    {outcome && outcome.listing_status !== "live" ? (
                      <Button
                        className="min-h-[2.75rem]"
                        onClick={() => {
                          setFiles([]);
                          setOutcome(null);
                          setStage("idle");
                          setStep(0);
                        }}
                      >
                        Try another photo
                      </Button>
                    ) : null}
                    <Button variant="outline" className="min-h-[2.75rem]" onClick={() => router.push("/seller/dashboard")}>
                      Dashboard
                    </Button>
                  </div>
                </motion.div>
              ) : null}
            </AnimatePresence>
          </section>

          <aside className="rounded-[1.375rem] border border-white/10 bg-[#13152A] p-6 text-sm text-primary/70 lg:sticky lg:top-24">
            <p className="hv-eyebrow">This listing</p>
            <p className="mt-2 font-display text-[1.6rem] font-extrabold uppercase leading-none text-white">{productName.trim() || "Untitled listing"}</p>
            <p className="mt-1">{brand || "Declared brand not selected"}</p>
            <p className="mt-4 text-xs leading-5 text-white/50">
              {files.length} image{files.length === 1 ? "" : "s"} selected. Photos that are not a watch, too small or look generated are
              turned away with the reason. A listed watch is screened, not verified authentic.
            </p>
            {listingId ? <p className="mt-4 break-all text-xs text-white/60">Listing {listingId}</p> : null}
          </aside>
        </div>
      )}
    </div>
    </div>
  );
}
