"use client";

import { AnimatePresence, motion } from "framer-motion";
import { useRouter } from "next/navigation";
import { useEffect, useMemo, useState } from "react";
import { useQuery } from "@tanstack/react-query";

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

  const progress = useMemo(() => ((step + 1) / steps.length) * 100, [step]);

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

  return (
    <div className="mx-auto max-w-3xl px-4 py-12">
      {!authed || !isSeller ? (
        <Card>
          <CardHeader>
            <h1 className="text-2xl font-semibold">Upload</h1>
          </CardHeader>
          <CardContent className="space-y-3">
            <p className="text-sm text-primary/65">Log in to upload a listing. The listing check uses legacy DINOv2, not the DINOv3 research demo.</p>
            <Button className="min-h-[44px]" onClick={() => router.push("/login")}>
              Log in
            </Button>
          </CardContent>
        </Card>
      ) : (
        <Card>
          <CardHeader>
            <p className="text-xs font-semibold uppercase tracking-wider text-primary/45">Seller flow</p>
            <h1 className="mt-2 text-3xl font-semibold tracking-tight">List an item</h1>
            <div className="mt-4 h-2 w-full overflow-hidden rounded-full bg-primary/10">
              <motion.div className="h-full bg-accent" animate={{ width: `${progress}%` }} transition={{ duration: 0.35 }} />
            </div>
            <p className="mt-2 text-xs font-semibold text-primary/50">
              Step {step + 1}/{steps.length}: {steps[step]}
            </p>
          </CardHeader>
          <CardContent className="space-y-6">
            <AnimatePresence mode="wait">
              {step === 0 ? (
                <motion.div key="s0" initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -8 }}>
                  <ImageUploader minImages={1} onChange={setFiles} />
                  <div className="mt-4 flex justify-end">
                    <Button
                      className="min-h-[44px]"
                      disabled={files.length < 1}
                      onClick={() => setStep(1)}
                    >
                      Continue
                    </Button>
                  </div>
                </motion.div>
              ) : null}

              {step === 1 ? (
                <motion.div key="s1" initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -8 }} className="space-y-3">
                  <div>
                    <label className="text-xs font-semibold text-primary/55">Product name</label>
                    <Input className="mt-2" value={productName} onChange={(e) => setProductName(e.target.value)} />
                  </div>
                  <div>
                    <label className="text-xs font-semibold text-primary/55">Category</label>
                    <p className="mt-2 text-sm font-medium text-primary/80">Luxury watch</p>
                  </div>
                  <div>
                    <label className="text-xs font-semibold text-primary/55">Declared brand</label>
                    <Input className="mt-2" value={brand} onChange={(e) => setBrand(e.target.value)} />
                    <p className="mt-2 text-xs text-primary/55">
                      {DECLARED_BRAND_NOTE} Legacy DINOv2 listing check. This is not the DINOv3 research prototype. An out-of-scope brand produces no authenticity result.
                    </p>
                  </div>
                  <div>
                    <label className="text-xs font-semibold text-primary/55">Condition</label>
                    <Input className="mt-2" value={condition} onChange={(e) => setCondition(e.target.value)} />
                  </div>
                  <div>
                    <label className="text-xs font-semibold text-primary/55">Size</label>
                    <Input className="mt-2" value={size} onChange={(e) => setSize(e.target.value)} />
                  </div>
                  <div className="flex justify-between gap-3 pt-2">
                    <Button variant="outline" className="min-h-[44px]" onClick={() => setStep(0)}>
                      Back
                    </Button>
                    <Button className="min-h-[44px]" onClick={() => setStep(2)}>
                      Continue
                    </Button>
                  </div>
                </motion.div>
              ) : null}

              {step === 2 ? (
                <motion.div key="s2" initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -8 }} className="space-y-4">
                  <p className="text-sm text-primary/65">
                    This runs the legacy DINOv2 classifier. It is not the frozen DINOv3 research prototype, and it is not a production authenticity guarantee. An unsupported brand returns no authentic or fake verdict.
                  </p>
                  <p className="text-sm text-primary/70" role="status">
                    {stage === "creating" ? "Creating listing." : null}
                    {stage === "screening" ? "Legacy screening in progress. The listing is not verified yet." : null}
                    {stage === "screening_failed" && listingId ? `Listing ${listingId} exists. Screening did not complete.` : null}
                    {stage === "idle" || stage === "create_failed" ? "Listing creation and screening are separate steps." : null}
                  </p>
                  {err ? <p className="text-sm font-semibold text-danger" role="alert">{err}</p> : null}
                  <div className="flex justify-between gap-3">
                    <Button variant="outline" className="min-h-[44px]" onClick={() => setStep(1)} disabled={stage === "creating" || stage === "screening"}>
                      Back
                    </Button>
                    <Button
                      className="min-h-[44px]"
                      disabled={stage === "creating" || stage === "screening"}
                      onClick={() => void runLegacyCheck()}
                    >
                      {listingId ? "Retry screening" : "Run legacy check"}
                    </Button>
                  </div>
                </motion.div>
              ) : null}

              {step === 3 ? (
                <motion.div key="s3" initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -8 }} className="space-y-5">
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
          </CardContent>
        </Card>
      )}
    </div>
  );
}
