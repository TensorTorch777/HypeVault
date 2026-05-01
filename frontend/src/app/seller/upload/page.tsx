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
import { api, fetchMe } from "@/lib/api";

const steps = ["Images", "Details", "Verify", "Result"] as const;

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
  const [category, setCategory] = useState<"sneaker" | "watch">("sneaker");
  const [brand, setBrand] = useState("");
  const [condition, setCondition] = useState("");
  const [size, setSize] = useState("");
  const [listingId, setListingId] = useState<string | null>(null);
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
    const { data } = await api.post("/verify/authenticate", fd);
    setVerdict(data.verdict);
    setConfidence(data.confidence);
    return data.verdict as "AUTHENTIC" | "FAKE";
  }

  return (
    <div className="mx-auto max-w-3xl px-4 py-12">
      {!authed || !isSeller ? (
        <Card>
          <CardHeader>
            <h1 className="text-2xl font-semibold">Upload</h1>
          </CardHeader>
          <CardContent className="space-y-3">
            <p className="text-sm text-primary/65">Log in to upload and verify listings.</p>
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
                    <div className="mt-2 grid grid-cols-2 gap-2">
                      <Button type="button" variant={category === "sneaker" ? "primary" : "outline"} onClick={() => setCategory("sneaker")}>
                        Sneaker
                      </Button>
                      <Button type="button" variant={category === "watch" ? "primary" : "outline"} onClick={() => setCategory("watch")}>
                        Watch
                      </Button>
                    </div>
                  </div>
                  <div>
                    <label className="text-xs font-semibold text-primary/55">Brand</label>
                    <Input className="mt-2" value={brand} onChange={(e) => setBrand(e.target.value)} />
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
                  <p className="text-sm text-primary/65">We’ll authenticate your lead image and update your listing.</p>
                  {err ? <p className="text-sm font-semibold text-danger">{err}</p> : null}
                  <div className="flex justify-between gap-3">
                    <Button variant="outline" className="min-h-[44px]" onClick={() => setStep(1)}>
                      Back
                    </Button>
                    <Button
                      className="min-h-[44px]"
                      onClick={() => {
                        void (async () => {
                          try {
                            setErr(null);
                            const id = listingId ?? (await createListing());
                            setListingId(id);
                            await runVerify(id);
                            setStep(3);
                          } catch {
                            setErr("Verification failed. Ensure Triton is running and your image meets requirements.");
                          }
                        })();
                      }}
                    >
                      Run verification
                    </Button>
                  </div>
                </motion.div>
              ) : null}

              {step === 3 ? (
                <motion.div key="s3" initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -8 }} className="space-y-5">
                  <AuthBadge verdict={verdict} confidence={confidence} />
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
