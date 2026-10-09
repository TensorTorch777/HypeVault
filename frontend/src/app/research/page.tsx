"use client";

import { useState } from "react";
import axios from "axios";

import { AuthBadge } from "@/components/AuthBadge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { api, getApiErrorMessage } from "@/lib/api";

const DINOV2_LEGACY = "dinov2_legacy";
const DINOV3_EXPERIMENTAL = "dinov3_experimental";

const MODEL_LABELS: Record<string, string> = {
  [DINOV2_LEGACY]: "DINOv2 — Legacy",
  [DINOV3_EXPERIMENTAL]: "DINOv3 — Experimental",
};

type ResearchResult = {
  status: "AUTHENTIC" | "FAKE" | "REVIEW";
  decision: "AUTHENTIC" | "FAKE" | "REVIEW";
  declared_brand: string;
  brand_verification: "NOT_PERFORMED";
  model_scope: string;
  research_only: true;
  production_ready: false;
  checkpoint_sha: string;
  model: string;
  model_version?: string;
};

export default function ResearchDemoPage() {
  const [brand, setBrand] = useState("");
  const [file, setFile] = useState<File | null>(null);
  const [modelId, setModelId] = useState(DINOV2_LEGACY);
  const [result, setResult] = useState<ResearchResult | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [pending, setPending] = useState(false);
  const experimental = modelId === DINOV3_EXPERIMENTAL;

  async function submit() {
    if (!file) {
      setErr("Choose an image before running the research demo.");
      return;
    }
    setErr(null);
    setResult(null);
    setPending(true);
    try {
      const body = new FormData();
      body.append("image", file);
      body.append("brand", brand);
      body.append("logical_model", modelId);
      const { data } = await api.post<ResearchResult>("/research/verify", body);
      setResult(data);
    } catch (error) {
      const fallback = "The research demo did not return a classification.";
      setErr(axios.isAxiosError(error) ? getApiErrorMessage(error, fallback) : fallback);
    } finally {
      setPending(false);
    }
  }

  return (
    <div className="mx-auto max-w-3xl px-4 py-12">
      <Card>
        <CardHeader>
          <p className="text-xs font-semibold uppercase tracking-wider text-primary/45">Research demo</p>
          <h1 className="mt-2 text-3xl font-semibold tracking-tight">Five-brand prototype</h1>
          <p className="mt-3 text-sm text-primary/70">
            Authorized research selector. The selected brand is user-declared and is not independently verified from
            the image. Neither model publishes a listing.
          </p>
        </CardHeader>
        <CardContent className="space-y-4">
          <div>
            <label className="text-xs font-semibold text-primary/55" htmlFor="research-model">
              Model
            </label>
            <select
              id="research-model"
              className="mt-2 flex h-11 w-full rounded-md border border-primary/15 bg-transparent px-3 text-sm"
              value={modelId}
              onChange={(event) => {
                setModelId(event.target.value);
                setResult(null);
              }}
            >
              <option value={DINOV2_LEGACY}>{MODEL_LABELS[DINOV2_LEGACY]}</option>
              <option value={DINOV3_EXPERIMENTAL}>{MODEL_LABELS[DINOV3_EXPERIMENTAL]}</option>
            </select>
            <p className="mt-2 text-sm text-primary/70">Selected model: {MODEL_LABELS[modelId]}</p>
            {experimental ? (
              <p className="mt-2 text-sm font-semibold text-danger">EXPERIMENTAL — NOT APPROVED FOR PRODUCTION</p>
            ) : null}
          </div>
          <div>
            <label className="text-xs font-semibold text-primary/55" htmlFor="research-brand">
              Declared brand
            </label>
            <Input
              id="research-brand"
              className="mt-2"
              value={brand}
              onChange={(event) => setBrand(event.target.value)}
              placeholder="Patek Philippe"
            />
          </div>
          <div>
            <label className="text-xs font-semibold text-primary/55" htmlFor="research-image">
              Image
            </label>
            <Input
              id="research-image"
              className="mt-2"
              type="file"
              accept="image/jpeg,image/png,image/webp"
              onChange={(event) => setFile(event.target.files?.[0] ?? null)}
            />
          </div>
          <Button className="min-h-[44px]" disabled={pending} onClick={() => void submit()}>
            {pending ? "Running research demo" : "Run research demo"}
          </Button>
          {err ? <p className="text-sm font-semibold text-danger">{err}</p> : null}
          {result ? (
            <div className="space-y-3">
              <AuthBadge
                lane="research"
                verdict={result.decision}
                confidence={null}
                declaredBrand={result.declared_brand}
              />
              <p className="text-sm text-primary/70">
                Model identity: {result.model}
                {result.model_version ? ` version ${result.model_version}` : ""}
              </p>
              {result.model === "DINOV3_RESEARCH_PROTOTYPE" ? (
                <p className="text-sm font-semibold text-danger">EXPERIMENTAL — NOT APPROVED FOR PRODUCTION</p>
              ) : null}
              <p className="text-sm text-primary/70">Brand verification: not performed</p>
              <p className="text-sm text-primary/70">Scope: {result.model_scope}</p>
              <p className="text-sm text-primary/70">Research only. This result cannot publish a listing.</p>
              <p className="break-all text-xs text-primary/50">Checkpoint {result.checkpoint_sha}</p>
            </div>
          ) : null}
        </CardContent>
      </Card>
    </div>
  );
}
