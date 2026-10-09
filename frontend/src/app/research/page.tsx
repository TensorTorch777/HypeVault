"use client";

import { useEffect, useState } from "react";
import axios from "axios";

import { AuthBadge } from "@/components/AuthBadge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { api, getApiErrorMessage } from "@/lib/api";

const DINOV2_LEGACY = "dinov2_legacy";
const DINOV3_EXPERIMENTAL = "dinov3_experimental";
const MODEL_IDS = [DINOV2_LEGACY, DINOV3_EXPERIMENTAL] as const;
type ModelId = (typeof MODEL_IDS)[number];

const MODEL_LABELS: Record<ModelId, string> = {
  [DINOV2_LEGACY]: "DINOv2 — Legacy",
  [DINOV3_EXPERIMENTAL]: "DINOv3 — Experimental — Not approved for production",
};

const EXPECTED_MODEL: Record<ModelId, ResearchResult["model"]> = {
  [DINOV2_LEGACY]: "LEGACY_DINOV2",
  [DINOV3_EXPERIMENTAL]: "DINOV3_RESEARCH_PROTOTYPE",
};

type ResearchResult = {
  status: "AUTHENTIC" | "FAKE" | "REVIEW";
  decision: "AUTHENTIC" | "FAKE" | "REVIEW";
  declared_brand: string;
  brand_verification: "NOT_PERFORMED";
  model_scope: string;
  research_only: true;
  production_ready: false;
  checkpoint_sha: string | null;
  model: "LEGACY_DINOV2" | "DINOV3_RESEARCH_PROTOTYPE";
  model_version?: string;
};

type ModelReadiness = {
  logical_id: string;
  model: string;
  version: string;
  architecture: string;
  same_model_as_live_route: boolean;
  ready: boolean;
};

type ReadinessReport = {
  server: { live: boolean; ready: boolean };
  models: ModelReadiness[];
};

export default function ResearchDemoPage() {
  const [brand, setBrand] = useState("");
  const [file, setFile] = useState<File | null>(null);
  const [modelId, setModelId] = useState<ModelId | null>(null);
  const [readiness, setReadiness] = useState<ReadinessReport | null>(null);
  const [readinessErr, setReadinessErr] = useState<string | null>(null);
  const [result, setResult] = useState<{ selected: ModelId; body: ResearchResult } | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [pending, setPending] = useState(false);

  useEffect(() => {
    let cancelled = false;
    api
      .get<ReadinessReport>("/research/models")
      .then(({ data }) => {
        if (cancelled) return;
        setReadiness(data);
        const firstReady = MODEL_IDS.find((id) => data.models.some((m) => m.logical_id === id && m.ready));
        setModelId(firstReady ?? null);
      })
      .catch((error) => {
        if (cancelled) return;
        const fallback = "Model readiness is unavailable. No model can be selected.";
        setReadinessErr(axios.isAxiosError(error) ? getApiErrorMessage(error, fallback) : fallback);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const statusFor = (id: ModelId) => readiness?.models.find((m) => m.logical_id === id) ?? null;
  const isReady = (id: ModelId) => Boolean(statusFor(id)?.ready);
  const experimental = modelId === DINOV3_EXPERIMENTAL;
  const selectedStatus = modelId ? statusFor(modelId) : null;

  async function submit() {
    if (!modelId || !isReady(modelId)) {
      setErr("The selected model is not ready. No other model will be used.");
      return;
    }
    if (!file) {
      setErr("Choose an image before running the research demo.");
      return;
    }
    const selected = modelId;
    setErr(null);
    setResult(null);
    setPending(true);
    try {
      const body = new FormData();
      body.append("image", file);
      body.append("brand", brand);
      body.append("logical_model", selected);
      const { data } = await api.post<ResearchResult>("/research/verify", body);
      if (data.model !== EXPECTED_MODEL[selected]) {
        setErr("The response came from a different model than the one selected. The result was discarded.");
        return;
      }
      setResult({ selected, body: data });
    } catch (error) {
      const fallback = "The selected model did not return a classification.";
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
              value={modelId ?? ""}
              disabled={!readiness || pending}
              onChange={(event) => {
                setModelId(event.target.value as ModelId);
                setResult(null);
                setErr(null);
              }}
            >
              {modelId === null ? <option value="">No model is ready</option> : null}
              {MODEL_IDS.map((id) => (
                <option key={id} value={id} disabled={!isReady(id)}>
                  {MODEL_LABELS[id]}
                  {readiness && !isReady(id) ? " (unavailable)" : ""}
                </option>
              ))}
            </select>
            {!readiness && !readinessErr ? <p className="mt-2 text-sm text-primary/60">Checking model readiness</p> : null}
            {readinessErr ? <p className="mt-2 text-sm font-semibold text-danger">{readinessErr}</p> : null}
            {readiness ? (
              <ul className="mt-2 space-y-1 text-xs text-primary/60">
                {MODEL_IDS.map((id) => {
                  const s = statusFor(id);
                  return (
                    <li key={id}>
                      {MODEL_LABELS[id]}: {s?.ready ? "ready" : "unavailable"}
                      {s ? ` · ${s.architecture}` : ""}
                      {s && !s.same_model_as_live_route ? " · not the live listing-check model" : ""}
                    </li>
                  );
                })}
              </ul>
            ) : null}
            {modelId ? <p className="mt-2 text-sm text-primary/70">Selected model: {MODEL_LABELS[modelId]}</p> : null}
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
          <Button
            className="min-h-[44px]"
            disabled={pending || !modelId || !selectedStatus?.ready}
            onClick={() => void submit()}
          >
            {pending ? "Running research demo" : "Run research demo"}
          </Button>
          {err ? <p className="text-sm font-semibold text-danger">{err}</p> : null}
          {result ? (
            <div className="space-y-3">
              <AuthBadge
                lane="research"
                verdict={result.body.decision}
                confidence={null}
                declaredBrand={result.body.declared_brand}
              />
              <p className="text-sm text-primary/70">
                Model that ran: {result.body.model}
                {result.body.model_version ? ` version ${result.body.model_version}` : ""} (selected{" "}
                {MODEL_LABELS[result.selected]})
              </p>
              {result.body.model === "DINOV3_RESEARCH_PROTOTYPE" ? (
                <p className="text-sm font-semibold text-danger">EXPERIMENTAL — NOT APPROVED FOR PRODUCTION</p>
              ) : null}
              <p className="text-sm text-primary/70">Declared brand: {result.body.declared_brand}</p>
              <p className="text-sm text-primary/70">
                brand_verification = {result.body.brand_verification} (the brand was not checked from the image)
              </p>
              <p className="text-sm text-primary/70">Scope: {result.body.model_scope}</p>
              <p className="text-sm text-primary/70">Research only. This result cannot publish a listing.</p>
              {result.body.checkpoint_sha ? (
                <p className="break-all text-xs text-primary/50">Checkpoint {result.body.checkpoint_sha}</p>
              ) : null}
            </div>
          ) : null}
        </CardContent>
      </Card>
    </div>
  );
}
