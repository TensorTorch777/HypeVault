"use client";

import { useEffect, useState } from "react";
import axios from "axios";

import { AuthBadge } from "@/components/AuthBadge";
import { ResearchExplanation, type ResearchExplanationPayload } from "@/components/ResearchExplanation";
import { DeclaredBrandSelect } from "@/components/DeclaredBrandSelect";
import { WatchMarketLookup } from "@/components/market/WatchMarketLookup";
import { Button } from "@/components/ui/button";
import { PageHero } from "@/components/layout/PageHero";
import { Card, CardContent, CardHeader } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { api } from "@/lib/api";
import { explanationForVerifiedResult } from "@/lib/researchExplanationConsistency";
import {
  DECLARED_BRAND_NOTE,
  SERVING_STATUS_LABEL,
  VALIDITY_NOT_ESTABLISHED,
  classifyHttpFailure,
  researchFailureMessage,
} from "@/lib/semanticCopy";

const DINOV2_LEGACY = "dinov2_legacy";
const DINOV3_EXPERIMENTAL = "dinov3_experimental";
// DINOv3 is the current model; DINOv2 is the earlier model, kept here so the two can be compared.
const MODEL_IDS = [DINOV3_EXPERIMENTAL, DINOV2_LEGACY] as const;
type ModelId = (typeof MODEL_IDS)[number];
const COMPARE = "compare";
type Choice = ModelId | typeof COMPARE;

const MODEL_LABELS: Record<ModelId, string> = {
  [DINOV3_EXPERIMENTAL]: "DINOv3 — Current model",
  [DINOV2_LEGACY]: "DINOv2 — Earlier model (research comparison)",
};

const MODEL_NOTES: Record<ModelId, string> = {
  [DINOV3_EXPERIMENTAL]: "Current model. Research result, not a production certificate.",
  [DINOV2_LEGACY]: "Earlier model, kept for research comparison. Not a production certificate.",
};

const EXPECTED_MODEL: Record<ModelId, ResearchResult["model"]> = {
  [DINOV2_LEGACY]: "LEGACY_DINOV2",
  [DINOV3_EXPERIMENTAL]: "DINOV3_RESEARCH_PROTOTYPE",
};

const RAN_LABEL: Record<ResearchResult["model"], string> = {
  LEGACY_DINOV2: "DINOv2 (earlier model)",
  DINOV3_RESEARCH_PROTOTYPE: "DINOv3 (current model)",
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
  unavailable_reason: string | null;
};

const UNAVAILABLE_REASONS: Record<string, string> = {
  TRITON_MODEL_NOT_READY: "not loaded in Triton",
  INSTANCE_KIND_UNKNOWN: "Triton instance kind unknown",
  PARITY_NOT_VALIDATED_FOR_SERVED_INSTANCE: "Triton parity not validated for this instance",
};

type LaneOutcome =
  | { state: "unavailable" }
  | { state: "error"; message: string }
  | { state: "ok"; body: ResearchResult; explanation: ResearchExplanationPayload };

type ReadinessReport = {
  server: { live: boolean; ready: boolean };
  models: ModelReadiness[];
};

export default function ResearchDemoPage() {
  const [brand, setBrand] = useState("");
  const [file, setFile] = useState<File | null>(null);
  const [modelId, setModelId] = useState<Choice | null>(null);
  const [readiness, setReadiness] = useState<ReadinessReport | null>(null);
  const [readinessErr, setReadinessErr] = useState<string | null>(null);
  const [result, setResult] = useState<{ selected: ModelId; body: ResearchResult } | null>(null);
  const [comparison, setComparison] = useState<Partial<Record<ModelId, LaneOutcome>> | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [pending, setPending] = useState(false);
  const [explanation, setExplanation] = useState<ResearchExplanationPayload | null>(null);
  const [explanationPending, setExplanationPending] = useState(false);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [previewSize, setPreviewSize] = useState({ width: 1, height: 1 });

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
        const status = axios.isAxiosError(error) ? error.response?.status : undefined;
        const bodyStatus = axios.isAxiosError(error) ? (error.response?.data as { status?: string } | undefined)?.status : undefined;
        setReadinessErr(researchFailureMessage(classifyHttpFailure(status, bodyStatus)));
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const statusFor = (id: ModelId) => readiness?.models.find((m) => m.logical_id === id) ?? null;
  const isReady = (id: ModelId) => Boolean(statusFor(id)?.ready);
  const single = modelId && modelId !== COMPARE ? modelId : null;
  const comparing = modelId === COMPARE;
  const canRun = comparing ? MODEL_IDS.some(isReady) : Boolean(single && isReady(single));

  async function submit() {
    if (!single || !isReady(single)) {
      setErr("Serving status: unavailable. No other model will be used, and no result was produced.");
      return;
    }
    if (!file) {
      setErr("Choose an image before running the research demo.");
      return;
    }
    const selected = single;
    setErr(null);
    setResult(null);
    setComparison(null);
    setExplanation(null);
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
      setExplanationPending(true);
      try {
        const explainBody = new FormData();
        explainBody.append("image", file);
        explainBody.append("brand", brand);
        explainBody.append("logical_model", selected);
        const explained = await api.post<ResearchExplanationPayload>("/research/explain", explainBody);
        setExplanation(
          explanationForVerifiedResult(
            { model: data.model, decision: data.decision },
            explained.data,
          ),
        );
      } catch {
        setExplanation({
          status: "unavailable",
          reason: "The explanation request failed.",
          publication_decision: "BLOCKED",
        });
      } finally {
        setExplanationPending(false);
      }
    } catch (error) {
      const status = axios.isAxiosError(error) ? error.response?.status : undefined;
      const bodyStatus = axios.isAxiosError(error) ? (error.response?.data as { status?: string } | undefined)?.status : undefined;
      setErr(researchFailureMessage(classifyHttpFailure(status, bodyStatus)));
    } finally {
      setPending(false);
    }
  }

  async function compareBoth() {
    if (!file) {
      setErr("Choose an image before comparing the two research models.");
      return;
    }
    setErr(null);
    setResult(null);
    setComparison(null);
    setExplanation(null);
    setPending(true);
    const next: Partial<Record<ModelId, LaneOutcome>> = {};
    await Promise.all(
      MODEL_IDS.map(async (id) => {
        if (!isReady(id)) {
          next[id] = { state: "unavailable" };
          return;
        }
        try {
          const body = new FormData();
          body.append("image", file);
          body.append("brand", brand);
          body.append("logical_model", id);
          const { data } = await api.post<ResearchResult>("/research/verify", body);
          if (data.model !== EXPECTED_MODEL[id]) {
            next[id] = {
              state: "error",
              message: "The response came from a different model than this lane. It was discarded.",
            };
            return;
          }
          // Each lane gets its own patch map, checked against that lane's classification.
          let explanation: ResearchExplanationPayload;
          try {
            const explainBody = new FormData();
            explainBody.append("image", file);
            explainBody.append("brand", brand);
            explainBody.append("logical_model", id);
            const explained = await api.post<ResearchExplanationPayload>("/research/explain", explainBody);
            explanation = explanationForVerifiedResult({ model: data.model, decision: data.decision }, explained.data);
          } catch {
            explanation = { status: "unavailable", reason: "The explanation request failed.", publication_decision: "BLOCKED" };
          }
          next[id] = { state: "ok", body: data, explanation };
        } catch (error) {
          const status = axios.isAxiosError(error) ? error.response?.status : undefined;
          const bodyStatus = axios.isAxiosError(error)
            ? (error.response?.data as { status?: string } | undefined)?.status
            : undefined;
          next[id] = { state: "error", message: researchFailureMessage(classifyHttpFailure(status, bodyStatus)) };
        }
      }),
    );
    setComparison(next);
    setPending(false);
  }

  return (
    <div className="pb-24">
    <PageHero
      eyebrow="Research console"
      title="Five-brand prototype"
      subtitle="Run a photo through DINOv2 or the DINOv3 candidate and see which patches drove the score. Research only: nothing here publishes a listing."
      scene="neon_rain"
      render="/renders/rm_rm011/hero.webp"
      compact
    />
    <div className="mx-auto max-w-3xl px-4">
      <Card>
        <CardHeader>
          <p className="hv-eyebrow">Research demo</p>
          <h2 className="mt-1 font-display text-[2rem] font-extrabold uppercase leading-none">Run a screening</h2>
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
              className="mt-2 flex h-11 w-full rounded-xl border border-white/12 bg-[#0E0F1E] px-3 text-[0.9375rem] text-primary focus:border-accent focus:outline-none"
             
              value={modelId ?? ""}
              disabled={!readiness || pending}
              onChange={(event) => {
                setModelId(event.target.value as Choice);
                setResult(null);
                setComparison(null);
                setErr(null);
              }}
            >
              {modelId === null ? (
                <option value="">
                  No model is available to serve
                </option>
              ) : null}
              {MODEL_IDS.map((id) => (
                <option key={id} value={id} disabled={!isReady(id)}>
                  {MODEL_LABELS[id]}
                  {readiness && !isReady(id) ? " (unavailable)" : ""}
                </option>
              ))}
              <option value={COMPARE} disabled={!MODEL_IDS.some(isReady)}>
                Compare both models (side by side)
              </option>
            </select>
            {!readiness && !readinessErr ? <p className="mt-2 text-sm text-primary/60">Checking serving status</p> : null}
            {readinessErr ? <p className="mt-2 text-sm font-semibold text-danger" role="alert">{readinessErr}</p> : null}
            <p className="mt-2 text-sm text-primary/70">{VALIDITY_NOT_ESTABLISHED}</p>
            {readiness ? (
              <ul className="mt-2 space-y-1 text-xs text-primary/60">
                {MODEL_IDS.map((id) => {
                  const s = statusFor(id);
                  return (
                    <li key={id}>
                      {SERVING_STATUS_LABEL} · {MODEL_LABELS[id]}:{" "}
                      {s?.ready
                        ? "available"
                        : `unavailable${s?.unavailable_reason ? ` (${UNAVAILABLE_REASONS[s.unavailable_reason] ?? s.unavailable_reason})` : ""}`}
                      {s ? ` · ${s.architecture}` : ""}
                      {s ? (s.same_model_as_live_route ? " · same model as the live listing check" : " · not the live listing-check model") : ""}
                    </li>
                  );
                })}
              </ul>
            ) : null}
            {single ? <p className="mt-2 text-sm text-primary/70">Selected model: {MODEL_LABELS[single]}</p> : null}
            {comparing ? <p className="mt-2 text-sm text-primary/70">Selected: both models, side by side</p> : null}
          </div>
          <div>
            <label className="text-xs font-semibold text-primary/55" htmlFor="research-brand">
              Declared brand
            </label>
            <DeclaredBrandSelect
              id="research-brand"
              value={brand}
              disabled={pending}
              onChange={(next) => {
                setBrand(next);
                setResult(null);
                setComparison(null);
                setErr(null);
              }}
            />
            <p className="mt-2 text-xs text-primary/60">{DECLARED_BRAND_NOTE}</p>
          </div>
          <div>
            <label className="text-xs font-semibold text-primary/55" htmlFor="research-image">
              Image
            </label>
            <Input
              id="research-image"
              className="mt-2 cursor-pointer py-2 file:mr-4 file:cursor-pointer file:rounded-full file:border-0 file:bg-accent file:px-4 file:py-1.5 file:text-[0.8125rem] file:font-bold file:text-ink"
              type="file"
              accept="image/jpeg,image/png,image/webp"
              onChange={(event) => {
                const next = event.target.files?.[0] ?? null;
                setFile(next);
                setExplanation(null);
                if (previewUrl) URL.revokeObjectURL(previewUrl);
                if (!next) {
                  setPreviewUrl(null);
                  return;
                }
                const url = URL.createObjectURL(next);
                setPreviewUrl(url);
                const img = new Image();
                img.onload = () => setPreviewSize({ width: img.naturalWidth, height: img.naturalHeight });
                img.src = url;
              }}
            />
          </div>
          <div className="flex flex-wrap gap-3">
            <Button
              className="min-h-[2.75rem]"
              disabled={pending || !brand || !file || !canRun}
              onClick={() => void (comparing ? compareBoth() : submit())}
            >
              {pending ? "Running" : comparing ? "Compare both models" : "Run research demo"}
            </Button>
          </div>
          <p className="text-xs text-primary/60">
            Comparison runs each research model separately on this image and declared brand. The two classifications are not combined, and neither one is treated as more accurate. Neither result publishes a listing.
          </p>
          {err ? <p className="text-sm font-semibold text-danger" role="alert">{err}</p> : null}
          {result ? (
            <div className="space-y-3">
              <AuthBadge
                lane="research"
                verdict={result.body.decision}
                confidence={null}
                declaredBrand={result.body.declared_brand}
              />
              <p className="text-sm text-primary/70">
                Model that ran: {RAN_LABEL[result.body.model]}
                {result.body.model_version ? ` version ${result.body.model_version}` : ""} (selected{" "}
                {MODEL_LABELS[result.selected]})
              </p>
              <p className="text-sm text-primary/70">{MODEL_NOTES[result.selected]}</p>
              <p className="text-sm text-primary/70">Declared brand: {result.body.declared_brand}</p>
              <p className="text-sm text-primary/70">
                brand_verification = {result.body.brand_verification} (the brand was not checked from the image)
              </p>
              <p className="text-sm text-primary/70">Scope: {result.body.model_scope}</p>
              <p className="text-sm text-primary/70">Research only. This result cannot publish a listing.</p>
              {result.body.checkpoint_sha ? (
                <p className="break-all text-xs text-primary/50">Checkpoint {result.body.checkpoint_sha}</p>
              ) : null}
              <ResearchExplanation
                imageUrl={previewUrl}
                imageWidth={previewSize.width}
                imageHeight={previewSize.height}
                payload={explanation}
                pending={explanationPending}
              />
            </div>
          ) : null}
          {comparison ? (
            <div className="space-y-3">
              <p className="text-sm font-semibold text-primary">Research-only side-by-side classification</p>
              <p className="text-sm text-primary/70">{DECLARED_BRAND_NOTE} {VALIDITY_NOT_ESTABLISHED}</p>
              <div className="grid gap-3 md:grid-cols-2">
                {MODEL_IDS.map((id) => {
                  const lane = comparison[id];
                  return (
                    <div key={id} className="rounded-xl border border-primary/15 p-4">
                      <p className="text-sm font-semibold">{MODEL_LABELS[id]}</p>
                      <p className="mt-1 text-xs text-primary/60">{MODEL_NOTES[id]}</p>
                      {lane?.state === "unavailable" ? (
                        <p className="mt-3 text-sm text-primary/70" role="status">Serving status: unavailable. No result was produced for this model.</p>
                      ) : null}
                      {lane?.state === "error" ? (
                        <p className="mt-3 text-sm font-semibold text-danger" role="alert">{lane.message}</p>
                      ) : null}
                      {lane?.state === "ok" ? (
                        <div className="mt-3 space-y-2">
                          <p className="text-sm">Model classification: {lane.body.decision} — research screening result, not verified.</p>
                          <p className="text-sm text-primary/70">Declared brand: {lane.body.declared_brand}</p>
                          <p className="text-sm text-primary/70">Model that ran: {RAN_LABEL[lane.body.model]}</p>
                          <p className="text-xs text-primary/55">This lane cannot publish a listing.</p>
                          <ResearchExplanation
                            imageUrl={previewUrl}
                            imageWidth={previewSize.width}
                            imageHeight={previewSize.height}
                            payload={lane.explanation}
                            pending={false}
                          />
                        </div>
                      ) : null}
                    </div>
                  );
                })}
              </div>
            </div>
          ) : null}
        </CardContent>
      </Card>
    </div>
    {/* Once the photo has been classified, show what the watch in it sells for. */}
    {file && (result || comparison) ? (
      <div className="mx-auto mt-8 max-w-6xl px-4">
        <WatchMarketLookup photo={file} />
      </div>
    ) : null}
    </div>
  );
}
