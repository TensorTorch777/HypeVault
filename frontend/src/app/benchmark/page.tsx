"use client";

import { useQuery } from "@tanstack/react-query";
import { Check, Circle, Clock } from "lucide-react";

import { Skeleton } from "@/components/ui/skeleton";
import { type BenchmarkModelRow, type BenchmarkSummary, type KN, type LiveModelEvidence, fetchBenchmarkSummary, formatDate } from "@/lib/market";
import { cn } from "@/lib/utils";

const MODEL_INFO: Record<string, { name: string; role: string; spec: string[] }> = {
  dinov2_legacy: {
    name: "DINOv2",
    role: "Live listing model until 30 Oct 2026",
    spec: ["ViT-B/14 backbone, 504 px input", "Uncalibrated sigmoid score", "FAKE at p ≥ 0.50; AUTHENTIC kept only at ≥ 0.88 confidence", "No review state"],
  },
  dinov3_experimental: {
    name: "DINOv3",
    role: "Research comparison",
    spec: ["ViT-B/16 backbone with CLS + patch attention, 512 px", "Temperature-calibrated score (T = 0.2404)", "Sends low-resolution or low-contrast images to REVIEW", "Checkpoint hash verified before every answer"],
  },
};

type Step = { label: string; detail: string; state: "done" | "waiting" | "todo" };

function steps(s: BenchmarkSummary): Step[] {
  const confirmed = Boolean(s.confirmatory);
  return [
    { label: "Protocol pre-registered", detail: `Written ${formatDate(s.protocol.written_on)}, before any image was collected`, state: s.protocol.status ? "done" : "todo" },
    {
      label: "Pipeline check",
      detail: s.pipeline_check ? `${s.pipeline_check.images_scored} test images scored end to end through both served models` : "Not run",
      state: s.pipeline_check ? "done" : "todo",
    },
    { label: "Real listing photos", detail: confirmed ? "300 eBay listings through the Browse API" : "300 eBay listings through the Browse API — waiting for an eBay developer key", state: confirmed ? "done" : "waiting" },
    { label: "Generated images", detail: confirmed ? "300 from a held-out generator (Kandinsky 2.2) and 150 from SDXL" : "300 from a held-out generator (Kandinsky 2.2) and 150 from SDXL — waiting for GPU time", state: confirmed ? "done" : "waiting" },
    { label: "Confirmatory run", detail: confirmed ? `Run ${s.confirmatory?.run_id}` : "Runs once both classes meet their minimums", state: confirmed ? "done" : "todo" },
  ];
}

const pct = (x: number | null | undefined) => (x == null ? "—" : `${(100 * x).toFixed(1)}%`);

function ResultsTable({ rows }: { rows: BenchmarkModelRow[] }) {
  return (
    <div className="overflow-x-auto">
      <table className="w-full min-w-[40rem] text-left text-[0.8125rem]">
        <thead className="text-[0.625rem] uppercase tracking-[0.18em] text-primary/55">
          <tr className="border-b border-primary/10">
            {["Model", "Coverage", "Balanced accuracy", "Generated passed as real", "Real flagged as generated", "ROC-AUC"].map((h) => (
              <th key={h} className="py-3 pr-4 font-medium">{h}</th>
            ))}
          </tr>
        </thead>
        <tbody className="hv-num">
          {rows.map((r) => (
            <tr key={r.model} className="border-b border-primary/10">
              <td className="py-4 pr-4 text-primary">{MODEL_INFO[r.model]?.name ?? r.model}</td>
              <td className="pr-4">{pct(r.coverage)}</td>
              <td className="pr-4">{pct(r.balanced_accuracy)}</td>
              <td className="pr-4">{r.false_authentic ? `${r.false_authentic.k}/${r.false_authentic.n} (≤ ${pct(r.false_authentic.upper95)})` : "—"}</td>
              <td className="pr-4">{r.false_fake ? `${r.false_fake.k}/${r.false_fake.n} (≤ ${pct(r.false_fake.upper95)})` : "—"}</td>
              <td>{r.roc_auc == null ? "—" : r.roc_auc.toFixed(3)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

const frac = (x: KN) => (x ? `${x.k}/${x.n}` : "—");

/** The model that screens listings now and the pre-registered evidence for it. Every number comes from the API. */
function LiveModel({ live }: { live: LiveModelEvidence }) {
  const b = live.benchmark;
  const systems = Object.entries(b.systems);
  const ours = "phase76_head";
  return (
    <section aria-labelledby="live-model" className="space-y-10">
      <div className="grid gap-10 lg:grid-cols-[1.1fr_1fr]">
        <div>
          <p className="hv-eyebrow">Current verdict · {b.name}</p>
          <h2 id="live-model" className="hv-display-md mt-4 text-primary">
            {live.name} screens every listing
          </h2>
          <p className="mt-5 text-[0.9375rem] leading-7 text-primary/70">{b.question}</p>
          <p className="mt-4 text-[0.8125rem] leading-6 text-primary/60">
            Written {formatDate(b.written_on)}, before any of its images existed · scored once, {formatDate(b.scored_at)} · pass rule:{" "}
            <span className="text-primary/80">{b.rule}</span>{" "}
            <span className={cn("ml-1 rounded-full px-2 py-0.5 text-[0.6875rem] font-bold", b.met ? "bg-success/15 text-success" : "bg-danger/15 text-danger")}>
              {b.met ? "MET" : "NOT MET"}
            </span>
          </p>
        </div>
        <div className="rounded-2xl border border-primary/10 bg-card p-6">
          <p className="text-[0.6875rem] uppercase tracking-[0.2em] text-gold">The live model</p>
          <p className="mt-3 text-[0.875rem] leading-6 text-primary/75">{live.features}.</p>
          <p className="mt-3 text-[0.8125rem] leading-6 text-primary/60">{b.caveat}</p>
        </div>
      </div>

      <div className="overflow-x-auto rounded-2xl border border-primary/10 bg-card p-6">
        <table className="w-full min-w-[36rem] text-left text-[0.8125rem]">
          <thead className="text-[0.625rem] uppercase tracking-[0.18em] text-primary/55">
            <tr className="border-b border-primary/10">
              <th className="py-3 pr-4 font-medium">Images</th>
              {systems.map(([id, name]) => (
                <th key={id} className={cn("py-3 pr-4 font-medium", id === ours && "text-gold")}>{name}</th>
              ))}
            </tr>
          </thead>
          <tbody className="hv-num">
            {b.groups.map((g) => (
              <tr key={g.id} className="border-b border-primary/10">
                <td className="py-4 pr-4 text-primary">
                  {g.label} <span className="text-primary/50">· {g.kind === "real" ? "wrongly called AI" : "caught as AI"}</span>
                </td>
                {systems.map(([id]) => (
                  <td key={id} className={cn("pr-4", id === ours ? "font-bold text-primary" : "text-primary/70")}>{frac(g.systems[id])}</td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div className="grid gap-4 md:grid-cols-3">
        {live.unseen_generators ? (
          <div className="rounded-2xl border border-primary/10 bg-card p-7">
            <p className="text-[0.6875rem] uppercase tracking-[0.2em] text-gold">Never-seen generators · {live.unseen_generators.name}</p>
            <p className="hv-num mt-4 font-display text-[2.4rem] font-extrabold leading-none text-primary">{frac(live.unseen_generators.caught)}</p>
            <p className="mt-2 text-[0.8125rem] leading-6 text-primary/70">
              caught from {live.unseen_generators.generators}; real eBay watches wrongly flagged {frac(live.unseen_generators.real_ebay_flagged)}.{" "}
              {live.unseen_generators.note}
            </p>
          </div>
        ) : null}
        {live.edited_copy_hold ? (
          <div className="rounded-2xl border border-primary/10 bg-card p-7">
            <p className="text-[0.6875rem] uppercase tracking-[0.2em] text-gold">Screenshots and crops · Phase 78</p>
            <table className="hv-num mt-4 w-full text-left text-[0.8125rem]">
              <thead className="text-[0.625rem] uppercase tracking-[0.16em] text-primary/55">
                <tr><th className="pb-2 font-medium">AI image</th><th className="pb-2 font-medium">Phase 76</th><th className="pb-2 font-medium">Phase 78</th></tr>
              </thead>
              <tbody>
                {live.edited_copy_hold.rows.map((r) => (
                  <tr key={r.edit}><td className="py-1 text-primary/80">{r.edit}</td><td className="text-primary/70">{r.phase76 ?? "—"}</td><td className="font-bold text-primary">{r.phase78 ?? "—"}</td></tr>
                ))}
                {live.edited_copy_hold.real_flagged ? (
                  <tr><td className="py-1 text-primary/80">Real photos flagged</td><td className="font-bold text-primary">{live.edited_copy_hold.real_flagged.phase76}</td><td className="text-primary/70">{live.edited_copy_hold.real_flagged.phase78}</td></tr>
                ) : null}
              </tbody>
            </table>
            <p className="mt-3 text-[0.75rem] leading-5 text-primary/60">{live.edited_copy_hold.role}</p>
          </div>
        ) : null}
        {live.known_ai ? (
          <div className="rounded-2xl border border-primary/10 bg-card p-7">
            <p className="text-[0.6875rem] uppercase tracking-[0.2em] text-gold">Known-AI copy check · SSCD</p>
            <p className="hv-num mt-4 font-display text-[2.4rem] font-extrabold leading-none text-primary">
              {live.known_ai.edited_matched == null ? "—" : pct(live.known_ai.edited_matched)}
            </p>
            <p className="mt-2 text-[0.8125rem] leading-6 text-primary/70">
              of edited copies of known AI images matched at similarity {live.known_ai.cut}; {live.known_ai.real_matched} of{" "}
              {live.known_ai.real_n?.toLocaleString("en-US")} real watch photos matched.
            </p>
          </div>
        ) : null}
      </div>
    </section>
  );
}

export default function BenchmarkPage() {
  const q = useQuery({ queryKey: ["benchmark-summary"], queryFn: fetchBenchmarkSummary });
  const s = q.data;

  return (
    <div className="pb-24">
      <section className="hv-guilloche border-b border-primary/10">
        <div className="mx-auto max-w-[77.5rem] px-4 pb-20 pt-36 sm:px-6">
          <p className="hv-eyebrow">Model benchmark</p>
          <h1 className="hv-display-lg mt-5 max-w-4xl text-primary">
            Which model screens listings? <span className="hv-gradient-text">Decided by evidence</span>, written down first.
          </h1>
          <p className="mt-7 max-w-2xl text-[0.9375rem] leading-7 text-primary/70">
            For every benchmark, the question, the data, the analysis and the pass rule were fixed before a single image was
            collected, and the results were scored once. The numbers decide; the rule cannot move after the results are in.
          </p>
        </div>
      </section>

      <div className="mx-auto max-w-[77.5rem] space-y-24 px-4 pt-20 sm:px-6">
        {q.isLoading ? <Skeleton className="h-80 rounded-2xl" /> : null}
        {q.isError ? (
          <p role="alert" className="rounded-2xl border border-danger/30 p-6 text-danger">The benchmark summary could not be loaded.</p>
        ) : null}

        {s ? (
          <>
            {s.live_model ? <LiveModel live={s.live_model} /> : null}

            <div className="border-t border-primary/10 pt-16">
              <p className="hv-eyebrow">History · benchmark_v1</p>
              <h2 className="hv-display-md mt-4 text-primary">The first fight: DINOv2 vs DINOv3</h2>
              <p className="mt-4 max-w-3xl text-[0.9375rem] leading-7 text-primary/70">
                The first pre-registered comparison, before the hybrid existed. Its verdict below is kept as it was written at the
                time; benchmark_v6 later put {s.live_model?.name ?? "the DINOv3 + Perception Encoder hybrid"} live.
              </p>
            </div>

            <section aria-labelledby="verdict" className="grid gap-10 lg:grid-cols-[1.1fr_1fr]">
              <div>
                <p className="hv-eyebrow">benchmark_v1 verdict, at the time</p>
                <h2 id="verdict" className="hv-display-md mt-4 text-primary">
                  {s.confirmatory?.frontier.frontier_model
                    ? `${MODEL_INFO[s.confirmatory.frontier.frontier_model]?.name} is the frontier model`
                    : "No frontier model named yet"}
                </h2>
                <p className="mt-5 text-[0.9375rem] leading-7 text-primary/70">
                  {s.confirmatory
                    ? s.confirmatory.frontier.statement
                    : "The confirmatory run did not happen, so neither model could be called better on benchmark_v1."}
                </p>
                {s.previous_evaluation ? (
                  <div className="mt-8 rounded-2xl border border-primary/10 bg-card p-6">
                    <p className="text-[0.6875rem] uppercase tracking-[0.2em] text-primary/55">{s.previous_evaluation.name}</p>
                    <p className="mt-3 text-[0.875rem] leading-6 text-primary/75">{s.previous_evaluation.summary}</p>
                  </div>
                ) : null}
              </div>
              <ol className="rounded-2xl border border-primary/10 bg-card p-7" aria-label="Progress">
                {steps(s).map((st, i) => (
                  <li key={st.label} className={cn("flex gap-4", i > 0 && "mt-6")}>
                    <span
                      className={cn(
                        "mt-0.5 grid h-7 w-7 shrink-0 place-items-center rounded-full border",
                        st.state === "done" && "border-success/50 text-success",
                        st.state === "waiting" && "border-review/50 text-review",
                        st.state === "todo" && "border-primary/20 text-primary/40"
                      )}
                      aria-hidden="true"
                    >
                      {st.state === "done" ? <Check className="h-3.5 w-3.5" /> : st.state === "waiting" ? <Clock className="h-3.5 w-3.5" /> : <Circle className="h-2.5 w-2.5" />}
                    </span>
                    <div>
                      <p className="text-[0.875rem] text-primary">
                        {st.label}
                        <span className="sr-only"> — {st.state === "done" ? "complete" : st.state === "waiting" ? "waiting" : "not started"}</span>
                      </p>
                      <p className="mt-1 text-[0.75rem] leading-5 text-primary/60">{st.detail}</p>
                    </div>
                  </li>
                ))}
              </ol>
            </section>

            {s.confirmatory?.primary ? (
              <section aria-labelledby="results">
                <p className="hv-eyebrow">Results</p>
                <h2 id="results" className="hv-display-md mt-4 text-primary">{s.confirmatory.primary.set} · {s.confirmatory.primary.n} images</h2>
                <div className="mt-8 rounded-2xl border border-primary/10 bg-card p-6">
                  <ResultsTable rows={s.confirmatory.primary.models} />
                  <p className="mt-5 text-[0.75rem] text-primary/60">
                    {s.confirmatory.primary.paired.discordant} images where the models disagreed · exact McNemar p ={" "}
                    {s.confirmatory.primary.paired.mcnemar_p.toFixed(4)}
                  </p>
                </div>
              </section>
            ) : null}

            <section aria-labelledby="contenders">
              <p className="hv-eyebrow">The contenders</p>
              <h2 id="contenders" className="hv-display-md mt-4 text-primary">The benchmark_v1 contenders: two frozen checkpoints</h2>
              <div className="mt-10 grid gap-4 md:grid-cols-2">
                {Object.entries(MODEL_INFO).map(([id, m]) => (
                  <article key={id} className="rounded-2xl border border-primary/10 bg-card p-7">
                    <p className="text-[0.6875rem] uppercase tracking-[0.22em] text-gold">{m.role}</p>
                    <h3 className="hv-serif mt-3 text-[2.2rem] text-primary">{m.name}</h3>
                    <ul className="mt-5 space-y-2.5 text-[0.8125rem] leading-6 text-primary/70">
                      {m.spec.map((line) => (
                        <li key={line} className="flex gap-3"><span className="mt-2.5 h-px w-3 shrink-0 bg-gold" aria-hidden="true" />{line}</li>
                      ))}
                    </ul>
                    <p className="mt-6 break-all font-mono text-[0.6875rem] text-primary/45">
                      SHA-256 {s.protocol.models[id]?.checkpoint_sha256}
                    </p>
                  </article>
                ))}
              </div>
            </section>

            <section aria-labelledby="rule" className="grid gap-10 lg:grid-cols-[1fr_1.2fr]">
              <div>
                <p className="hv-eyebrow">The decision rule</p>
                <h2 id="rule" className="hv-display-md mt-4 text-primary">How a winner can be named</h2>
                <p className="mt-5 text-[0.875rem] leading-7 text-primary/70">
                  Both models see the same images. The comparison counts only the images where they disagree and tests that
                  split with an exact McNemar test. With 49 disagreements the test has 80% power to detect a 70/30 split.
                </p>
              </div>
              <dl className="divide-y divide-primary/10 rounded-2xl border border-primary/10 bg-card">
                {Object.entries(s.protocol.frontier_rule ?? {}).map(([k, v]) => (
                  <div key={k} className="p-6">
                    <dt className="text-[0.6875rem] uppercase tracking-[0.2em] text-gold">{k.replace(/_/g, " ").toLowerCase()}</dt>
                    <dd className="mt-2 text-[0.8125rem] leading-6 text-primary/75">{v}</dd>
                  </div>
                ))}
              </dl>
            </section>

            <section aria-labelledby="design" className="grid gap-4 md:grid-cols-3">
              <h2 id="design" className="sr-only">Benchmark design</h2>
              <div className="rounded-2xl border border-primary/10 bg-card p-7">
                <p className="text-[0.6875rem] uppercase tracking-[0.2em] text-gold">Real photographs</p>
                <p className="mt-4 text-[0.8125rem] leading-6 text-primary/75">{s.protocol.classes.REAL_PHOTO}.</p>
              </div>
              <div className="rounded-2xl border border-primary/10 bg-card p-7">
                <p className="text-[0.6875rem] uppercase tracking-[0.2em] text-gold">Generated images</p>
                <ul className="mt-4 space-y-2 text-[0.8125rem] leading-6 text-primary/75">
                  {Object.entries(s.protocol.classes.GENERATED).map(([k, v]) => (
                    <li key={k}><span className="text-primary">{k.startsWith("heldout") ? "Primary · " : "Secondary · "}</span>{v}</li>
                  ))}
                </ul>
              </div>
              <div className="rounded-2xl border border-primary/10 bg-card p-7">
                <p className="text-[0.6875rem] uppercase tracking-[0.2em] text-gold">Same treatment for every image</p>
                <ul className="mt-4 space-y-2 text-[0.8125rem] leading-6 text-primary/75">
                  {(s.protocol.normalisation ?? []).map((n) => <li key={n}>{n}</li>)}
                </ul>
              </div>
              <p className="text-[0.75rem] leading-6 text-primary/55 md:col-span-3">{s.protocol.scope_limit}</p>
            </section>
          </>
        ) : null}
      </div>
    </div>
  );
}
