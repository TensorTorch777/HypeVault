"use client";

import { useQuery } from "@tanstack/react-query";
import { Check, Circle, Clock } from "lucide-react";

import { Skeleton } from "@/components/ui/skeleton";
import { type BenchmarkModelRow, type BenchmarkSummary, fetchBenchmarkSummary, formatDate } from "@/lib/market";
import { cn } from "@/lib/utils";

const MODEL_INFO: Record<string, { name: string; role: string; spec: string[] }> = {
  dinov2_legacy: {
    name: "DINOv2",
    role: "Live listing model",
    spec: ["ViT-B/14 backbone, 504 px input", "Uncalibrated sigmoid score", "FAKE at p ≥ 0.50; AUTHENTIC kept only at ≥ 0.88 confidence", "No review state"],
  },
  dinov3_experimental: {
    name: "DINOv3",
    role: "Research candidate",
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
    { label: "Real listing photos", detail: "300 eBay listings through the Browse API — waiting for an eBay developer key", state: confirmed ? "done" : "waiting" },
    { label: "Generated images", detail: "300 from a held-out generator (Kandinsky 2.2) and 150 from SDXL — waiting for GPU time", state: confirmed ? "done" : "waiting" },
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
            <tr key={r.model} className="border-b border-primary/8">
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

export default function BenchmarkPage() {
  const q = useQuery({ queryKey: ["benchmark-summary"], queryFn: fetchBenchmarkSummary });
  const s = q.data;

  return (
    <div className="pb-24">
      <section className="hv-guilloche border-b border-primary/10">
        <div className="mx-auto max-w-[77.5rem] px-4 pb-20 pt-36 sm:px-6">
          <p className="hv-eyebrow">Model benchmark</p>
          <h1 className="hv-display-lg mt-5 max-w-4xl text-primary">
            DINOv2 or DINOv3? <span className="hv-gradient-text">Decided by evidence</span>, written down first.
          </h1>
          <p className="mt-7 max-w-2xl text-[0.9375rem] leading-7 text-primary/70">
            Before a single image was collected, we fixed the question, the data, the analysis and the rule that names the
            frontier model. The numbers decide; the rule cannot move after the results are in.
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
            <section aria-labelledby="verdict" className="grid gap-10 lg:grid-cols-[1.1fr_1fr]">
              <div>
                <p className="hv-eyebrow">Current verdict</p>
                <h2 id="verdict" className="hv-display-md mt-4 text-primary">
                  {s.confirmatory?.frontier.frontier_model
                    ? `${MODEL_INFO[s.confirmatory.frontier.frontier_model]?.name} is the frontier model`
                    : "No frontier model named yet"}
                </h2>
                <p className="mt-5 text-[0.9375rem] leading-7 text-primary/70">
                  {s.confirmatory
                    ? s.confirmatory.frontier.statement
                    : "The confirmatory run has not happened, so no model may be called better. DINOv2 stays on the live listing check and DINOv3 stays a research candidate until the rule below is met."}
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
              <h2 id="contenders" className="hv-display-md mt-4 text-primary">Two frozen checkpoints, unchanged by the test</h2>
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
