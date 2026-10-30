"use client";

import { useState } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import axios from "axios";

import { Button } from "@/components/ui/button";
import { api, type Listing } from "@/lib/api";
import { formatDate } from "@/lib/market";

const AWAITING = new Set(["NEEDS_PROOF", "PROOF_SUBMITTED"]);

function pct(value: unknown): string {
  return typeof value === "number" ? `${Math.round(value * 100)}%` : "—";
}

/** Held listings: the first photo next to the seller's proof photo (watch beside today's date), with approve and reject. */
export function ReviewQueue({ queryKey }: { queryKey: readonly unknown[] }) {
  const client = useQueryClient();
  const q = useQuery({ queryKey, queryFn: async () => (await api.get<Listing[]>("/listings/all")).data });
  const [busy, setBusy] = useState<string | null>(null);
  const [notes, setNotes] = useState<Record<string, string>>({});
  const [err, setErr] = useState<string | null>(null);
  const held = (q.data ?? []).filter((l) => l.status === "pending" && AWAITING.has(l.screening_code ?? ""));

  async function decide(id: string, decision: "approve" | "reject") {
    setBusy(id);
    setErr(null);
    try {
      await api.post(`/listings/${id}/review`, { decision, note: notes[id] || null });
      await client.invalidateQueries({ queryKey });
    } catch (e) {
      const detail = axios.isAxiosError(e) ? (e.response?.data as { detail?: string } | undefined)?.detail : undefined;
      setErr(detail ?? "The decision was not saved.");
    } finally {
      setBusy(null);
    }
  }

  return (
    <section className="mx-auto mt-10 max-w-[73.75rem] px-4 sm:px-6" aria-labelledby="review-title">
      <h2 id="review-title" className="font-display text-[2rem] font-extrabold uppercase leading-none text-primary">
        Review queue <span className="text-primary/45">{q.isLoading ? "" : held.length}</span>
      </h2>
      <p className="mt-2 max-w-3xl text-sm text-primary/60">
        Listings screening was not sure about. Approve only when the proof photo shows the same watch as the first photo, next to a
        handwritten note with the date and the seller&apos;s name.
      </p>
      {err ? <p className="mt-3 text-sm font-semibold text-danger" role="alert">{err}</p> : null}
      {!q.isLoading && held.length === 0 ? <p className="mt-6 text-sm text-primary/55">Nothing is waiting for review.</p> : null}
      <div className="mt-6 grid gap-4">
        {held.map((l) => {
          const d = (l.screening_details ?? {}) as Record<string, unknown>;
          const proof = (d.proof ?? {}) as Record<string, unknown>;
          return (
            <div key={l.id} className="hv-hud grid gap-5 p-5 md:grid-cols-[1fr_1fr_1.1fr]">
              {[
                ["First photo", l.s3_url],
                ["Proof photo", l.proof_url],
              ].map(([label, src]) => (
                <figure key={label} className="min-w-0">
                  <figcaption className="hv-mono text-[0.6875rem] uppercase tracking-[0.18em] text-primary/55">{label}</figcaption>
                  {src ? (
                    // eslint-disable-next-line @next/next/no-img-element
                    <img src={src} alt={`${label} for ${l.product_name}`} className="mt-2 aspect-square w-full rounded-lg object-cover" />
                  ) : (
                    <div className="mt-2 flex aspect-square w-full items-center justify-center rounded-lg border border-dashed border-white/15 text-sm text-primary/50">
                      Not sent yet
                    </div>
                  )}
                </figure>
              ))}
              <div className="min-w-0 text-sm text-primary/75">
                <p className="font-display text-[1.3rem] font-extrabold uppercase leading-none text-primary">{l.product_name}</p>
                <p className="mt-1 text-[0.75rem] text-primary/55">
                  {l.brand ?? "No brand"} · {l.seller_email ?? "unknown seller"} · {formatDate(l.created_at)}
                </p>
                <dl className="mt-4 space-y-1 text-[0.8125rem]">
                  <div className="flex justify-between gap-4"><dt>First photo, chance it is AI</dt><dd>{pct(d.p_ai)}</dd></div>
                  <div className="flex justify-between gap-4"><dt>Proof photo, chance it is AI</dt><dd>{pct(proof.p_ai)}</dd></div>
                  <div className="flex justify-between gap-4"><dt>Closest known AI image</dt><dd>{pct(d.known_ai_similarity)}</dd></div>
                  <div className="flex justify-between gap-4">
                    <dt>Held because</dt>
                    <dd>{d.held_by === "edited_copy_model" ? "looks like an edited AI copy" : "borderline AI score"}</dd>
                  </div>
                </dl>
                <input
                  className="mt-4 w-full rounded-lg border border-white/12 bg-[#0E0F1E] px-3 py-2 text-sm text-primary"
                  placeholder="Note to the seller (optional)"
                  aria-label={`Note for ${l.product_name}`}
                  value={notes[l.id] ?? ""}
                  onChange={(e) => setNotes((n) => ({ ...n, [l.id]: e.target.value }))}
                />
                <div className="mt-3 flex flex-wrap gap-2">
                  <Button
                    className="min-h-[2.5rem]"
                    disabled={busy === l.id || l.screening_code !== "PROOF_SUBMITTED"}
                    onClick={() => void decide(l.id, "approve")}
                  >
                    Approve
                  </Button>
                  <Button variant="outline" className="min-h-[2.5rem]" disabled={busy === l.id} onClick={() => void decide(l.id, "reject")}>
                    Reject
                  </Button>
                </div>
              </div>
            </div>
          );
        })}
      </div>
    </section>
  );
}
