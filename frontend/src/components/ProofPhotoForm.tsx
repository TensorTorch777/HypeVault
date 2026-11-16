"use client";

import { useState } from "react";
import axios from "axios";

import { Button } from "@/components/ui/button";
import { api, type Listing } from "@/lib/api";

/** A held listing's proof photo: the watch beside a paper with today's date and the seller's name. An admin then decides. */
export function ProofPhotoForm({ listingId, onSent }: { listingId: string; onSent: (listing: Listing) => void }) {
  const [file, setFile] = useState<File | null>(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const today = new Date().toLocaleDateString(undefined, { day: "numeric", month: "long", year: "numeric" });

  async function send() {
    if (!file) return;
    setBusy(true);
    setErr(null);
    try {
      const fd = new FormData();
      fd.append("image", file);
      const { data } = await api.post<Listing>(`/listings/${listingId}/proof`, fd);
      onSent(data);
    } catch (e) {
      const detail = axios.isAxiosError(e) ? (e.response?.data as { detail?: string } | undefined)?.detail : undefined;
      setErr(detail ?? "The proof photo could not be sent. Try again.");
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="rounded-xl border border-[#F5C451]/40 bg-[#F5C451]/[0.06] p-5">
      <p className="font-display text-[1.2rem] font-extrabold uppercase leading-none text-primary">Send a proof photo</p>
      <ol className="mt-3 list-decimal space-y-1 pl-5 text-[0.8125rem] leading-6 text-primary/80">
        <li>Write today&apos;s date ({today}) and your name on a piece of paper.</li>
        <li>Photograph the watch next to the paper, dial facing the camera.</li>
        <li>Upload it here. An admin compares it with your first photo.</li>
      </ol>
      <input
        type="file"
        accept="image/jpeg,image/png,image/webp"
        aria-label="Proof photo"
        className="mt-4 block w-full cursor-pointer text-sm text-primary/80 file:mr-4 file:cursor-pointer file:rounded-full file:border-0 file:bg-accent file:px-4 file:py-1.5 file:text-[0.8125rem] file:font-bold file:text-ink"
        onChange={(e) => setFile(e.target.files?.[0] ?? null)}
      />
      {err ? <p className="mt-3 text-sm font-semibold text-danger" role="alert">{err}</p> : null}
      <Button className="mt-4 min-h-[2.75rem]" disabled={!file || busy} onClick={() => void send()}>
        {busy ? "Sending…" : "Send proof photo"}
      </Button>
    </div>
  );
}
