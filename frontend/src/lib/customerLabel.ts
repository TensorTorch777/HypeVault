export const LEGACY_SCREENING_LABEL = "Legacy screening — not verified";
export const DEMO_LISTING_LABEL = "Demo listing — not verified";
export const PENDING_LABEL = "Pending";
export const REJECTED_LABEL = "Rejected";
export const PUBLISHED_UNVERIFIED_LABEL = "Published — not verified";
export const SCREENED_LISTED_LABEL = "Listed — photo screened, authenticity not verified";

const SEED_CONFIDENCE = 0.965;

export type LabelSource = {
  customer_label?: string | null;
  status?: string | null;
  verdict?: string | null;
  confidence?: number | null;
  s3_url?: string | null;
  screening_code?: string | null;
  screening_reason?: string | null;
};

function isSeedConstant(listing: LabelSource): boolean {
  if (listing.s3_url) return false;
  if (listing.verdict !== "AUTHENTIC") return false;
  if (listing.confidence == null) return false;
  return Math.abs(listing.confidence - SEED_CONFIDENCE) <= 1e-9;
}

/** Publication visibility is not an authenticity claim. */
export function customerLabel(listing: LabelSource): string {
  if (listing.customer_label) return listing.customer_label;
  if (listing.status === "live" && (listing.screening_code === "PASSED" || listing.screening_code === "REVIEW_APPROVED")) {
    return SCREENED_LISTED_LABEL;
  }
  if (listing.status === "pending") return PENDING_LABEL;
  if (listing.status === "rejected") return REJECTED_LABEL;
  if (isSeedConstant(listing)) return DEMO_LISTING_LABEL;
  if (listing.status === "live" && listing.s3_url) return LEGACY_SCREENING_LABEL;
  if (listing.status === "live") return PUBLISHED_UNVERIFIED_LABEL;
  return PENDING_LABEL;
}

/** Card, badge, and detail copy. Visibility is never an authenticity certificate. */
export function customerStatusCopy(listing: LabelSource): { title: string; detail: string } {
  const raw = listing.customer_label || customerLabel(listing);
  if (raw === DEMO_LISTING_LABEL || isSeedConstant(listing)) {
    return {
      title: DEMO_LISTING_LABEL,
      detail: "Demo data — not a live listing and not a verified authentic item.",
    };
  }
  if (listing.status === "rejected" && listing.screening_reason) {
    return { title: "Not listed", detail: listing.screening_reason };
  }
  if (listing.status === "pending" && listing.screening_code === "NEEDS_PROOF") {
    return {
      title: "Held — proof photo needed",
      detail: listing.screening_reason ?? "Send a photo of the watch beside today's date. An admin then checks it.",
    };
  }
  if (listing.status === "pending" && listing.screening_code === "PROOF_SUBMITTED") {
    return {
      title: "Held — waiting for an admin check",
      detail: "Proof photo received. Buyers cannot see this listing until an admin approves it.",
    };
  }
  if (listing.status === "live" && listing.screening_code === "REVIEW_APPROVED") {
    return {
      title: "Listed — checked by an admin",
      detail: "Listed after an admin checked the seller's proof photo. This is not a certificate of authenticity.",
    };
  }
  if (listing.status === "live" && listing.screening_code === "PASSED") {
    return {
      title: "Listed — photo screened",
      detail:
        "A wristwatch was found and the photo looks like an original photograph. Screening is not a certificate of authenticity.",
    };
  }
  if (listing.status === "pending" && listing.screening_code === null && listing.s3_url) {
    return {
      title: "Held — not listed",
      detail: "Screening did not finish, so buyers cannot see this listing. Retry screening from the upload page.",
    };
  }
  if (listing.status === "rejected" || raw === REJECTED_LABEL) {
    return {
      title: "Rejected by screening workflow",
      detail: "This is a legacy workflow state, not independent proof that the item is counterfeit. Authenticity is not verified.",
    };
  }
  if (listing.status === "live" || raw === PUBLISHED_UNVERIFIED_LABEL || raw === LEGACY_SCREENING_LABEL) {
    return {
      title: "Visible in marketplace — authenticity not verified",
      detail: "A visible or published listing is not a certificate of authenticity.",
    };
  }
  return {
    title: "Pending — not verified",
    detail: "This listing has not been screened yet, so buyers cannot see it.",
  };
}
