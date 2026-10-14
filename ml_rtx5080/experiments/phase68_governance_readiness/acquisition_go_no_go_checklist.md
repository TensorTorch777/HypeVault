# Acquisition go / no-go checklist

**Status:** Checklist for a future human decision. Completing rows does not authorize acquisition.

Use with `templates/acquisition_decision_record.template.json` and `decision_flow.md`.
Mark each item `PASS`, `FAIL`, `PENDING`, or `NOT_APPLICABLE` (N/A requires rationale + reviewer).

---

## 1. Research purpose and permitted use

- [ ] Intended research questions stated in writing.
- [ ] Permitted model-development purpose bounded (evaluation / non-public research / other—explicit).
- [ ] Target product/category scope listed (brands, families, geographies if any).
- [ ] Intended audience and publication boundaries stated (including aggregates-only if applicable).
- [ ] Explicit out-of-scope uses listed (commercial resale of images, open redistribution, etc.).
- [ ] Purpose compatible with each proposed source’s terms (pending source review in §2).

**Stop if:** purpose is unclear, contradicts source terms, or implies production authenticity claims without separate promotion approval.

---

## 2. Source-by-source approval

For **each** proposed source (no provider contact in Phase 68):

- [ ] Canonical provider/source identity recorded.
- [ ] Public URL(s) recorded from desk review only.
- [ ] Intended acquisition method named (API, bulk download, manual capture, etc.) with scope limit (volume, date range, content types).
- [ ] Terms of service / license evidence referenced (document id, URI, retrieval date, checksum if retained).
- [ ] Copyright and redistribution constraints summarized.
- [ ] Required permission from provider or rights holder identified (`REQUIRED` / `NOT_REQUIRED` with rationale).
- [ ] Restrictions on storage, transformation, research use, derivatives, and publication recorded.
- [ ] Reviewer, review date, evidence refs, and expiration/review-by date set.

**Do not** treat publicly visible images as automatically licensed for the proposed research use.

**Stop if:** license/terms unclear, incompatible, expired, or permission required but missing.

---

## 3. Privacy, consent, and ethics

- [ ] Privacy risk assessment completed or marked N/A with rationale.
- [ ] Consent requirements identified (`REQUIRED` / `NOT_REQUIRED` with rationale).
- [ ] Institutional / faculty / ethics approval identified where applicable.
- [ ] Evidence refs for each applicable approval (not fabricated; Phase 68 leaves these empty).
- [ ] Faculty approval is **not** treated as a substitute for licenses or website terms.

**Stop if:** required privacy/consent/ethics review is missing, denied, expired, or withdrawn.

---

## 4. Label verification and ground truth

- [ ] Independent verification procedure defined (aligned with Phase 55 evidence categories and Phase 61 SOP).
- [ ] Reviewer qualification requirements stated (pseudonymous refs only in Git).
- [ ] Dual-review + adjudication / disagreement handling defined.
- [ ] Provenance and chain-of-custody requirements defined.
- [ ] Immutable evidence references required for verified labels.
- [ ] Explicit exclusion: filename, directory name, scraped description, seller claim, model prediction, or synthetic generation is **not** independently verified ground truth.
- [ ] Distinction documented between (a) eligible for ground-truth *review* and (b) eligible for a specific ML *admission* use (Phase 65 / `data_admission`).

**Stop if:** no credible independent verification procedure exists.

---

## 5. Dataset design and leakage controls

Prospective plan only—**do not** inspect or unlock final-test / OOD content:

- [ ] Record and product identity scheme defined.
- [ ] Brand / model / seller / marketplace / source provenance fields specified where available.
- [ ] Exact-duplicate and near-duplicate detection policy defined.
- [ ] Product-disjoint and source-disjoint split rules defined when applicable.
- [ ] Calibration / validation / final-test / OOD separation plan stated at policy level only.
- [ ] Leakage prevention for duplicate, product, seller, source, and brand documented.
- [ ] Missing metadata and unverifiable labels reported separately (not silently coerced).

**Stop if:** leakage or duplicate controls are unresolved for the proposed acquisition scope.

---

## 6. Security, privacy, and retention

- [ ] Designated storage location and access controls (least privilege) defined.
- [ ] Checksums and audit-log requirements defined.
- [ ] Retention period and backup policy defined.
- [ ] Deletion / takedown / quarantine procedures defined for unauthorized, mislicensed, mislabelled, or disputed artifacts.
- [ ] Incident handling owner named (or vacant → PENDING).
- [ ] Treatment of personal or sensitive information defined.

**Stop if:** storage security, retention, or deletion procedures are unapproved or missing.

---

## 7. Budget and operational ownership

- [ ] Budget ceiling stated (currency + amount) or explicit `ZERO_BUDGET_DESK_ONLY`.
- [ ] Funding source identified (or N/A with rationale for desk-only).
- [ ] Accountable owner named (or vacant → PENDING).
- [ ] Time window and volume/rate limits stated.
- [ ] Storage cost estimate and stop condition before expenditure stated.

**Phase 68 itself must not incur expenses, request funding, contact suppliers, or activate paid services.**

**Stop if:** spending would be required without approved budget/owner/stop condition.

---

## 8. Safety and scope boundaries

- [ ] Locked final-test / OOD access remains prohibited without a separate approved plan.
- [ ] No bypass of Phase 65 path guards, admission, or publication gates proposed.
- [ ] Unverified scraper/synthetic labels will not be treated as ground truth.
- [ ] Model predictions will not determine human labels.
- [ ] Research evaluation, dataset readiness, inference authorization, public claims, and production promotion remain **separate** decisions.

**Stop if:** any request implies bypassing final-test/OOD locks, elevating scraper labels to GT, or collapsing separate authorization decisions.

---

## Mandatory NO-GO / blocked triggers (summary)

Any applicable critical requirement that is missing, contradictory, expired, withdrawn, or unverifiable → `NO_GO` or `CONDITIONAL_NO_GO`. See `decision_flow.md` for remediation ownership. Proposed numeric thresholds in any attached plan are **proposals requiring explicit human approval**, not established project policy.
