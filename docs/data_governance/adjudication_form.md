# Authentic / counterfeit adjudication form

Complete one form per physical watch. Store it under the evidence ID, and reference it from each image record (`verification.evidence_id`).

| Field | Value |
| --- | --- |
| Evidence ID | |
| Contribution ID(s) | |
| Brand (as examined) | |
| Basis for brand | e.g. case-back engraving, movement calibre, papers. A readable dial wordmark alone is not enough |
| Reference / model | |
| Serial checked against | e.g. manufacturer service record, or "not checked" |
| Examiner reference | access-controlled ID, not a public name |
| Examiner qualification evidence ID | |
| Examiner independence | Confirm no stake in the sale, and that you did not use HypeVault model output |
| Examination date | |
| Examination method | Tick all that apply: movement inspection, case-back opening, weight/dimension check, serial verification, papers/box check, other |
| Conclusion | `authentic` / `counterfeit` / `undetermined` |
| Reasons for the conclusion | What was observed |
| Confidence | `certain` / `probable`. A `probable` conclusion goes to second review |
| Images taken of this item | count, and the image SHA-256 list |
| Second examiner (if required) | reference, date, conclusion |
| Signature / attestation | |

## Rules

- `undetermined` is a valid conclusion. It is never converted to `authentic` or `counterfeit` later without a new examination.
- A conclusion based only on photos, listing text, price, or seller claims is not admissible. Record it as `undetermined`.
- If two examiners disagree, follow `disputed_label_policy.md`.
