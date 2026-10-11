# Split manifest contract

**Status:** `PROTOCOL_ONLY — NO DATA ACQUISITION OR INFERENCE AUTHORIZED`

This contract describes a future manifest. This phase does not build that manifest from images. It does not open, enumerate, hash, inspect, or score the locked final-test set or excluded OOD files.

`DATA_READINESS` remains `NO_GO_FOR_AUTHENTICITY_CLAIMS`. Independently verified labels remain zero. No real watch image, certificate, raw serial number, personal identity, credential, or private evidence belongs in this document or in a committed manifest.

## What a future manifest is

A versioned, reviewable list of pseudonymous records. One record is one image. Many image records may share one `product_id`. The manifest is immutable after its checksum is recorded. A later change is a new version, not an edit of the frozen bytes.

The committed artifact for a future study would be the schema, the checksum, the eligibility counts, and the exclusion reasons. Per-image paths and private evidence stay in an access-controlled store that is not committed. Locked evaluation membership is checksummed so a later audit can prove what was held out without printing that contents in this phase.

## Required fields

| Field | Rule |
| --- | --- |
| `image_sha256` | SHA-256 of the exact image bytes. Used for exact duplicate detection. Not a physical-watch identity. |
| Near-duplicate group | A documented perceptual or byte-near match. Method, threshold, and version are frozen before eligibility. An unresolved near-duplicate link blocks primary eligibility. |
| `product_id` | Pseudonymous physical-watch identity. Every photo, crop, duplicate, and alternate view of that watch uses this id and the same partition. `UNKNOWN` cannot enter the primary benchmark. |
| `brand` | Verified brand, not the historical folder name. `UNKNOWN` cannot enter the primary benchmark. The historical folder brand, if any, is a separate field and is not ground truth. |
| `reference_family` | Verified reference family or `UNKNOWN`. `UNKNOWN` does not by itself prove ineligibility, but it cannot be filled by guessing. |
| `source_id`, `seller_id`, `marketplace_id` | Pseudonymous. `source_id` must be known for primary eligibility. Seller and marketplace stay `UNKNOWN` when they are not lawfully known, and unknown values are counted, not invented. |
| `verified_label` | `AUTHENTIC` or `COUNTERFEIT` only after independent evidence. Otherwise `INDETERMINATE`. Never copied from a directory name. |
| Evidence-record reference | Pseudonymous id of the evidence packet. The packet itself, including reviewer notes and certificates, is not committed. |
| Image-use rights | `license_status` approved for the intended use, including `ML_EVALUATION` in `permitted_purposes`. |
| Consent or legal basis | `consent_status` of `APPROVED` or a documented `NOT_REQUIRED`. |
| `retention_until` | A real calendar date for any primary-eligible record. `UNKNOWN` is allowed on pending records and is not allowed on primary-eligible records. |
| `deletion_status` | `RETENTION_ACTIVE` for any primary-eligible record. `NOT_RECORDED`, `PENDING_DELETION`, `DELETED`, and `NOT_APPLICABLE` are not eligible unless a later written exception policy says so. No such policy exists. |
| Partition | Exactly one of `training`, `calibration`, `development`, or `locked_evaluation`. |
| DINOv2 membership check | `PASS`, `FAIL`, or `UNKNOWN` against that model's task-specific training and calibration records. Primary eligibility requires `PASS`. |
| DINOv3 membership check | Same, separately. Primary eligibility requires `PASS` for both models when the record is in a paired cohort. |
| Preprocessing version | The native preprocessing id for the model that will be scored. DINOv2 and DINOv3 ids are not interchangeable. |
| Frozen model identifier | Checkpoint SHA-256 of the model the manifest is authorized to score. |
| Protocol version | `phase56_validation_protocol_v1` or a later version that replaces it before data collection. |

These eligibility gates match the fail-closed Phase 55A evidence-ledger schema now on `main`. Satisfying the schema is necessary and not sufficient. Schema tests do not verify a real watch.

## One watch, one partition

All images that share a `product_id`, an exact `image_sha256`, or an unresolved near-duplicate group are assigned together. A crop or alternate view cannot be placed in `locked_evaluation` while another view of the same watch is in `training`, `calibration`, or `development`.

## Overlap handling

Before a record can be primary-eligible, the manifest build compares:

- `product_id` across partitions and against both models' task-specific training and calibration memberships.
- Exact image hashes and near-duplicate groups across partitions.
- `source_id` against sources used in task-specific training or calibration. A reused source is reported. If the approved plan treats that source as leakage, the product is excluded from the primary analysis.
- `seller_id` and `marketplace_id` the same way when those ids are known. Unknown ids are not treated as proof of separation.

Any unresolved overlap blocks primary-benchmark eligibility or is explicitly excluded from the primary analysis. Exclusions are counted by reason. They are not dropped silently, and they are not described as evaluated.

## Eligible paired cohort

The paired cohort is the set of `product_id` values that:

- sit in `locked_evaluation`;
- have a verified brand and a resolved `verified_label`;
- meet the Phase 55A eligibility fields, including rights, consent, retention, deletion, evidence, and `PASS` disjointness for both models;
- include every image of that product, all in `locked_evaluation`.

If a product is eligible for only one model, it is reported in that model's cohort only. It is not called a paired comparison.

The published derivation records the rule, the protocol version, the manifest checksum, and the counts. It does not list locked-evaluation paths or image bytes. A reviewer can recompute the checksum and the counts from the access-controlled manifest without this phase reading the locked final-test or OOD files.

## Checksums

- `manifest_sha256` over the canonical serialized manifest.
- `protocol_sha256` over `validation_protocol_v1.md` at the version named in the manifest.
- A separate checksum of the locked-evaluation id list, stored with the manifest, so later audit can detect substitution.

A study that cannot reproduce these checksums does not have an immutable manifest. It cannot support a primary claim.

## What this phase does not do

No script in this phase reads a dataset directory or lists locked test files. No image is hashed here. No membership check against the live image store is executed here. Those steps belong to a later authorized phase.
