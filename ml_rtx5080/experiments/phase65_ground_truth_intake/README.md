# Phase 65 — Ground-truth intake and evidence-review pipeline

**Status:** `OFFLINE_METADATA_ONLY — NO ACQUISITION, IMAGE ACCESS, OR INFERENCE AUTHORIZED`

This package connects merged `hypevault_scraper` manifests to the existing Phase 55 evidence ledger and `ml_rtx5080.data_admission` contracts. It prepares human review queues and readiness reports. It does **not** create authenticity labels automatically and does **not** freeze or score a dataset.

## Contracts reused

| Contract | Role |
|---|---|
| `hypevault_scraper.common` (`scraper_manifest_v1`) | Source manifest validation; `UNVERIFIED_WEB_IMAGE` / `SYNTHETIC_AI_GENERATED` |
| Phase 55 `evidence_ledger.schema.json` | Authoritative eligibility for evidence-backed labels |
| Phase 61 label/evidence SOP | Physical-product linkage and “no sensitive data in Git” |
| `ml_rtx5080/data_admission.py` | Required ADMIT for ML-use eligibility (candidate_image_record_v2) |
| Phase 62 readiness invariants | `DATA_READINESS=NO_GO_FOR_AUTHENTICITY_CLAIMS` preserved |

No competing authenticity schema is introduced. Review decisions live in a **separate append-only ledger**.

## Invariants

- Scraper manifests are never rewritten.
- `UNVERIFIED_WEB_IMAGE` and `SYNTHETIC_AI_GENERATED` never become verified by metadata alone.
- Synthetic imagery cannot become counterfeit (or authentic) ground truth.
- Rights review is independent of authenticity adjudication.
- A verified authenticity outcome can still be ML-ineligible without approved rights / `admission_decision` ADMIT.
- Review authenticity decisions must match Phase 55 `verified_label` and adjudication outcome.
- No mapping into `Label_0_*` / `Label_1_*`.
- Intake guards all CLI paths (including symlinks) before any read/write; never opens image files or locked final-test / OOD paths.

## CLI

```bash
# From the repository root
python ml_rtx5080/experiments/phase65_ground_truth_intake/intake.py \
  --manifest /path/to/manifest.jsonl \
  --output-dir /tmp/phase65_out

# Optional: append one human decision, then re-run intake
python ml_rtx5080/experiments/phase65_ground_truth_intake/intake.py \
  --manifest /path/to/manifest.jsonl \
  --review-ledger /path/to/review_ledger.jsonl \
  --append-decision /path/to/decision.json \
  --output-dir /tmp/phase65_out
```

Prefer metadata paths under ignored local artifact trees.

### Outputs

| File | Contents |
|---|---|
| `intake_report.json` | Valid/invalid counts, duplicate hashes, manifest checksums |
| `review_queue.jsonl` | One queue row per record for human review |
| `review_queue.csv` | Compact CSV summary |
| `readiness_report.json` | Counts, blockers, verified-label totals, no-go readiness |

## Review ledger

Append-only JSONL (`phase65_review_ledger_v1`). A proposal to treat a record as authentic/counterfeit **must** attach a Phase 55 `phase55_evidence_record` that:

- Validates against `evidence_ledger.schema.json` with `eligible_primary_benchmark=true`
- Matches the scraper’s `stored_sha256`
- Uses agreed/adjudicated reviewer outcomes

Rights must be `APPROVED` and `candidate_image_record_v2` must receive `admission_decision` → `ADMIT` before ML-use eligibility. Schema-valid evidence alone does not count as admitted.

## Readiness meaning

With only scraper manifests and no admitted evidence ledger rows, the report shows:

- `verified_authentic_count = 0`
- `verified_counterfeit_count = 0`
- `data_readiness = NO_GO_FOR_AUTHENTICITY_CLAIMS`
- `dataset_frozen = false`
- `benchmark_ready = false`

Passing validators does **not** mean the dataset is frozen or claim-ready.

## Tests

```bash
python -m unittest discover -s ml_rtx5080/experiments/phase65_ground_truth_intake/tests -v
```

## Governance gate crosswalk

Descriptive names in reports are not always executable symbols. Authoritative runtime
enforcement for this repository is:

| Concern | Authoritative source | Runtime enforcement | Notes |
|---|---|---|---|
| Data readiness / authenticity claims | `DATA_READINESS = "NO_GO_FOR_AUTHENTICITY_CLAIMS"` in `intake.py` (and related phase READMEs) | Emitted into Phase 65 readiness/intake reports; does not authorize claims | Remains NO_GO until an explicit later governance change |
| Acquisition | `ACQUISITION_AUTHORIZED = False` in `intake.py` | Report field only in Phase 65; scraper network still requires separate CLI opt-in | Not flipped by tests or merges |
| Inference authorization (research intake) | `INFERENCE_AUTHORIZED = False` in `intake.py` | Report field; intake never runs inference | Separate from Triton serving availability |
| Marketplace publication / production authenticity model | `AUTHENTICITY_MODEL_PRODUCTION_APPROVED = False` in `backend/inference/publication_gate.py` and `ml_rtx5080/experiments/publication_safety_v1/publication_policy.json` | `can_publish_listing()` returns false while the flag is false | Source of truth for listing go-live |
| Research prototype promotion scope | `PRODUCTION_PROMOTION: "FORBIDDEN"` in `ml_rtx5080/experiments/scope_lock_v1/scope_contract.json` | Scope-lock / research gating docs and tests assert `FORBIDDEN` | Not the same key as descriptive `PRODUCTION_PROMOTION_ALLOWED` |
| Descriptive `DINOv3_PRODUCTION_ALLOWED` / `PRODUCTION_PROMOTION_ALLOWED` | Appear in historical reports (`docs/engineering/*`, dual-model reports) | **Not** standalone Python/JSON constants | Treat as documentation aliases for the closed state above; do not invent a bypass |

Human approvals, website terms, licenses, consent, and faculty permission remain external requirements. They are not satisfied by schema validity, CI green, or a readiness report.

### Review-ledger append durability (Phase 67)

`append_review_decision` requires a non-empty `protected_inputs` list and commits each new
row via a same-directory temp file + `os.replace` so readers do not observe a truncated
final line from that writer. Prior ledger bytes are preserved if the commit fails before
replace. Concurrent multi-writer appends are not locked; last successful replace wins.
Single-writer use is assumed.