# Phase 65 — Ground-truth intake and evidence-review pipeline

**Status:** `OFFLINE_METADATA_ONLY — NO ACQUISITION, IMAGE ACCESS, OR INFERENCE AUTHORIZED`

This package connects merged `hypevault_scraper` manifests to the existing Phase 55 evidence ledger and `ml_rtx5080.data_admission` contracts. It prepares human review queues and readiness reports. It does **not** create authenticity labels automatically and does **not** freeze or score a dataset.

## Contracts reused

| Contract | Role |
|---|---|
| `hypevault_scraper.common` (`scraper_manifest_v1`) | Source manifest validation; `UNVERIFIED_WEB_IMAGE` / `SYNTHETIC_AI_GENERATED` |
| Phase 55 `evidence_ledger.schema.json` | Authoritative eligibility for evidence-backed labels |
| Phase 61 label/evidence SOP | Physical-product linkage and “no sensitive data in Git” |
| `ml_rtx5080/data_admission.py` | Optional candidate_image_record_v2 admission check |
| Phase 62 readiness invariants | `DATA_READINESS=NO_GO_FOR_AUTHENTICITY_CLAIMS` preserved |

No competing authenticity schema is introduced. Review decisions live in a **separate append-only ledger**.

## Invariants

- Scraper manifests are never rewritten.
- `UNVERIFIED_WEB_IMAGE` and `SYNTHETIC_AI_GENERATED` never become verified by metadata alone.
- Synthetic imagery cannot become counterfeit (or authentic) ground truth.
- Rights review is independent of authenticity adjudication.
- A verified authenticity outcome can still be ML-ineligible without approved rights / admission.
- No mapping into `Label_0_*` / `Label_1_*`.
- Intake never opens image files or locked final-test / OOD paths.

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

Rights must be `APPROVED` before ML-use eligibility. Optional `candidate_image_record_v2` payloads are checked with `admission_decision`.

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
