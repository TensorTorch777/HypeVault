# HypeVault Phase 55A — Ground-Truth Source Qualification

**Status: PRE-FLIGHT ONLY.** This package is a local review starter, not an authorization to collect images, contact organizations, spend money, accept terms, or run inference.

## Why this phase exists

Phase 53/54 established `DATA_READINESS = NO_GO_FOR_AUTHENTICITY_CLAIMS`. The common held-out cohort has 315 historically authentic-labeled A. Lange & Söhne images and zero historically fake-labeled images. Independently verified labels remain zero. Existing folder labels are historical metadata, not authenticity ground truth.

## Contents

- `source_due_diligence.md` — public source leads and unresolved questions.
- `source_tracker.csv` — non-sensitive lead tracking with no outreach sent.
- `outreach_draft.md` — information request draft; do not send without separate approval.
- `evidence_ledger.schema.json` — schema for a future access-controlled, pseudonymized evidence ledger.
- `tests/test_evidence_ledger_schema.py` — synthetic-only schema tests.
- `phase55a_review_checklist.md` — review gate before any real data collection.

## Safety boundaries

- No watch images, certificates, serial numbers, customer records, reviewer identities, credentials, or private correspondence are included.
- Do not scrape or download publicly visible watch images. Public availability does not imply permission for model evaluation.
- Do not send messages or forms, create accounts, accept terms, or spend money from this phase.
- Do not open locked final-test or excluded OOD image files, enumerate them, or run inference.
- Do not alter checkpoints, weights, thresholds, temperature, calibration, serving, production flags, access control, or publication policy.
- `eligible_primary_benchmark` must remain `false` until independent review establishes every gate in the Phase 54 plan.

## Tests

The included test uses Python `jsonschema` and only synthetic metadata. Run from this folder:

```bash
python -m unittest discover -s tests -v
```

When integrating into the repository, check whether `jsonschema` is already available in the CI environment; if not, add a pinned compatible dependency to the relevant test requirements rather than allowing the test to silently skip.
