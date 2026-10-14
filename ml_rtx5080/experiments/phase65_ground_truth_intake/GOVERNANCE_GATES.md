# Governance gate crosswalk (Phase 67)

This note clarifies naming from the Phase 66 audit. It does **not** authorize
acquisition, inference, authenticity claims, or production promotion.

## Authoritative runtime / config symbols

| Gate | Where it lives | Effective closed value |
|---|---|---|
| `DATA_READINESS` | `ml_rtx5080/experiments/phase65_ground_truth_intake/intake.py` | `NO_GO_FOR_AUTHENTICITY_CLAIMS` |
| `ACQUISITION_AUTHORIZED` | same `intake.py` module | `False` |
| `INFERENCE_AUTHORIZED` | same `intake.py` module | `False` |
| `AUTHENTICITY_MODEL_PRODUCTION_APPROVED` | `backend/inference/publication_gate.py` and `ml_rtx5080/experiments/publication_safety_v1/publication_policy.json` | `False` |
| `PRODUCTION_PROMOTION` | `ml_rtx5080/experiments/scope_lock_v1/scope_contract.json` | `FORBIDDEN` |

Marketplace publication is refused by `publication_gate.can_publish_listing()` while
`AUTHENTICITY_MODEL_PRODUCTION_APPROVED` is false.

## Descriptive aliases (not executable constants)

These strings appear in historical reports and summaries:

- `DINOv3_PRODUCTION_ALLOWED`
- `PRODUCTION_PROMOTION_ALLOWED`

They document the closed intent. They are **not** Python module constants and must
not be treated as alternate switches that can open a gate. Prefer the authoritative
symbols above.

## What remains human / external

Website terms, licenses, permissions, consent, label verification, and faculty
approval are separate requirements. Schema-valid metadata, CI success, or an intake
readiness report does not satisfy them and does not change the closed gates above.
