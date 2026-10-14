# Phase 68 — Governance & acquisition go/no-go readiness package

**Status:** `GOVERNANCE_PREPARATION_ONLY — NO ACQUISITION, CONTACT, EXPENDITURE, OR INFERENCE AUTHORIZED`

Phase 68 defines what HypeVault would need before a human could *consider* authorizing future data acquisition. It turns Phase 66/67 findings into a decision framework: approvals, evidence, ownership, rights, source eligibility, scope limits, storage, cost ceilings, and stop conditions.

## What this phase is

- A documentation and template package for a future explicit human decision.
- A JSON Schema and blank draft record for an acquisition *recommendation* review.
- Focused schema/template tests using synthetic governance metadata only.

## What this phase is not

- Not authorization to acquire, download, generate, or inspect images or datasets.
- Not permission to contact sellers, websites, marketplaces, or rights holders.
- Not approval to spend money or use paid APIs.
- Not a change to any runtime gate, policy JSON, scope contract, CLI flag, or environment variable.
- Not ground-truth label creation, training, inference, calibration, or production promotion.
- Not a claim that faculty permission, website terms, licenses, consent, or verification are satisfied.

## Authoritative closed gates (inspected, unchanged)

| Concern | Authoritative source | Closed value |
|---|---|---|
| Data readiness / authenticity claims | `ml_rtx5080/experiments/phase65_ground_truth_intake/intake.py` | `DATA_READINESS = "NO_GO_FOR_AUTHENTICITY_CLAIMS"` |
| Acquisition | same `intake.py` | `ACQUISITION_AUTHORIZED = False` |
| Inference (intake context) | same `intake.py` | `INFERENCE_AUTHORIZED = False` |
| Marketplace publication | `backend/inference/publication_gate.py` + `publication_safety_v1/publication_policy.json` | `AUTHENTICITY_MODEL_PRODUCTION_APPROVED = False`; `can_publish_listing()` fails closed |
| Research prototype promotion | `scope_lock_v1/scope_contract.json` | `PRODUCTION_PROMOTION = "FORBIDDEN"` |
| Descriptive aliases | Historical reports only | `DINOv3_PRODUCTION_ALLOWED` / `PRODUCTION_PROMOTION_ALLOWED` are **not** executable constants (see Phase 67 `GOVERNANCE_GATES.md`) |

Scraper network access still requires an explicit CLI opt-in (`hypevault_scraper.common.require_network_opt_in`). That opt-in is **not** `ACQUISITION_AUTHORIZED` and does not authorize authenticity-ground-truth collection under this package.

## Package contents

| Artifact | Role |
|---|---|
| `acquisition_go_no_go_checklist.md` | Human review checklist covering purpose, sources, rights, privacy, labels, leakage, security, budget, and boundaries |
| `decision_flow.md` | Fail-closed state machine for draft → no-go / conditional / recommendation |
| `schemas/acquisition_decision_record.schema.json` | Machine-checkable shape for a future decision record |
| `templates/acquisition_decision_record.template.json` | Blank draft template (status `DRAFT`; no fabricated approvals) |
| `tests/test_phase68_governance_schema.py` | Offline schema/template consistency tests |

## How to complete and review a decision record

1. **Prepare (DRAFT):** Custodian drafts the record from the template; attaches evidence references (paths/URIs/checksums) without acquiring images.
2. **Review:** Required reviewers record outcome, evidence refs, and timestamps. Conflicts → unresolved / NO_GO until adjudicated.
3. **Outcome vocabulary (non-authorizing):**
   - `DRAFT` — incomplete.
   - `NO_GO` — mandatory condition fails.
   - `CONDITIONAL_NO_GO` — remediation required before reconsideration.
   - `GO_RECOMMENDED_SEPARATE_AUTHORIZATION_REQUIRED` — reviewers recommend a *separate* authorization decision; **acquisition remains unauthorized**.
4. **Separate authorization:** Only a later, explicit human/institutional phase may consider changing `ACQUISITION_AUTHORIZED` or related gates. Filling this package never flips those gates.

### Roles (proposed; require human confirmation)

| Role | Responsibility |
|---|---|
| Record preparer / custodian | Assembles evidence references and completes the draft |
| Rights / license reviewer | Source terms, copyright, redistribution |
| Privacy / ethics reviewer | Consent, PII, institutional requirements |
| Ground-truth / adjudication lead | Independent verification procedure |
| Dataset integrity reviewer | Duplicates, leakage, split plan |
| Security / retention owner | Storage, access, deletion, incidents |
| Budget owner | Ceiling, funding source, stop conditions |
| Authorizing decision-maker (future phase) | Alone may decide whether a separate authorization phase begins |

Until named individuals are assigned in a completed record, roles remain vacant and the outcome stays `DRAFT` or `NO_GO`.

## Evidence handling

- Prefer durable references: document title, URI or repository path, retrieval date, and SHA-256 of retained copies where retained.
- Do not commit watch images, certificates, raw serials, customer PII, credentials, or private correspondence to Git (aligned with Phase 55 / 61).
- Incomplete, contradictory, expired, or withdrawn approvals → fail closed (`NO_GO` or `CONDITIONAL_NO_GO`).
- Schema validity proves structure only—not legal rights, ethical clearance, or factual truth.

## Why completion does not authorize

A completed checklist, schema-valid record, green CI run, Phase 65 readiness report, or `GO_RECOMMENDED_SEPARATE_AUTHORIZATION_REQUIRED` outcome:

- does **not** set `ACQUISITION_AUTHORIZED=true`;
- does **not** enable scraper network collection for authenticity GT;
- does **not** change `DATA_READINESS`, inference, publication, or promotion gates;
- does **not** replace website terms, licenses, consent, or faculty approval.

**Readiness to consider acquisition ≠ authorization to acquire.**

## Related prior phases

- Phase 55 — evidence-ledger schema and eligibility (no acquisition).
- Phase 60 — public source-route desk review (no contact).
- Phase 61 — ground-truth SOP package (blank procedures).
- Phase 62 — freeze/audit protocol (no freeze authorized).
- Phase 65 — offline intake / review ledger (metadata only).
- Phase 67 — gate naming crosswalk and ledger append hardening.
