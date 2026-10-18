# Gap-closure plan

**Status:** `PLAN_ONLY — NO DATA ACQUISITION OR INFERENCE AUTHORIZED`

Actions are ordered. A later action does not start because an earlier document exists. Independently verified labels remain zero. No lead in `source_tracker.csv` is qualified.

Roles below are functions, not named people. "Project owner" approves spend and claims. "Counsel" reviews terms. "Evidence custodian" holds records outside Git. "Protocol owner" keeps the written rules. None of these roles is authorized by this file to contact a provider.

## P0 — Decisions that must be written before any collection

### P0-1 Record the acceptance decisions that Phase 56 left open

- Evidence required: a signed approval naming deployment use, maximum acceptable false-accept risk, confidence level, minimum products per brand and per class, abstention policy, and what would support or reject a claim. The 300-product illustration in `metrics_and_acceptance.md` is not that approval.
- Depends on: nothing empirical. It is a decision.
- Accountable role: project owner.
- Done when: the approval is a versioned document with a checksum, and it does not change `DATA_READINESS` by itself.
- Human approval: required. This plan does not grant it.

### P0-2 Freeze the product-level aggregation rule for a future study

- Evidence required: the default in `metrics_and_acceptance.md` (any review/abstain makes the product review; otherwise any counterfeit image makes the product counterfeit; else authentic), or a written replacement made before evaluation labels exist.
- Depends on: P0-1 only if the replacement changes how abstentions affect a claim.
- Accountable role: protocol owner.
- Done when: the rule and protocol version are named in the future manifest header. Review/abstain is not an authentic success.
- Human approval: required before any replacement of the default. Not required to keep the written default as the candidate rule.

## P1 — Source qualification for both outcome classes

These actions describe diligence. They do not authorize sending the outreach draft.

### P1-1 Qualify a route that can supply both authentic and counterfeit physical watches

- Evidence required: written answers on examiner qualifications, both outcome classes plus indeterminate, brand coverage, redacted sample report, conflict controls, and a quote. Public web pages are self-published claims. Hourmark remains `LEAD_QUALIFICATION_REQUIRED`. Rolex Certified Pre-Owned remains positive-only and cannot supply counterfeit negatives. Phillips catalogue text remains not primary evidence. WatchCSA remains `HOLD_DO_NOT_UPLOAD` until counsel-reviewed terms replace the published Member Content clause.
- Depends on: P0-1 for any spend. No contact until a separate authorization says outreach may be sent.
- Accountable role: protocol owner prepares the question list. Project owner authorizes contact. Counsel reviews any terms.
- Done when: a due-diligence record says qualified or rejected, with the written reply stored outside Git. Until then the lead stays unqualified.
- Human approval: required before any message, account, or payment.

### P1-2 Secure image-use rights, consent or legal basis, retention, and deletion

- Evidence required: `license_status` approved for `ML_EVALUATION`, consent `APPROVED` or documented `NOT_REQUIRED`, `retention_until` a real date, `deletion_status` exactly `RETENTION_ACTIVE`. These match the Phase 55A schema. A schema-shaped blank record is not this evidence.
- Depends on: P1-1 identifying a rights-holder. WatchCSA's published terms are a reason to stop, not a license.
- Accountable role: counsel and project owner.
- Done when: the rights instrument covers storage, duration, evaluation use, and deletion, and no customer identity or raw serial is required in Git.
- Human approval: required.

## P2 — Review design

### P2-1 Name reviewer qualifications and blinding without naming people in Git

- Evidence required: a role description of qualifications and conflict controls, plus the rule that reviewers do not see model scores or each other's decisions. Adjudication matches the schema: opposite outcomes, and the adjudication outcome equals `verified_label`.
- Depends on: P1-1, because qualifications are about the actual examiners.
- Accountable role: protocol owner drafts the rule. Project owner accepts the qualification standard.
- Done when: the standard is written and the evidence store, outside Git, can hold role codes rather than names.
- Human approval: required before any watch is examined.

### P2-2 Tie every image set to one physical watch

- Evidence required: a pseudonymous `product_id` used by every view, crop, and duplicate of that watch. Hash groups in the Phase 54 report are not product ids.
- Depends on: P1-2, because the images are not yet authorized.
- Accountable role: evidence custodian.
- Done when: a non-committed manifest can show one partition for that product id. `UNKNOWN` product ids stay ineligible.
- Human approval: required to receive images. Not required to keep this rule.

## P3 — Leakage controls

### P3-1 Record source, and seller or marketplace when lawfully known

- Evidence required: pseudonymous `source_id` for every primary candidate. Seller and marketplace ids when known. Unknown is recorded and is not treated as separation.
- Depends on: P1-1 and P2-2.
- Accountable role: evidence custodian.
- Done when: overlap checks against task-specific training and calibration sources are possible. Unresolved overlap blocks primary eligibility.
- Human approval: required to collect the underlying identifiers. The recording rule itself is already in `split_manifest_contract.md`.

### P3-2 Produce model-specific disjointness without opening locked files

- Evidence required: `PASS` or `FAIL` for DINOv2 `fe1daa0…` and DINOv3 `5a38c93…` task-specific training and calibration memberships, at product level. Phase 57 records only a split-count summary: DINOv2 has no calibration split on the live checkpoint, and 2,682 DINOv3-validation images sit in DINOv2 training folders. That summary is not a product-level pass. The locked 3,006-row membership stays unread.
- Depends on: P2-2 and a future authorized comparison that uses checksums rather than printing locked paths.
- Accountable role: protocol owner.
- Done when: each primary product is `PASS` for both models, or it is excluded and counted. A single-model pass is not a paired comparison.
- Human approval: required before any membership file that contains locked-test paths is opened.

## P4 — What each validation type still needs

### P4-1 Within-brand

- Evidence required: a brand that was in task-specific training, plus evaluation products of that brand that were not, with both verified classes before a counterfeit-detection claim.
- Depends on: P1 through P3.
- Accountable role: protocol owner.
- Done when: per-brand product counts, source counts, and both class counts can be reported. One missing class is an incomplete claim, not a zero error rate.
- Human approval: required to acquire the products. The definition is already written.

### P4-2 Cross-brand

- Evidence required: every brand named in the claim has both verified classes, and brand is not a proxy for authenticity. The 315-image A. Lange & Söhne historical cohort does not meet this.
- Depends on: P4-1 for more than one brand, plus the paired-cohort rule.
- Accountable role: protocol owner.
- Done when: per-brand counterfeit results sit beside any aggregate. Unmatched DINOv2 and DINOv3 sets are labeled as separate cohorts.
- Human approval: required for the claim threshold in P0-1. Not satisfied by writing this plan.

### P4-3 Project-task-held-out brand

- Evidence required: one brand absent from task-specific training, model selection, threshold selection, and calibration, with both verified classes or an explicit single-class limit. Call it project-task-held-out. Do not call it globally unseen. Foundation-pretraining exposure is not known from the files on `main`.
- Depends on: P0-2 frozen before that brand's labels are used, and P3-2.
- Accountable role: protocol owner.
- Done when: the freeze checksum exists before scoring, and only tested brands are reported. The older Vacheron Constantin / Richard Mille folder split on checkpoint `19e832ce` is not this evidence for live DINOv2.
- Human approval: required before scoring. Pretraining membership remains unauditable until a cited record exists.

## P5 — Statistical and human-review policy

### P5-1 Choose interval method and abstention handling

- Evidence required: a written choice of product-level interval, how source clustering is handled, and how indeterminate labels and abstentions affect the claim. The approximate `3/n` bound is an illustration, not a guarantee and not a minimum.
- Depends on: P0-1.
- Accountable role: project owner, with the protocol owner writing the method.
- Done when: the method is frozen before evaluation labels are used. No pass/fail sentence is allowed before that.
- Human approval: required.

No action in this plan changes production promotion gates.
