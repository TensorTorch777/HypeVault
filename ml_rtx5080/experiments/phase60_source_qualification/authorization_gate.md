# Phase 60 — Authorization gate

**Current authorization:** none of the operations below is approved.

This file lists decisions a person must record before any external action. A merged pull request, a green CI run, or this desk review does not supply that record.

## Stop until a separate written approval names the operation

| Operation | Why it is stopped | What an approval must name |
|---|---|---|
| Send email, submit a form, or call Hourmark, Rolex, WatchCSA, Phillips, or any other provider | Outreach status remains `NOT_SENT` | Provider, purpose, sender, and the exact text or script |
| Create an account or accept terms, including WatchCSA | Published WatchCSA Member Content terms assign submitted images and watch information to WCSA | Counsel reviewer, the terms being accepted, and confirmation they were replaced or are acceptable |
| Request or accept a quote, or spend money | Published Hourmark prices are not a HypeVault commitment | Budget cap, payer, and what is being purchased |
| Receive, download, or store watch photos, certificates, or private reports | No image right or privacy basis is confirmed | Rights holder, permitted use `ML_EVALUATION`, storage location, retention date, and deletion rule |
| Handle or solicit suspected counterfeit goods | No lawful custody process is approved | Custodian, legal basis, and handling limits |
| Run inference or open locked final-test or excluded OOD data | Inference and protected-data access stay closed | Model, cohort, and prohibited operations |

## Decisions still required even after contact is someday approved

Contact approval would not authorize collection. Collection approval would not authorize ML evaluation. Label approval would not authorize inference.

1. Who examines the watch, with what qualifications, and how conflicts are handled.
2. Whether outcomes can be `AUTHENTIC`, `COUNTERFEIT`, and `INDETERMINATE`, with a redacted report tied to one physical watch.
3. Whether two reviewers can work blinded to each other and to model predictions.
4. Who owns the images and whether a written grant covers non-public ML evaluation, storage, access, and any aggregate publication.
5. The privacy basis, retention date, deletion status, and redisclosure limits.
6. Which brands and both classes can be supplied, as independent physical products.
7. A redacted sample report and written terms before any data exchange.

The existing file `ml_rtx5080/experiments/phase55_ground_truth_acquisition/outreach_draft.md` remains `DRAFT ONLY — NOT SENT`.
