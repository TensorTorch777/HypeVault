# DINOv3 cross-brand v2 — pre-registered research plan

Branch: `research/dinov3-cross-brand-v2` (from `main` at merge `f2acf4a`).

This is a research plan, not a training run and not a production-promotion proposal. Serving infrastructure results (Triton GPU parity, mixed concurrency, API latency) are not cross-brand authenticity evidence.

`DINOv3_PRODUCTION_ALLOWED = false`

`AUTHENTICITY_MODEL_PRODUCTION_APPROVED = false`

`PRODUCTION_PROMOTION_ALLOWED = false`

The frozen production candidate stays frozen: checkpoint `5a38c93fd442b03653c65d2a5ecc9c2687ef152f7c5c020763e4ce1fd9c7d28f`, temperature `0.24038200410185356`, threshold `0.50`, FP32. The original locked final test is not opened.

## Current data status

Phase 49 verified-data pilot: `NO_GO`.

The existing ~30,000-image catalog has no independent authenticity-label provenance. Folder names (`Label_0_Watches` / `Label_1_Watches`) are **not** verified authentic/counterfeit labels.

Phase 48 documented a severe class/geometry confound: class is entangled with crop, background, and listing template. Within-catalog metrics can look strong because of that shortcut.

Existing five-fold / cross-brand numbers on this catalog are **historical, within-catalog evidence with those label-quality limitations**. They must not be overwritten, silently upgraded, or cited as verified real-world authenticity performance.

## Scientific question

When independently verified authentic and counterfeit images are available, and evaluation is brand-disjoint and source-disjoint, does the frozen DINOv3 `cls_patch_attention` candidate change false-authentic rate, calibration, or shortcut reliance relative to the live DINOv2 ViT-B/14 listing-check model — without using the locked final test and without changing the frozen candidate?

Until that data exists, the question is not answerable.

## Hypotheses (pre-registered; not tested on the current catalog)

H1. On independently verified, brand-disjoint evaluation, DINOv3 does not automatically inherit the catalog five-fold ranking versus DINOv2.

H2. Geometry, template, and marketplace-source shortcuts remain the primary risk; a model that looks strong on folder labels can fail when those cues are removed.

H3. Cross-brand transfer from the five in-scope brands to a held-out brand is weaker than within-brand catalog scores, even with verified labels.

H4. If verified data cannot be obtained at a pre-registered minimum size and provenance standard, the comparison stops at `NO_GO` rather than training on defective labels.

## Workstreams (in order)

### 1. Exploratory analysis of existing model behaviour and shortcut risk

Allowed now, without training:

- Re-read Phase 48 geometry-shortcut and Phase 49 `NO_GO` artifacts as historical record.
- Score already-allowed non-test, non-final-test images with the **frozen** DINOv2 and DINOv3 serving paths for qualitative error review.
- Document likely shortcuts (tight product crop vs lifestyle, watermark, marketplace template, brand-name overlay, class-correlated aspect ratio).

Forbidden:

- Supervised training or fine-tuning on the current catalog.
- Treating folder labels as verified authenticity.
- Claiming cross-brand performance from Triton/API tests.
- Opening locked final-test images.
- Changing the frozen checkpoint, temperature, or threshold.
- Enabling production approval.

### 2. Independently verified authentic / counterfeit data

A legitimate comparison requires a new dataset with, for every image:

- Independent authenticity judgement (not inferred from a folder, a scraper tag, or a model score).
- Documented source, license/consent, and acquisition date.
- Brand identity from a declared, auditable source — not from the authenticity head.
- Separation of authentic vs counterfeit by human or institutional evidence, with an adjudication trail (`docs/data_governance/`).

Minimum bar before any labelled training experiment: written provenance for both classes, enough per-brand samples for a pre-registered split, and an explicit statement of remaining confounders. Until then, data status stays `NO_GO`.

### 3. Brand-disjoint and source-disjoint evaluation

When verified data exists, freeze the split **before** measuring:

- At least one in-scope brand held out of training entirely (brand-disjoint).
- Sources (marketplace, photographer, archive) do not cross the train / eval cut (source-disjoint).
- No image, near-duplicate, or listing template from eval in train.
- The locked original final test remains untouched and is not a substitute for this split.

### 4. Future controlled DINOv2-versus-DINOv3 comparison

Only after (2) and (3):

- Same verified eval set, same preprocessing contracts, same FP32 policy.
- Live DINOv2: `dinov2_vitb14_live` / `fe1daa0b…fa66`, 504, existing verdict + 0.88 floor.
- Frozen DINOv3 candidate: `dinov3_authenticity_candidate` / `5a38c93f…d28f`, 512, temperature `0.24038200410185356`, threshold `0.50`.
- Do not change those artifacts to win the comparison. A new candidate, if any, is a separately named experiment — never a silent replacement of the frozen production candidate.

## Metrics (when a verified split exists)

Primary:

- False-authentic rate (counterfeit predicted authentic), overall and per brand.
- Decision mismatches vs the verified label (`AUTHENTIC` / `FAKE` / `REVIEW` as defined by each model's frozen policy).

Secondary:

- Calibration (temperature remains frozen for the candidate; ECE is diagnostic only).
- Per-brand recall of authentic and of counterfeit.
- Shortcut audits: performance drop when geometry/template cues are ablated or stratified.

Tolerances for **serving parity** stay `1e-4` logit and `1e-5` probability. Those are infrastructure numbers, not authenticity proof.

## Shortcuts to control

From Phase 48 and related audits, at least:

- Class-correlated crop and padding
- Background / studio vs wrist photography
- Marketplace watermark and listing chrome
- Image size and compression
- Brand-name text in frame
- Near-duplicates across splits

A result that disappears after controlling these is not authenticity generalization.

## Stop conditions

Stop and do not train if any of the following hold:

1. Independently verified labels are still unavailable (`NO_GO` remains).
2. The only remaining labels are the current catalog folder names.
3. A proposed split is not brand-disjoint or not source-disjoint.
4. An experiment would require opening the locked final test.
5. An experiment would modify the frozen checkpoint, temperature, threshold, or production flags.
6. Results would be presented as verified real-world authenticity without meeting (2)–(3).

## Exploratory exception (not default)

Training on the known-defective catalog labels is allowed only if the user **explicitly** approves a clearly labelled exploratory run. Even then:

- The run title must include `EXPLORATORY_DEFECTIVE_LABELS`.
- Folder labels stay described as unverified.
- Outputs must not be called verified authenticity performance, a production candidate, or a reason to set `DINOv3_PRODUCTION_ALLOWED`.
- The frozen candidate is not overwritten.

No such approval exists at the time this plan is written. Default action: **do not train**.

## Historical five-fold results

Catalog five-fold / `v3_cross_brand` figures may be cited only as:

> historical within-catalog scores on folder labels with a documented geometry confound; not independently verified authenticity and not brand-general performance.

Do not retcon them into the v2 verified-data study.

## Deliverables for this branch (plan stage)

- This file.
- No new training jobs.
- No production-flag changes.
- Optional: shortcut-stratified error notes on eligible non-test images via the frozen serving path.

Next action after this plan: acquire or refuse independently verified data. Not `train.py`.
