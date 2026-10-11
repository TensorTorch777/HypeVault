# Metrics and acceptance

**Status:** `PROTOCOL_ONLY — NO DATA ACQUISITION OR INFERENCE AUTHORIZED`

This file defines how a future study would report results and which decisions are still unmade. It does not set a passing score. `DATA_READINESS` remains `NO_GO_FOR_AUTHENTICITY_CLAIMS`. Independently verified labels remain zero. No model is scored in this phase.

The unit of analysis is the physical watch (`product_id`). Photographs are not independent observations. Source-level clustering is reported when more than one product shares a `source_id`, and interval methods that ignore that clustering are labeled as if-independent illustrations only.

## Product-level aggregation

Multiple images of one watch are reduced to one product-level outcome **before** evaluation labels are inspected. The rule is written into the frozen protocol and is not chosen by trying aggregations and keeping the best.

Until an approval names a different rule, the predeclared default is:

1. Each image receives the model's native decision: authentic, counterfeit, or review/abstain. DINOv3 review stays review. It is not relabeled.
2. The product is **review/abstain** if any image is review/abstain.
3. Otherwise the product is **counterfeit** if any decided image is counterfeit.
4. Otherwise the product is **authentic**.

This default is conservative about false accepts: one counterfeit image decision makes the product counterfeit, and any abstention prevents a silent authentic verdict. An approved study may replace this default only before evaluation, with the replacement written down. It may not be replaced after results are seen.

## Thresholds

Thresholds, temperature, and native policies of the frozen checkpoints stay as already frozen. Any additional operating threshold is selected only on approved `development` or `calibration` products. Evaluation labels, including locked-evaluation labels, do not influence that choice. The selected threshold is recorded with the calibration manifest checksum before evaluation starts.

## Metrics

All primary rates below use product-level outcomes on the eligible cohort. `INDETERMINATE` labels are excluded from the binary denominators and counted separately. Review/abstain is not a successful authentic verdict.

**False-accept rate (FAR).** Independently confirmed counterfeit physical watches classified authentic, divided by all eligible confirmed counterfeit physical watches. Watches that are review/abstain stay in the denominator and are not in the numerator. The count of abstentions is also reported.

**False-reject rate.** Independently confirmed authentic physical watches classified counterfeit, divided by all eligible confirmed authentic physical watches. Review/abstain stays in the denominator and is not counted as a rejection or as a success.

**Counterfeit recall / sensitivity.** Confirmed counterfeit watches classified counterfeit, divided by all eligible confirmed counterfeit watches. Abstentions are not counted as detections. A selective recall that drops abstentions is reported separately and labeled as selective.

**Authentic specificity.** Confirmed authentic watches classified authentic, divided by all eligible confirmed authentic watches. Abstentions are not counted as true authentic decisions.

**Balanced accuracy.** The mean of counterfeit recall and authentic specificity, using the denominators above. If either class is absent, balanced accuracy is undefined.

**Confusion.** For each brand, and for the aggregate: counts of authentic, counterfeit, and review/abstain decisions against authentic and counterfeit labels, plus indeterminate labels held out of the matrix.

**Counts beside every rate.** Per brand and overall: number of products, number of sources, number of authentic products, number of counterfeit products, number of images, and number of abstentions.

**Calibration.** On products with a resolved label, report a predeclared calibration summary of the native scores (for example reliability of the operational score). DINOv2's sigmoid and DINOv3's temperature-scaled sigmoid are not one calibrated scale and are not compared as if they were. Abstention rate is reported separately from calibration.

**Aggregate.** Overall rates are published in addition to every per-brand result. An aggregate without the per-brand counterfeit result is not a complete report.

## Uncertainty

Intervals are computed on products, not images. The method (for example a product-level bootstrap, or a source-clustered bootstrap when sources repeat) is named before evaluation.

Illustration only: if `n` independent counterfeit products have zero false accepts, a simple one-sided 95% binomial upper bound is approximately `3/n`. About 300 such products would put that illustration near 1%. This is not a guarantee, not a required sample size, and not valid when products share sources or are otherwise dependent. Source clustering widens uncertainty. The illustration is not permission to claim a 1% risk.

## Acceptance decisions that are not yet made

No universal false-accept rate, accuracy bar, or sample-size minimum is adopted here. Before any future study can call a result passing, a recorded approval must set:

- The intended deployment use.
- The maximum acceptable false-accept risk.
- The required confidence level.
- The minimum product counts per brand and per class.
- How indeterminate labels and abstentions affect the claim.
- The human-review policy for abstentions.
- What evidence would support an authenticity-related claim, and what evidence would reject it.

Until that approval exists, this protocol defines measurement only. It does not authorize a claim, a deployment, or a change to production promotion gates.

## Failure conditions for a reported claim

- Evaluation labels were used to choose the threshold, the aggregation rule, or the checkpoint.
- Review/abstain outcomes were counted as authentic successes.
- A per-brand counterfeit result was omitted for a brand named in the claim.
- Image-level rates were presented as if photographs were independent products.
- Cross-brand results on task-trained brands were described as unseen-brand generalization.
- A project-task-held-out brand was called globally unseen to the pretrained model without evidence of pretraining membership.
