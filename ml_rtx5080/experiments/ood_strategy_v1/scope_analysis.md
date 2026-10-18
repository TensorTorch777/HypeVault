# Scope analysis

The frozen checkpoint was trained on five brands: A. Lange & Söhne, Audemars Piguet, Patek Philippe, Richard Mille, and Vacheron Constantin. The catalog has no product identifier and no marketplace field. Evaluation inside those brands is a closed-set authenticity task. Behavior on other watches is an open-set task.

Phase 35 measured that open-set task. On 194 unseen-luxury images the authentic-escape rate was 0.8814, with a Wilson interval from 0.8284 to 0.9197. Rolex was 159/180 AUTHENTIC, Omega 10/12, and Cartier 2/2. Watch-like groups C, D, and E together escaped at 0.8405 on 257 images. No research OOD gate brought the unsafe groups to a 0.05 escape rate without raising calibration review by more than 0.10. `production_ood_threshold` stayed null.

Phase 36 did not confirm or overturn that result. Its holdout was 14 unseen-luxury images. The Wilson interval for 11/14 runs from 0.5241 to 0.9243.

Phase 37 admitted no new images. Route A did not reach the registered floors. Route B was not collected and is not evidenced at that scale. Route C has `LEGAL_BASIS = UNKNOWN`. Route D datasets are derivatives, unlabeled, or metadata without an image license.

A general luxury-watch authenticity system would need an open-set rejection claim. The evidence does not support that claim, and an acceptable confirmatory corpus is not available. The accurate description is a five-brand authenticity research prototype. That description limits the supported domain to the five training brands. It does not say the prototype is safe inside those brands beyond the existing closed-set results, and it does not say Phase 35 was invalid.

No weight, threshold, calibration constant, or serving path was changed. Promotion remains disallowed.

## Minimum evidence before any later promotion review

Promotion is not under review. If a later review is opened, the package that would have to exist first is:

1. The supported domain is written down, either the five training brands or a wider domain with its own corpus.
2. For any open-set claim, a versioned protocol and a corpus that meets the existing holdout floors, with image licenses that are explicit or institutionally approved. `unknown` is not approval.
3. Phase 35 remains in the record as warning evidence, with its source limits stated.
4. Phase 36 remains an underpowered observation and is not a tuning set.
5. `production_ood_threshold` is still null unless a separate, pre-registered decision sets it. This review does not set it.

That package does not exist. `PROMOTION_ALLOWED = false`.
