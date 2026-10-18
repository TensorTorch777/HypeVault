# Phase 46 report

The frozen checkpoint hash matches `5a38c93fd442b03653c65d2a5ecc9c2687ef152f7c5c020763e4ce1fd9c7d28f`. It was not retrained, recalibrated, or replaced. The locked final-test images were not opened.

## Sixth brand

`EXTERNAL_CLASSIFICATION_VALIDATION = BLOCKED`.

No brand outside the five-brand catalog and the Phase 35 and Phase 36 lists had a corpus that also had traceable provenance, an explicit research license, original photographs, independent source groups, and authentic-versus-counterfeit labels established without this model. Commons categories, Roboflow derivatives, Kaggle brand folders, Chanel fashion scrapes, and synthetic text sets were rejected in `data_audit.json`. No external image was downloaded or scored, so PR-AUC and the false-authentic rate are not reported.

## Richard Mille errors

The fold 4 model counted 20 counterfeit images as authentic, a rate of 0.007407 on 2700 counterfeits. The saved prediction file identifies the 15 lowest P(fake) cases. The other five identities were not stored, and the fold weights are gone, so they were not reconstructed with `epoch_018.pt`.

All 15 identified files were inspected. Nine have P(fake) below 0.01. Each file is its own duplicate group. The nearest dHash neighbor is Hamming 8 to 19, so the locked Hamming-0 rule does not show leakage. Every file is a 512×512 catalog image, and the catalog source field is UNKNOWN.

The lowest score, 0.000274, is a multi-panel advertisement sheet. Other low scores include studio photographs of tonneau watches, a side view of a movement, dial crops with broken wordmarks, and at least three watches whose case shape is not a Richard Mille tonneau. The typical Richard Mille counterfeit in this fold has median P(fake) 0.999839. Visible text errors did not by themselves produce a high P(fake). That last sentence is an observation about these images. A causal claim about what the head uses is a hypothesis, recorded in the analysis file.

Per-image Mahalanobis and cosine distances for these 15 files were not saved. The fold-level feature summary still shows Richard Mille shifted relative to the training brands: 29.3 percent of held-out images sit above the training Mahalanobis p95, and the mean P(fake) on that distant slice is 0.000177.

## Separation of claims

Phase 45 STRONG applies only inside the original five-brand catalog. It does not answer the sixth-brand question, and it does not overturn the Phase 35 open-set failure.

## Phase 47

Do not promote the model. Obtain an independently labeled, licensed, original-photograph corpus for one brand that is outside both the five-brand catalog and the Phase 35 and Phase 36 brand lists, with at least 100 authentic and 100 counterfeit images, and lock that manifest before any inference. Until that corpus exists, the sixth-brand classification claim stays blocked.
