# Phase 35 real OOD validation

OOD_FAIL

The HypeVault repository did not contain a real OOD corpus. Phase 25 recorded zero samples.
Images were then taken from CC BY 4.0 Rolex Roboflow exports already on disk, a bounded local sneaker sample with license UNKNOWN, and Wikimedia Commons category downloads.
No threshold is promoted. production_ood_threshold is null.

## Status reasons

- authentic escape stays at or above 0.15 on B_product_non_watch, C_watch_outside_scope, D_unseen_luxury_brand, and no research gate brings those groups to 0.05 without raising calibration review by more than 0.10

## Corpus counts

- A_unrelated: 68 scored; mean P(fake) 9.040898088712738e-07; fraction AUTHENTIC at 0.50 1.0; fraction FAKE at 0.50 0.0
- B_product_non_watch: 83 scored; mean P(fake) 3.7459395245850573e-08; fraction AUTHENTIC at 0.50 1.0; fraction FAKE at 0.50 0.0
- C_watch_outside_scope: 45 scored; mean P(fake) 2.1815035288862103e-07; fraction AUTHENTIC at 0.50 1.0; fraction FAKE at 0.50 0.0
- D_unseen_luxury_brand: 194 scored; mean P(fake) 4.3559058348691635e-09; fraction AUTHENTIC at 0.50 1.0; fraction FAKE at 0.50 0.0
- E_structurally_different: 18 scored; mean P(fake) 4.875608666664034e-07; fraction AUTHENTIC at 0.50 1.0; fraction FAKE at 0.50 0.0

## Threshold sweep

- none: calibration review increase 0.0; authentic escape {"A_unrelated": 0.75, "B_product_non_watch": 0.8433734939759037, "C_watch_outside_scope": 0.6888888888888889, "D_unseen_luxury_brand": 0.8814432989690721, "E_structurally_different": 0.7777777777777778}
- conservative: calibration review increase 0.010006671114076049; authentic escape {"A_unrelated": 0.0, "B_product_non_watch": 0.0, "C_watch_outside_scope": 0.3111111111111111, "D_unseen_luxury_brand": 0.8247422680412371, "E_structurally_different": 0.0}
- moderate: calibration review increase 0.038358905937291535; authentic escape {"A_unrelated": 0.0, "B_product_non_watch": 0.0, "C_watch_outside_scope": 0.15555555555555556, "D_unseen_luxury_brand": 0.6907216494845361, "E_structurally_different": 0.0}
- aggressive: calibration review increase 0.16110740493662443; authentic escape {"A_unrelated": 0.0, "B_product_non_watch": 0.0, "C_watch_outside_scope": 0.044444444444444446, "D_unseen_luxury_brand": 0.18556701030927836, "E_structurally_different": 0.0}

## Strongest authentic OOD predictions

- D_unseen_luxury_brand / unseen_luxury_brand / Rolex: P(fake) 3.4674459997824313e-09, Mahalanobis 34.830990487753354, flags []
- D_unseen_luxury_brand / unseen_luxury_brand / Rolex: P(fake) 3.490225597009913e-09, Mahalanobis 34.2336893699416, flags []
- D_unseen_luxury_brand / unseen_luxury_brand / Rolex: P(fake) 3.4917283058681634e-09, Mahalanobis 34.77624443096816, flags []
- C_watch_outside_scope / fashion_watches / Swatch: P(fake) 3.516417814739771e-09, Mahalanobis 35.31825147338134, flags []
- D_unseen_luxury_brand / unseen_luxury_brand / Rolex: P(fake) 3.528268001896423e-09, Mahalanobis 35.1029437754471, flags []
- D_unseen_luxury_brand / unseen_luxury_brand / Rolex: P(fake) 3.535554369824674e-09, Mahalanobis 36.27479845819408, flags []
- D_unseen_luxury_brand / unseen_luxury_brand / Rolex: P(fake) 3.5692093120225744e-09, Mahalanobis 34.521827955555345, flags []
- D_unseen_luxury_brand / unseen_luxury_brand / Rolex: P(fake) 3.599255629876323e-09, Mahalanobis 34.95584378450747, flags []
- D_unseen_luxury_brand / unseen_luxury_brand / Rolex: P(fake) 3.630282138089595e-09, Mahalanobis 34.604889030368746, flags []
- D_unseen_luxury_brand / unseen_luxury_brand / Rolex: P(fake) 3.639504243952381e-09, Mahalanobis 37.60961576017493, flags []
- E_structurally_different / wrist_shots / UNKNOWN: P(fake) 3.642624427990935e-09, Mahalanobis 38.753240055388105, flags []
- D_unseen_luxury_brand / unseen_luxury_brand / Rolex: P(fake) 3.6482721072495413e-09, Mahalanobis 36.25502765353255, flags []
- D_unseen_luxury_brand / unseen_luxury_brand / Rolex: P(fake) 3.6612710553215383e-09, Mahalanobis 34.08742869829489, flags []
- D_unseen_luxury_brand / unseen_luxury_brand / Rolex: P(fake) 3.66419183800438e-09, Mahalanobis 36.20583668427768, flags []
- D_unseen_luxury_brand / unseen_luxury_brand / Rolex: P(fake) 3.6669767411864608e-09, Mahalanobis 35.265837276495965, flags []

Rolex files are Roboflow derivatives. dHash does not undo flips, so the independent photograph count is unknown.
The frozen checkpoint, temperature, policy, and final test were not modified.
