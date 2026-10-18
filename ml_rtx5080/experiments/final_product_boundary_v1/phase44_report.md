# Phase 44

Customer labels are computed when a listing is read. The 19 image-backed live rows read as `Legacy screening — not verified`. The 20 seed rows read as `Demo listing — not verified`. Their stored status, verdict, and confidence were not rewritten.

A seed row still contains the historical constant `AUTHENTIC` at confidence `0.965`. The API marks that value `SEED_CONSTANT`, so it is not treated as a model prediction. Image-backed verdicts stay `HISTORICAL_MODEL_EVIDENCE`.

`status = live` remains publication state. It does not mean the listing is authenticity-verified. New model results still cannot publish a listing. `AUTHENTICITY_MODEL_PRODUCTION_APPROVED` remains false.

No manual authorization workflow was added. A future listing can move from draft to pending, then through a separate authorization step, then to live. That step is not implemented.
