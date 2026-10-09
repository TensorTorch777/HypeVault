# Scope lock report

`SYSTEM_SCOPE = FIVE_BRAND_AUTHENTICITY_RESEARCH_PROTOTYPE`

`GENERAL_LUXURY_WATCH_AUTHENTICITY = UNSUPPORTED`

`OPEN_SET_AUTHENTICITY = UNSUPPORTED`

`OOD_PROTECTION = UNVALIDATED`

`PRODUCTION_PROMOTION = FORBIDDEN`

## Currently supported

Five-brand authenticity research prototype for A. Lange & Söhne, Audemars Piguet, Patek Philippe, Richard Mille, and Vacheron Constantin, and only when that brand is explicitly declared.

## Currently not supported

General luxury-watch authenticity.

## Currently not validated

Open-set rejection.

## Currently not allowed

Customer-facing production promotion.

## Future expansion

A wider brand list would require a new versioned protocol and new independently governed evidence. That expansion was not started.

## Enforcement

`backend/inference/scope_gate.py` compares the declared brand string with the five names. It does not open an image and it does not read an authenticity score. `POST /verify/authenticate` runs that check before preprocessing and before `classify_image`. An unsupported declaration returns HTTP 422:

```text
status = UNSUPPORTED_SCOPE
decision = null
reason = "Brand is outside the validated five-brand research scope."
```

The live success body is still `AuthenticateResponse` with `AUTHENTIC` or `FAKE` only after the gate returns `SUPPORTED`. The DINOv2 files `verdict.py`, `dinov2_model.py`, `local_torch.py`, and `triton_client.py` were not edited.

The frozen checkpoint, temperature, threshold, and final test were not modified.
