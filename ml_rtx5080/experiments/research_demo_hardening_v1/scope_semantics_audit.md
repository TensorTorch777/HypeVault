# Scope semantics audit

The authenticity model scores a watch image only after the caller declares a brand. That declaration is a research scope filter. It does not establish that the image depicts the declared brand.

`POST /verify/authenticate` reads `brand` from the form, or from a stored listing when the form omits it. `evaluate_declared_brand` accepts an exact normalized match of the five supported brands. Anything else returns HTTP 422 `UNSUPPORTED_SCOPE` with `decision = null` before `Image.open` and before `classify_image`.

A supported response keeps the existing DINOv2 verdict and adds `declared_brand`, `brand_verification = NOT_PERFORMED`, `model_scope = FIVE_BRAND_RESEARCH_PROTOTYPE`, `research_only = true`, and `production_ready = false`. There is no `verified_brand` field.

No path assigns a brand from `P(fake)`. No CLIP, OCR, zero-shot, or extra image classifier was added. Brand verification is outside the current model scope.

The result UI says the brand is declared, that brand verification was not performed, and that the result is a five-brand research classification rather than a guarantee.
