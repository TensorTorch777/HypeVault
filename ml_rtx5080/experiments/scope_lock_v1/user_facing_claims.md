# User-facing claims

## Approved

"Research prototype classification within five evaluated watch brands."

"The selected brand is user-declared and is not independently verified from the image."

The five brands are A. Lange & Söhne, Audemars Piguet, Patek Philippe, Richard Mille, and Vacheron Constantin.

An in-scope result may be reported as an authenticity classification for that declared brand. It is a research result.

## Not approved

- "The image was verified to be a Patek Philippe."
- "Brand verified."
- "AI verified."
- "Certified authentic."
- "AI-powered universal luxury-watch authentication."
- "Guaranteed authentic."
- "Works for all luxury watches."
- "OOD-safe."
- "Production validated."
- "100% accurate" outside the locked five-brand final test.
- Any statement that zero false-authentic errors extend past the evaluated five-brand test and the tested stress suite.

## Response words

`AUTHENTIC` and `FAKE` are allowed only after the request explicitly names a supported brand and the frozen decision path runs.

`UNSUPPORTED_SCOPE` means the declared brand is outside that list, missing, or ambiguous. It is not an authenticity verdict.

`REVIEW` and `POLICY_ERROR` belong to the shadow policy. They are not customer-promotion states.
