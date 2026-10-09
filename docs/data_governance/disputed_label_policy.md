# Disputed-label policy

A label is disputed when any of these happens:
- Two qualified examiners reach different conclusions.
- An examiner records `probable` rather than `certain`.
- New evidence contradicts an admitted label.
- The image may show a different watch from the one examined.

## Handling

1. Set the record's `disposition` to `HOLD_DISPUTED` and keep `label` unchanged. A disputed item is excluded from every split, including evaluation.
2. Ask a third qualified examiner, independent of the first two, to examine the physical item. Do not use photos alone.
3. Admit the item only if two of three independent examiners agree with `certain` confidence. Otherwise set `disposition` to `EXCLUDE` with reason `UNRESOLVED_DISPUTE`.
4. Keep the full history: every conclusion, examiner reference, date, and evidence ID. Never overwrite an earlier conclusion.
5. If an admitted label is later overturned, exclude the item. Then list every protocol, split, and model that used it, and re-evaluate those results.

## What never resolves a dispute

Model predictions, majority votes of non-examiners, directory names, listing text, price, file geometry, or JPEG settings.
