# Phase 37 acquisition report

Phase 37 collected and audited sources. It did not score the frozen authenticity model and it did not change the Phase 36 protocol.

Protocol hash: `babaaf2b5c30d252c39765886ee5db032c09c4e3f25cf0e9eacc63f17128dde9`

## Source table

| Source | Category coverage | Brands | Candidate | Admitted | Rejected | Rejection reason | Provenance |
|---|---|---|---:|---:|---:|---|---|
| Pixnio | none of the four protocol groups | UNKNOWN | 1 | 0 | 1 | ambiguous_category | CC0 page, no source-declared brand |
| Openverse | not collected | not collected | 0 | 0 | 0 | resized Flickr rendition | aggregator URL is not the original |
| Flickr | not collected | uploader title only | 0 | 0 | 0 | original above the 1.5 MB cap | one CC BY-SA original was larger than the frozen cap |
| Wikimedia Commons | ceiling only, not collected | category names | 0 | 0 | 0 | bulk originals are not dependable and the ceiling misses the targets | Phase 36 HTTP 429 record |
| Internet Archive | not a watch-photo corpus | UNKNOWN | 0 | 0 | 0 | search hits are not brand-verified photographs | license not established for a watch corpus |
| Met, Smithsonian, Europeana, Hippopx | unavailable | UNKNOWN | 0 | 0 | 0 | endpoint denied or gone | bytes and license were not established |
| WatchBase CDN | not collected | not used | 0 | 0 | 0 | photograph license not established | CC0 claim covers the dataset text, not the shown CDN rendition |

## Independence table

| Check | Count |
|---|---:|
| Exact duplicates rejected | 0 |
| Near duplicates rejected | 0 |
| Derivative files rejected | 0 |
| Overlapping source IDs or URLs rejected | 0 |
| HypeVault catalog overlap in the admitted set | 0 |

The one stored pilot did not match a Phase 16, Phase 35, Phase 36, or Phase 25 SHA-256, source URL, filename, or dHash within Hamming 4. It was still rejected because the source does not verify a brand or a protocol category.

Openverse and the WatchBase `/watch/lg/` example were refused before download because they are resized or license-unproven derivatives. Those refusals are source-level. They are not counted as downloaded derivative files.

## Coverage table

| Group | Target | Candidate | Admitted | Holdout target | Ready |
|---|---:|---:|---:|---:|---|
| Unseen luxury | 300 | 0 | 0 | 60 | no |
| Ordinary | 200 | 0 | 0 | 40 | no |
| Smart/digital | 100 | 0 | 0 | 25 | no |
| Wrist/store | 50 | 0 | 0 | 25 | no |

Unseen luxury brands with at least 8 holdout images: 0.

Images per brand: none admitted.

Independent source groups per brand: none admitted.

## Commons ceiling that was not collected

Phase 36 recorded 531 original downloads that failed with HTTP 429 after the size and title filters. Those files were not added to Phase 37. If every one of them had later passed license and content checks, the new-file ceiling would still be:

- unseen luxury: 258 (target 300)
- ordinary watches: 155 (target 200)
- smart/digital: 104 (target 100)
- wrist/store: 14 (target 50, useful holdout 25)

The Phase 36 admitted files stay in the Phase 36 holdout and are not members of this corpus.

## Stop

Acquisition stopped because no qualified source can supply original, provenance-backed photographs at the registered targets. Wikimedia original delivery is rate-limited, and its recorded ceiling misses the unseen, ordinary, and wrist targets. The other pilots failed original-byte, license, or source-verified brand checks. The protocol was not relaxed.

`WATCH_OOD_V2_CORPUS_READY = false`
