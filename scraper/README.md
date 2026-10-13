# HypeVault scraper tools (research-only)

**Warning:** Scraped Bing/DuckDuckGo images are **not** independently verified authentic products. SDXL-generated branded imagery is **not** verified counterfeit ground truth. Public visibility of an image does **not** establish permission for arbitrary reuse. Assess access and rights for each source before any network collection.

These tools never write into the legacy training directories `Label_0_*` / `Label_1_*`, and they never assign `VERIFIED_AUTHENTIC` or `VERIFIED_COUNTERFEIT`.

## Label semantics

| Status | Assigned by these tools? | Meaning |
|---|---|---|
| `UNVERIFIED_WEB_IMAGE` | Yes (scrapers) | Downloaded from the web; authenticity unknown |
| `SYNTHETIC_AI_GENERATED` | Yes (generators) | Created by a local generative model |
| `VERIFIED_AUTHENTIC` | No | Requires separate evidence-backed human review |
| `VERIFIED_COUNTERFEIT` | No | Requires separate evidence-backed human review |

Unverified and synthetic records always use `target_label: null` and `intended_use_eligibility.authenticity_benchmark: false`.

## Scripts

| Script | Purpose | Default output |
|---|---|---|
| `scraper.py` | Bing watch image search via icrawler | `local_data/scraper_artifacts/unverified_web_images/watches/<brand>/` |
| `balance_scraper.py` | DuckDuckGo watch top-up downloads | same watches tree |
| `sneaker_scraper.py` | Bing sneaker image search | `.../unverified_web_images/sneakers/<brand>/` |
| `generate_label1.py` | SDXL synthetic watches | `.../synthetic_research_only/watches/<brand>/` |
| `generate_label1_sneakers.py` | SDXL synthetic sneakers | `.../synthetic_research_only/sneakers/<brand>/` |

Shared helpers live in `common.py` (validation, SHA-256, manifests, resume, atomic writes).

## Installation

Lightweight collection and validation (CPU):

```bash
python -m pip install -r scraper/requirements.txt
```

Optional GPU generation (install a CUDA-capable PyTorch build for your machine first; do not assume the CI CPU wheel):

```bash
python -m pip install -r scraper/requirements-generation.txt
```

Pinned application Torch remains defined by the root `requirements-ci.txt` / training requirements. Generation extras are optional and local.

## Dry-run and network / generation opt-in

Importing a module never starts collection or model load. Actual work requires an explicit flag.

```bash
# Plan only — no network, no writes
python scraper/scraper.py --dry-run
python scraper/balance_scraper.py --dry-run
python scraper/sneaker_scraper.py --dry-run
python scraper/generate_label1.py --dry-run
python scraper/generate_label1_sneakers.py --dry-run

# Network collection (only after rights review)
python scraper/scraper.py --allow-network --csv /path/to/Watches.csv
python scraper/balance_scraper.py --allow-network
python scraper/sneaker_scraper.py --allow-network

# Synthetic generation (GPU; research-only directory)
python scraper/generate_label1.py --allow-generation
python scraper/generate_label1_sneakers.py --allow-generation
```

Use `--help` on each script for path overrides (`--artifact-root`, `--csv`, targets, timeouts).

## Paths and manifests

Artifact root (ignored by Git): `local_data/scraper_artifacts/`

Each brand folder stores:

- Normalized JPEG files named by content SHA-256 prefix
- `manifest.jsonl` — one provenance record per stored image
- `content_index.json` — exact SHA map and perceptual-hash review candidates

Resume / failure state lives under `local_data/scraper_artifacts/state/` (also ignored).

### Manifest fields (schema `scraper_manifest_v1`)

Includes record id, relative path, stored and original SHA-256, dimensions, format, byte size, family, claimed brand/model, search query, UTC timestamps, provider, source URLs (or `UNKNOWN`/`NOT_RECORDED`), provenance completeness, `label_status`, null `target_label`, evidence/rights review statuses, eligibility flags, duplicate / near-duplicate review status, and generation metadata for synthetic rows.

Missing source metadata stays explicit (`UNKNOWN`, `NOT_RECORDED`, `INCOMPLETE`). Nothing invents URLs.

### icrawler URL limitation

`icrawler` download tasks include `file_url`, but the Bing wrappers validate files after the crawl from a temp directory whose numbered filenames do not preserve that URL. When the URL cannot be recovered, manifests record `source_image_url=UNKNOWN` and document the limitation.

## Validation, duplicates, resume

- Responses are size-bounded, content-type checked, and decoded with Pillow. HTML error pages, truncated files, and unsupported formats are rejected.
- Exact duplicates (content SHA-256) do not increment the unique-image count and do not overwrite existing files.
- Near-duplicates (average-hash Hamming distance ≤ 6) are stored with `duplicate_status=NEAR_DUPLICATE_CANDIDATE` for human review; they are not auto-merged.
- Counts include only decodable images in the intended output directory.
- Failed queries are tracked separately and are not permanently marked completed, so resume can retry them.
- Targets are aspirational: if a provider returns too few unique valid images, the run reports the shortfall instead of inventing success.
- Transient rate limits use bounded exponential backoff. Access refusals are not circumvented.

## Reviewing near-duplicate candidates

Inspect `manifest.jsonl` rows with `duplicate_status=NEAR_DUPLICATE_CANDIDATE` and the `near_duplicate_of` path. Decide keep/drop manually; the tooling does not auto-delete candidates.

## Rights reminder

Before `--allow-network`, confirm that collecting and storing images from the chosen provider is permitted for your intended research use. Catalog CSVs and downloaded bytes remain local and must stay out of Git.
