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

Shared helpers live in the **`hypevault_scraper`** package (`hypevault_scraper/common.py`).
That name is intentional: the backend already owns the Python package name `scraper`
(`backend/scraper/`). CLI scripts under `scraper/*.py` remain the entry points and import
`hypevault_scraper.common` so they never collide with backend imports during
`unittest discover`.

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

Artifact root also stores:

- `corpus_content_index.json` — corpus-wide exact SHA map and perceptual-hash review candidates (shared across brands)
- `state/exact_duplicate_events.jsonl` — traceable skips when identical bytes are offered under another brand folder

Exact duplicates are counted once corpus-wide. Near-duplicates are still stored with their own provenance and flagged for review (never auto-deleted).

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

## Dealer crawlers (The 1916 Company, Chronext, European Watch Co., Shopify dealers)

Our own crawlers, no scraping service. Code: `backend/market/dealers/` (`http.py` polite fetcher,
`extract.py` parsing, `sites.py` one class per HTML dealer, `shopify.py` a generic Shopify crawler,
`crawl.py` runner/export, `service.py` site search) and `scraper/dealer_scraper.py` (CLI).

Two kinds of source:

- **HTML dealers** (`sites.py`): The 1916 Company, Chronext, European Watch Co. Discovery uses the
  dealer's XML sitemaps (then category pages as fallback); product pages are read from schema.org
  JSON-LD, OpenGraph and labelled spec rows — no fragile CSS selectors.
- **Shopify dealers** (`shopify.py`): any Shopify storefront, read through its public `products.json`
  feed (clean structured data: vendor, price, tags, images). One class covers all of them — add a store
  by declaring a 3-line subclass in `SHOPIFY_SITES`. Watch Collectors UK is included.

Politeness (both kinds): robots.txt honoured (including Crawl-delay), one request per site every 3 s by
default (`--delay`, min 1 s), backoff on 429/5xx. A 403 or bot-challenge page **stops** that site; the
crawler never tries to get past it. It identifies itself as `HypeVaultResearchBot/1.0`.

Output: `local_data/dealers/<source>/listings.{csv,jsonl}`; with `--save-db`, rows for the five supported
brands go into `market_listings`. `--sites` defaults to every registered dealer.

```bash
python scraper/dealer_scraper.py --probe --allow-network                   # 1 item per site; shows parsed fields
python scraper/dealer_scraper.py --allow-network --per-brand 30 --save-db   # five brands, all dealers
python scraper/dealer_scraper.py --allow-network --sites europeanwatch,watchcollectors --per-brand 50
python scraper/dealer_scraper.py --allow-network --more-brands --download-images
```

Always run `--probe` first against the live sites: it confirms which dealers respond and which fields
parse, so a site that is bot-walled or has changed its markup is caught before a full crawl.

Dealers are part of the normal market search, the same pipeline as eBay: `search_market` returns
`sources.dealers` next to `sources.ebay` and `sources.chrono24_dataset`, for the same query and brand
(`GET /listings/{id}/comparison`, `GET /market/search?q=...`). The shared market panel shows them under
"From dealers", always after the eBay listings, on the listing page, the upload result and the market page.
They never change the eBay results, the eBay median or whether the shorter fallback query runs, and a failed
dealer lookup is reported as `status: "error"` without affecting eBay. Rows come from the last
`DEALER_MAX_AGE_DAYS` (default 30). Re-run the crawl (e.g. weekly) to keep it current.

Unlike eBay, dealers are not queried live on a page view: eBay has a search API that answers in one call,
while a dealer crawl walks sitemaps one request every few seconds. So the crawl runs on a schedule and the
site reads its snapshot, the way it reads the Chrono24 dataset.
Prices are in each dealer's currency.

Status from a live probe (Nov 2026):

- **The 1916 Company** — works. Full data from the embedded product JSON (price, reference, condition,
  box/papers, case size and material, dial). Prices USD. Production year is not always published.
- **Chronext** — works. Title, price, currency, reference, condition and product photos from JSON-LD.
  Year, box/papers and case specs are rendered by JavaScript and are not in the server HTML, so they
  stay blank (we do not run a headless browser to get them).
- **European Watch Co.** and **Shopify dealers (e.g. Watch Collectors UK)** — added as extra sources;
  confirm each with `--probe` on your network before a full crawl, since stock and markup vary.

**Watchfinder & Co. was evaluated and dropped.** Every page sits behind an AWS WAF JavaScript bot
challenge that a polite HTTP crawler cannot pass without a real browser, which would conflict with the
responsible-crawling stance here. The challenge is still detected by the fetcher (`looks_like_challenge`)
so any site that newly deploys one is reported as blocked rather than silently returning nothing.

### Terms-of-use review (do before relying on a dealer in production)

Following robots.txt and backing off on bot checks is polite, but it is not the same as permission: each
site's Terms of Use decide whether automated access and re-displaying their listing photos is allowed.
Chrono24 was removed from this project precisely because its terms forbid scraping. Before this feature is
used beyond a local research prototype, read each dealer's Terms of Use (search them for "automated
access", "scraping", "crawling", "reproduce content") and record the decision here. Showing only title,
price and a link — no photo — is the lowest-risk option if a site's terms are unclear on images.

| Dealer | Terms reviewed? | Automated access allowed? | Show photos? | Decision |
| --- | --- | --- | --- | --- |
| The 1916 Company | ☐ not yet | ? | ? | pending review |
| Chronext | ☐ not yet | ? | ? | pending review |
| European Watch Co. | ☐ not yet | ? | ? | pending review |
| Watch Collectors UK | ☐ not yet | ? | ? | pending review |

### Keeping the data fresh

Nothing runs the crawler automatically. The site hides rows older than `DEALER_MAX_AGE_DAYS` (30), so if
nobody reruns `dealer_scraper.py --save-db`, every dealer panel will say "No dealer listings matched"
about a month after the last run. Decide and record one of: a weekly manual run, a scheduled job, or
surfacing the data's age in the UI. The crawler identifies itself (in `http.py`) as a KJSSE final-year
project with a link to this repo, so each dealer can see who is crawling — keep that only if you're
comfortable with it.
