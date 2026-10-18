"""Balance unverified web watch coverage via DuckDuckGo image search.

Writes only under local_data/scraper_artifacts/unverified_web_images/watches/.
Never maps into Label_0_Watches. Never assigns verified authenticity labels.
Importing this module does not start downloads.
"""

from __future__ import annotations

from pathlib import Path
import sys


def _ensure_repo_import_path() -> None:
    """Prefer the repo package path over the script directory shadowing the package name."""
    repo = Path(__file__).resolve().parents[1]
    script_dir = Path(__file__).resolve().parent
    sys.path[:] = [p for p in sys.path if Path(p).resolve() != script_dir]
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))



_ensure_repo_import_path()

import argparse
import logging
import random
import time

from hypevault_scraper.common import (
    ARTIFACT_ROOT_REL,
    DEFAULT_MAX_BYTES,
    DEFAULT_TIMEOUT_SECONDS,
    STATUS_UNVERIFIED_WEB,
    STATE_REL,
    load_corpus_index,
    StoreStats,
    assert_not_legacy_training_dir,
    count_valid_images,
    download_bounded,
    load_resume_state,
    mark_query_completed,
    require_network_opt_in,
    resolve_under_repo,
    save_resume_state,
    store_validated_image,
    safe_brand_name,
)

LOGGER = logging.getLogger("hypevault.scraper.balance")

TARGET_PER_CLASS = 1500
BASE_DELAY = 3.5
MAX_DELAY = 180
BATCH_SIZE = 15
MAX_RETRIES = 5
IMAGES_PER_QUERY = 15

BRANDS_TO_BALANCE = [
    "Richard Mille",
    "A. Lange & Söhne",
    "Vacheron Constantin",
]


def build_search_queries(brand: str, models: list[str]) -> list[str]:
    queries: list[str] = []
    for model in models:
        queries.append(f"{brand} {model} watch")
        queries.append(f"{brand} {model} wristwatch official")
        queries.append(f"{brand} {model} watch dial closeup")
    queries.extend(
        [
            f"{brand} watch collection",
            f"{brand} watches official",
            f"{brand} timepiece",
        ]
    )
    return queries


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download unverified DuckDuckGo image results to fill shortfall brands. "
            "Outputs are research-only and unlabeled for authenticity."
        )
    )
    parser.add_argument("--csv", default="Watches.csv")
    parser.add_argument("--artifact-root", default=str(ARTIFACT_ROOT_REL))
    parser.add_argument("--target-per-class", type=int, default=TARGET_PER_CLASS)
    parser.add_argument("--images-per-query", type=int, default=IMAGES_PER_QUERY)
    parser.add_argument("--max-retries", type=int, default=MAX_RETRIES)
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_SECONDS)
    parser.add_argument("--max-bytes", type=int, default=DEFAULT_MAX_BYTES)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--allow-network", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    artifact_root = resolve_under_repo(args.artifact_root, ARTIFACT_ROOT_REL)
    csv_path = resolve_under_repo(args.csv, "Watches.csv")
    state_path = resolve_under_repo(
        STATE_REL / "balance_scraper_resume.json",
        STATE_REL / "balance_scraper_resume.json",
    )

    if args.dry_run:
        print("DRY-RUN: no network and no image writes.")
        print(f"CSV: {csv_path}")
        print(f"Artifact root: {artifact_root}")
        print(f"Brands: {', '.join(BRANDS_TO_BALANCE)}")
        return 0

    require_network_opt_in(args.allow_network)

    import pandas as pd
    import requests
    from duckduckgo_search import DDGS

    if not csv_path.exists():
        LOGGER.error("CSV not found: %s", csv_path)
        return 2

    df = pd.read_csv(csv_path, low_memory=False).dropna(subset=["brand", "ref"])
    state = load_resume_state(state_path)
    completed = set(state.get("completed_queries") or [])
    stats = StoreStats(**(state.get("stats") or {})) if isinstance(state.get("stats"), dict) else StoreStats()
    session = requests.Session()
    session.headers.update({"User-Agent": "HypeVaultResearchCollector/1.0"})
    ddgs = DDGS()
    consecutive_ratelimits = 0
    query_count = 0

    for brand in BRANDS_TO_BALANCE:
        dest_dir = artifact_root / "unverified_web_images" / "watches" / safe_brand_name(brand)
        dest_dir.mkdir(parents=True, exist_ok=True)
        assert_not_legacy_training_dir(dest_dir)
        index = load_corpus_index(artifact_root)
        current = count_valid_images(dest_dir)
        if current >= args.target_per_class:
            print(f"[{brand}] Already at {current}/{args.target_per_class}. Skipping.")
            continue

        models = df[df["brand"] == brand]["model"].dropna().unique().tolist()
        queries = build_search_queries(brand, models)
        print(f"[{brand}] current={current} need={args.target_per_class - current} queries={len(queries)}")

        for query in queries:
            if query in completed:
                continue
            current = count_valid_images(dest_dir)
            if current >= args.target_per_class:
                break

            success = False
            for attempt in range(args.max_retries):
                try:
                    results = list(
                        ddgs.images(
                            query,
                            safesearch="off",
                            size="Large",
                            max_results=args.images_per_query,
                        )
                    )
                    added = 0
                    for result in results:
                        image_url = result.get("image") or result.get("url")
                        page_url = result.get("url") or result.get("source")
                        if not image_url:
                            stats.invalid += 1
                            continue
                        try:
                            data, content_type = download_bounded(
                                session,
                                image_url,
                                timeout=args.timeout,
                                max_bytes=args.max_bytes,
                            )
                        except Exception as exc:  # noqa: BLE001
                            LOGGER.warning("download failed url=%s err=%s", image_url, exc)
                            stats.errors += 1
                            continue
                        before = stats.stored
                        store_validated_image(
                            data,
                            dest_dir=dest_dir,
                            artifact_root=artifact_root,
                            family="watches",
                            claimed_brand=brand,
                            claimed_model=None,
                            label_status=STATUS_UNVERIFIED_WEB,
                            search_query=query,
                            search_provider="duckduckgo",
                            source_image_url=image_url,
                            source_page_url=page_url if isinstance(page_url, str) else "UNKNOWN",
                            content_type=content_type,
                            max_bytes=args.max_bytes,
                            index=index,
                            stats=stats,
                        )
                        if stats.stored > before:
                            added += 1
                    consecutive_ratelimits = 0
                    print(
                        f"  [{query_count + 1}] {query!r} stored=+{added} "
                        f"total={count_valid_images(dest_dir)} stats={stats.as_dict()}"
                    )
                    success = True
                    break
                except Exception as exc:  # noqa: BLE001
                    err = str(exc)
                    if "Ratelimit" in err or "403" in err or "429" in err:
                        consecutive_ratelimits += 1
                        backoff = min(MAX_DELAY, BASE_DELAY * (2 ** consecutive_ratelimits)) + random.uniform(2, 8)
                        LOGGER.warning("rate limited attempt=%s backoff=%.0fs", attempt + 1, backoff)
                        time.sleep(backoff)
                        ddgs = DDGS()
                        continue
                    LOGGER.error("query error %r: %s", query, exc)
                    break

            mark_query_completed(state, query, success=success)
            query_count += 1
            state["stats"] = stats.as_dict()
            save_resume_state(state_path, state)

            delay = BASE_DELAY + random.uniform(1.0, 3.0)
            if consecutive_ratelimits > 0:
                delay = min(MAX_DELAY, delay * (2 ** min(consecutive_ratelimits, 5)))
            time.sleep(delay)
            if query_count % BATCH_SIZE == 0:
                pause = random.uniform(15, 30)
                print(f"  --- batch pause {pause:.0f}s ---")
                time.sleep(pause)
                ddgs = DDGS()
                consecutive_ratelimits = 0

        final = count_valid_images(dest_dir)
        print(f"[{brand}] final={final}/{args.target_per_class} shortfall={max(0, args.target_per_class - final)}")

    print(f"BALANCE COMPLETE stats={stats.as_dict()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
