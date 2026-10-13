"""Collect unverified web sneaker images into research-only artifact directories.

Outputs: local_data/scraper_artifacts/unverified_web_images/sneakers/<brand>/.
Never writes Label_0_Sneakers and never assigns verified authenticity labels.
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
import shutil
import time

from hypevault_scraper.common import (
    ARTIFACT_ROOT_REL,
    STATUS_UNVERIFIED_WEB,
    STATE_REL,
    ContentIndex,
    load_corpus_index,
    StoreStats,
    assert_not_legacy_training_dir,
    count_valid_images,
    load_resume_state,
    mark_query_completed,
    require_network_opt_in,
    resolve_under_repo,
    safe_brand_name,
    save_resume_state,
    store_validated_image,
)

LOGGER = logging.getLogger("hypevault.scraper.sneakers")

ULTRA_LUXURY_SNEAKERS = {
    "Dior": [
        "B27", "B22", "B30", "B23", "D-Connect", "Walk'n'Dior",
        "Dior Vibe", "CD Diamond",
    ],
    "Louis Vuitton": [
        "LV Trainer", "LV Skate", "Run Away", "Archlight", "Time Out",
        "Charlie", "Trainer Maxi",
    ],
    "Balenciaga": [
        "Triple S", "Track", "Speed", "Defender", "Runner", "Cargo",
        "3XL", "Tyrex",
    ],
    "Gucci": [
        "Ace", "Rhyton", "Screener", "Basket", "Run", "Tennis 1977",
    ],
    "Alexander McQueen": [
        "Oversized", "Sprint Runner", "Court", "Tread Slick",
        "Hybrid", "Deck Plimsoll",
    ],
}

TARGET_PER_CLASS = 3000
IMAGES_PER_QUERY = 100
MAX_ROUNDS_PER_BRAND = 4
EARLY_STOP_MIN_GAIN = 25
EARLY_STOP_ONLY_IF_STILL_NEED_GT = 80
TOPUP_MAX_QUERIES = 120

SEARCH_ALIASES = {
    "Louis Vuitton": ["Louis Vuitton", "LV"],
    "Alexander McQueen": ["Alexander McQueen", "McQueen"],
    "Dior": ["Dior", "Christian Dior"],
}

ICRAWLER_PROVENANCE_NOTE = (
    "icrawler BingImageCrawler exposes task['file_url'] while downloading, but "
    "post-crawl ingestion from the temp directory cannot reliably recover that URL "
    "from the numbered filename alone; source_image_url is UNKNOWN unless captured "
    "another way."
)


def build_queries(brand: str, models: list[str], round_idx: int) -> list[str]:
    brand_names = SEARCH_ALIASES.get(brand, [brand])
    scene_terms = [
        "on feet", "street style", "side view", "close up", "product photo",
        "editorial", "lookbook", "retail listing",
    ]
    shift = round_idx % len(scene_terms)
    scene_terms = scene_terms[shift:] + scene_terms[:shift]
    queries: list[str] = []
    for brand_name in brand_names:
        for model in models:
            queries.append(f"{brand_name} {model} sneakers")
            queries.append(f"{brand_name} {model} sneaker product photo black background")
            queries.append(f"{brand_name} {model} {scene_terms[0]}")
            queries.append(f"{brand_name} {model} {scene_terms[1]}")
            queries.append(f"{brand_name} {model} {scene_terms[2]}")
            queries.append(f"{brand_name} {model} mens")
            queries.append(f"{brand_name} {model} womens")
            queries.append(f"{brand_name} {model} low top")
            queries.append(f"{brand_name} {model} high top")
        queries.extend(
            [
                f"{brand_name} luxury sneakers",
                f"{brand_name} designer sneakers",
                f"{brand_name} premium sneakers collection",
                f"{brand_name} sneaker catalog",
                f"{brand_name} sneakers official website",
                f"{brand_name} sneakers high resolution black background",
                f"{brand_name} sneakers close up",
                f"{brand_name} sneakers review",
                f"{brand_name} sneakers street style",
                f"{brand_name} sneakers runway",
                f"{brand_name} sneakers campaign",
                f"{brand_name} sneakers studio black background",
                f"{brand_name} sneakers dark background",
            ]
        )
    random.shuffle(queries)
    return queries


def build_topup_queries(brand: str, models: list[str]) -> list[str]:
    brand_names = SEARCH_ALIASES.get(brand, [brand])
    queries: list[str] = []
    extras = ["white", "black", "cream", "grey", "2023", "2024", "2025", "SS", "FW"]
    for bn in brand_names:
        for model in models:
            for extra in extras:
                queries.append(f"{bn} {model} sneakers {extra}")
                queries.append(f"{bn} {model} sneaker {extra} black background")
            queries.append(f"{bn} {model} trainers")
            queries.append(f"{bn} {model} trainers product photo")
        for extra in extras:
            queries.append(f"{bn} {extra} sneakers black background")
            queries.append(f"{bn} luxury {extra} sneakers")
    random.shuffle(queries)
    return queries


def _crawl_bing(query: str, tmp_folder: Path, max_num: int) -> None:
    from icrawler.builtin import BingImageCrawler

    if tmp_folder.exists():
        shutil.rmtree(tmp_folder)
    tmp_folder.mkdir(parents=True, exist_ok=True)
    crawler = BingImageCrawler(
        storage={"root_dir": str(tmp_folder)},
        downloader_threads=4,
        log_level=logging.CRITICAL,
    )
    crawler.crawl(keyword=query, max_num=max_num, filters={"type": "photo", "size": "large"})


def ingest_temp_dir(
    tmp_folder: Path,
    *,
    dest_dir: Path,
    artifact_root: Path,
    brand: str,
    query: str,
    stats: StoreStats,
    index: ContentIndex,
) -> int:
    if not tmp_folder.exists():
        return 0
    before = stats.stored
    for path in sorted(tmp_folder.iterdir()):
        if not path.is_file():
            continue
        try:
            data = path.read_bytes()
        except OSError:
            stats.errors += 1
            continue
        store_validated_image(
            data,
            dest_dir=dest_dir,
            artifact_root=artifact_root,
            family="sneakers",
            claimed_brand=brand,
            claimed_model=None,
            label_status=STATUS_UNVERIFIED_WEB,
            search_query=query,
            search_provider="bing_icrawler",
            source_image_url="UNKNOWN",
            source_page_url="UNKNOWN",
            provenance_note=ICRAWLER_PROVENANCE_NOTE,
            index=index,
            stats=stats,
        )
    return stats.stored - before


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download unverified Bing sneaker images into "
            "local_data/scraper_artifacts/unverified_web_images/sneakers/."
        )
    )
    parser.add_argument("--artifact-root", default=str(ARTIFACT_ROOT_REL))
    parser.add_argument("--target-per-class", type=int, default=TARGET_PER_CLASS)
    parser.add_argument("--images-per-query", type=int, default=IMAGES_PER_QUERY)
    parser.add_argument("--temp-dir", default="/tmp/hypevault_sneaker_crawl_tmp")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--allow-network", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    artifact_root = resolve_under_repo(args.artifact_root, ARTIFACT_ROOT_REL)
    temp_dir = Path(args.temp_dir)
    state_path = resolve_under_repo(
        STATE_REL / "sneaker_scraper_resume.json",
        STATE_REL / "sneaker_scraper_resume.json",
    )

    if args.dry_run:
        print("DRY-RUN: no network and no image writes.")
        print(f"Artifact root: {artifact_root}")
        print(f"Brands: {', '.join(ULTRA_LUXURY_SNEAKERS)}")
        return 0

    require_network_opt_in(args.allow_network)
    state = load_resume_state(state_path)
    completed = set(state.get("completed_queries") or [])
    stats = StoreStats(**(state.get("stats") or {})) if isinstance(state.get("stats"), dict) else StoreStats()
    start = time.time()

    for brand, models in ULTRA_LUXURY_SNEAKERS.items():
        dest_dir = artifact_root / "unverified_web_images" / "sneakers" / safe_brand_name(brand)
        dest_dir.mkdir(parents=True, exist_ok=True)
        assert_not_legacy_training_dir(dest_dir)
        index = load_corpus_index(artifact_root)
        current = count_valid_images(dest_dir)
        if current >= args.target_per_class:
            print(f"[{brand}] Already at {current}/{args.target_per_class}. Skipping.")
            continue

        print(f"[{brand}] current={current} target={args.target_per_class}")
        for round_idx in range(MAX_ROUNDS_PER_BRAND):
            current = count_valid_images(dest_dir)
            if current >= args.target_per_class:
                break
            start_round = current
            queries = build_queries(brand, models, round_idx)
            for query in queries:
                if count_valid_images(dest_dir) >= args.target_per_class:
                    break
                if query in completed:
                    continue
                fetch_count = min(args.images_per_query, args.target_per_class - count_valid_images(dest_dir))
                success = False
                try:
                    _crawl_bing(query, temp_dir, fetch_count)
                    added = ingest_temp_dir(
                        temp_dir,
                        dest_dir=dest_dir,
                        artifact_root=artifact_root,
                        brand=brand,
                        query=query,
                        stats=stats,
                        index=index,
                    )
                    print(f"  +{added} -> {count_valid_images(dest_dir)}/{args.target_per_class}")
                    success = True
                except Exception as exc:  # noqa: BLE001
                    LOGGER.error("query failed %r: %s", query, exc)
                finally:
                    if temp_dir.exists():
                        shutil.rmtree(temp_dir, ignore_errors=True)
                mark_query_completed(state, query, success=success)
                state["stats"] = stats.as_dict()
                save_resume_state(state_path, state)
                time.sleep(1)

            gained = count_valid_images(dest_dir) - start_round
            still_short = args.target_per_class - count_valid_images(dest_dir)
            if gained < EARLY_STOP_MIN_GAIN and still_short > EARLY_STOP_ONLY_IF_STILL_NEED_GT:
                print("  Early stop: low yield while still far from target.")
                break

        short = args.target_per_class - count_valid_images(dest_dir)
        if short > 0:
            for query in build_topup_queries(brand, models)[:TOPUP_MAX_QUERIES]:
                if count_valid_images(dest_dir) >= args.target_per_class:
                    break
                if query in completed:
                    continue
                try:
                    _crawl_bing(query, temp_dir, min(args.images_per_query, short))
                    ingest_temp_dir(
                        temp_dir,
                        dest_dir=dest_dir,
                        artifact_root=artifact_root,
                        brand=brand,
                        query=query,
                        stats=stats,
                        index=index,
                    )
                    mark_query_completed(state, query, success=True)
                except Exception as exc:  # noqa: BLE001
                    LOGGER.error("top-up failed %r: %s", query, exc)
                    mark_query_completed(state, query, success=False)
                finally:
                    if temp_dir.exists():
                        shutil.rmtree(temp_dir, ignore_errors=True)
                state["stats"] = stats.as_dict()
                save_resume_state(state_path, state)

        final = count_valid_images(dest_dir)
        print(f"[{brand}] final={final}/{args.target_per_class} shortfall={max(0, args.target_per_class - final)}")

    print(f"DONE in {(time.time() - start) / 60:.1f}m stats={stats.as_dict()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
