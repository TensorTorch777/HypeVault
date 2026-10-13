import hashlib
import logging
import os
import random
import shutil
import time

from icrawler.builtin import BingImageCrawler

logging.getLogger("icrawler").setLevel(logging.CRITICAL)

output_dir = "Label_0_Sneakers"
temp_dir = "/tmp/sneaker_crawl_tmp"

# Ultra-luxury sneaker classes
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
# Do not bail on "low yield" when we are still short of the target (fixes 2994/3000 cases).
EARLY_STOP_MIN_GAIN = 25
EARLY_STOP_ONLY_IF_STILL_NEED_GT = 80
TOPUP_MAX_QUERIES = 120

SEARCH_ALIASES = {
    "Louis Vuitton": ["Louis Vuitton", "LV"],
    "Alexander McQueen": ["Alexander McQueen", "McQueen"],
    "Dior": ["Dior", "Christian Dior"],
}


def count_images(folder):
    if not os.path.exists(folder):
        return 0
    return len(
        [f for f in os.listdir(folder) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    )


def move_new_images(src_folder, dest_folder):
    """Move downloaded images to class folder with content-hash names."""
    if not os.path.exists(src_folder):
        return 0

    moved = 0
    for fname in os.listdir(src_folder):
        if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        src_path = os.path.join(src_folder, fname)
        with open(src_path, "rb") as f:
            file_hash = hashlib.md5(f.read()).hexdigest()[:12]

        ext = os.path.splitext(fname)[1].lower() or ".jpg"
        dest_path = os.path.join(dest_folder, f"{file_hash}{ext}")
        if not os.path.exists(dest_path):
            shutil.move(src_path, dest_path)
            moved += 1

    return moved


def crawl_bing(query, tmp_folder, max_num):
    if os.path.exists(tmp_folder):
        shutil.rmtree(tmp_folder)
    os.makedirs(tmp_folder, exist_ok=True)

    crawler = BingImageCrawler(
        storage={"root_dir": tmp_folder},
        downloader_threads=4,
        log_level=logging.CRITICAL,
    )
    crawler.crawl(keyword=query, max_num=max_num, filters={"type": "photo", "size": "large"})


def build_queries(brand, models, round_idx):
    brand_names = SEARCH_ALIASES.get(brand, [brand])
    scene_terms = [
        "on feet",
        "street style",
        "side view",
        "close up",
        "product photo",
        "editorial",
        "lookbook",
        "retail listing",
    ]
    # Rotate scene terms each round to force broader coverage.
    shift = round_idx % len(scene_terms)
    scene_terms = scene_terms[shift:] + scene_terms[:shift]

    queries = []
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


def build_topup_queries(brand, models):
    """Extra diverse queries to grab the last few unique images."""
    brand_names = SEARCH_ALIASES.get(brand, [brand])
    queries = []
    extras = ["white", "black", "cream", "grey", "2023", "2024", "2025", "SS", "FW"]
    for bn in brand_names:
        for model in models:
            for x in extras:
                queries.append(f"{bn} {model} sneakers {x}")
                queries.append(f"{bn} {model} sneaker {x} black background")
            queries.append(f"{bn} {model} trainers")
            queries.append(f"{bn} {model} trainers product photo")
        for x in extras:
            queries.append(f"{bn} {x} sneakers black background")
            queries.append(f"{bn} luxury {x} sneakers")
    random.shuffle(queries)
    return queries


def main():
    os.makedirs(output_dir, exist_ok=True)
    start_time = time.time()

    print(f"Ultra-luxury sneaker classes: {len(ULTRA_LUXURY_SNEAKERS)}")
    for brand in ULTRA_LUXURY_SNEAKERS:
        print(f"  - {brand}")
    print()

    for brand, models in ULTRA_LUXURY_SNEAKERS.items():
        safe_brand = brand.replace("/", "_")
        brand_folder = os.path.join(output_dir, safe_brand)
        os.makedirs(brand_folder, exist_ok=True)

        current = count_images(brand_folder)
        needed = TARGET_PER_CLASS - current
        if needed <= 0:
            print(f"[{brand}] Already at {current}/{TARGET_PER_CLASS}. Skipping.")
            continue

        print(f"\n{'=' * 60}")
        print(f"[{brand}] Current: {current} | Target: {TARGET_PER_CLASS} | Need: {needed}")
        print(f"{'=' * 60}")

        for round_idx in range(MAX_ROUNDS_PER_BRAND):
            current = count_images(brand_folder)
            if current >= TARGET_PER_CLASS:
                print(f"  Target reached ({current}/{TARGET_PER_CLASS})!")
                break

            queries = build_queries(brand, models, round_idx)
            print(f"  Round {round_idx + 1}/{MAX_ROUNDS_PER_BRAND}: {len(queries)} queries")
            start_round_count = current

            for q_idx, query in enumerate(queries):
                current = count_images(brand_folder)
                still_needed = TARGET_PER_CLASS - current
                if still_needed <= 0:
                    print(f"  Target reached ({current}/{TARGET_PER_CLASS})!")
                    break

                fetch_count = min(IMAGES_PER_QUERY, still_needed)
                print(f'    [{q_idx + 1}/{len(queries)}] "{query}" ...', end=" ", flush=True)

                try:
                    crawl_bing(query, temp_dir, fetch_count)
                    moved = move_new_images(temp_dir, brand_folder)
                    new_total = count_images(brand_folder)
                    print(f"+{moved} images -> {new_total}/{TARGET_PER_CLASS}")
                except Exception as e:
                    print(f"Error: {e}")

                time.sleep(1)

            end_round_count = count_images(brand_folder)
            gained = end_round_count - start_round_count
            print(f"  Round {round_idx + 1} gained: {gained} images")

            still_short = TARGET_PER_CLASS - end_round_count
            if gained < EARLY_STOP_MIN_GAIN and still_short > EARLY_STOP_ONLY_IF_STILL_NEED_GT:
                print("  Early stop: low new-image yield and still far from target.")
                break

        # Top-up pass: never leave a class a few images short because of duplicate saturation.
        short = TARGET_PER_CLASS - count_images(brand_folder)
        if short > 0:
            print(f"  Top-up: need {short} more images...")
            top_queries = build_topup_queries(brand, models)
            for q_idx, query in enumerate(top_queries[:TOPUP_MAX_QUERIES]):
                current = count_images(brand_folder)
                if current >= TARGET_PER_CLASS:
                    break
                still_needed = TARGET_PER_CLASS - current
                fetch_count = min(IMAGES_PER_QUERY, still_needed)
                print(f'    [top-up {q_idx + 1}] "{query}" ...', end=" ", flush=True)
                try:
                    crawl_bing(query, temp_dir, fetch_count)
                    moved = move_new_images(temp_dir, brand_folder)
                    print(f"+{moved} -> {count_images(brand_folder)}/{TARGET_PER_CLASS}")
                except Exception as e:
                    print(f"Error: {e}")
                time.sleep(1)

        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

        final = count_images(brand_folder)
        print(f"\n  [{brand}] Final: {final}/{TARGET_PER_CLASS}")

    elapsed = time.time() - start_time
    print(f"\n{'=' * 60}")
    print(f"DONE in {elapsed / 60:.1f} minutes")
    print(f"{'=' * 60}")
    for brand in ULTRA_LUXURY_SNEAKERS:
        safe_brand = brand.replace("/", "_")
        folder = os.path.join(output_dir, safe_brand)
        cnt = count_images(folder)
        status = "OK" if cnt >= TARGET_PER_CLASS else f"need {TARGET_PER_CLASS - cnt} more"
        print(f"  {brand:<25} {cnt:>5}/{TARGET_PER_CLASS}  {status}")


if __name__ == "__main__":
    main()
