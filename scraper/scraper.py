import pandas as pd
import os
import time
import shutil
import logging
import hashlib
from icrawler.builtin import BingImageCrawler

logging.getLogger("icrawler").setLevel(logging.CRITICAL)

csv_file = "Watches.csv"
output_dir = "Label_0_Watches"
temp_dir = "/tmp/watch_crawl_tmp"

PREMIUM_BRANDS = [
    "Patek Philippe",
    "Audemars Piguet",
    "Vacheron Constantin",
    "A. Lange & Söhne",
    "Richard Mille",
]

TARGET_PER_CLASS = 3000
IMAGES_PER_QUERY = 100

SEARCH_NAMES = {
    "A. Lange & Söhne": ["A Lange Sohne", "Lange Sohne", "A. Lange Sohne"],
    "Richard Mille": ["Richard Mille"],
    "Patek Philippe": ["Patek Philippe"],
    "Audemars Piguet": ["Audemars Piguet"],
    "Vacheron Constantin": ["Vacheron Constantin"],
}

def count_images(folder):
    if not os.path.exists(folder):
        return 0
    return len([f for f in os.listdir(folder) if f.lower().endswith((".jpg", ".jpeg", ".png"))])

def move_new_images(src_folder, dest_folder):
    """Move downloaded images from temp to destination with unique names."""
    if not os.path.exists(src_folder):
        return 0
    moved = 0
    for fname in os.listdir(src_folder):
        if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        src_path = os.path.join(src_folder, fname)
        with open(src_path, "rb") as f:
            file_hash = hashlib.md5(f.read()).hexdigest()[:12]
        ext = os.path.splitext(fname)[1]
        dest_path = os.path.join(dest_folder, f"{file_hash}{ext}")
        if not os.path.exists(dest_path):
            shutil.move(src_path, dest_path)
            moved += 1
    return moved

def crawl_bing(query, tmp_folder, max_num):
    if os.path.exists(tmp_folder):
        shutil.rmtree(tmp_folder)
    os.makedirs(tmp_folder)
    crawler = BingImageCrawler(
        storage={"root_dir": tmp_folder},
        downloader_threads=4,
        log_level=logging.CRITICAL,
    )
    crawler.crawl(keyword=query, max_num=max_num, filters={"type": "photo", "size": "large"})

# --- Main ---
print("Loading CSV...")
df = pd.read_csv(csv_file, low_memory=False)
df = df.dropna(subset=["brand", "ref"])
df = df[df["brand"].isin(PREMIUM_BRANDS)]

unique_watches = df.drop_duplicates(subset=["brand", "model", "ref"])[["brand", "model", "ref"]].reset_index(drop=True)

print(f"\nClasses ({len(PREMIUM_BRANDS)} brands):")
for b in PREMIUM_BRANDS:
    cnt = unique_watches[unique_watches["brand"] == b].shape[0]
    print(f"  {b:<25} {cnt:>5} unique refs in CSV")
print()

os.makedirs(output_dir, exist_ok=True)
start_time = time.time()

for brand in PREMIUM_BRANDS:
    safe_brand = brand.replace("/", "_")
    brand_folder = os.path.join(output_dir, safe_brand)
    os.makedirs(brand_folder, exist_ok=True)

    current = count_images(brand_folder)
    needed = TARGET_PER_CLASS - current

    if needed <= 0:
        print(f"\n[{brand}] Already at {current}/{TARGET_PER_CLASS}. Skipping.")
        continue

    models = df[df["brand"] == brand]["model"].dropna().unique().tolist()
    refs = df[df["brand"] == brand]["ref"].dropna().unique().tolist()
    search_names = SEARCH_NAMES.get(brand, [brand])

    queries = []
    for name in search_names:
        for model in models:
            queries.append(f"{name} {model} watch")
            queries.append(f"{name} {model} wristwatch official photo")
            queries.append(f"{name} {model} dial close up")
        for ref in refs[:50]:
            queries.append(f"{name} {ref} watch")
        queries.append(f"{name} luxury watch")
        queries.append(f"{name} watch collection")
        queries.append(f"{name} timepiece high resolution")
        queries.append(f"{name} watch face close up")
        queries.append(f"{name} wristwatch product photo")
        queries.append(f"{name} watch review")
        queries.append(f"{name} watch catalog")
        queries.append(f"{name} latest watches")
        queries.append(f"{name} watch official website")

    print(f"\n{'='*60}")
    print(f"[{brand}] Current: {current} | Target: {TARGET_PER_CLASS} | Need: {needed}")
    print(f"[{brand}] {len(queries)} search queries")
    print(f"{'='*60}")

    for q_idx, query in enumerate(queries):
        current = count_images(brand_folder)
        still_needed = TARGET_PER_CLASS - current
        if still_needed <= 0:
            print(f"  Target reached ({current}/{TARGET_PER_CLASS})!")
            break

        fetch_count = min(IMAGES_PER_QUERY, still_needed)
        print(f"  [{q_idx+1}/{len(queries)}] \"{query}\" ...", end=" ", flush=True)

        try:
            crawl_bing(query, temp_dir, fetch_count)
            moved = move_new_images(temp_dir, brand_folder)
            new_total = count_images(brand_folder)
            print(f"+{moved} images -> {new_total}/{TARGET_PER_CLASS}")
        except Exception as e:
            print(f"Error: {e}")

        time.sleep(1)

    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)

    final = count_images(brand_folder)
    print(f"\n  [{brand}] Final: {final}/{TARGET_PER_CLASS}")

elapsed = time.time() - start_time
print(f"\n{'='*60}")
print(f"DONE in {elapsed/60:.1f} minutes")
print(f"{'='*60}")
for brand in PREMIUM_BRANDS:
    safe_brand = brand.replace("/", "_")
    folder = os.path.join(output_dir, safe_brand)
    cnt = count_images(folder)
    status = "OK" if cnt >= TARGET_PER_CLASS else f"need {TARGET_PER_CLASS - cnt} more"
    print(f"  {brand:<25} {cnt:>5}/{TARGET_PER_CLASS}  {status}")
