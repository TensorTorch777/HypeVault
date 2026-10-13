import pandas as pd
import requests
import os
import time
import random
import json
import hashlib
from duckduckgo_search import DDGS

csv_file = "Watches.csv"
output_dir = "Label_0_Watches"
progress_file = "balance_progress.json"

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

def load_progress():
    if os.path.exists(progress_file):
        with open(progress_file, "r") as f:
            return json.load(f)
    return {"done_queries": [], "downloaded": 0, "failed": 0}

def save_progress(prog):
    with open(progress_file, "w") as f:
        json.dump(prog, f)

def download_image(url, path):
    try:
        resp = requests.get(url, timeout=15, headers={
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"
        })
        if resp.status_code == 200 and len(resp.content) > 2000:
            with open(path, "wb") as f:
                f.write(resp.content)
            return True
    except Exception:
        pass
    return False

def count_existing(brand_folder):
    if not os.path.exists(brand_folder):
        return 0
    return len([f for f in os.listdir(brand_folder) if f.endswith(".jpg")])

def build_search_queries(brand, models):
    queries = []
    for model in models:
        queries.append(f"{brand} {model} watch")
        queries.append(f"{brand} {model} wristwatch official")
        queries.append(f"{brand} {model} watch dial closeup")
    queries.append(f"{brand} watch collection")
    queries.append(f"{brand} watches official")
    queries.append(f"{brand} timepiece")
    return queries

# --- Main ---
print("Loading CSV to get model names...")
df = pd.read_csv(csv_file, low_memory=False)
df = df.dropna(subset=["brand", "ref"])

progress = load_progress()
done_queries = set(progress["done_queries"])
total_downloaded = progress["downloaded"]
total_failed = progress["failed"]

ddgs = DDGS()
consecutive_ratelimits = 0
query_count = 0

for brand in BRANDS_TO_BALANCE:
    safe_brand = brand.replace("/", "_")
    brand_folder = os.path.join(output_dir, safe_brand)
    os.makedirs(brand_folder, exist_ok=True)

    current_count = count_existing(brand_folder)
    needed = TARGET_PER_CLASS - current_count

    if needed <= 0:
        print(f"\n[{brand}] Already has {current_count}/{TARGET_PER_CLASS} images. Skipping.")
        continue

    models = df[df["brand"] == brand]["model"].dropna().unique().tolist()
    queries = build_search_queries(brand, models)

    print(f"\n{'='*60}")
    print(f"[{brand}] Current: {current_count} | Target: {TARGET_PER_CLASS} | Need: {needed}")
    print(f"[{brand}] {len(queries)} search queries from {len(models)} models")
    print(f"{'='*60}")

    brand_downloaded = 0

    for q_idx, query in enumerate(queries):
        if query in done_queries:
            continue

        current_count = count_existing(brand_folder)
        if current_count >= TARGET_PER_CLASS:
            print(f"[{brand}] Reached target ({current_count}/{TARGET_PER_CLASS}). Moving on.")
            break

        success = False
        for attempt in range(MAX_RETRIES):
            try:
                results = list(ddgs.images(
                    query,
                    safesearch="off",
                    size="Large",
                    max_results=IMAGES_PER_QUERY,
                ))

                img_count = 0
                for r in results:
                    image_url = r["image"]
                    url_hash = hashlib.md5(image_url.encode()).hexdigest()[:12]
                    file_path = os.path.join(brand_folder, f"web_{url_hash}.jpg")

                    if os.path.exists(file_path):
                        continue

                    if download_image(image_url, file_path):
                        img_count += 1
                        brand_downloaded += 1
                        total_downloaded += 1

                consecutive_ratelimits = 0
                remaining = TARGET_PER_CLASS - count_existing(brand_folder)
                print(f"  [{q_idx+1}/{len(queries)}] \"{query}\" -> {img_count} new images (need {remaining} more)")
                success = True
                break

            except Exception as e:
                err_str = str(e)
                if "Ratelimit" in err_str or "403" in err_str:
                    consecutive_ratelimits += 1
                    backoff = min(MAX_DELAY, BASE_DELAY * (2 ** consecutive_ratelimits)) + random.uniform(2, 8)
                    print(f"  Rate limited (attempt {attempt+1}/{MAX_RETRIES}), waiting {backoff:.0f}s...")
                    time.sleep(backoff)
                    ddgs = DDGS()
                else:
                    print(f"  Error on \"{query}\": {e}")
                    total_failed += 1
                    success = True
                    break

        if not success:
            total_failed += 1
            print(f"  Gave up on \"{query}\" after {MAX_RETRIES} retries")

        done_queries.add(query)
        query_count += 1

        if query_count % 5 == 0:
            progress["done_queries"] = list(done_queries)
            progress["downloaded"] = total_downloaded
            progress["failed"] = total_failed
            save_progress(progress)

        delay = BASE_DELAY + random.uniform(1.0, 3.0)
        if consecutive_ratelimits > 0:
            delay = min(MAX_DELAY, delay * (2 ** min(consecutive_ratelimits, 5)))
        time.sleep(delay)

        if query_count % BATCH_SIZE == 0:
            pause = random.uniform(15, 30)
            print(f"  --- Batch pause {pause:.0f}s ---")
            time.sleep(pause)
            ddgs = DDGS()
            consecutive_ratelimits = 0

    final_count = count_existing(brand_folder)
    print(f"\n[{brand}] Done: {final_count}/{TARGET_PER_CLASS} images (+{brand_downloaded} new)")

progress["done_queries"] = list(done_queries)
progress["downloaded"] = total_downloaded
progress["failed"] = total_failed
save_progress(progress)

print(f"\n{'='*60}")
print("BALANCE COMPLETE")
print(f"{'='*60}")
for brand in BRANDS_TO_BALANCE:
    safe_brand = brand.replace("/", "_")
    brand_folder = os.path.join(output_dir, safe_brand)
    cnt = count_existing(brand_folder)
    print(f"  {brand:<25} {cnt:>5}/{TARGET_PER_CLASS}")
print(f"\nTotal new downloads: {total_downloaded} | Failed: {total_failed}")
