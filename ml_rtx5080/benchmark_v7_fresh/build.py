"""Assemble benchmark_v7_fresh (protocol.json): OpenFake core/test images and fresh real eBay photos.

Every image goes through benchmark_v1's admission code with the benchmark_v2+ settings: centre square,
Lanczos to 512 px, JPEG q90, training-set and in-benchmark duplicate checks. The duplicate targets also
include every frontier-pool and OpenFake development image that Phase 76 or Phase 77 trained on.

    .venv/bin/python ml_rtx5080/benchmark_v7_fresh/build.py openfake
    .venv/bin/python ml_rtx5080/benchmark_v7_fresh/build.py real --per-brand 40

Run only after ml_rtx5080/experiments/phase77/frozen_rule.json exists and lock.json records it.
No model scores anything here.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ML = HERE.parent
REPO = ML.parent
sys.path.insert(0, str(ML / "benchmark_v2_fresh"))
sys.path.insert(0, str(ML))
sys.path.insert(0, str(REPO / "backend"))

import build as v2  # noqa: E402  (sets the 512 px normalisation)
from common import BenchmarkDedup, TrainingIndex, append_manifest, read_manifest  # noqa: E402

DATA = HERE / "data"
MANIFEST = DATA / "manifest.jsonl"
NORMALISED = DATA / "normalised"
RAW = DATA / "openfake_raw"
LOCK = HERE / "lock.json"
EARLIER = [
    ML / "benchmark_v1_real_vs_generated" / "data" / "manifest.jsonl",
    ML / "benchmark_v2_fresh" / "data" / "manifest.jsonl",
    ML / "benchmark_v3_fresh" / "data" / "manifest.jsonl",
    ML / "benchmark_v4_fresh" / "data" / "manifest.jsonl",
    ML / "benchmark_v5_frontier" / "data" / "manifest.jsonl",
    ML / "benchmark_v6_fresh" / "data" / "manifest.jsonl",
]


def _require_lock() -> None:
    if not LOCK.exists():
        raise SystemExit("lock.json is missing: freeze Phase 77 and record it before fetching any benchmark_v7 image")


def dedup_all() -> tuple[BenchmarkDedup, list]:
    """Earlier benchmarks, every admitted v7 image so far, and every frontier/OpenFake training image."""
    from common import dhash_image, sha256_bytes  # the helpers admit() uses
    from PIL import Image, ImageOps

    dedup = BenchmarkDedup()
    earlier = [s for m in EARLIER if m.exists() for s in read_manifest(m)]
    mine = [s for s in read_manifest(MANIFEST) if s.excluded is None] if MANIFEST.exists() else []
    for s in earlier + mine:
        dedup.sha256s.add(s.original_sha256)
        dedup.dhashes.append(int(s.original_dhash, 16))
    cache = DATA / "training_hashes.json"
    if not cache.exists():
        import phase77_openfake as of

        files = sorted((ML / "experiments" / "frontier_v5" / "train_images").rglob("*"))
        files += [REPO / r["path"] for r in of.manifest_rows(of.OUT / "dev")]
        hashes = []
        for f in files:
            if f.is_file() and f.suffix.lower() not in (".jsonl", ".json"):
                raw = f.read_bytes()
                try:
                    with Image.open(f) as im:
                        hashes.append([sha256_bytes(raw), f"{dhash_image(ImageOps.exif_transpose(im).convert('RGB')):016x}"])
                except Exception:
                    continue
        DATA.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(hashes))
    for sha, dh in json.loads(cache.read_text()):
        dedup.sha256s.add(sha)
        dedup.dhashes.append(int(dh, 16))
    return dedup, earlier + (list(read_manifest(MANIFEST)) if MANIFEST.exists() else [])


def openfake() -> None:
    import phase77_openfake as of

    _require_lock()
    rows = of.plan_test()
    of.fetch(rows, RAW)
    planned = {r["row_id"]: r for r in rows}
    dedup, _ = dedup_all()
    training = TrainingIndex.load()
    done = {s.sample_id for s in read_manifest(MANIFEST)} if MANIFEST.exists() else set()
    samples = []
    for r in of.manifest_rows(RAW):
        sid = f"of_{r['row_id']}"
        if sid in done or r["row_id"] not in planned:
            continue
        group = planned[r["row_id"]]["test_group"]
        samples.append(v2._admit(
            (REPO / r["path"]).read_bytes(), sample_id=sid, label="REAL_PHOTO" if r["label"] == "real" else "GENERATED",
            group=group, brand="n/a", unit_id=r["row_id"], out_dir=NORMALISED / group, training=training, dedup=dedup,
            meta={"model": r["model"], "type": r["type"], "release_date": r["release_date"], "openfake_sha256": r["image_sha256"],
                  "source": "ComplexDataLab/OpenFake core/test", "parquet": [r["file"], r["rg"], r["idx"]]},
        ))
    append_manifest(MANIFEST, samples)
    kept = [s for s in samples if s.excluded is None]
    print(f"admitted {len(kept)}/{len(samples)}", flush=True)


def real(per_brand: int) -> None:
    _require_lock()
    # The relaxation from the start (protocol.json): gallery and market_listings listings are allowed for every
    # brand, so only earlier benchmark listings (v1-v6) and training images are excluded.
    v2.DATA, v2.MANIFEST = DATA, MANIFEST
    v2.dedup_with_v1 = dedup_all
    original_admit = v2._admit
    v2._admit = lambda raw, **kw: original_admit(raw, **{**kw, "group": "real_ebay_v7", "out_dir": NORMALISED / "real_ebay_v7"})
    asyncio.run(v2.collect_real(per_brand, max_pages=10, delay=0.5))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("what", choices=("openfake", "real"))
    parser.add_argument("--per-brand", type=int, default=40)
    args = parser.parse_args()
    if args.what == "openfake":
        openfake()
    else:
        real(args.per_brand)
