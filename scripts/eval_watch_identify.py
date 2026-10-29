"""Measure watch identification on two test sets.

A. Leave-one-out on the gallery: each labelled eBay photo is hidden in turn and named from the
   others. Near-copies (cosine >= 0.95, usually the same photo relisted) are hidden too, so a
   listing cannot be matched to its own copy. Look-alikes only; no dial text (photos are not kept).
B. Unseen photos (optionally also with eBay's visual search as a clue, --ebay-photo): N real photos per brand from Label_0_Watches, which the gallery never saw.
   These have brand labels only, so this checks the brand, through the full pipeline
   (watch check + dial text + look-alikes).

    .venv/bin/python scripts/eval_watch_identify.py --photos 20
"""

from __future__ import annotations

import argparse
import asyncio
import collections
import json
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from database import AsyncSessionLocal, engine  # noqa: E402
from identify import recognize  # noqa: E402
from identify.catalog import load_catalog  # noqa: E402
from identify.embedder import Embedder  # noqa: E402
from identify.gallery import Gallery  # noqa: E402
from inference.content_gate import score_watch  # noqa: E402
from inference.dial_text import read_dial_lines  # noqa: E402

NEAR_COPY = 0.95
OUT = ROOT / "ml_rtx5080" / "experiments" / "watch_identify_v1"


def leave_one_out(g: Gallery) -> dict:
    hits1 = hits3 = brand1 = n = 0
    sims_right, sims_wrong = [], []
    for i in range(len(g)):
        truth = (g.meta["brand"][i], g.meta["model"][i] or None)
        if truth[1] is None:
            continue
        sims = g.vectors @ g.vectors[i]
        hidden = set(np.flatnonzero(sims >= NEAR_COPY).tolist()) | {i}
        nb = g.neighbours(g.vectors[i], recognize.K, exclude=hidden)
        guesses = recognize.rank(nb if nb and nb[0].similarity >= recognize.MIN_SIMILARITY else [])
        names = [(c["brand"], c["model"]) for c in guesses]
        n += 1
        top1 = bool(names) and names[0] == truth
        hits1 += top1
        hits3 += truth in names
        brand1 += bool(names) and names[0][0] == truth[0]
        (sims_right if top1 else sims_wrong).append(nb[0].similarity if nb else 0.0)
    return {
        "labelled_photos": n,
        "model_top1": round(hits1 / n, 3),
        "model_top3": round(hits3 / n, 3),
        "brand_top1": round(brand1 / n, 3),
        "median_top_similarity_when_right": round(float(np.median(sims_right)), 3) if sims_right else None,
        "median_top_similarity_when_wrong": round(float(np.median(sims_wrong)), 3) if sims_wrong else None,
    }


async def unseen(g: Gallery, catalog, per_brand: int, seed: int, image_search=None) -> dict:
    rng = random.Random(seed)
    embedder = Embedder()
    rows = []
    for brand_dir in sorted((ROOT / "Label_0_Watches").iterdir()):
        files = sorted(p for p in brand_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"})
        for path in rng.sample(files, min(per_brand, len(files))):
            with Image.open(path) as im:
                pil = im.convert("RGB")
            res = await recognize.identify(catalog=catalog, gallery=g, image=pil, embedder=embedder,
                                           read_lines=read_dial_lines, watch_check=score_watch,
                                           image_search=image_search)
            top = res["candidates"][0] if res["candidates"] else None
            rows.append({"brand": brand_dir.name, "status": res["status"],
                         "guess_brand": top["brand"] if top else None, "dial_used": res["used"]["dial_text"] if "used" in res else False})
    n = len(rows)
    status = collections.Counter(r["status"] for r in rows)
    right = sum(r["guess_brand"] == r["brand"] for r in rows)
    per_brand_right = collections.Counter(r["brand"] for r in rows if r["guess_brand"] == r["brand"])
    return {
        "photos": n,
        "brand_top1": round(right / n, 3),
        "status_counts": dict(status),
        "dial_text_named_a_brand_or_model": sum(r["dial_used"] for r in rows),
        "brand_top1_by_brand": {b: f"{per_brand_right[b]}/{per_brand}" for b in sorted({r['brand'] for r in rows})},
    }


async def run(per_brand: int, seed: int, ebay_photo: bool) -> None:
    from build_watch_gallery import gallery_path

    base = gallery_path()
    report: dict = {"leave_one_out": {}}
    for pool, path in (("cls", base), ("patch_mean", base.with_name(f"{base.stem}_patch_mean{base.suffix}"))):
        if path.exists():
            g = Gallery.load(path)
            report["gallery_size"] = len(g)
            report["gallery_built_on"] = g.built_on
            report["leave_one_out"][pool] = leave_one_out(g)
            print(pool, report["leave_one_out"][pool])
    if per_brand:
        async with AsyncSessionLocal() as session:
            catalog = await load_catalog(session)
        await engine.dispose()
        report["unseen_label0_photos"] = await unseen(Gallery.load(base), catalog, per_brand, seed)
        print("unseen", report["unseen_label0_photos"])
        if ebay_photo:
            from identify.service import ebay_photo_search

            report["unseen_label0_photos_with_ebay_photo_search"] = await unseen(
                Gallery.load(base), catalog, per_brand, seed, image_search=ebay_photo_search)
            print("unseen + eBay photo search", report["unseen_label0_photos_with_ebay_photo_search"])
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "eval_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {(OUT / 'eval_report.json').relative_to(ROOT)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--photos", type=int, default=0, help="Unseen Label_0 photos per brand (0 skips part B)")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--ebay-photo", action="store_true", help="Also run part B with eBay photo search (one API call per photo)")
    args = parser.parse_args()
    sys.path.insert(0, str(ROOT / "scripts"))
    asyncio.run(run(args.photos, args.seed, args.ebay_photo))


if __name__ == "__main__":
    main()
