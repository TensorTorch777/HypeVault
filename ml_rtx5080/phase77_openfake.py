"""Phase 77: OpenFake (ComplexDataLab/OpenFake v2, CC-BY-NC-4.0) images for development and for benchmark_v7.

OpenFake's `core/test` split holds generators that are not in its own train/validation splits. Every
generator family that appears in `core/test` is kept out of Phase 77 training, so the families in the
test split are unseen by Phase 76 and Phase 77 alike, except those Phase 76 already met through the
frontier pool (reported apart). Development images come from `core/validation` only.

Only the text columns are read to plan; image bytes are then fetched one parquet row group at a time,
for the row groups that hold planned rows.

    .venv/bin/python ml_rtx5080/phase77_openfake.py meta        # text columns -> experiments/phase77_openfake/*_meta.jsonl
    .venv/bin/python ml_rtx5080/phase77_openfake.py fetch-dev   # development rows -> experiments/phase77_openfake/dev/
"""

from __future__ import annotations

import collections
import json
import random
import re
import sys
from pathlib import Path

ML = Path(__file__).resolve().parent
OUT = ML / "experiments" / "phase77_openfake"
REPO_ID = "ComplexDataLab/OpenFake"
REVISION = "main"
SPLITS = {"test": 13, "validation": 15}
TEXT = ["label", "model", "type", "release_date", "row_id", "image_sha256", "prompt"]
SEED = 77
DEV_PER_MODEL = 40
DEV_REALS = 1200
MAX_DEV_ROW_GROUPS = 60

# Families of the generators in core/test. A development generator matching any of these is not used.
TEST_FAMILIES = re.compile(
    r"flux|chroma|z-image|illustrious|gpt-image|dall-?e|^4o|midjourney|veo|wan|recraft|ernie|lumina|sora|seedream|"
    r"nano-banana|ideogram|aurora|frames|halfmoon", re.IGNORECASE)
# Junk, and checkpoints built on a test family: Illustrious-based anime SDXL ("_il", One Obsession) and FLUX-based Mystic.
JUNK = re.compile(r"tiny-random|_il|obsession|mystic", re.IGNORECASE)


def _path(split: str, i: int) -> str:
    n = SPLITS[split]
    return f"datasets/{REPO_ID}@{REVISION}/core/{split}-{i:05d}-of-{n:05d}.parquet"


def meta() -> None:
    from concurrent.futures import ThreadPoolExecutor

    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem

    def one(args):
        split, i = args
        rows = []
        with HfFileSystem().open(_path(split, i), "rb", block_size=2**16, cache_type="readahead") as f:
            pf = pq.ParquetFile(f)
            for g in range(pf.metadata.num_row_groups):
                for k, r in enumerate(pf.read_row_group(g, columns=TEXT).to_pylist()):
                    rows.append({**r, "split": split, "file": i, "rg": g, "idx": k})
        return rows

    OUT.mkdir(parents=True, exist_ok=True)
    for split, n in SPLITS.items():
        target = OUT / f"{split}_meta.jsonl"
        if target.exists():
            continue
        with ThreadPoolExecutor(n) as ex:
            rows = [r for part in ex.map(one, [(split, i) for i in range(n)]) for r in part]
        target.write_text("".join(json.dumps(r) + "\n" for r in rows))
        print(split, len(rows), flush=True)


def read_meta(split: str) -> list[dict]:
    return [json.loads(x) for x in (OUT / f"{split}_meta.jsonl").read_text().splitlines()]


def dev_eligible(r: dict) -> bool:
    return r["label"] == "real" or not (TEST_FAMILIES.search(r["model"]) or JUNK.search(r["model"]))


def plan_dev() -> list[dict]:
    """Up to DEV_PER_MODEL images per eligible validation generator and DEV_REALS reals.

    Row groups are taken greedily (most still-needed rows first, seeded ties) up to MAX_DEV_ROW_GROUPS,
    so rare generators may fall short; the shortfall is what it is.
    """
    rows = [r for r in read_meta("validation") if dev_eligible(r)]
    by_rg = collections.defaultdict(list)
    for r in rows:
        by_rg[(r["file"], r["rg"])].append(r)
    order = sorted(by_rg)
    random.Random(SEED).shuffle(order)
    need = collections.Counter({m: DEV_PER_MODEL for m in {r["model"] for r in rows if r["label"] == "fake"}})
    need["__real__"] = DEV_REALS

    def key(r):
        return "__real__" if r["label"] == "real" else r["model"]

    picked = []
    for _ in range(MAX_DEV_ROW_GROUPS):
        gain = {g: sum(1 for r in by_rg[g] if need[key(r)] > 0) for g in order}
        best = max(order, key=lambda g: gain[g])
        if gain[best] == 0:
            break
        for r in by_rg[best]:
            if need[key(r)] > 0:
                need[key(r)] -= 1
                picked.append(r)
        order.remove(best)
    return picked


# benchmark_v7_fresh/protocol.json quotas (core/test only)
UNSEEN_BOTH = ("veo-3", "sora-2", "wan-video-2.5", "ernie-image", "ernie-image-turbo")
SEEN_BY_P76 = ("flux.2-klein-9b", "z-image-turbo", "gpt-image-1.5", "gpt-image-2", "midjourney-7", "recraft-v2", "recraft-v3",
               "lumina-17-2-25", "aurora-20-1-25", "frames-23-1-25", "halfmoon-4-4-25", "ideogram-2.0", "seedream-v5.0",
               "nano-banana-pro")
TEST_SEED = 7
MAX_TEST_ROW_GROUPS = 120


def test_group(r: dict) -> str | None:
    if r["label"] == "real":
        return "real_general"
    if r["model"] in UNSEEN_BOTH:
        return "unseen_both"
    if r["model"] == "illustrious":
        return "illustrious"
    if r["model"] in SEEN_BY_P76:
        return "seen_family_by_phase76"
    return None


def plan_test() -> list[dict]:
    """Seeded greedy row-group plan over core/test filling the protocol quotas (shortfalls accepted)."""
    quota = collections.Counter({**{m: 120 for m in UNSEEN_BOTH}, "illustrious": 60, **{m: 30 for m in SEEN_BY_P76},
                                 "__real__": 300})
    rows = [r for r in read_meta("test") if test_group(r)]
    by_rg = collections.defaultdict(list)
    for r in rows:
        by_rg[(r["file"], r["rg"])].append(r)
    order = sorted(by_rg)
    random.Random(TEST_SEED).shuffle(order)

    def key(r):
        return "__real__" if r["label"] == "real" else r["model"]

    picked = []
    for _ in range(MAX_TEST_ROW_GROUPS):
        # primary rows first: an unseen_both row counts 10, any other needed row 1
        gain = {g: sum((10 if r["model"] in UNSEEN_BOTH else 1) for r in by_rg[g] if quota[key(r)] > 0) for g in order}
        best = max(order, key=lambda g: gain[g])
        if gain[best] == 0:
            break
        for r in by_rg[best]:
            if quota[key(r)] > 0:
                quota[key(r)] -= 1
                picked.append({**r, "test_group": test_group(r)})
        order.remove(best)
    return picked


def fetch(rows: list[dict], folder: Path) -> None:
    """Write each planned row's original image bytes to folder/<label>/<model>/<row_id><ext>, plus manifest.jsonl."""
    from concurrent.futures import ThreadPoolExecutor

    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem

    by_rg = collections.defaultdict(list)
    for r in rows:
        by_rg[(r["split"], r["file"], r["rg"])].append(r)
    manifest = folder / "manifest.jsonl"
    done = {json.loads(x)["row_id"] for x in manifest.read_text().splitlines()} if manifest.exists() else set()
    todo = [g for g in sorted(by_rg) if any(r["row_id"] not in done for r in by_rg[g])]
    folder.mkdir(parents=True, exist_ok=True)

    def one(g):
        split, i, rg = g
        with HfFileSystem().open(_path(split, i), "rb", block_size=2**22, cache_type="readahead") as f:
            images = pq.ParquetFile(f).read_row_group(rg, columns=["image", "row_id"]).to_pylist()
        by_id = {x["row_id"]: x["image"] for x in images}
        out = []
        for r in by_rg[g]:
            raw = (by_id.get(r["row_id"]) or {}).get("bytes")
            if not raw:
                continue
            ext = Path((by_id[r["row_id"]].get("path") or "")).suffix.lower() or ".img"
            target = folder / r["label"] / re.sub(r"[^A-Za-z0-9._-]+", "_", r["model"]) / f"{r['row_id']}{ext}"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(raw)
            out.append({**{k: r[k] for k in ("row_id", "label", "model", "type", "release_date", "image_sha256",
                                             "split", "file", "rg", "idx")},
                        "path": str(target.relative_to(ML.parent))})
        return out

    with ThreadPoolExecutor(6) as ex, manifest.open("a") as w:
        for n, part in enumerate(ex.map(one, todo), 1):
            for x in part:
                w.write(json.dumps(x) + "\n")
            w.flush()
            if n % 5 == 0 or n == len(todo):
                print(f"{folder.name}: {n}/{len(todo)} row groups", flush=True)


def manifest_rows(folder: Path) -> list[dict]:
    return [json.loads(x) for x in (folder / "manifest.jsonl").read_text().splitlines()]


def sets() -> dict:
    """Development sets for phase77_probe: name -> (ids, loader)."""
    import kandinsky_v4 as k4

    folder = OUT / "dev"
    rows = manifest_rows(folder)
    paths = {r["row_id"]: ML.parent / r["path"] for r in rows}
    return {"openfake_dev": ([r["row_id"] for r in rows], lambda i, p=paths: k4.jpeg90(k4.square(p[i], 512)))}


if __name__ == "__main__":
    sys.path.insert(0, str(ML))
    if sys.argv[1] == "meta":
        meta()
    elif sys.argv[1] == "fetch-dev":
        rows = plan_dev()
        c = collections.Counter(r["model"] for r in rows)
        print(len(rows), "rows from", len({(r["file"], r["rg"]) for r in rows}), "row groups;", len(c), "models", flush=True)
        fetch(rows, OUT / "dev")
