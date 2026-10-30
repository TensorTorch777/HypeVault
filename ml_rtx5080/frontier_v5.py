"""Phase 75: teach DINOv3 to recognise frontier generators (ChatGPT/GPT Image, Gemini/Imagen, FLUX...).

Why: a frontier-generator image of a Patek Philippe Grandmaster Chime passed listing screening.
DINOv2 (live), DINOv3 epoch_018, the Phase 74 head and two open detectors all called it a real photo
(ml_rtx5080/experiments/external_detectors_v1/report.json).

Development data (training only; never used to report accuracy):
  Thermostatic/frontier-synthetic-images-2026 (40,290 frontier images, Apache-2.0 / CDLA-Permissive-2.0
  per row). Only the text columns are read first; images are fetched just for the rows selected.

    .venv/bin/python ml_rtx5080/frontier_v5.py index      # text columns -> experiments/frontier_v5/metadata.jsonl
    .venv/bin/python ml_rtx5080/frontier_v5.py fetch      # selected rows -> experiments/frontier_v5/train_images/
    .venv/bin/python ml_rtx5080/frontier_v5.py extract    # DINOv3 features -> experiments/frontier_v5/features/frontier.npz
    .venv/bin/python ml_rtx5080/frontier_v5.py design     # development comparison (leave one generator out)
    .venv/bin/python ml_rtx5080/frontier_v5.py freeze     # final head + thresholds from real photos only

The head is the Phase 74 recipe (StandardScaler + logistic regression on frozen DINOv3 epoch_018 features)
with the frontier images added as a further generated source. Every image gets the benchmark
normalisation (short side 512, centre square, JPEG q90) so both classes are handled identically.
"""

from __future__ import annotations

import argparse
import collections
import json
import re
import sys
from pathlib import Path

_PKG = Path(__file__).resolve().parent
OUT = _PKG / "experiments" / "frontier_v5"
META = OUT / "metadata.jsonl"
REPO = "datasets/Thermostatic/frontier-synthetic-images-2026"
TEXT_COLUMNS = ["sample_id", "generator", "prompt", "source_license", "source_repo", "width", "height", "group_id", "phash"]

# Prompts that put a watch (or a wrist, the usual seller photo) in the picture.
WATCH_RE = re.compile(
    r"\b(?:wrist\s?watch\w*|watch(?:es)?|timepiece\w*|chronograph\w*|horolog\w*|rolex|patek|audemars|omega|cartier|"
    r"vacheron|richard mille|lange|tourbillon|watch face|dial)\b",
    re.IGNORECASE,
)
# "watch" as a verb ("watching the sunset", "watch out") is not a watch.
NOT_A_WATCH_RE = re.compile(r"\bwatch(?:ing|ed|es the|ing the)\b|\bwatch (?:out|over|the|as|tv|a movie)\b", re.IGNORECASE)


def _fs():
    from huggingface_hub import HfFileSystem

    return HfFileSystem()


def _index_file(fs, f: str, attempts: int = 4) -> list[dict]:
    """One file's text columns, retried when the connection drops."""
    import time

    for attempt in range(1, attempts + 1):
        try:
            return _read_text_columns(fs, f)
        except Exception as exc:
            if attempt == attempts:
                raise
            print(f"  retry {Path(f).name} ({attempt}/{attempts - 1}): {type(exc).__name__}", flush=True)
            time.sleep(5 * attempt)
    return []


def _read_text_columns(fs, f: str) -> list[dict]:
    import pyarrow.parquet as pq

    rows = []
    with fs.open(f, "rb", block_size=1 << 20, cache_type="readahead") as fh:
        pf = pq.ParquetFile(fh)
        row = 0
        for rg in range(pf.num_row_groups):
            for r in pf.read_row_group(rg, columns=TEXT_COLUMNS).to_pylist():
                r.update(file=f, row_group=rg, row=row)
                rows.append(r)
                row += 1
    return rows


def index(workers: int = 4) -> None:
    """Read only the text columns of every parquet file (no image bytes). Resumable: one shard per file."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    fs = _fs()
    files = sorted(fs.glob(f"{REPO}/data/train-*.parquet"))
    shards = OUT / "metadata_shards"
    shards.mkdir(parents=True, exist_ok=True)
    todo = [f for f in files if not (shards / (Path(f).stem + ".jsonl")).exists()]
    print(f"{len(files)} files, {len(todo)} to read", flush=True)
    with ThreadPoolExecutor(workers) as pool:
        futures = {pool.submit(_index_file, _fs(), f): f for f in todo}
        failed = []
        for done, fut in enumerate(as_completed(futures), 1):
            f = futures[fut]
            try:
                rows = fut.result()
            except Exception as exc:  # leave it for the next run; shards make this resumable
                failed.append(f)
                print(f"  FAILED {Path(f).name}: {type(exc).__name__}", flush=True)
                continue
            tmp = shards / (Path(f).stem + ".jsonl.tmp")
            tmp.write_text("".join(json.dumps(r) + "\n" for r in rows))
            tmp.rename(shards / (Path(f).stem + ".jsonl"))
            print(f"  {done}/{len(todo)} files", flush=True)
    if failed:
        print(f"{len(failed)} files failed; run index again to finish them", flush=True)
        return
    with open(META, "w") as w:
        for shard in sorted(shards.glob("*.jsonl")):
            w.write(shard.read_text())
    print(f"wrote {META.relative_to(_PKG.parent)} ({sum(1 for _ in open(META))} rows)")


def load_meta() -> list[dict]:
    return [json.loads(line) for line in META.read_text().splitlines()]


def is_watch_prompt(prompt: str | None) -> bool:
    p = prompt or ""
    return bool(WATCH_RE.search(p)) and not (NOT_A_WATCH_RE.search(p) and not re.search(r"wrist|timepiece|dial", p, re.I))


N_OTHER_PER_GENERATOR = 60  # non-watch images per generator, for generator diversity
SEED = 75
IMAGES = OUT / "train_images"
MANIFEST = IMAGES / "manifest.jsonl"


def select() -> list[dict]:
    """Every watch-prompt row, plus a seeded sample of N_OTHER_PER_GENERATOR other rows per generator."""
    import random

    rows = load_meta()
    rng = random.Random(SEED)
    picked = [r for r in rows if is_watch_prompt(r["prompt"])]
    by_gen: dict[str, list[dict]] = collections.defaultdict(list)
    for r in rows:
        if not is_watch_prompt(r["prompt"]):
            by_gen[r["generator"]].append(r)
    # Rows sit in large parquet row groups (about 80 MB each). Take each generator's sample from as few
    # row groups as possible: first the groups that already hold its watch rows, then its largest groups.
    watch_groups = {(r["file"], r["row_group"]) for r in picked}
    for gen in sorted(by_gen):
        groups: dict[tuple[str, int], list[dict]] = collections.defaultdict(list)
        for r in by_gen[gen]:
            groups[(r["file"], r["row_group"])].append(r)
        order = sorted(groups, key=lambda k: (k not in watch_groups, -len(groups[k]), k))
        taken: list[dict] = []
        for key in order:
            if len(taken) >= N_OTHER_PER_GENERATOR:
                break
            members = groups[key]
            taken += rng.sample(members, min(len(members), N_OTHER_PER_GENERATOR - len(taken)))
        picked += taken
    return picked


PER_FILE_REPO = "Qwen/Qwen-Image-Bench"  # stores each image as its own file (Apache-2.0)


def _save(r: dict, data: bytes, man, done: set) -> None:
    ext = ".jpg" if data[:3] == b"\xff\xd8\xff" else ".webp" if data[8:12] == b"WEBP" else ".png"
    folder = IMAGES / re.sub(r"[^A-Za-z0-9._-]+", "_", r["generator"])
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{r['sample_id']}{ext}"
    path.write_bytes(data)
    man.write(json.dumps({"sample_id": r["sample_id"], "path": str(path.relative_to(_PKG.parent)),
                          "generator": r["generator"], "group_id": r["group_id"],
                          "watch_prompt": is_watch_prompt(r["prompt"]), "prompt": r["prompt"],
                          "source_license": r["source_license"], "source_repo": r["source_repo"]}) + "\n")
    done.add(r["sample_id"])


def _fetch_per_file(rows: list[dict]) -> None:
    """Read just the locator column for these rows' groups, then download each image file on its own."""
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    done = set()
    if MANIFEST.exists():
        done = {json.loads(line)["sample_id"] for line in MANIFEST.read_text().splitlines()}
    rows = [r for r in rows if r["sample_id"] not in done]
    by_group: dict[tuple[str, int], dict[str, dict]] = collections.defaultdict(dict)
    for r in rows:
        by_group[(r["file"], r["row_group"])][r["sample_id"]] = r
    fs = _fs()
    located = []
    for (f, rg), members in by_group.items():
        with fs.open(f, "rb", block_size=1 << 20, cache_type="readahead") as fh:
            cols = ["sample_id", "original_locator", "source_revision"]
            for t in pq.ParquetFile(fh).read_row_group(rg, columns=cols).to_pylist():
                if t["sample_id"] in members:
                    located.append((members[t["sample_id"]], t["original_locator"], t["source_revision"]))
    print(f"{len(located)} files to download from {PER_FILE_REPO}", flush=True)
    IMAGES.mkdir(parents=True, exist_ok=True)
    tmp = OUT / "tmp_download"
    with open(MANIFEST, "a") as man:
        for n, (r, locator, revision) in enumerate(located, 1):
            local = hf_hub_download(PER_FILE_REPO, locator, revision=revision, repo_type="dataset", local_dir=tmp)
            _save(r, Path(local).read_bytes(), man, done)
            Path(local).unlink(missing_ok=True)
            if n % 100 == 0:
                man.flush()
                print(f"  {n}/{len(located)} files", flush=True)


def fetch(dry_run: bool = False) -> None:
    """Download only the image column of row groups that hold selected rows; keep just those rows."""
    import pyarrow.parquet as pq

    picked = select()
    per_file = [r for r in picked if r["source_repo"] == PER_FILE_REPO]
    if not dry_run:
        _fetch_per_file(per_file)
    wanted: dict[tuple[str, int], dict[str, dict]] = collections.defaultdict(dict)
    for r in picked:
        if r["source_repo"] != PER_FILE_REPO:
            wanted[(r["file"], r["row_group"])][r["sample_id"]] = r
    fs = _fs()
    done = set()
    if MANIFEST.exists():
        done = {json.loads(line)["sample_id"] for line in MANIFEST.read_text().splitlines()}
    todo = {k: v for k, v in wanted.items() if set(v) - done}
    total = 0
    for f, rg in todo:
        with fs.open(f, "rb") as fh:
            md = pq.ParquetFile(fh).metadata.row_group(rg)
            names = [md.column(i).path_in_schema for i in range(md.num_columns)]
            total += sum(md.column(i).total_compressed_size for i, n in enumerate(names) if n.startswith("image"))
    print(f"{len(picked)} rows selected ({sum(is_watch_prompt(r['prompt']) for r in picked)} watch prompts); "
          f"{len(per_file)} fetched one file at a time from {PER_FILE_REPO}; "
          f"{len(todo)} row groups to read, about {total / 1e9:.1f} GB of image data", flush=True)
    if dry_run:
        return
    IMAGES.mkdir(parents=True, exist_ok=True)
    with open(MANIFEST, "a") as man:
        for n, ((f, rg), rows) in enumerate(sorted(todo.items()), 1):
            with fs.open(f, "rb", block_size=1 << 22) as fh:
                table = pq.ParquetFile(fh).read_row_group(rg, columns=["sample_id", "image"]).to_pylist()
            for t in table:
                r = rows.get(t["sample_id"])
                if r is None or t["sample_id"] in done:
                    continue
                _save(r, t["image"]["bytes"], man, done)
            man.flush()
            print(f"  {n}/{len(todo)} row groups, {len(done)} images", flush=True)


FEATURES = OUT / "features"
P74 = _PKG / "experiments" / "kandinsky_v4"
FAKE_BUDGET = 0.01
C = 0.05
CANDIDATES = {
    # name: (feature view, training sources). "coco" adds real non-watch photos, so that "not a watch"
    # cannot stand in for "generated" (nearly all frontier images are not watches; all our reals are).
    "p74_ft": ("ft", ("bench",)),                      # Phase 74 recipe, no frontier images (baseline)
    "front_ft": ("ft", ("bench", "frontier")),
    "front_pre+ft": ("pre+ft", ("bench", "frontier")),
    "front_ft+coco": ("ft", ("bench", "frontier", "coco")),
    "front_pre+ft+coco": ("pre+ft", ("bench", "frontier", "coco")),
}
COCO = OUT / "coco"
# The generator families behind the two test apps (reported separately; selection uses all generators).
FAMILIES = {
    "OpenAI": re.compile(r"gpt|dall-?e|^4o", re.IGNORECASE),
    "Google": re.compile(r"imagen|nano-banana|gemini", re.IGNORECASE),
}
N_COCO = 2000


N_COCO_HOLDOUT = 1000


def coco_rows(holdout: bool = False) -> list[dict]:
    """A seeded sample of N_COCO COCO val2017 photos (real, mostly not watches), unzipped on first use.
    holdout=True gives N_COCO_HOLDOUT different photos that no candidate trains on."""
    import random
    import zipfile

    folder = COCO / "val2017"
    if not folder.exists():
        with zipfile.ZipFile(COCO / "val2017.zip") as z:
            z.extractall(COCO)
    files = sorted(folder.glob("*.jpg"))
    picked = random.Random(SEED).sample(files, N_COCO)
    if holdout:
        rest = sorted(set(files) - set(picked))
        picked = random.Random(SEED + 1).sample(rest, N_COCO_HOLDOUT)
    return [{"sample_id": f.stem, "path": str(f.relative_to(_PKG.parent)), "generator": "real_coco",
             "group_id": f.stem, "watch_prompt": False} for f in picked]


def extract(which: str = "frontier") -> None:
    """DINOv3 features (pretrained and fine-tuned epoch_018) for the frontier images or the COCO reals."""
    import numpy as np

    sys.path.insert(0, str(_PKG))
    import kandinsky_v4 as k4

    rows = (coco_rows() if which == "coco" else coco_rows(holdout=True) if which == "coco_holdout"
            else [json.loads(line) for line in MANIFEST.read_text().splitlines()])
    FEATURES.mkdir(parents=True, exist_ok=True)
    target = FEATURES / f"{which}.npz"
    pre, ft = k4.Pretrained(), k4.FineTuned()
    store: dict[str, list] = collections.defaultdict(list)
    kept = []
    for start in range(0, len(rows), k4.BATCH):
        chunk, images = [], []
        for r in rows[start:start + k4.BATCH]:
            try:
                images.append(k4.jpeg90(k4.square(_PKG.parent / r["path"], 512)))
                chunk.append(r)
            except Exception as exc:
                print(f"  skip {r['path']}: {exc}", flush=True)
        if not chunk:
            continue
        for feats in (pre.features(images), ft.features(images)):
            for k, v in feats.items():
                store[k].append(v)
        kept += chunk
        if (start // k4.BATCH) % 50 == 0:
            print(f"  {start + len(chunk)}/{len(rows)}", flush=True)
    meta = {"id": np.array([r["sample_id"] for r in kept]), "generator": np.array([r["generator"] for r in kept]),
            "group": np.array([r["group_id"] for r in kept]), "watch": np.array([r["watch_prompt"] for r in kept]),
            "label": np.full(len(kept), 0 if which.startswith("coco") else 1, dtype=int)}
    np.savez(target, **meta, **{k: np.concatenate(v) for k, v in store.items()})
    print(f"wrote {target.relative_to(_PKG.parent)}: {len(kept)} images", flush=True)


def _dev():
    """Development features: Phase 74 caches plus the frontier set."""
    sys.path.insert(0, str(_PKG))
    import kandinsky_v4_design as d74
    import numpy as np

    bench = [d74.load(f"bench_{v}") for v in ("v1", "v2", "v3", "v4")]
    train, val_l0 = d74.load("train_l0"), d74.load("val_l0")
    k_train = d74.load("k_train")
    z = np.load(FEATURES / "frontier.npz")
    frontier = {k: z[k] for k in z.files}
    z = np.load(FEATURES / "coco.npz")
    coco = {k: z[k] for k in z.files}
    ebay = d74.concat([d74.subset(b, b["label"] == 0) for b in bench])
    bench_gen = d74.concat([d74.subset(b, b["label"] == 1) for b in bench] + [k_train])
    return d74, train, val_l0, ebay, bench_gen, frontier, coco


def _train_xy(d74, feats, sources, train, ebay, bench_gen, frontier, coco):
    import numpy as np

    reals = [train, ebay] + ([coco] if "coco" in sources else [])
    xs = [d74.view(r, feats) for r in reals]
    ys = [np.zeros(len(r["id"])) for r in reals]
    gens = ([bench_gen] if "bench" in sources else []) + ([frontier] if "frontier" in sources and frontier is not None
                                                          and len(frontier["id"]) else [])
    for g in gens:
        xs.append(d74.view(g, feats))
        ys.append(np.ones(len(g["id"])))
    return np.concatenate(xs), np.concatenate(ys)


def _fit_threshold(d74, feats, sources, train, val_l0, ebay, bench_gen, frontier, coco):
    """Fit, then pick the FAKE threshold on real watch photos only: Label_0 validation + eBay out-of-fold."""
    import numpy as np

    model = d74.fit_lr(*_train_xy(d74, feats, sources, train, ebay, bench_gen, frontier, coco), C)
    folds = np.arange(len(ebay["id"])) % 5
    oof = np.zeros(len(ebay["id"]))
    for f in range(5):
        m = d74.fit_lr(*_train_xy(d74, feats, sources, train, d74.subset(ebay, folds != f), bench_gen, frontier, coco), C)
        oof[folds == f] = m.decision_function(d74.view(d74.subset(ebay, folds == f), feats))
    real = np.concatenate([model.decision_function(d74.view(val_l0, feats)), oof])
    return model, d74.threshold(real), real


def design() -> None:
    """Leave one frontier generator out: train without it, then count how many of its images are caught."""
    import numpy as np

    d74, train, val_l0, ebay, bench_gen, frontier, coco = _dev()
    gens = sorted(set(frontier["generator"]))
    results = {}
    for name, (feats, sources) in CANDIDATES.items():
        caught = n = watch_caught = watch_n = 0
        per_gen = {}
        real_fake = []
        for g in gens:
            held = frontier["generator"] == g
            model, t, real = _fit_threshold(d74, feats, sources, train, val_l0, ebay, bench_gen,
                                            d74.subset(frontier, ~held), coco)
            s = model.decision_function(d74.view(d74.subset(frontier, held), feats))
            w = frontier["watch"][held]
            per_gen[g] = [int((s >= t).sum()), int(held.sum())]
            caught += int((s >= t).sum()); n += int(held.sum())
            watch_caught += int((s[w] >= t).sum()); watch_n += int(w.sum())
            real_fake.append(float((real >= t).mean()))
        family = {fam: [sum(per_gen[g][0] for g in per_gen if FAMILIES[fam].search(g)),
                        sum(per_gen[g][1] for g in per_gen if FAMILIES[fam].search(g))] for fam in FAMILIES}
        results[name] = {"unseen_generator_caught": [caught, n], "unseen_watch_prompts_caught": [watch_caught, watch_n],
                         "real_fake_rate_mean": round(float(np.mean(real_fake)), 4), "families": family,
                         "per_generator": per_gen}
        fam_txt = "  ".join(f"{k} {v[0]}/{v[1]}" for k, v in family.items())
        print(f"{name:18s} held-out generators caught {caught}/{n} ({caught / max(n, 1):.1%}); "
              f"watch prompts {watch_caught}/{watch_n}; {fam_txt}; real FAKE {np.mean(real_fake):.2%}", flush=True)
    (OUT / "design_compare.json").write_text(json.dumps(results, indent=2) + "\n")


def shortcut() -> None:
    """Content-shortcut check: each candidate, fitted on all development data, scored on held-out real COCO photos."""
    import numpy as np

    d74, train, val_l0, ebay, bench_gen, frontier, coco = _dev()
    z = np.load(FEATURES / "coco_holdout.npz")
    holdout = {k: z[k] for k in z.files}
    path = OUT / "design_compare.json"
    results = json.loads(path.read_text())
    for name, (feats, sources) in CANDIDATES.items():
        model, t, _real = _fit_threshold(d74, feats, sources, train, val_l0, ebay, bench_gen, frontier, coco)
        s = model.decision_function(d74.view(holdout, feats))
        k, n = int((s >= t).sum()), len(s)
        results[name]["coco_holdout_fake"] = [k, n]
        print(f"{name:18s} held-out real COCO photos called FAKE: {k}/{n} ({k / n:.1%})", flush=True)
    path.write_text(json.dumps(results, indent=2) + "\n")


COCO_FAKE_BUDGET = 0.05


def freeze() -> None:
    """Pick the candidate by the pre-registered rule, fit it on all development data, record its hash."""
    import hashlib

    import joblib
    import numpy as np

    results = json.loads((OUT / "design_compare.json").read_text())
    eligible = {k: v for k, v in results.items() if v["real_fake_rate_mean"] <= FAKE_BUDGET
                and v["coco_holdout_fake"][0] / v["coco_holdout_fake"][1] <= COCO_FAKE_BUDGET}
    rate = lambda k: eligible[k]["unseen_generator_caught"][0] / max(eligible[k]["unseen_generator_caught"][1], 1)  # noqa: E731
    selected = max(eligible, key=lambda k: (rate(k), CANDIDATES[k][0] == "ft"))
    feats, sources = CANDIDATES[selected]
    d74, train, val_l0, ebay, bench_gen, frontier, coco = _dev()
    model, t_fake, real = _fit_threshold(d74, feats, sources, train, val_l0, ebay, bench_gen, frontier, coco)
    path = OUT / "dinov3_frontier_v5_head.joblib"
    joblib.dump(model, path)
    rule = {
        "rule": "dinov3_frontier_v5", "phase": 75, "selected_candidate": selected, "features": feats,
        "generated_sources": list(sources), "head": f"StandardScaler + logistic regression (C = {C}, balanced classes)",
        "model": {"path": str(path.relative_to(_PKG.parent)), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()},
        "t_fake": t_fake, "threshold_source": "real photos only: Label_0 validation + benchmark_v1-v4 eBay (out-of-fold), at most 1% FAKE",
        "development_real_fake_rate": round(float((real >= t_fake).mean()), 4),
        "normalisation": "RGB, Lanczos short side 512, centre square, JPEG q90",
        "frozen_before_any_v5_scoring": True, "design_results": results,
    }
    (OUT / "frozen_rule.json").write_text(json.dumps(rule, indent=2) + "\n")
    print(f"froze {selected}: t_fake={t_fake:.3f}, dev real FAKE {rule['development_real_fake_rate']:.2%}, sha256 {rule['model']['sha256'][:12]}")


class FrozenHead:
    """The frozen Phase 75 rule: hash-checked head, benchmark normalisation, FAKE if score >= t_fake."""

    def __init__(self) -> None:
        import hashlib

        import joblib

        sys.path.insert(0, str(_PKG))
        import kandinsky_v4 as k4
        import kandinsky_v4_design as d74

        self.rule = json.loads((OUT / "frozen_rule.json").read_text())
        path = _PKG.parent / self.rule["model"]["path"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != self.rule["model"]["sha256"]:
            raise RuntimeError("frozen head does not match its recorded SHA-256")
        self.k4, self.d74 = k4, d74
        self.model = joblib.load(path)
        self.pre, self.ft = k4.Pretrained(), k4.FineTuned()

    def scores(self, paths: list[Path]) -> list[float]:
        import numpy as np

        out = []
        for start in range(0, len(paths), self.k4.BATCH):
            images = [self.k4.jpeg90(self.k4.square(p, 512)) for p in paths[start:start + self.k4.BATCH]]
            feats = {**self.pre.features(images), **self.ft.features(images)}
            out += self.model.decision_function(self.d74.view(feats, self.rule["features"])).tolist()
        return [float(x) for x in np.asarray(out)]

    def verdict(self, score: float) -> str:
        return "FAKE" if score >= self.rule["t_fake"] else "AUTHENTIC"


def score_files(paths: list[str]) -> None:
    """Score individual files with the frozen head (for spot checks such as the motivating case)."""
    head = FrozenHead()
    for p, sc in zip(paths, head.scores([Path(x) for x in paths])):
        print(f"{Path(p).name:36s} score {sc:+.2f} (FAKE at >= {head.rule['t_fake']:+.2f}) -> {head.verdict(sc)}")


def summary() -> None:
    rows = load_meta()
    gen = collections.Counter(r["generator"] for r in rows)
    watch = [r for r in rows if is_watch_prompt(r["prompt"])]
    wgen = collections.Counter(r["generator"] for r in watch)
    print(f"{len(rows)} rows, {len(gen)} generators; {len(watch)} rows with a watch prompt")
    for g, c in gen.most_common():
        print(f"  {g:45s} {c:6d}  watch prompts: {wgen.get(g, 0)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("step", choices=["index", "summary", "fetch", "extract", "extract-coco", "extract-coco-holdout",
                                         "design", "shortcut", "freeze", "score-files"])
    parser.add_argument("--dry-run", action="store_true", help="fetch: only report how much would be downloaded")
    parser.add_argument("files", nargs="*", help="score-files: images to score with the frozen head")
    args = parser.parse_args()
    if args.step == "fetch":
        fetch(args.dry_run)
    elif args.step == "extract-coco":
        extract("coco")
    elif args.step == "score-files":
        score_files(args.files)
    elif args.step == "extract-coco-holdout":
        extract("coco_holdout")
    else:
        {"index": index, "summary": summary, "extract": extract, "design": design, "shortcut": shortcut,
         "freeze": freeze}[args.step]()


if __name__ == "__main__":
    sys.exit(main())
