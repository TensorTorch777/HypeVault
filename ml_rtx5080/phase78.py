"""Phase 78: the Phase 76 head retrained with screenshot-style edited copies, so a crop or screenshot of an AI
image is still called AI.

Same features as Phase 76 (DINOv3 pre+ft view + Perception Encoder L/14-336) and the same training data;
each training image also gets two edited copies (random crop 55-95% at a random position, shrink to 40-100%,
PNG or JPEG q60-95, sometimes brightness/contrast +-15%), normalised like any upload. Same head (standardise,
logistic regression C = 0.05, balanced classes, app images weighted x10), fitted with L-BFGS on the GPU and
frozen as plain arrays. FAKE cut on original real watch photos only (Label_0 validation + benchmark_v1-v5
eBay out-of-fold, at most 1% FAKE), like Phase 76.

The test (fixed before training): benchmark_v6 AI images and benchmark_v6/v7 real eBay photos (none in
training), originals and three edited copies each. Phase 78 passes if it catches more screenshot-style copies
(> 37/60) and 70% crops (> 9/60) than Phase 76, at least 57/60 originals, and flags at most one more original
real photo than Phase 76 (5/224).

    .venv/bin/python ml_rtx5080/phase78.py extract
    .venv/bin/python ml_rtx5080/phase78.py freeze
    .venv/bin/python ml_rtx5080/phase78.py evaluate <screenshot.png>
"""

from __future__ import annotations

import hashlib
import io
import json
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageOps

ML = Path(__file__).resolve().parent
REPO = ML.parent
OUT = ML / "experiments" / "phase78"
AUG = OUT / "aug_features.npz"
RULE = OUT / "frozen_rule.json"
HEAD = OUT / "phase78_head.npz"
sys.path.insert(0, str(ML))
sys.path.insert(0, str(ML / "benchmark_v1_real_vs_generated"))

COPIES = 2
REPEAT = 10
C = 0.05
FAKE_BUDGET = 0.01
COCO_FAKE_BUDGET = 0.05
VIEW = "pre+ft"


def normalise(img: Image.Image) -> Image.Image:
    """What every upload gets: RGB, centre square, Lanczos 512 px, JPEG q90 (FrozenFusionHead.score_image)."""
    import kandinsky_v4 as k4

    img = img.convert("RGB")
    s = min(img.size)
    left, top = (img.width - s) // 2, (img.height - s) // 2
    return k4.jpeg90(img.crop((left, top, left + s, top + s)).resize((512, 512), Image.LANCZOS))


def random_edit(img: Image.Image, seed: str) -> Image.Image:
    rng = random.Random(seed)
    img = img.convert("RGB")
    img.info = {}  # a screenshot carries no source metadata (an oversized ICC profile once broke the PNG reload)
    W, H = img.size
    f = rng.uniform(0.55, 0.95)
    w, h = max(64, int(W * f)), max(64, int(H * f))
    x, y = rng.randint(0, W - w), rng.randint(0, H - h)
    img = img.crop((x, y, x + w, y + h))
    s = rng.uniform(0.4, 1.0)
    img = img.resize((max(64, int(w * s)), max(64, int(h * s))), rng.choice([Image.BILINEAR, Image.BICUBIC, Image.LANCZOS]))
    if rng.random() < 0.3:
        img = ImageEnhance.Brightness(ImageEnhance.Contrast(img).enhance(rng.uniform(0.85, 1.15))).enhance(rng.uniform(0.85, 1.15))
    buf = io.BytesIO()
    if rng.random() < 0.5:
        img.save(buf, "PNG")
    else:
        img.save(buf, "JPEG", quality=rng.randint(60, 95))
    return Image.open(io.BytesIO(buf.getvalue())).convert("RGB")


class Features:
    """Phase 76 features: DINOv3 pretrained + fine-tuned (pre+ft view) and Perception Encoder L/14-336."""

    def __init__(self) -> None:
        import fusion_probe as fp
        import kandinsky_v4 as k4
        import kandinsky_v4_design as d74

        self.k4, self.d74 = k4, d74
        self.pre, self.ft, self.enc = k4.Pretrained(), k4.FineTuned(), fp.Encoders(("pe",))

    def __call__(self, images: list[Image.Image]) -> np.ndarray:
        out = []
        for s in range(0, len(images), self.k4.BATCH):
            chunk = images[s:s + self.k4.BATCH]
            feats = {**self.pre.features(chunk), **self.ft.features(chunk)}
            out.append(np.concatenate([self.d74.view(feats, VIEW), self.enc(chunk)["pe"]], axis=1))
        return np.concatenate(out)


def _original_paths() -> dict[str, dict[str, Path]]:
    """Source file for every cached training image, per set (the paths fusion_probe loads from)."""
    import fusion_probe as fp
    import phase77_probe as p7

    out = {}
    for name, (ids, load) in fp._sets().items():
        paths = load.__defaults__[0]
        out[name] = {i: Path(paths[i]) for i in ids}
    v5_ids, v5_load = p7._v5_items()
    from common import read_manifest

    b5 = ML / "benchmark_v5_frontier"
    v5_paths = {s.sample_id: (b5 / s.path if not Path(s.path).is_absolute() else Path(s.path))
                for s in read_manifest(b5 / "data" / "manifest.jsonl") if s.excluded is None}
    out["v5"] = {i: v5_paths[i] for i in v5_ids if i in v5_paths}
    return out


TRAIN_SETS = ("train_l0", "coco", "frontier", "k_train", "bench_v1", "bench_v2", "bench_v3", "bench_v4", "v5")


def extract() -> None:
    from concurrent.futures import ThreadPoolExecutor

    OUT.mkdir(parents=True, exist_ok=True)
    if AUG.exists():
        raise SystemExit(f"{AUG} exists")
    paths = _original_paths()
    jobs = [(name, i, c) for name in TRAIN_SETS for i in paths[name] for c in range(COPIES)]
    print(len(jobs), "edited copies", flush=True)
    feat = Features()

    def make(job):
        name, i, c = job
        try:
            return normalise(random_edit(ImageOps.exif_transpose(Image.open(paths[name][i])), f"{i}|{c}"))
        except Exception as exc:  # recorded and dropped, never silently replaced
            print("skip", name, i, c, type(exc).__name__, flush=True)
            return None

    parts = OUT / "aug_parts"
    parts.mkdir(exist_ok=True)
    step = 64 * 50
    with ThreadPoolExecutor(8) as pool:
        for s in range(0, len(jobs), step):
            target = parts / f"part_{s:06d}.npz"
            if target.exists():
                continue
            chunk = jobs[s:s + step]
            images = list(pool.map(make, chunk))
            ok = [k for k, im in enumerate(images) if im is not None]
            xs = [feat([images[k] for k in ok[b:b + 64]]) for b in range(0, len(ok), 64)]
            np.savez(target, set=np.array([chunk[k][0] for k in ok]), id=np.array([chunk[k][1] for k in ok]),
                     copy=np.array([chunk[k][2] for k in ok]), x=np.concatenate(xs).astype(np.float32))
            print(s + len(chunk), "/", len(jobs), flush=True)
    zs = [np.load(f) for f in sorted(parts.glob("part_*.npz"))]
    np.savez(AUG, **{k: np.concatenate([z[k] for z in zs]) for k in ("set", "id", "copy", "x")})
    print("saved", AUG, flush=True)


def _phase76_training():
    """Phase 76's training sets as (x, label, weight, set name, id, is_ebay_real), from the cached features."""
    import fusion_probe as fp
    import frontier_v5 as f5
    import kandinsky_v4_design as d74

    feats75 = f5.CANDIDATES["front_pre+ft+coco"][0]
    assert feats75 == VIEW

    def with_pe(d, name):
        z = np.load(fp.CACHE / f"{name}.npz")
        assert z["id"].tolist() == d["id"].tolist(), name
        return np.concatenate([d74.view(d, feats75), z["pe"]], axis=1)

    _, train, val_l0, _, _, frontier, coco = f5._dev()
    rows = []

    def add(name, d, x, label, weight=1.0, ebay=None):
        n = len(d["id"])
        rows.append({"set": np.full(n, name), "id": np.asarray(d["id"]), "x": x, "y": np.asarray(label) * np.ones(n),
                     "w": np.full(n, weight), "ebay": np.zeros(n, bool) if ebay is None else ebay})

    add("train_l0", train, with_pe(train, "train_l0"), 0)
    add("coco", coco, with_pe(coco, "coco"), 0)
    add("frontier", frontier, with_pe(frontier, "frontier"), 1)
    k = d74.load("k_train")
    add("k_train", k, with_pe(k, "k_train"), 1)
    for v in ("v1", "v2", "v3", "v4"):
        b = d74.load(f"bench_{v}")
        lab = b["label"].astype(int)
        add(f"bench_{v}", b, with_pe(b, f"bench_{v}"), lab, ebay=lab == 0)
    v5z = np.load(ML / "experiments" / "phase76_probe" / "v5_features.npz")
    v5 = {k_: v5z[k_] for k_ in v5z.files}
    bz = np.load(ML / "experiments" / "phase76_probe" / "backbone_features.npz")
    g = v5["group"]
    keep = np.isin(g, ["chatgpt_v5", "gemini_v5", "real_ebay_v5"])
    x5 = np.concatenate([d74.view(v5, feats75), bz["pe_core_large"]], axis=1)[keep]
    lab5 = np.isin(g[keep], ["chatgpt_v5", "gemini_v5"]).astype(int)
    rows.append({"set": np.full(keep.sum(), "v5"), "id": v5["id"][keep], "x": x5, "y": lab5.astype(float),
                 "w": np.where(lab5 == 1, REPEAT, 1.0), "ebay": lab5 == 0})
    val = {"x": with_pe(val_l0, "val_l0")}
    z = np.load(f5.FEATURES / "coco_holdout.npz")
    hold = {k_: z[k_] for k_ in z.files}
    val["coco_holdout"] = with_pe(hold, "coco_holdout")
    cat = {k_: np.concatenate([r[k_] for r in rows]) for k_ in rows[0]}
    return cat, val


def _fit(x, y, w):
    from phase77_compare import GpuLR

    m = GpuLR(x, y, w, c=C)
    return {"mu": m.mu.cpu().numpy(), "sd": m.sd.cpu().numpy(), "w": m.w.cpu().numpy(), "b": float(m.b.cpu().numpy()[0])}


def decision(head: dict, x: np.ndarray) -> np.ndarray:
    return ((x - head["mu"]) / head["sd"]) @ head["w"] + head["b"]


def freeze() -> None:
    if RULE.exists():
        raise SystemExit(f"{RULE.relative_to(REPO)} exists; Phase 78 is already frozen")
    base, val = _phase76_training()
    z = np.load(AUG)
    key = {(s, i): k for k, (s, i) in enumerate(zip(base["set"].tolist(), base["id"].tolist()))}
    src = np.array([key[(s, i)] for s, i in zip(z["set"].tolist(), z["id"].tolist())])
    x = np.concatenate([base["x"], z["x"]])
    y = np.concatenate([base["y"], base["y"][src]])
    w = np.concatenate([base["w"], base["w"][src]])
    owner = np.concatenate([np.arange(len(base["y"])), src])  # which original each row comes from
    head = _fit(x, y, w)

    ebay = np.flatnonzero(base["ebay"])
    folds = np.random.default_rng(78).integers(0, 5, len(ebay))
    oof = []
    for f in range(5):
        held = ebay[folds == f]
        keep = ~np.isin(owner, held)
        m = _fit(x[keep], y[keep], w[keep])
        oof.append(decision(m, base["x"][held]))
    real = np.concatenate([decision(head, val["x"])] + oof)
    grid = np.unique(real)[::-1]
    t_fake = float(min(t for t in grid if (real >= t).mean() <= FAKE_BUDGET))
    coco_fake = float((decision(head, val["coco_holdout"]) >= t_fake).mean())
    if coco_fake > COCO_FAKE_BUDGET:
        raise SystemExit(f"not eligible: {coco_fake:.1%} of held-out COCO photos FAKE")
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(HEAD, **head)
    rule = {
        "rule": "phase78", "phase": 78, "features": "Phase 76: DINOv3 pre+ft view + Perception Encoder L/14-336",
        "head": f"standardise + logistic regression (C = {C}, balanced, app images x{REPEAT}), L-BFGS on GPU, frozen as arrays",
        "training": {"originals": int(len(base["y"])), "edited_copies": int(len(src)), "copies_per_image": COPIES,
                     "edit": "random crop 55-95% at a random position, shrink 40-100%, PNG or JPEG q60-95, 30% brightness/contrast +-15%"},
        "model": {"path": str(HEAD.relative_to(REPO)), "sha256": hashlib.sha256(HEAD.read_bytes()).hexdigest()},
        "t_fake": t_fake,
        "threshold_source": "original real watch photos: Label_0 validation + benchmark_v1-v5 eBay (5-fold out-of-fold), at most 1% FAKE",
        "development_real_fake_rate": round(float((real >= t_fake).mean()), 4),
        "coco_holdout_fake_rate": round(coco_fake, 4),
    }
    RULE.write_text(json.dumps(rule, indent=2) + "\n")
    print(json.dumps({k: rule[k] for k in ("t_fake", "development_real_fake_rate", "coco_holdout_fake_rate")}, indent=1))


def load_head() -> tuple[dict, float]:
    """The frozen Phase 78 arrays and cut, hash-checked. It reads Phase 76's features, so no model is loaded here."""
    rule = json.loads(RULE.read_text())
    path = REPO / rule["model"]["path"]
    if hashlib.sha256(path.read_bytes()).hexdigest() != rule["model"]["sha256"]:
        raise RuntimeError("Phase 78 head does not match its recorded SHA-256")
    z = np.load(path)
    return {k: z[k] for k in z.files}, float(rule["t_fake"])


class FrozenPhase78Head:
    def __init__(self, features: Features | None = None) -> None:
        self.rule = json.loads(RULE.read_text())
        path = REPO / self.rule["model"]["path"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != self.rule["model"]["sha256"]:
            raise RuntimeError("Phase 78 head does not match its recorded SHA-256")
        z = np.load(path)
        self.head = {k: z[k] for k in z.files}
        self.features = features or Features()

    def score_image(self, img: Image.Image) -> float:
        return float(decision(self.head, self.features([normalise(img)]))[0])


def evaluate(screenshot: str) -> None:
    import joblib

    import phase76

    rule76 = json.loads(phase76.RULE.read_text())
    head76 = joblib.load(REPO / rule76["model"]["path"])
    head78 = FrozenPhase78Head.__new__(FrozenPhase78Head)
    head78.rule = json.loads(RULE.read_text())
    z = np.load(REPO / head78.rule["model"]["path"])
    h78 = {k: z[k] for k in z.files}
    feat = Features()

    def edits(img):
        W, H = img.size
        def crop(f):
            w, h = int(W * f), int(H * f)
            return img.crop(((W - w) // 2, (H - h) // 2, (W + w) // 2, (H + h) // 2))
        a = crop(0.85)
        a = a.resize((int(a.width * 0.8), int(a.height * 0.8)), Image.BILINEAR)
        b = io.BytesIO()
        a.save(b, "PNG")
        return {"screenshot_style": Image.open(io.BytesIO(b.getvalue())).convert("RGB"), "crop70": crop(0.70),
                "half_size": img.resize((W // 2, H // 2), Image.BILINEAR)}

    items = []
    for bench, groups, label in (("benchmark_v6_fresh", {"chatgpt_v6", "gemini_v6", "gemini_edu_v6"}, 1),
                                 ("benchmark_v6_fresh", {"real_ebay_v6"}, 0), ("benchmark_v7_fresh", {"real_ebay_v7"}, 0)):
        for line in (ML / bench / "data" / "manifest.jsonl").read_text().splitlines():
            r = json.loads(line)
            if r.get("excluded") or not r.get("path") or r["group"] not in groups:
                continue
            p = Path(r["path"])
            img = Image.open(p if p.is_absolute() else ML / bench / p).convert("RGB")
            items.append((label, "original", img))
            items += [(label, k, v) for k, v in edits(img).items()]
    items.append((1, "your_screenshot", Image.open(screenshot).convert("RGB")))
    items.append((1, "grandmaster_original", Image.open(REPO / "testtest.jpeg").convert("RGB")))
    x = feat([normalise(im) for _, _, im in items])
    s76 = head76.decision_function(x) - rule76["t_fake"]
    s78 = decision(h78, x) - head78.rule["t_fake"]
    labels = np.array([i[0] for i in items])
    kinds = np.array([i[1] for i in items])
    rep = {}
    for lab, name in ((1, "AI caught"), (0, "real photos wrongly flagged")):
        for k in ("original", "screenshot_style", "crop70", "half_size", "your_screenshot", "grandmaster_original"):
            sel = (labels == lab) & (kinds == k)
            if sel.any():
                rep[f"{name} · {k}"] = {"phase76": f"{int((s76[sel] >= 0).sum())}/{int(sel.sum())}",
                                        "phase78": f"{int((s78[sel] >= 0).sum())}/{int(sel.sum())}"}
    ai_o, re_o = (labels == 1) & (kinds == "original"), (labels == 0) & (kinds == "original")
    def caught(s, k):
        sel = (labels == 1) & (kinds == k)
        return int((s[sel] >= 0).sum())
    passed = (caught(s78, "screenshot_style") > 37 and caught(s78, "crop70") > 9 and caught(s78, "original") >= 57
              and int((s78[re_o] >= 0).sum()) <= int((s76[re_o] >= 0).sum()) + 1)
    rep["pass_rule_met"] = bool(passed)
    rep["auc_original_ai_vs_real"] = {n: round(float(np.mean([(a > r) + 0.5 * (a == r) for a in s[ai_o] for r in s[re_o]])), 4)
                                      for n, s in (("phase76", s76), ("phase78", s78))}
    (OUT / "screenshot_benchmark.json").write_text(json.dumps(rep, indent=1) + "\n")
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    {"extract": extract, "freeze": freeze}.get(sys.argv[1], lambda: evaluate(sys.argv[2]))()
