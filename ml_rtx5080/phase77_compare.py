"""Phase 77 development comparison (exploratory; written before it was first run).

Each candidate is (features, training data). For every generator family F in the development pool,
the head is trained without any image of F, the FAKE cut is set on Label_0 validation real watch
photos (at most 1% FAKE), and the share of F's images caught is recorded.

Selection rule, fixed here before any result:
  1. eligible: COCO hold-out FAKE <= 5% and real eBay (benchmark_v1-v6, 5-fold out-of-fold) FAKE <= 3%;
  2. best = highest macro mean caught over families; within 1 point, the one with fewer feature dimensions.
Phase 76 itself (its features, its training data) is the reference row.

The heads here are the Phase 76 head (standardise, then logistic regression, C = 0.05, balanced
classes, app images weighted x10) fitted with L-BFGS on the GPU for speed; the frozen head uses
scikit-learn, as Phase 76 did.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ML = Path(__file__).resolve().parent
REPO = ML.parent
OUT = ML / "experiments" / "phase77_probe"
CACHE = OUT / "features"
sys.path.insert(0, str(ML))

import numpy as np  # noqa: E402

C = 0.05
REPEAT = 10
FAKE_BUDGET = 0.01
SEED = 77

FEATURES = {
    "p76": ("d3", "pe"),
    "d3+pe512": ("d3", "pe512"),
    "d3+pe+pe512": ("d3", "pe", "pe512"),
    "d3+pe512+mid": ("d3", "pe512", "pe512_m12", "pe512_m18"),
    "d3+pe+pe512+mid": ("d3", "pe", "pe512", "pe512_m12", "pe512_m18"),
}
DATA = ("p76", "p77")  # p77 adds benchmark_v6 (spent) and OpenFake development images

FAMILY_RULES = [
    ("OpenAI", r"gpt|dall-?e|^4o|chatgpt"), ("Google", r"imagen|nano-banana|gemini"),
    ("FLUX", r"flux|chroma"), ("Midjourney", r"midjourney"), ("ByteDance", r"seedream"),
    ("Qwen", r"qwen"), ("Hunyuan", r"hunyuan"), ("Recraft", r"recraft"), ("Ideogram", r"ideogram"),
    ("xAI", r"aurora|grok"), ("Kling", r"kling"), ("GLM", r"glm"), ("Janus", r"janus"), ("Sana", r"sana"),
    ("Z-Image", r"z-image"), ("HiDream", r"hidream"), ("Lumina", r"lumina"), ("Kolors", r"kolors"),
    ("Playground", r"playground"), ("Anime-SDXL", r"anima"), ("Pony-SDXL", r"pony"),
    ("Stable Cascade", r"cascade"), ("aMUSEd", r"amused"), ("Kandinsky", r"kandinsky"),
    ("SD3", r"sd-3|stable-diffusion-3"),
    ("SD-family", r"sd|stable-diffusion|juggernaut|realvis|realistic_vision|cyberrealistic"),
]


def family(generator: str) -> str:
    for name, pattern in FAMILY_RULES:
        if re.search(pattern, generator, re.IGNORECASE):
            return name
    return generator  # its own family (frames, halfmoon, ...)


def load() -> dict:
    """Every development set as dicts with features d3, pe, pe512, pe512_m12, pe512_m18 and meta (label, family, role)."""
    import frontier_v5 as f5
    import kandinsky_v4_design as d74
    import phase77_openfake as of
    import phase77_probe as p7

    view = f5.CANDIDATES["front_pre+ft+coco"][0]
    fusion = ML / "experiments" / "phase76_probe" / "fusion_features"

    def new(name):
        z = np.load(CACHE / f"{name}.npz")
        return {k: z[k] for k in z.files}

    def attach(d, name, pe_from=None):
        z = new(name)
        assert z["id"].tolist() == d["id"].tolist(), name
        out = {"id": d["id"], "d3": d74.view(d, view) if "d3" not in d else d["d3"],
               **{k: z[k] for k in ("pe512", "pe512_m12", "pe512_m18")}}
        if pe_from is not None:
            out["pe"] = pe_from
        else:
            zp = np.load(fusion / f"{name}.npz")
            assert zp["id"].tolist() == d["id"].tolist(), name
            out["pe"] = zp["pe"]
        return out

    def meta(d, label, fam, role, weight=1.0):
        n = len(d["id"])
        d["label"] = np.full(n, label) if np.isscalar(label) else np.asarray(label)
        d["family"] = np.full(n, fam, dtype=object) if isinstance(fam, str) else np.asarray(fam, dtype=object)
        d["role"] = np.full(n, role, dtype=object)
        d["weight"] = np.full(n, weight, dtype=np.float32)
        return d

    _, train, val_l0, _, _, frontier, coco = f5._dev()
    sets = {
        "train_l0": meta(attach(train, "train_l0"), 0, "real", "train_real"),
        "val_l0": meta(attach(val_l0, "val_l0"), 0, "real", "cut_real"),
        "coco": meta(attach(coco, "coco"), 0, "real", "train_real"),
    }
    z = np.load(f5.FEATURES / "coco_holdout.npz")
    sets["coco_holdout"] = meta(attach({k: z[k] for k in z.files}, "coco_holdout"), 0, "real", "holdout_real")
    sets["frontier"] = meta(attach(frontier, "frontier"), 1, [family(g) for g in frontier["generator"]], "gen")
    bench_fam = {"heldout_kandinsky22": "Kandinsky", "sdxl_base10": "SD-family", "amused512": "aMUSEd",
                 "stable_cascade_lite": "Stable Cascade", "stable_cascade_v4": "Stable Cascade"}
    for v in ("v1", "v2", "v3", "v4"):
        b = d74.load(f"bench_{v}")
        a = attach(b, f"bench_{v}")
        lab = b["label"].astype(int)
        fam = [("real" if l == 0 else bench_fam.get(g, "Kandinsky")) for l, g in zip(lab, b["group"])]
        sets[f"bench_{v}"] = meta(a, lab, fam, "bench")
    k = d74.load("k_train")
    sets["k_train"] = meta(attach(k, "k_train"), 1, "Kandinsky", "gen")

    v5z = np.load(ML / "experiments" / "phase76_probe" / "v5_features.npz")
    v5 = {k: v5z[k] for k in v5z.files}
    bz = np.load(ML / "experiments" / "phase76_probe" / "backbone_features.npz")
    a = attach(v5, "v5", pe_from=bz["pe_core_large"])
    g = v5["group"]
    fam = np.where(g == "chatgpt_v5", "OpenAI", np.where(g == "gemini_v5", "Google", "real"))
    lab = np.isin(g, ["chatgpt_v5", "gemini_v5"]).astype(int)
    keep = g != "motivating"
    sets["v5"] = meta({k: v[keep] for k, v in a.items()}, lab[keep], fam[keep], "app_v5")
    sets["v5"]["weight"] = np.where(sets["v5"]["label"] == 1, REPEAT, 1).astype(np.float32)
    sets["testtest"] = meta({k: v[~keep] for k, v in a.items()}, 1, "OpenAI-or-Google", "motivating")

    v6 = new("v6")
    v6rows = {r["id"]: r for r in p7._v6_rows()}
    g6 = np.array([v6rows[i]["group"] for i in v6["id"]])
    d = {"id": v6["id"], "d3": d74.view(v6, view), **{k: v6[k] for k in ("pe", "pe512", "pe512_m12", "pe512_m18")}}
    lab = (g6 != "real_ebay_v6").astype(int)
    fam = np.where(g6 == "chatgpt_v6", "OpenAI", np.where(lab == 1, "Google", "real"))
    sets["v6"] = meta(d, lab, fam, "app_v6")
    sets["v6"]["weight"] = np.where(lab == 1, REPEAT, 1).astype(np.float32)

    of_path = CACHE / "openfake_dev.npz"
    if of_path.exists():
        od = new("openfake_dev")
        rows = {r["row_id"]: r for r in of.manifest_rows(of.OUT / "dev")}
        lab = np.array([int(rows[i]["label"] == "fake") for i in od["id"]])
        fam = [family(rows[i]["model"]) if l else "real" for i, l in zip(od["id"], lab)]
        d = {"id": od["id"], "d3": d74.view(od, view), **{k: od[k] for k in ("pe", "pe512", "pe512_m12", "pe512_m18")}}
        sets["openfake_dev"] = meta(d, lab, fam, "openfake")
    return sets


class GpuLR:
    """StandardScaler + L2 logistic regression (sklearn's objective: 0.5|w|^2 + C * sum_i s_i * loss_i)."""

    def __init__(self, x, y, s, c=C):
        import torch

        self.torch = torch
        x = torch.as_tensor(x, device="cuda", dtype=torch.float32)
        y = torch.as_tensor(y, device="cuda", dtype=torch.float32)
        s = torch.as_tensor(s, device="cuda", dtype=torch.float32)
        # balanced classes on the weighted counts, as class_weight="balanced" on repeated rows
        n, pos = s.sum(), (s * y).sum()
        s = s * torch.where(y > 0, n / (2 * pos), n / (2 * (n - pos)))
        self.mu, self.sd = x.mean(0), x.std(0).clamp_min(1e-6)
        x = (x - self.mu) / self.sd
        w = torch.zeros(x.shape[1], device="cuda", requires_grad=True)
        b = torch.zeros(1, device="cuda", requires_grad=True)
        opt = torch.optim.LBFGS([w, b], lr=1, max_iter=500, history_size=20, tolerance_grad=1e-6,
                                tolerance_change=1e-9, line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad()
            z = x @ w + b
            loss = 0.5 * (w * w).sum() + c * (s * torch.nn.functional.binary_cross_entropy_with_logits(z, y, reduction="none")).sum()
            loss.backward()
            return loss

        opt.step(closure)
        self.w, self.b = w.detach(), b.detach()

    def decision_function(self, x):
        torch = self.torch
        x = (torch.as_tensor(x, device="cuda", dtype=torch.float32) - self.mu) / self.sd
        return (x @ self.w + self.b).cpu().numpy()


def _cat(ds, keys):
    return np.concatenate([np.concatenate([d[k] for k in keys], axis=1) for d in ds])


def _pool(sets, data):
    """Training pool for a data variant: (list of dicts)."""
    from kandinsky_v4_design import subset

    pool = [sets["train_l0"], sets["coco"], sets["k_train"], sets["frontier"], sets["v5"]]
    pool += [sets[f"bench_{v}"] for v in ("v1", "v2", "v3", "v4")]
    if data == "p77":
        pool.append(sets["v6"])
        if "openfake_dev" in sets:
            pool.append(sets["openfake_dev"])
    return [subset(d, np.ones(len(d["id"]), bool)) for d in pool]


def run() -> dict:
    from kandinsky_v4_design import subset

    sets = load()
    rng = np.random.default_rng(SEED)
    gen_sets = [s for s in ("frontier", "k_train", "v5", "v6", "openfake_dev", "bench_v1", "bench_v2", "bench_v3", "bench_v4")
                if s in sets]
    families = sorted({f for s in gen_sets for f, l in zip(sets[s]["family"], sets[s]["label"]) if l == 1})
    ebay_ids = np.concatenate([sets[s]["id"][sets[s]["label"] == 0] for s in ("bench_v1", "bench_v2", "bench_v3", "bench_v4", "v5", "v6")])
    ebay_fold = dict(zip(ebay_ids.tolist(), rng.integers(0, 5, len(ebay_ids)).tolist()))
    results = {}
    for data in DATA:
        pool = _pool(sets, data)
        for fname, keys in FEATURES.items():
            name = f"{fname} / {data}"
            row = {"features": list(keys), "data": data, "caught": {}}
            dims = None
            for fam in families:
                tr = [subset(d, d["family"] != fam) for d in pool]
                x, y = _cat(tr, keys), np.concatenate([d["label"] for d in tr])
                dims = x.shape[1]
                m = GpuLR(x, y, np.concatenate([d["weight"] for d in tr]))
                t = _cut(m.decision_function(_cat([sets["val_l0"]], keys)))
                test = [subset(sets[s], (sets[s]["family"] == fam) & (sets[s]["label"] == 1)) for s in gen_sets]
                sc = m.decision_function(_cat([d for d in test if len(d["id"])], keys))
                row["caught"][fam] = [int((sc >= t).sum()), int(len(sc))]
            # full model: COCO hold-out, testtest; 5-fold eBay out-of-fold
            m = GpuLR(_cat(pool, keys), np.concatenate([d["label"] for d in pool]), np.concatenate([d["weight"] for d in pool]))
            t = _cut(m.decision_function(_cat([sets["val_l0"]], keys)))
            row["coco_holdout_fake"] = float((m.decision_function(_cat([sets["coco_holdout"]], keys)) >= t).mean())
            row["testtest_margin"] = round(float(m.decision_function(_cat([sets["testtest"]], keys))[0] - t), 3)
            # eBay reals of the v6 set are trained on only in p77; out-of-fold covers them either way
            fp = []
            for f in range(5):
                def in_fold(d):
                    return (d["label"] == 0) & np.array([ebay_fold.get(i, -1) == f for i in d["id"].tolist()])
                tr = [subset(d, ~in_fold(d)) for d in pool]
                mm = GpuLR(_cat(tr, keys), np.concatenate([d["label"] for d in tr]), np.concatenate([d["weight"] for d in tr]))
                tt = _cut(mm.decision_function(_cat([sets["val_l0"]], keys)))
                held = [subset(sets[s], in_fold(sets[s])) for s in ("bench_v1", "bench_v2", "bench_v3", "bench_v4", "v5", "v6")]
                fp += (mm.decision_function(_cat([h for h in held if len(h["id"])], keys)) >= tt).tolist()
            row["ebay_oof_fake"] = [int(sum(fp)), len(fp)]
            rates = [k / n for k, n in row["caught"].values() if n]
            row["macro_caught"] = round(float(np.mean(rates)), 4)
            row["dims"] = dims
            results[name] = row
            print(f"{name:28s} macro {row['macro_caught']:.3f}  OpenAI {_r(row, 'OpenAI')}  Google {_r(row, 'Google')}  "
                  f"eBay FAKE {row['ebay_oof_fake'][0]}/{row['ebay_oof_fake'][1]}  COCO {row['coco_holdout_fake']:.3f}  "
                  f"testtest {row['testtest_margin']:+.2f}", flush=True)
    return results


def _r(row, fam):
    k, n = row["caught"].get(fam, [0, 0])
    return f"{k}/{n}"


def _cut(real_scores: np.ndarray) -> float:
    grid = np.unique(real_scores)[::-1]
    return float(min(t for t in grid if (real_scores >= t).mean() <= FAKE_BUDGET))


def main() -> None:
    results = run()
    eligible = {k: v for k, v in results.items()
                if v["coco_holdout_fake"] <= 0.05 and v["ebay_oof_fake"][0] <= 0.03 * v["ebay_oof_fake"][1]}
    best = max(eligible.values(), key=lambda v: v["macro_caught"]) if eligible else None
    pick = None
    if best:
        close = [k for k, v in eligible.items() if v["macro_caught"] >= best["macro_caught"] - 0.01]
        pick = min(close, key=lambda k: results[k]["dims"])
    (OUT / "compare.json").write_text(json.dumps({"results": results, "eligible": sorted(eligible), "selected": pick}, indent=1) + "\n")
    print("selected:", pick)
