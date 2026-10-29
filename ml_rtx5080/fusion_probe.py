"""Phase 76 fusion probe (exploratory): do Perception Encoder and CLIP features add to DINOv3?

Same design as phase76_probe.py: the Phase 75 training data plus one app's benchmark_v5 watch
images (x10), FAKE threshold on real photos only (at most 1%), tested on the other app's images,
all 144 real_ebay_v5 photos (never trained on) and 1,000 held-out COCO photos.
benchmark_v5 is spent, so this is development evidence, not a confirmatory result.

    .venv/bin/python ml_rtx5080/fusion_probe.py extract
    .venv/bin/python ml_rtx5080/fusion_probe.py compare
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ML = Path(__file__).resolve().parent
REPO = ML.parent
OUT = ML / "experiments" / "phase76_probe"
CACHE = OUT / "fusion_features"
sys.path.insert(0, str(ML))

import numpy as np  # noqa: E402

BACKBONES = ("pe", "clip")
PE_MODEL = "vit_pe_core_large_patch14_336.fb"
REPEAT = 10
COMBOS = {"dinov3": ("d3",), "pe": ("pe",), "clip": ("clip",), "dinov3+pe": ("d3", "pe"),
          "dinov3+clip": ("d3", "clip"), "dinov3+pe+clip": ("d3", "pe", "clip")}


def _sets():
    """Every development set as (cached DINOv3 ids, id -> image loader), in cache order."""
    import frontier_v5 as f5
    import kandinsky_v4 as k4
    import kandinsky_v4_design as d74

    sq = lambda p: k4.jpeg90(k4.square(p, 512))  # noqa: E731
    out = {}
    for name in ("train_l0", "val_l0", "k_train", "bench_v1", "bench_v2", "bench_v3", "bench_v4"):
        ids = d74.load(name)["id"].tolist()
        if name in ("train_l0", "val_l0"):
            paths = {i: REPO / i for i in ids}
            out[name] = (ids, lambda i, p=paths: sq(p[i]))
        else:
            if name == "k_train":
                rows = k4.k_train_rows()
            else:
                root = ML / ("benchmark_v1_real_vs_generated" if name == "bench_v1" else f"benchmark_{name[-2:]}_fresh")
                k4.BENCH.setdefault(name[-2:], root)
                rows = k4.bench_rows(name[-2:])
            paths = {r["id"]: r["path"] for r in rows}
            out[name] = (ids, lambda i, p=paths: k4.stored(p[i]))
    for name in ("frontier", "coco", "coco_holdout"):
        ids = np.load(f5.FEATURES / f"{name}.npz")["id"].tolist()
        rows = (f5.coco_rows() if name == "coco" else f5.coco_rows(holdout=True) if name == "coco_holdout"
                else [json.loads(x) for x in f5.MANIFEST.read_text().splitlines()])
        paths = {r["sample_id"]: REPO / r["path"] for r in rows}
        out[name] = (ids, lambda i, p=paths: sq(p[i]))
    return out


class Encoders:
    def __init__(self, which: tuple[str, ...] = BACKBONES) -> None:
        import timm
        import torch

        self.torch, self.which = torch, which
        if "pe" in which:
            self.pe = timm.create_model(PE_MODEL, pretrained=True, num_classes=0).cuda().eval().half()
            self.pe_tf = timm.data.create_transform(**timm.data.resolve_data_config({}, model=self.pe))
        if "clip" in which:
            from transformers import AutoProcessor, CLIPModel

            self.clip = CLIPModel.from_pretrained("openai/clip-vit-large-patch14", dtype=torch.float16).cuda().eval()
            self.clip_proc = AutoProcessor.from_pretrained("openai/clip-vit-large-patch14")

    def __call__(self, images) -> dict[str, np.ndarray]:
        torch = self.torch
        out = {}
        with torch.no_grad():
            if "pe" in self.which:
                out["pe"] = self.pe(torch.stack([self.pe_tf(i) for i in images]).cuda().half()).float().cpu().numpy()
            if "clip" in self.which:
                x = self.clip_proc(images=images, return_tensors="pt")["pixel_values"].cuda().half()
                out["clip"] = self.clip.vision_model(pixel_values=x).pooler_output.float().cpu().numpy()
        return out


def extract() -> None:
    from concurrent.futures import ThreadPoolExecutor

    CACHE.mkdir(parents=True, exist_ok=True)
    enc = Encoders()
    with ThreadPoolExecutor(8) as pool:
        for name, (ids, load) in _sets().items():
            target = CACHE / f"{name}.npz"
            if target.exists():
                print(f"{name}: cached", flush=True)
                continue
            store = {b: [] for b in BACKBONES}
            for s in range(0, len(ids), 64):
                for k, v in enc(list(pool.map(load, ids[s:s + 64]))).items():
                    store[k].append(v)
            np.savez(target, id=np.array(ids), **{k: np.concatenate(v) for k, v in store.items()})
            print(f"{name}: {len(ids)} images", flush=True)


def compare() -> None:
    import frontier_v5 as f5
    import kandinsky_v4_design as d74
    from sklearn.metrics import roc_auc_score

    feats75 = f5.CANDIDATES["front_pre+ft+coco"][0]

    def with_extra(d: dict, name: str) -> dict:
        z = np.load(CACHE / f"{name}.npz")
        assert z["id"].tolist() == d["id"].tolist(), name
        return {**d, "d3": d74.view(d, feats75), "pe": z["pe"], "clip": z["clip"]}

    _, train, val_l0, _, _, frontier, coco = f5._dev()
    train, val_l0 = with_extra(train, "train_l0"), with_extra(val_l0, "val_l0")
    frontier, coco = with_extra(frontier, "frontier"), with_extra(coco, "coco")
    benches = [with_extra(d74.load(f"bench_{v}"), f"bench_{v}") for v in ("v1", "v2", "v3", "v4")]
    ebay = d74.concat([d74.subset(b, b["label"] == 0) for b in benches])
    bench_gen = d74.concat([d74.subset(b, b["label"] == 1) for b in benches] + [with_extra(d74.load("k_train"), "k_train")])
    z = np.load(f5.FEATURES / "coco_holdout.npz")
    holdout = with_extra({k: z[k] for k in z.files}, "coco_holdout")

    v5z = np.load(OUT / "v5_features.npz")
    v5 = {k: v5z[k] for k in v5z.files}
    bz = np.load(OUT / "backbone_features.npz")
    v5 = {**v5, "d3": d74.view(v5, feats75), "pe": bz["pe_core_large"], "clip": bz["clip_vit_l14"]}
    g = v5["group"]

    def x(d, parts):
        return np.concatenate([d[p] for p in parts], axis=1)

    def fit(parts, reals, gens):
        return d74.fit_lr(np.concatenate([x(r, parts) for r in reals + gens]),
                          np.concatenate([np.zeros(len(r["id"])) for r in reals] + [np.ones(len(q["id"])) for q in gens]), f5.C)

    results = {}
    for combo, parts in COMBOS.items():
        for a, b in (("chatgpt_v5", "gemini_v5"), ("gemini_v5", "chatgpt_v5")):
            extra = d74.subset(v5, g == a)
            extra = {k: np.concatenate([extra[k]] * REPEAT) for k in ("id", "d3", "pe", "clip")}
            gens = [bench_gen, frontier, extra]
            model = fit(parts, [train, ebay, coco], gens)
            folds = np.arange(len(ebay["id"])) % 5
            oof = np.zeros(len(ebay["id"]))
            for f in range(5):
                m = fit(parts, [train, d74.subset(ebay, folds != f), coco], gens)
                oof[folds == f] = m.decision_function(x(d74.subset(ebay, folds == f), parts))
            t = d74.threshold(np.concatenate([model.decision_function(x(val_l0, parts)), oof]))
            s = lambda m: model.decision_function(x(d74.subset(v5, m), parts))  # noqa: E731
            test, reals, tt = s(g == b), s(g == "real_ebay_v5"), s(g == "motivating")
            ch = model.decision_function(x(holdout, parts))
            r = {"caught": [int((test >= t).sum()), len(test)], "real_ebay_v5_fake": [int((reals >= t).sum()), len(reals)],
                 "coco_holdout_fake": [int((ch >= t).sum()), len(ch)],
                 "auc_vs_real_ebay": round(float(roc_auc_score(np.r_[np.zeros(len(reals)), np.ones(len(test))], np.r_[reals, test])), 3),
                 "testtest_fake": bool(tt[0] >= t), "testtest_margin": round(float(tt[0] - t), 2)}
            results[f"{combo}: {a}->{b}"] = r
            print(f"{combo:16s} {a[:-3]:>7s}->{b[:-3]:7s} caught {r['caught'][0]:2d}/75  real eBay FAKE {r['real_ebay_v5_fake'][0]}/144  "
                  f"COCO FAKE {r['coco_holdout_fake'][0]}/1000  AUC {r['auc_vs_real_ebay']}  Grandmaster {'CAUGHT' if r['testtest_fake'] else 'missed'} ({r['testtest_margin']:+})", flush=True)
    (OUT / "fusion_compare.json").write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    {"extract": extract, "compare": compare}[sys.argv[1]]()
