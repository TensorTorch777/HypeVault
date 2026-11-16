"""Phase 76 probe (exploratory): add AI *watch* images to the Phase 75 recipe.
Train on chatgpt_v5, test on gemini_v5, and the reverse. real_ebay_v5 is never trained on (FAKE-rate check).
benchmark_v5 is spent, so using it as development data is allowed; these are not confirmatory numbers."""
import json, sys
from pathlib import Path
REPO = Path("/home/tensortorch26/Desktop/HypeVault"); ML = REPO / "ml_rtx5080"; B5 = ML / "benchmark_v5_frontier"
OUT = ML / "experiments" / "phase76_probe"
for p in (str(ML / "benchmark_v1_real_vs_generated"), str(ML), str(REPO / "backend")):
    sys.path.insert(0, p)
import numpy as np
from common import read_manifest
import frontier_v5 as f5, kandinsky_v4 as k4

cache = OUT / "v5_features.npz"
if not cache.exists():
    pre, ft = k4.Pretrained(), k4.FineTuned()
    samples = [s for s in read_manifest(B5 / "data/manifest.jsonl") if s.excluded is None]
    items = [(s.sample_id, s.group, k4.stored(str(B5 / s.path) if not Path(s.path).is_absolute() else s.path)) for s in samples]
    items.append(("testtest", "motivating", k4.jpeg90(k4.square(REPO / "testtest.jpeg", 512))))
    store = {}
    for i in range(0, len(items), k4.BATCH):
        ims = [x[2] for x in items[i:i + k4.BATCH]]
        for d in (pre.features(ims), ft.features(ims)):
            for k, v in d.items(): store.setdefault(k, []).append(v)
    np.savez(cache, id=np.array([x[0] for x in items]), group=np.array([x[1] for x in items]),
             **{k: np.concatenate(v) for k, v in store.items()})
z = np.load(cache); v5 = {k: z[k] for k in z.files}

d74, train, val_l0, ebay, bench_gen, frontier, coco = f5._dev()
zh = np.load(f5.FEATURES / "coco_holdout.npz"); holdout = {k: zh[k] for k in zh.files}
feats, sources = f5.CANDIDATES["front_pre+ft+coco"]
g = v5["group"]

def as_gen(mask, repeat):
    d = d74.subset(v5, mask)
    n = len(d["id"])
    d = {**d, "generator": np.full(n, "v5"), "group": d["id"], "watch": np.ones(n, bool), "label": np.ones(n, int)}
    return d74.concat([d] * repeat)

res = {}
for repeat in (1, 10, 30):
    for train_on, test_on in (("chatgpt_v5", "gemini_v5"), ("gemini_v5", "chatgpt_v5")):
        extra = as_gen(g == train_on, repeat)
        fr = d74.concat([frontier, extra])
        model, t, real = f5._fit_threshold(d74, feats, sources, train, val_l0, ebay, bench_gen, fr, coco)
        s = lambda m: model.decision_function(d74.view(d74.subset(v5, m), feats))
        test, reals, tt = s(g == test_on), s(g == "real_ebay_v5"), s(g == "motivating")
        ch = model.decision_function(d74.view(holdout, feats))
        from sklearn.metrics import roc_auc_score
        r = {"caught": [int((test >= t).sum()), len(test)], "real_ebay_v5_fake": [int((reals >= t).sum()), len(reals)],
             "coco_holdout_fake": [int((ch >= t).sum()), len(ch)], "dev_real_fake": round(float((real >= t).mean()), 4),
             "auc_vs_real_ebay": round(float(roc_auc_score(np.r_[np.zeros(len(reals)), np.ones(len(test))], np.r_[reals, test])), 3),
             "testtest_score": round(float(tt[0]), 3), "t_fake": round(t, 3)}
        res[f"x{repeat}_{train_on}->{test_on}"] = r
        print(f"x{repeat:<3d} {train_on:>10s} -> {test_on:10s} caught {r['caught'][0]}/{r['caught'][1]}  real eBay FAKE {r['real_ebay_v5_fake'][0]}/{r['real_ebay_v5_fake'][1]}  COCO FAKE {r['coco_holdout_fake'][0]}/{r['coco_holdout_fake'][1]}  AUC {r['auc_vs_real_ebay']}  testtest {r['testtest_score']} (t {r['t_fake']})", flush=True)
(OUT / "probe_results.json").write_text(json.dumps(res, indent=2) + "\n")
