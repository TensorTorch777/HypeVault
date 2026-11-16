"""Exploratory backbone comparison on the spent benchmark_v5: same images, same small linear probe.
Train: one app's 75 AI watch images + half the real eBay photos. Test: the other app + the other half."""
import json, sys
from pathlib import Path
REPO = Path("/home/tensortorch26/Desktop/HypeVault"); ML = REPO / "ml_rtx5080"; B5 = ML / "benchmark_v5_frontier"
OUT = ML / "experiments" / "phase76_probe"
for p in (str(ML / "benchmark_v1_real_vs_generated"), str(ML), str(REPO / "backend")):
    sys.path.insert(0, p)
import numpy as np, torch
from PIL import Image
from common import read_manifest
import kandinsky_v4 as k4
from kandinsky_v4_design import view
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

z = np.load(OUT / "v5_features.npz"); v5 = {k: z[k] for k in z.files}
ids, group = v5["id"].tolist(), v5["group"]
paths = {s.sample_id: (B5 / s.path if not Path(s.path).is_absolute() else Path(s.path))
         for s in read_manifest(B5 / "data/manifest.jsonl") if s.excluded is None}
def image(i):
    return k4.jpeg90(k4.square(REPO / "testtest.jpeg", 512)) if i == "testtest" else Image.open(paths[i]).convert("RGB")
dev = "cuda"

@torch.no_grad()
def hf_feats(name, kind):
    from transformers import AutoModel, AutoProcessor, CLIPModel
    model = (CLIPModel if kind == "clip" else AutoModel).from_pretrained(name, torch_dtype=torch.float16).to(dev).eval()
    proc = AutoProcessor.from_pretrained(name)
    out = []
    for s in range(0, len(ids), 32):
        x = proc(images=[image(i) for i in ids[s:s + 32]], return_tensors="pt")["pixel_values"].to(dev, torch.float16)
        o = model.vision_model(pixel_values=x)
        out.append(o.pooler_output.float().cpu().numpy())
    del model; torch.cuda.empty_cache()
    return np.concatenate(out)

@torch.no_grad()
def timm_feats(name):
    import timm
    m = timm.create_model(name, pretrained=True, num_classes=0).to(dev).eval().half()
    cfg = timm.data.resolve_data_config({}, model=m); tf = timm.data.create_transform(**cfg)
    out = []
    for s in range(0, len(ids), 32):
        x = torch.stack([tf(image(i)) for i in ids[s:s + 32]]).to(dev).half()
        out.append(m(x).float().cpu().numpy())
    del m; torch.cuda.empty_cache()
    return np.concatenate(out)

cache = OUT / "backbone_features.npz"
feats = dict(np.load(cache)) if cache.exists() else {}
todo = {
    "dinov3_pretrained_cls": lambda: v5["d3pre_cls"] if "d3pre_cls" in v5 else view(v5, "pre"),
    "dinov3_pre+ft (Phase 75 view)": lambda: view(v5, "pre+ft"),
    "dinov2_base_pretrained": lambda: timm_feats("vit_base_patch14_dinov2.lvd142m"),
    "clip_vit_l14": lambda: hf_feats("openai/clip-vit-large-patch14", "clip"),
    "siglip2_so400m": lambda: hf_feats("google/siglip2-so400m-patch14-384", "siglip"),
    "pe_core_large": lambda: timm_feats("vit_pe_core_large_patch14_336.fb"),
}
for k, f in todo.items():
    if k not in feats:
        print("extracting", k, flush=True); feats[k] = f(); np.savez(cache, **feats)

real = np.where(group == "real_ebay_v5")[0]
halves = [real[::2], real[1::2]]
res = {}
for k, x in feats.items():
    rows = []
    for a, b in (("chatgpt_v5", "gemini_v5"), ("gemini_v5", "chatgpt_v5")):
        for h in (0, 1):
            tr = np.r_[np.where(group == a)[0], halves[h]]; te_ai = np.where(group == b)[0]; te_real = halves[1 - h]
            y = np.r_[np.ones((group == a).sum()), np.zeros(len(halves[h]))]
            m = make_pipeline(StandardScaler(), LogisticRegression(C=0.05, class_weight="balanced", max_iter=5000)).fit(x[tr], y)
            s_ai, s_real = m.decision_function(x[te_ai]), m.decision_function(x[te_real])
            t = np.quantile(s_real, 0.97)  # oracle cut at 3% real FAKE, same rule for every model
            rows.append({"dir": f"{a}->{b}", "auc": roc_auc_score(np.r_[np.ones(len(s_ai)), np.zeros(len(s_real))], np.r_[s_ai, s_real]),
                         "caught_at_3pct": float((s_ai > t).mean()),
                         "testtest_above_cut": bool(m.decision_function(x[[ids.index("testtest")]])[0] > t)})
    by = lambda d: [r for r in rows if r["dir"].startswith(d)]
    res[k] = {"auc_mean": round(np.mean([r["auc"] for r in rows]), 3),
              "chatgpt->gemini_caught": round(np.mean([r["caught_at_3pct"] for r in by("chatgpt")]), 3),
              "gemini->chatgpt_caught": round(np.mean([r["caught_at_3pct"] for r in by("gemini")]), 3),
              "testtest_caught_in_runs": sum(r["testtest_above_cut"] for r in rows)}
    print(f"{k:32s} AUC {res[k]['auc_mean']:.3f}  ChatGPT->Gemini caught {res[k]['chatgpt->gemini_caught']:.0%}  Gemini->ChatGPT {res[k]['gemini->chatgpt_caught']:.0%}  Grandmaster caught {res[k]['testtest_caught_in_runs']}/4", flush=True)
(OUT / "backbone_compare.json").write_text(json.dumps(res, indent=2) + "\n")
