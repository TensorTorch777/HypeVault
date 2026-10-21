"""Phase 71: a generator-agnostic detector from frozen features and autoencoder reconstructions.

Protocol: experiments/generator_agnostic_v1/protocol.json (written before any extraction).

    .venv/bin/python ml_rtx5080/generator_agnostic.py extract   # features for every set (GPU)
    .venv/bin/python ml_rtx5080/generator_agnostic.py select    # train both probes, pick on validation, fix threshold
    .venv/bin/python ml_rtx5080/generator_agnostic.py final     # score the final sets once

Reconstructed images are never written to disk. Each image is built, post-processed and
embedded in memory; only the feature arrays are saved.
"""

from __future__ import annotations

import argparse
import io
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter, ImageOps

_PKG = Path(__file__).resolve().parent
if str(_PKG) not in sys.path:
    sys.path.insert(0, str(_PKG))

from normalize_surface import roc_auc  # noqa: E402
from unseen_generator import clopper_pearson  # noqa: E402

REPO = _PKG.parent
OUT = _PKG / "experiments" / "generator_agnostic_v1"
FEATURES = OUT / "features"
MANIFEST = _PKG / "experiments" / "dataset_audit" / "split_manifest_v2.json"
BENCH = _PKG / "benchmark_v1_real_vs_generated"
BENCH_RUN = BENCH / "results" / "20261020T110244Z"
SIZE = 512
TRAIN_PER_BRAND = 1000
VAL_PER_BRAND = 200
BATCH = 16
VAE_BATCH = 4
VAES = {
    "sd15": ("runwayml/stable-diffusion-v1-5", "fp16"),
    "sdxl": ("stabilityai/stable-diffusion-xl-base-1.0", "fp32"),
}
CANDIDATES = ("clip_vitl14_linear", "dinov3_vitb16_linear")


# ---------- images ----------

def base_image(path: str | Path) -> Image.Image:
    img = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    side = min(img.size)
    left, top = (img.width - side) // 2, (img.height - side) // 2
    return img.crop((left, top, left + side, top + side)).resize((SIZE, SIZE), Image.LANCZOS)


def jpeg(img: Image.Image, quality: int) -> Image.Image:
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def clean_view(img: Image.Image) -> Image.Image:
    return jpeg(img, 90)


def augmented_view(img: Image.Image, seed: str) -> Image.Image:
    rng = random.Random(seed)
    if rng.random() < 0.5:
        factor = rng.uniform(0.5, 1.0)
        small = max(16, int(round(SIZE * factor)))
        img = img.resize((small, small), Image.BICUBIC).resize((SIZE, SIZE), Image.BICUBIC)
    if rng.random() < 0.3:
        img = img.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.2, 1.5)))
    return jpeg(img, rng.randint(60, 95))


# ---------- models ----------

class Reconstructor:
    def __init__(self, name: str) -> None:
        import torch
        from diffusers import AutoencoderKL

        repo, precision = VAES[name]
        self.dtype = torch.float16 if precision == "fp16" else torch.float32
        self.vae = AutoencoderKL.from_pretrained(repo, subfolder="vae", torch_dtype=self.dtype).to("cuda").eval()
        self.torch = torch

    def __call__(self, images: list[Image.Image]) -> list[Image.Image]:
        # The fp32 SDXL decoder needs about 4 GiB per 16 images at 512 px, so decode in small chunks.
        return [r for start in range(0, len(images), VAE_BATCH) for r in self._run(images[start:start + VAE_BATCH])]

    def _run(self, images: list[Image.Image]) -> list[Image.Image]:
        torch = self.torch
        x = torch.from_numpy(np.stack([np.asarray(i, dtype=np.float32) for i in images])).permute(0, 3, 1, 2)
        x = (x / 127.5 - 1.0).to("cuda", self.dtype)
        with torch.inference_mode():
            latent = self.vae.encode(x).latent_dist.mode()
            out = self.vae.decode(latent).sample
        out = ((out.float().clamp(-1, 1) + 1) * 127.5).round().byte().permute(0, 2, 3, 1).cpu().numpy()
        return [Image.fromarray(a) for a in out]


class Embedders:
    def __init__(self) -> None:
        import timm
        import torch
        from transformers import CLIPModel, CLIPProcessor

        self.torch = torch
        self.clip = CLIPModel.from_pretrained("openai/clip-vit-large-patch14", torch_dtype=torch.float16).to("cuda").eval()
        self.clip_proc = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
        self.dino = timm.create_model("vit_base_patch16_dinov3.lvd1689m", pretrained=True, num_classes=0, img_size=SIZE)
        self.dino = self.dino.to("cuda").eval().half()
        cfg = timm.data.resolve_data_config({}, model=self.dino)
        self.mean = torch.tensor(cfg["mean"]).view(1, 3, 1, 1).to("cuda").half()
        self.std = torch.tensor(cfg["std"]).view(1, 3, 1, 1).to("cuda").half()

    def __call__(self, images: list[Image.Image]) -> dict[str, np.ndarray]:
        torch = self.torch
        with torch.inference_mode():
            pixels = self.clip_proc(images=images, return_tensors="pt")["pixel_values"].to("cuda", torch.float16)
            clip = self.clip.get_image_features(pixel_values=pixels)
            clip = getattr(clip, "pooler_output", clip)
            clip = torch.nn.functional.normalize(clip.float(), dim=-1)
            x = torch.from_numpy(np.stack([np.asarray(i) for i in images])).permute(0, 3, 1, 2).to("cuda").half() / 255.0
            dino = self.dino((x - self.mean) / self.std).float()
        return {"clip_vitl14_linear": clip.cpu().numpy(), "dinov3_vitb16_linear": dino.cpu().numpy()}


# ---------- sets ----------

def split_rows(split: str, label: int, per_brand: int | None) -> list[dict]:
    samples = json.loads(MANIFEST.read_text())["samples"]
    rows, counts = [], {}
    for s in samples:
        if s["split"] != split or s["label"] != label:
            continue
        if per_brand is not None and counts.get(s["brand"], 0) >= per_brand:
            continue
        counts[s["brand"]] = counts.get(s["brand"], 0) + 1
        rows.append({"id": s["sample_id"], "path": s["path"], "brand": s["brand"]})
    return rows


def bench_rows(group: str) -> list[dict]:
    rows = []
    for line in (BENCH / "data" / "manifest.jsonl").read_text().splitlines():
        s = json.loads(line)
        if s["group"] == group and s.get("excluded") is None:
            rows.append({"id": s["sample_id"], "path": str(BENCH / s["path"]), "brand": s["brand"]})
    return rows


def extract_set(name: str, rows: list[dict], embed: Embedders, recon: dict[str, Reconstructor] | None, augment: bool) -> None:
    """Writes features/<name>__<view>.npz. Views: real (and sd15 / sdxl when recon is given), each clean (+ aug)."""
    target = FEATURES / f"{name}__real_clean.npz"
    if target.exists():
        print(f"{name}: cached", flush=True)
        return
    store: dict[str, dict[str, list]] = {}

    def add(view: str, images: list[Image.Image]) -> None:
        feats = embed(images)
        bucket = store.setdefault(view, {k: [] for k in CANDIDATES})
        for k in CANDIDATES:
            bucket[k].append(feats[k])

    for start in range(0, len(rows), BATCH):
        chunk = rows[start:start + BATCH]
        bases = [base_image(r["path"]) for r in chunk]
        sources = {"real": bases}
        for vae_name, vae in (recon or {}).items():
            sources[vae_name] = vae(bases)
        for source, images in sources.items():
            add(f"{source}_clean", [clean_view(i) for i in images])
            if augment:
                add(f"{source}_aug", [augmented_view(i, f"{name}:{source}:{r['id']}") for i, r in zip(images, chunk)])
        if (start // BATCH) % 25 == 0:
            print(f"{name}: {start + len(chunk)}/{len(rows)}", flush=True)
    ids = np.array([r["id"] for r in rows])
    brands = np.array([r["brand"] for r in rows])
    for view, bucket in store.items():
        np.savez(FEATURES / f"{name}__{view}.npz", ids=ids, brands=brands, **{k: np.concatenate(v) for k, v in bucket.items()})
    print(f"{name}: wrote {sorted(store)}", flush=True)


def extract() -> None:
    FEATURES.mkdir(parents=True, exist_ok=True)
    embed = Embedders()
    recon = {name: Reconstructor(name) for name in VAES}
    extract_set("train", split_rows("train", 0, TRAIN_PER_BRAND), embed, recon, augment=True)
    extract_set("val_real", split_rows("validation", 0, VAL_PER_BRAND), embed, recon, augment=False)
    extract_set("val_label1", split_rows("validation", 1, VAL_PER_BRAND), embed, None, augment=False)
    extract_set("test_label0", split_rows("test", 0, None), embed, None, augment=False)
    extract_set("test_label1", split_rows("test", 1, None), embed, None, augment=False)
    for group in ("real_ebay", "heldout_kandinsky22", "sdxl_base10"):
        extract_set(f"bench_{group}", bench_rows(group), embed, None, augment=False)


# ---------- probes ----------

def load(name: str, view: str, candidate: str) -> np.ndarray:
    return np.load(FEATURES / f"{name}__{view}.npz")[candidate]


def fit(candidate: str):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    real = [load("train", v, candidate) for v in ("real_clean", "real_aug")]
    fake = [load("train", f"{s}_{v}", candidate) for s in VAES for v in ("clean", "aug")]
    x = np.concatenate(real + fake)
    y = np.concatenate([np.zeros(sum(len(a) for a in real)), np.ones(sum(len(a) for a in fake))])
    model = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, class_weight="balanced", max_iter=5000))
    model.fit(x, y)
    return model


def p_fake(model, x: np.ndarray) -> np.ndarray:
    return model.predict_proba(x)[:, 1]


def select() -> None:
    import joblib

    result = {"candidates": {}}
    for candidate in CANDIDATES:
        model = fit(candidate)
        real = p_fake(model, load("val_real", "real_clean", candidate))
        label1 = p_fake(model, load("val_label1", "real_clean", candidate))
        recon = np.concatenate([p_fake(model, load("val_real", f"{s}_clean", candidate)) for s in VAES])
        threshold = float(np.percentile(real, 99))
        result["candidates"][candidate] = {
            "val_auc_label0_vs_label1": roc_auc(np.concatenate([real, label1]), np.concatenate([np.zeros(len(real)), np.ones(len(label1))])),
            "val_auc_real_vs_reconstruction": roc_auc(np.concatenate([real, recon]), np.concatenate([np.zeros(len(real)), np.ones(len(recon))])),
            "threshold_p99_real_val": threshold,
            "val_label1_detected_at_threshold": int((label1 >= threshold).sum()),
            "val_label1_n": len(label1),
        }
        joblib.dump(model, OUT / f"{candidate}.joblib")
        print(candidate, json.dumps(result["candidates"][candidate]), flush=True)
    best = max(CANDIDATES, key=lambda c: result["candidates"][c]["val_auc_label0_vs_label1"])
    result["selected"] = best
    result["threshold"] = result["candidates"][best]["threshold_p99_real_val"]
    (OUT / "selection.json").write_text(json.dumps(result, indent=2) + "\n")
    print("selected", best, flush=True)


def rate(k: int, n: int) -> dict:
    lo, hi = clopper_pearson(k, n)
    return {"k": k, "n": n, "rate": k / n if n else math.nan, "ci95": [lo, hi]}


def final() -> None:
    import joblib

    out = OUT / "final_metrics.json"
    if out.exists():
        sys.exit("final_metrics.json exists: the final evaluation runs once")
    selection = json.loads((OUT / "selection.json").read_text())
    result = {"selected": selection["selected"], "threshold": selection["threshold"], "models": {}}
    real_ebay_n = None
    for candidate in CANDIDATES:
        model = joblib.load(OUT / f"{candidate}.joblib")
        threshold = selection["candidates"][candidate]["threshold_p99_real_val"]
        scores = {name: p_fake(model, load(name, "real_clean", candidate))
                  for name in ("bench_real_ebay", "bench_heldout_kandinsky22", "bench_sdxl_base10", "test_label0", "test_label1")}
        real_ebay_n = len(scores["bench_real_ebay"])
        block = {}
        for real, fake in (("bench_real_ebay", "bench_heldout_kandinsky22"), ("bench_real_ebay", "bench_sdxl_base10"), ("test_label0", "test_label1")):
            r, f = scores[real], scores[fake]
            block[f"{real}__vs__{fake}"] = {
                "generated_detected": rate(int((f >= threshold).sum()), len(f)),
                "real_rejected": rate(int((r >= threshold).sum()), len(r)),
                "roc_auc": roc_auc(np.concatenate([r, f]), np.concatenate([np.zeros(len(r)), np.ones(len(f))])),
            }
        result["models"][candidate] = {"threshold": threshold, "sets": block}
        print(candidate, json.dumps({k: (v["generated_detected"]["k"], v["real_rejected"]["k"], round(v["roc_auc"], 4)) for k, v in block.items()}), flush=True)
    result["earlier_models_same_benchmark"] = earlier_models()
    result["real_ebay_n"] = real_ebay_n
    out.write_text(json.dumps(result, indent=2) + "\n")


def earlier_models() -> dict:
    import csv

    rows = list(csv.DictReader((BENCH_RUN / "predictions.csv").open()))
    summary = {}
    for model in ("dinov2_legacy", "dinov3_experimental"):
        for group in ("heldout_kandinsky22", "sdxl_base10", "real_ebay"):
            decisions = [r[f"{model}_decision"] for r in rows if r["group"] == group]
            summary[f"{model}__{group}"] = {d: decisions.count(d) for d in sorted(set(decisions))}
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=("extract", "select", "final"))
    args = parser.parse_args()
    {"extract": extract, "select": select, "final": final}[args.command]()


if __name__ == "__main__":
    main()
