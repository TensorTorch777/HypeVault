"""Phase 74: make DINOv3 catch Kandinsky 2.2 (a seen generator from now on).

Protocol: benchmark_v4_fresh/protocol.json (written before any v4 data). Development data only here.

Per image, three groups of signals:
- d3pre: frozen pretrained DINOv3 ViT-B/16 tokens: final CLS, mean patch token, and CLS after blocks 4 and 8
  (intermediate layers, as in RINE).
- d3ft: the frozen fine-tuned DINOv3 epoch_018: its raw logit, CLS and attention-pooled patch embedding.
- movq: AEROBLADE-style distance between the image and its reconstruction by Kandinsky's own MoVQ
  autoencoder (pixel L1, and DINOv3 patch-token cosine distance standing in for LPIPS), at 512 px and after
  a 768 px upscale (Kandinsky renders at 768 and benchmark images are downscaled to 512).

Training views add Rajan et al.'s aligned reconstructions: each Label_0 training photo is rendered at 768,
passed through MoVQ, then given the benchmark normalisation (Lanczos to 512, JPEG q90), exactly like a
Kandinsky output. No denoising and no prompts.

    .venv/bin/python ml_rtx5080/kandinsky_v4.py extract [--sets ...]
"""

from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

_PKG = Path(__file__).resolve().parent
if str(_PKG) not in sys.path:
    sys.path.insert(0, str(_PKG))

import generator_agnostic as ga  # noqa: E402

OUT = _PKG / "experiments" / "kandinsky_v4"
FEATURES = OUT / "features"
BATCH = 8
MOVQ_BATCH = 2
KANDINSKY_SIDE = 768
INTERMEDIATE_BLOCKS = (3, 7)  # zero-based: outputs of blocks 4 and 8 of 12
BENCH = {
    "v1": _PKG / "benchmark_v1_real_vs_generated",
    "v2": _PKG / "benchmark_v2_fresh",
    "v3": _PKG / "benchmark_v3_fresh",
}
K_TRAIN_MANIFEST = OUT / "train_images" / "manifest.jsonl"
LABEL0_VAL = _PKG / "experiments" / "watch_forensics_v2" / "label0_val_models.csv"


# ---------- images ----------

def jpeg90(img: Image.Image) -> Image.Image:
    return ga.jpeg(img, 90)


def square(path: str | Path, side: int) -> Image.Image:
    img = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    s = min(img.size)
    left, top = (img.width - s) // 2, (img.height - s) // 2
    return img.crop((left, top, left + s, top + s)).resize((side, side), Image.LANCZOS)


def stored(path: str | Path) -> Image.Image:
    """A benchmark image as stored (512 px JPEG q90). benchmark_v1 stored 768 px; those get the v2 normalisation."""
    img = Image.open(path).convert("RGB")
    if img.size != (ga.SIZE, ga.SIZE):
        img = jpeg90(img.resize((ga.SIZE, ga.SIZE), Image.LANCZOS))
    return img


# ---------- models ----------

class MoVQ:
    """Kandinsky 2.2's MoVQ autoencoder: encode, then decode without quantising, as the pipeline decodes."""

    def __init__(self) -> None:
        import torch
        from diffusers import VQModel

        self.torch = torch
        self.model = VQModel.from_pretrained("kandinsky-community/kandinsky-2-2-decoder", subfolder="movq",
                                             torch_dtype=torch.float16).to("cuda").eval()

    def __call__(self, images: list[Image.Image]) -> list[Image.Image]:
        return [r for s in range(0, len(images), MOVQ_BATCH) for r in self._run(images[s:s + MOVQ_BATCH])]

    def _run(self, images: list[Image.Image]) -> list[Image.Image]:
        torch = self.torch
        x = torch.from_numpy(np.stack([np.asarray(i, dtype=np.float32) for i in images])).permute(0, 3, 1, 2)
        x = (x / 127.5 - 1.0).to("cuda", torch.float16)
        with torch.inference_mode():
            out = self.model.decode(self.model.encode(x).latents, force_not_quantize=True).sample
        out = ((out.float().clamp(-1, 1) + 1) * 127.5).round().byte().permute(0, 2, 3, 1).cpu().numpy()
        return [Image.fromarray(a) for a in out]


class Pretrained:
    """Frozen pretrained DINOv3 ViT-B/16 at 512 px."""

    def __init__(self) -> None:
        import timm
        import torch

        self.torch = torch
        self.model = timm.create_model("vit_base_patch16_dinov3.lvd1689m", pretrained=True, num_classes=0, img_size=ga.SIZE)
        self.model = self.model.to("cuda").eval().half()
        cfg = timm.data.resolve_data_config({}, model=self.model)
        self.mean = torch.tensor(cfg["mean"]).view(1, 3, 1, 1).to("cuda").half()
        self.std = torch.tensor(cfg["std"]).view(1, 3, 1, 1).to("cuda").half()
        self.prefix = 1 + getattr(self.model, "num_reg_tokens", 4)

    def tokens(self, images: list[Image.Image]):
        """(final tokens after norm, [intermediate tokens]) as fp32 tensors on the GPU."""
        torch = self.torch
        x = torch.from_numpy(np.stack([np.asarray(i) for i in images])).permute(0, 3, 1, 2).to("cuda").half() / 255.0
        x = (x - self.mean) / self.std
        with torch.inference_mode():
            final, inter = self.model.forward_intermediates(x, indices=list(INTERMEDIATE_BLOCKS), output_fmt="NLC",
                                                            return_prefix_tokens=True, norm=True)
        return final.float(), [torch.cat([p, t], dim=1).float() for t, p in inter]

    def features(self, images: list[Image.Image]) -> dict[str, np.ndarray]:
        final, inter = self.tokens(images)
        patches = final[:, self.prefix:]
        out = {"d3pre_cls": final[:, 0], "d3pre_patch": patches.mean(1)}
        for block, t in zip(INTERMEDIATE_BLOCKS, inter):
            out[f"d3pre_cls_b{block + 1}"] = t[:, 0]
        return {k: v.cpu().numpy().astype(np.float32) for k, v in out.items()}

    def patch_distance(self, a: list[Image.Image], b: list[Image.Image]) -> np.ndarray:
        """Mean cosine distance between corresponding final patch tokens (an LPIPS stand-in)."""
        ta, tb = self.tokens(a)[0][:, self.prefix:], self.tokens(b)[0][:, self.prefix:]
        return (1 - self.torch.nn.functional.cosine_similarity(ta, tb, dim=-1)).mean(1).cpu().numpy()


class FineTuned:
    """The frozen fine-tuned DINOv3 epoch_018 (hash-checked), FP32 as served."""

    def __init__(self) -> None:
        import torch

        from reference_inference import load_reference_model

        self.torch = torch
        ckpt = _PKG / "experiments" / "v2_dinov3_cls_patch_attention" / "epoch_018.pt"
        self.model, self.digest = load_reference_model(ckpt, torch.device("cuda"))

    def features(self, images: list[Image.Image]) -> dict[str, np.ndarray]:
        torch = self.torch
        from reference_inference import preprocess_image

        m = self.model
        x = torch.stack([preprocess_image(i) for i in images]).to("cuda")
        with torch.inference_mode():
            tokens = m.backbone.forward_features(x)
            cls = m.backbone.get_cls_token(tokens)
            pooled, _ = m.attention_pool(m.backbone.get_patch_tokens(tokens))
            logit = m.trunk(m.fusion(torch.cat((cls, pooled), dim=-1))).reshape(-1)
        return {"d3ft_cls": cls.float().cpu().numpy(), "d3ft_pool": pooled.float().cpu().numpy(),
                "d3ft_logit": logit.float().cpu().numpy()}


# ---------- sets ----------

def bench_rows(version: str) -> list[dict]:
    root = BENCH[version]
    rows = []
    for line in (root / "data" / "manifest.jsonl").read_text().splitlines():
        s = json.loads(line)
        if s.get("excluded") is None:
            path = Path(s["path"]) if Path(s["path"]).is_absolute() else root / s["path"]
            rows.append({"id": s["sample_id"], "path": str(path), "brand": s["brand"], "group": s["group"],
                         "label": int(s["label"] == "GENERATED")})
    return rows


def k_train_rows() -> list[dict]:
    rows = []
    for line in K_TRAIN_MANIFEST.read_text().splitlines():
        s = json.loads(line)
        if s.get("excluded") is None:
            rows.append({"id": s["sample_id"], "path": s["path"], "brand": s["brand"], "group": s["group"], "label": 1})
    return rows


def label_rows(split: str, label: int, per_brand: int | None) -> list[dict]:
    return [{**r, "path": str(ga.REPO / r["path"]) if not Path(r["path"]).is_absolute() else r["path"],
             "group": f"{split}_label{label}", "label": label} for r in ga.split_rows(split, label, per_brand)]


def label0_val_rows() -> list[dict]:
    """The 1,000 Label_0 validation photos Phase 73 used for thresholds."""
    import csv

    ids = {r["sample_id"] for r in csv.DictReader(LABEL0_VAL.open())}
    return [r for r in label_rows("validation", 0, None) if r["id"] in ids]


SETS = {
    "train_l0": lambda: label_rows("train", 0, ga.TRAIN_PER_BRAND),
    "val_l0": label0_val_rows,
    "val_l1": lambda: label_rows("validation", 1, ga.VAL_PER_BRAND),
    "bench_v1": lambda: bench_rows("v1"),
    "bench_v2": lambda: bench_rows("v2"),
    "bench_v3": lambda: bench_rows("v3"),
    "k_train": k_train_rows,
}
# Sets that also get aligned MoVQ reconstructions as extra (fake) training views.
ALIGNED = {"train_l0", "val_l0"}


def extract_set(name: str, rows: list[dict], pre: Pretrained, ft: FineTuned, movq: MoVQ) -> None:
    target = FEATURES / f"{name}.npz"
    if target.exists():
        print(f"{name}: cached", flush=True)
        return
    store: dict[str, list] = {}

    def put(prefix: str, feats: dict[str, np.ndarray]) -> None:
        for k, v in feats.items():
            store.setdefault(f"{prefix}{k}", []).append(v)

    for start in range(0, len(rows), BATCH):
        chunk = rows[start:start + BATCH]
        if name in ALIGNED:
            # Training photos get the benchmark normalisation here, as admitted benchmark images did.
            images = [jpeg90(square(r["path"], ga.SIZE)) for r in chunk]
            aligned = [jpeg90(i.resize((ga.SIZE, ga.SIZE), Image.LANCZOS)) for i in movq([square(r["path"], KANDINSKY_SIDE) for r in chunk])]
            put("aligned__", pre.features(aligned))
            put("aligned__", ft.features(aligned))
        else:
            images = [stored(r["path"]) for r in chunk]
        put("", pre.features(images))
        put("", ft.features(images))
        # AEROBLADE-style distances to the MoVQ reconstruction.
        rec512 = movq(images)
        up = [i.resize((KANDINSKY_SIDE, KANDINSKY_SIDE), Image.LANCZOS) for i in images]
        rec768 = [i.resize((ga.SIZE, ga.SIZE), Image.LANCZOS) for i in movq(up)]
        a = np.stack([np.asarray(i, dtype=np.float32) for i in images])
        for tag, rec in (("512", rec512), ("768", rec768)):
            b = np.stack([np.asarray(i, dtype=np.float32) for i in rec])
            put("", {f"movq_l1_{tag}": np.abs(a - b).mean((1, 2, 3)), f"movq_cos_{tag}": pre.patch_distance(images, rec)})
        if (start // BATCH) % 50 == 0:
            print(f"{name}: {start + len(chunk)}/{len(rows)}", flush=True)
    arrays = {k: np.concatenate(v) for k, v in store.items()}
    meta = {k: np.array([r[k] for r in rows]) for k in ("id", "brand", "group", "label")}
    np.savez(target, **meta, **arrays)
    print(f"{name}: wrote {len(rows)} rows, {sorted(arrays)}", flush=True)


def extract(names: list[str]) -> None:
    FEATURES.mkdir(parents=True, exist_ok=True)
    pre, ft, movq = Pretrained(), FineTuned(), MoVQ()
    print("fine-tuned checkpoint", ft.digest, flush=True)
    for name in names:
        extract_set(name, SETS[name](), pre, ft, movq)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=("extract",))
    parser.add_argument("--sets", nargs="+", default=list(SETS), choices=list(SETS))
    args = parser.parse_args()
    extract(args.sets)


if __name__ == "__main__":
    main()
