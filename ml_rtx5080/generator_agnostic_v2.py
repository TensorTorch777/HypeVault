"""Phase 73 fix 4: the Phase 71 probe retrained on reconstructions from five decoders.

Adds aMUSEd's VQ-GAN (token-based), TAESD and TAESDXL to Phase 71's SD 1.5 and SDXL VAEs. Kandinsky's
MoVQ and Stable Cascade's Paella VQ-GAN are never used, because benchmark_v3 tests those generators.
Only the frozen pretrained DINOv3 backbone is used (it was selected in Phase 71).

    .venv/bin/python ml_rtx5080/generator_agnostic_v2.py extract
    .venv/bin/python ml_rtx5080/generator_agnostic_v2.py fit
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

_PKG = Path(__file__).resolve().parent
if str(_PKG) not in sys.path:
    sys.path.insert(0, str(_PKG))

import generator_agnostic as ga  # noqa: E402

OUT = _PKG / "experiments" / "generator_agnostic_v2"
FEATURES = OUT / "features"
OLD_FEATURES = ga.FEATURES
CANDIDATE = "dinov3_vitb16_linear"
NEW_DECODERS = ("amused_vq", "taesd", "taesdxl")
ALL_DECODERS = ("sd15", "sdxl", *NEW_DECODERS)
V2_BENCH = _PKG / "benchmark_v2_fresh"


class NewReconstructor:
    """aMUSEd VQ-GAN (encode, quantise, decode) or a tiny autoencoder (TAESD / TAESDXL)."""

    def __init__(self, name: str) -> None:
        import torch

        self.torch, self.name = torch, name
        if name == "amused_vq":
            from diffusers import VQModel

            self.model = VQModel.from_pretrained("amused/amused-512", subfolder="vqvae", variant="fp16", torch_dtype=torch.float16)
        else:
            from diffusers import AutoencoderTiny

            self.model = AutoencoderTiny.from_pretrained(f"madebyollin/{name}", torch_dtype=torch.float16)
        self.model = self.model.to("cuda").eval()

    def __call__(self, images: list[Image.Image]) -> list[Image.Image]:
        return [r for start in range(0, len(images), ga.VAE_BATCH) for r in self._run(images[start:start + ga.VAE_BATCH])]

    def _run(self, images: list[Image.Image]) -> list[Image.Image]:
        torch = self.torch
        x = torch.from_numpy(np.stack([np.asarray(i, dtype=np.float32) for i in images])).permute(0, 3, 1, 2)
        x = (x / 127.5 - 1.0).to("cuda", torch.float16)
        with torch.inference_mode():
            if self.name == "amused_vq":
                out = self.model.decode(self.model.encode(x).latents, force_not_quantize=False).sample
            else:
                # AutoencoderTiny works in [0, 1].
                out = self.model.decode(self.model.encode((x + 1) / 2).latents).sample * 2 - 1
        out = ((out.float().clamp(-1, 1) + 1) * 127.5).round().byte().permute(0, 2, 3, 1).cpu().numpy()
        return [Image.fromarray(a) for a in out]


class DinoOnly(ga.Embedders):
    """Only the frozen DINOv3 embedding; CLIP is not loaded."""

    def __init__(self) -> None:
        import timm
        import torch

        self.torch = torch
        self.dino = timm.create_model("vit_base_patch16_dinov3.lvd1689m", pretrained=True, num_classes=0, img_size=ga.SIZE).to("cuda").eval().half()
        cfg = timm.data.resolve_data_config({}, model=self.dino)
        self.mean = torch.tensor(cfg["mean"]).view(1, 3, 1, 1).to("cuda").half()
        self.std = torch.tensor(cfg["std"]).view(1, 3, 1, 1).to("cuda").half()

    def __call__(self, images: list[Image.Image]) -> dict[str, np.ndarray]:
        torch = self.torch
        with torch.inference_mode():
            x = torch.from_numpy(np.stack([np.asarray(i) for i in images])).permute(0, 3, 1, 2).to("cuda").half() / 255.0
            return {CANDIDATE: self.dino((x - self.mean) / self.std).float().cpu().numpy()}


def v2_rows(group: str) -> list[dict]:
    rows = []
    for line in (V2_BENCH / "data" / "manifest.jsonl").read_text().splitlines():
        s = json.loads(line)
        if s["group"] == group and s.get("excluded") is None:
            rows.append({"id": s["sample_id"], "path": s["path"], "brand": s["brand"]})
    return rows


def extract() -> None:
    FEATURES.mkdir(parents=True, exist_ok=True)
    ga.CANDIDATES = (CANDIDATE,)
    ga.FEATURES = FEATURES
    embed = DinoOnly()
    recon = {name: NewReconstructor(name) for name in NEW_DECODERS}
    # Training photos: only the three new decoders (Phase 71 already holds real, SD 1.5 and SDXL views).
    ga.extract_set("train_new", ga.split_rows("train", 0, ga.TRAIN_PER_BRAND), embed, recon, augment=True)
    ga.extract_set("val_real_new", ga.split_rows("validation", 0, ga.VAL_PER_BRAND), embed, recon, augment=False)
    for group in ("real_ebay_v2", "kandinsky22_fresh", "amused512"):
        ga.extract_set(f"bench2_{group}", v2_rows(group), embed, None, augment=False)


def load(folder: Path, name: str, view: str) -> np.ndarray:
    return np.load(folder / f"{name}__{view}.npz")[CANDIDATE]


def fit() -> None:
    import joblib
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    from normalize_surface import roc_auc

    real = [load(OLD_FEATURES, "train", v) for v in ("real_clean", "real_aug")]
    fake = [load(OLD_FEATURES, "train", f"{d}_{v}") for d in ("sd15", "sdxl") for v in ("clean", "aug")]
    fake += [load(FEATURES, "train_new", f"{d}_{v}") for d in NEW_DECODERS for v in ("clean", "aug")]
    x = np.concatenate(real + fake)
    y = np.concatenate([np.zeros(sum(len(a) for a in real)), np.ones(sum(len(a) for a in fake))])
    model = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, class_weight="balanced", max_iter=5000))
    model.fit(x, y)
    joblib.dump(model, OUT / "dinov3_vitb16_linear_5dec.joblib")
    real_val = model.predict_proba(load(OLD_FEATURES, "val_real", "real_clean"))[:, 1]
    label1 = model.predict_proba(load(OLD_FEATURES, "val_label1", "real_clean"))[:, 1]
    summary = {
        "decoders": ALL_DECODERS,
        "val_auc_label0_vs_label1": roc_auc(np.concatenate([real_val, label1]), np.concatenate([np.zeros(len(real_val)), np.ones(len(label1))])),
        "threshold_p99_real_val": float(np.percentile(real_val, 99)),
    }
    (OUT / "fit.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=("extract", "fit"))
    {"extract": extract, "fit": fit}[parser.parse_args().command]()


if __name__ == "__main__":
    main()
