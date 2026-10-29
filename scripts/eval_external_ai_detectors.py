"""Score open-source AI-image detectors on our real and generated watch photos.

Why: a ChatGPT/Gemini-style image of a Patek Philippe Grandmaster Chime passed listing screening.
DINOv2 (live), DINOv3 epoch_018 and the Phase 74 head all called it a real photo, because none
of them saw a frontier generator in training. These detectors were trained on thousands of
generators, including GPT Image, DALL-E 3, FLUX, Imagen and Midjourney.

Detectors (MIT licence, weights downloaded to models/external/, loaded by our own code):
  frontier  Thermostatic/community-forensics-frontier-detector-2026-08 (ViT-S/16 384;
            AI if raw logit >= 1.359375, the published frozen boundary)
  commfor   buildborderless/CommunityForensics-DeepfakeDet-ViT (CVPR 2025 Community Forensics; AI if p > 0.5)

    .venv/bin/python scripts/eval_external_ai_detectors.py [--per-set 200]
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from inference.ai_image_detector import FrontierDetector  # noqa: E402

OUT = ROOT / "ml_rtx5080" / "experiments" / "external_detectors_v1"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def files(folder: Path, n: int | None, rng: random.Random) -> list[Path]:
    found = sorted(p for p in folder.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES)
    return found if n is None or len(found) <= n else sorted(rng.sample(found, n))


def label0(per_brand: int, rng: random.Random) -> list[Path]:
    out: list[Path] = []
    for brand in sorted((ROOT / "Label_0_Watches").iterdir()):
        out += files(brand, per_brand, rng)
    return out


class CommFor:
    def __init__(self) -> None:
        import torch
        from transformers import ViTForImageClassification, ViTImageProcessor

        path = ROOT / "models" / "external" / "commfor_vit_buildborderless"
        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = ViTForImageClassification.from_pretrained(path).to(self.device).eval()
        self.proc = ViTImageProcessor.from_pretrained(path)

    def prob(self, images) -> list[float]:
        with self.torch.inference_mode():
            x = self.proc(images, return_tensors="pt").to(self.device)
            return self.torch.sigmoid(self.model(**x).logits.reshape(-1)).float().cpu().tolist()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--per-set", type=int, default=200)
    parser.add_argument("--seed", type=int, default=11)
    args = parser.parse_args()
    rng = random.Random(args.seed)
    from PIL import Image

    sets = {
        "REAL eBay photos (benchmark_v4 real_ebay_v4)": (0, files(ROOT / "ml_rtx5080/benchmark_v4_fresh/data/real_ebay_v4", None, rng)),
        "REAL Label_0 photos": (0, label0(args.per_set // 5, rng)),
        "REAL user photo 1_REAL_vacheron_constantin.jpg": (0, [ROOT / "1_REAL_vacheron_constantin.jpg"]),
        "AI user image testtest.jpeg (frontier generator)": (1, [ROOT / "testtest.jpeg"]),
        "AI Kandinsky 2.2 (benchmark_v4)": (1, files(ROOT / "ml_rtx5080/benchmark_v4_fresh/data/kandinsky22_v4", None, rng)),
        "AI Stable Cascade (benchmark_v4)": (1, files(ROOT / "ml_rtx5080/benchmark_v4_fresh/data/stable_cascade_v4", None, rng)),
        "AI SDXL": (1, files(ROOT / "data_unseen_generator_v1/sdxl_base_1_0", args.per_set, rng)),
        "AI SD 1.5": (1, files(ROOT / "data_unseen_generator_v1/sd_1_5", args.per_set, rng)),
    }
    frontier, commfor = FrontierDetector(), CommFor()
    report = {"detectors": {"frontier": FrontierDetector.REPO, "commfor": "buildborderless/CommunityForensics-DeepfakeDet-ViT"},
              "sets": {}}
    for name, (label, paths) in sets.items():
        flagged = {"frontier": 0, "commfor": 0}
        scores = []
        for start in range(0, len(paths), 16):
            chunk = paths[start:start + 16]
            ims = []
            for p in chunk:
                with Image.open(p) as im:
                    ims.append(im.convert("RGB"))
            fr = frontier.score_batch(ims)
            cf = commfor.prob(ims)
            for p, f, c in zip(chunk, fr, cf):
                flagged["frontier"] += f.is_ai
                flagged["commfor"] += c > 0.5
                scores.append({"file": p.name, "frontier_prob": round(f.probability, 4), "commfor_prob": round(c, 4)})
        n = len(paths)
        row = {"label": "AI" if label else "REAL", "n": n,
               "flagged_as_ai": {k: f"{v}/{n}" for k, v in flagged.items()},
               "correct_rate": {k: round((v if label else n - v) / n, 3) for k, v in flagged.items()}}
        if n <= 2:
            row["scores"] = scores
        report["sets"][name] = row
        print(f"{name:52s} n={n:4d}  flagged AI: frontier {flagged['frontier']:4d}  commfor {flagged['commfor']:4d}"
              + (f"  {scores}" if n <= 2 else ""))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {(OUT / 'report.json').relative_to(ROOT)}")


if __name__ == "__main__":
    main()
