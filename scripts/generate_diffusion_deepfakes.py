#!/usr/bin/env python3
"""
Generate synthetic listing-style product photos with Stable Diffusion.

These images are fully out-of-distribution vs scraped training data and are useful
for probing the authentic vs deepfake binary classifier.

Install (after your CUDA PyTorch venv is active):
  pip install -r scripts/requirements_diffusion.txt

Examples:
  python scripts/generate_diffusion_deepfakes.py --num 8 --out-dir synthetic_deepfakes
  python scripts/generate_diffusion_deepfakes.py --model-id runwayml/stable-diffusion-v1-5 --steps 30
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import torch

DEFAULT_PROMPTS = [
    # Sneaker-style (generic — avoid trademark names in prompts)
    "professional ecommerce photo of a luxury white leather low-top sneaker, seamless gray studio backdrop, soft diffused lighting, 85mm product photography, ultra sharp",
    "close-up handheld photo of a black knit slip-on runner sneaker in a bright shoe store, shallow depth of field, phone camera realism",
    "studio product shot of a low-top sneaker with brown checkered textile panels and off-white midsole, white background, catalog lighting",
    # Watch-style
    "macro product photo of an ultra-thin tonneau shaped skeleton luxury watch with visible gears, black rubber strap, dark studio background, dramatic rim light",
    "wristwatch held on a small white display stand, skeleton dial, star-shaped bezel screws, indoor shop lighting, smartphone photo perspective",
    "close-up of a sporty tonneau watch case side profile with textured crown, blue ribbed rubber strap, hand holding stand, shallow DOF",
    # Mixed / harder scenes
    "busy sneaker store shelf with multiple colorful athletic shoes on wooden planks, slight motion blur, harsh overhead LEDs, wide angle phone snapshot",
    "flat lay of a single cream colored leather sneaker on tiled floor, top-down view, marketplace listing aesthetic",
]


def _slug(s: str, max_len: int = 48) -> str:
    t = re.sub(r"[^a-zA-Z0-9]+", "_", s.lower()).strip("_")
    return t[:max_len] or "prompt"


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic images with Stable Diffusion")
    repo_root = Path(__file__).resolve().parents[1]
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=repo_root / "synthetic_deepfakes",
        help="Output directory (created if missing)",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="runwayml/stable-diffusion-v1-5",
        help="HF model id (SD 1.5 family recommended for 12–16 GB GPUs)",
    )
    parser.add_argument("--num", type=int, default=8, help="How many images to generate")
    parser.add_argument("--steps", type=int, default=28, help="DDIM inference steps")
    parser.add_argument("--guidance", type=float, default=7.5, help="Classifier-free guidance scale")
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42, help="Base seed (incremented per image)")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--fp16",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use float16 on CUDA (recommended)",
    )
    parser.add_argument(
        "--prompt-file",
        type=Path,
        default=None,
        help="Optional text file: one prompt per line (overrides built prompts if set)",
    )
    args = parser.parse_args()

    try:
        from diffusers import StableDiffusionPipeline
    except ImportError as e:
        raise SystemExit(
            "Missing diffusers. Install with:\n  pip install -r scripts/requirements_diffusion.txt\n"
        ) from e

    if args.prompt_file is not None:
        lines = [ln.strip() for ln in args.prompt_file.read_text().splitlines() if ln.strip()]
        prompts = lines
    else:
        prompts = list(DEFAULT_PROMPTS)

    if args.num < 1:
        raise SystemExit("--num must be >= 1")

    dtype = torch.float16 if args.fp16 and args.device == "cuda" else torch.float32

    print(f"Loading {args.model_id} on {args.device} ({dtype})...")
    pipe = StableDiffusionPipeline.from_pretrained(
        args.model_id,
        torch_dtype=dtype,
        safety_checker=None,
        requires_safety_checker=False,
    )
    pipe = pipe.to(args.device)
    if hasattr(pipe, "enable_attention_slicing"):
        pipe.enable_attention_slicing()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    for i in range(args.num):
        prompt = prompts[i % len(prompts)]
        gen = torch.Generator(device=args.device).manual_seed(args.seed + i)
        with torch.inference_mode():
            out = pipe(
                prompt=prompt,
                num_inference_steps=args.steps,
                guidance_scale=args.guidance,
                width=args.width,
                height=args.height,
                generator=gen,
            )
        img = out.images[0]
        name = f"synth_{i:03d}_{_slug(prompt)}.png"
        path = args.out_dir / name
        img.save(path)
        print(f"  wrote {path}")

    print(f"Done. {args.num} images in {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()
