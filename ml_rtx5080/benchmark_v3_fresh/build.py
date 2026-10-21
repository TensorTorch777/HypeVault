"""Collect and generate the fresh benchmark_v3 images (protocol.json, written before any data).

Same admission code as benchmark_v2 (512 px normalisation and the training exclusions); every
benchmark_v1 and benchmark_v2 listing and image is excluded too.

    .venv/bin/python ml_rtx5080/benchmark_v3_fresh/build.py real --per-brand 60
    .venv/bin/python ml_rtx5080/benchmark_v3_fresh/build.py stable_cascade_lite --per-brand 60
    .venv/bin/python ml_rtx5080/benchmark_v3_fresh/build.py kandinsky22_v3 --per-brand 60

No model scores anything here.
"""

from __future__ import annotations

import argparse
import asyncio
import io
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
V2 = HERE.parent / "benchmark_v2_fresh"
sys.path.insert(0, str(V2))

import build as v2  # noqa: E402  (sets the 512 px normalisation)
from common import BenchmarkDedup, TrainingIndex, read_manifest  # noqa: E402

DATA = HERE / "data"
MANIFEST = DATA / "manifest.jsonl"
GROUPS = {
    "stable_cascade_lite": {"model_id": "stabilityai/stable-cascade", "size": 768, "steps": 20, "guidance": 4.0, "seed": 74_000},
    "kandinsky22_v3": {"model_id": "kandinsky-community/kandinsky-2-2-decoder", "size": 768, "steps": 50, "guidance": 4.0, "seed": 75_000},
}


def dedup_all() -> tuple[BenchmarkDedup, list]:
    """Every v1 and v2 image, admitted or not, and every admitted v3 image is a duplicate target."""
    dedup = BenchmarkDedup()
    earlier = list(read_manifest(v2.V1_MANIFEST)) + list(read_manifest(v2.MANIFEST))
    v3 = list(read_manifest(MANIFEST)) if MANIFEST.exists() else []
    for s in earlier + [s for s in v3 if s.excluded is None]:
        dedup.sha256s.add(s.original_sha256)
        dedup.dhashes.append(int(s.original_dhash, 16))
    return dedup, earlier + v3


def _cascade_pipeline():
    """Prior and decoder built separately: diffusers 0.38 cannot resolve the repo's 'wuerstchen' VQ-GAN entry."""
    import torch
    from diffusers import DDPMWuerstchenScheduler, StableCascadeDecoderPipeline, StableCascadePriorPipeline, StableCascadeUNet
    from diffusers.pipelines.deprecated.wuerstchen.modeling_paella_vq_model import PaellaVQModel
    from transformers import CLIPTextModelWithProjection, CLIPTokenizer

    bf16 = torch.bfloat16
    prior = StableCascadePriorPipeline.from_pretrained(
        "stabilityai/stable-cascade-prior", variant="bf16", torch_dtype=bf16,
        prior=StableCascadeUNet.from_pretrained("stabilityai/stable-cascade-prior", subfolder="prior_lite", variant="bf16", torch_dtype=bf16),
    )
    repo = "stabilityai/stable-cascade"
    decoder = StableCascadeDecoderPipeline(
        decoder=StableCascadeUNet.from_pretrained(repo, subfolder="decoder_lite", variant="bf16", torch_dtype=bf16),
        tokenizer=CLIPTokenizer.from_pretrained(repo, subfolder="tokenizer"),
        text_encoder=CLIPTextModelWithProjection.from_pretrained(repo, subfolder="text_encoder", variant="bf16", torch_dtype=bf16),
        scheduler=DDPMWuerstchenScheduler.from_pretrained(repo, subfolder="scheduler"),
        vqgan=PaellaVQModel.from_pretrained(repo, subfolder="vqgan", torch_dtype=bf16),
    )
    for pipe in (prior, decoder):
        pipe.enable_model_cpu_offload()
        pipe.set_progress_bar_config(disable=True)

    def run(prompt, negative_prompt, height, width, generator, prior_num_inference_steps, prior_guidance_scale, num_inference_steps, guidance_scale):
        embeddings = prior(prompt=prompt, negative_prompt=negative_prompt, height=height, width=width, generator=generator,
                           num_inference_steps=prior_num_inference_steps, guidance_scale=prior_guidance_scale).image_embeddings
        return decoder(image_embeddings=embeddings.to(bf16), prompt=prompt, negative_prompt=negative_prompt, generator=generator,
                       num_inference_steps=num_inference_steps, guidance_scale=guidance_scale)

    return run


def generate(group: str, per_brand: int) -> None:
    import torch

    cfg = GROUPS[group]
    dedup, _ = dedup_all()
    done = {s.sample_id for s in read_manifest(MANIFEST)} if MANIFEST.exists() else set()
    training = TrainingIndex.load()
    jobs = [j for j in v2.plan(group, per_brand, v2.models_from_dataset(Path.home() / "Downloads/archive/chrono24_unique_brand_plus_ref.csv"), cfg["seed"])
            if f"{group}_{j['seed']}" not in done]
    print(f"{group}: {len(jobs)} images to generate with {cfg['model_id']}", flush=True)
    if not jobs:
        return
    if group == "stable_cascade_lite":
        pipe = _cascade_pipeline()
    else:
        pipe = v2.load_pipeline(cfg["model_id"], "cuda", True)
    for n, job in enumerate(jobs, 1):
        generator = torch.Generator(device="cpu").manual_seed(job["seed"])
        kwargs = dict(prompt=job["prompt"], negative_prompt=v2.NEGATIVE, height=cfg["size"], width=cfg["size"], generator=generator)
        if group == "stable_cascade_lite":
            kwargs.update(prior_num_inference_steps=cfg["steps"], prior_guidance_scale=cfg["guidance"], num_inference_steps=10, guidance_scale=0.0)
        else:
            kwargs.update(num_inference_steps=cfg["steps"], guidance_scale=cfg["guidance"])
        image = pipe(**kwargs).images[0]
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        sample = v2._admit(
            buf.getvalue(), sample_id=f"{group}_{job['seed']}", label="GENERATED", group=group,
            brand=job["brand"], unit_id=f"{group}:{job['scene']}", out_dir=DATA / group, training=training, dedup=dedup,
            meta={**job, "negative_prompt": v2.NEGATIVE, "generator": cfg["model_id"], "steps": cfg["steps"],
                  "guidance": cfg["guidance"], "size": cfg["size"]},
        )
        v2.append_manifest(MANIFEST, [sample])
        print(f"[{n}/{len(jobs)}] {sample.sample_id} {job['brand']} -> {sample.excluded or 'admitted'}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("what", choices=("real", *GROUPS))
    parser.add_argument("--per-brand", type=int, default=60)
    args = parser.parse_args()
    if args.what == "real":
        # Reuse v2's collector with v3's paths, group name and duplicate set.
        v2.DATA, v2.MANIFEST, v2.dedup_with_v1 = DATA, MANIFEST, dedup_all
        real_group = "real_ebay_v3"
        original = v2._admit
        v2._admit = lambda raw, **kw: original(raw, **{**kw, "group": real_group, "out_dir": DATA / real_group})
        asyncio.run(v2.collect_real(args.per_brand, max_pages=10, delay=0.5))
    else:
        generate(args.what, args.per_brand)


if __name__ == "__main__":
    main()
