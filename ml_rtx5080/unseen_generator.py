"""Phase 70: score images from generators that did not produce the training folders.

The generator behind Label_1_Watches is not recorded. Every Label_1 file is
512×512, so SDXL base 1.0 (native 1024×1024, two text encoders) is the primary
unseen generator. SD 1.5 (native 512×512) is reported beside it because it may
be close to the training generator.

Every image here is generated, so the only measurable rate is detection: the
fraction scored P(generated) >= 0.50. Temperature does not change that decision
(it is the sign of the logit). Real photos are not re-scored; Phase 69 already
measured the false-fake rate.

Scorers:
  random_split   Phase 69 brand-mixed model on normalized images (primary)
  held_out_fold  Phase 69 fold model that never saw that brand, on normalized images
  frozen_raw     frozen production epoch_018 on the raw generator PNG, production eval transform
  frozen_norm    frozen production epoch_018 on the normalized image

    .venv/bin/python ml_rtx5080/unseen_generator.py generate
    .venv/bin/python ml_rtx5080/unseen_generator.py score
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

_PKG = Path(__file__).resolve().parent
_REPO = _PKG.parent
if str(_PKG) not in sys.path:
    sys.path.insert(0, str(_PKG))

from cross_brand import FOLDS
from normalize_surface import normalize_image, save_jpeg

PROTOCOL_ID = "unseen_generator_v1"
OUT = _PKG / "experiments" / PROTOCOL_ID
DATA = _REPO / "data_unseen_generator_v1"
PHASE69_WEIGHTS = _PKG / "experiments" / "normalized_cross_brand_v1" / "weights"
FROZEN = _PKG / "experiments" / "v2_dinov3_cls_patch_attention" / "epoch_018.pt"
FROZEN_SHA256 = "5a38c93fd442b03653c65d2a5ecc9c2687ef152f7c5c020763e4ce1fd9c7d28f"
SEED = 57
PER_PROMPT = 2

GENERATORS = {
    "sdxl_base_1_0": {
        "model_id": "stabilityai/stable-diffusion-xl-base-1.0",
        "size": 1024,
        "steps": 30,
        "guidance": 6.0,
        "variant": "fp16",
    },
    "sd_1_5": {
        "model_id": "runwayml/stable-diffusion-v1-5",
        "size": 512,
        "steps": 30,
        "guidance": 7.5,
        "variant": None,
    },
}
MODEL_LINES = {
    "A. Lange & Söhne": ["Lange 1", "Saxonia", "Zeitwerk", "Datograph", "Odysseus"],
    "Audemars Piguet": ["Royal Oak", "Royal Oak Offshore", "Code 11.59", "Royal Oak Concept", "Jules Audemars"],
    "Patek Philippe": ["Nautilus", "Aquanaut", "Calatrava", "Grand Complications", "Annual Calendar"],
    "Richard Mille": ["RM 011", "RM 035", "RM 055", "RM 67-01", "RM 027"],
    "Vacheron Constantin": ["Overseas", "Patrimony", "Traditionnelle", "Fiftysix", "Historiques"],
}
SCENES = [
    "professional studio product photo on a seamless white background, soft box lighting",
    "worn on a wrist outdoors, smartphone photo, natural daylight",
    "inside a jewelry store display case, glass reflections, warm spotlights",
    "flat lay on a wooden table, marketplace listing photo, slightly off-center",
    "macro close-up of the dial and hands, shallow depth of field",
    "held in a hand against a plain wall, indoor lighting, phone camera",
    "next to its presentation box and papers, top-down photo",
    "side profile of the case and crown on a watch stand, studio lighting",
    "on a leather watch roll under a warm desk lamp at night",
    "on a dark slate surface, dramatic rim light, luxury catalog style",
]
NEGATIVE = "cartoon, illustration, drawing, render, text, watermark, logo overlay, blurry, deformed"
THRESHOLD = 0.5


def prompts() -> list[dict]:
    rows = []
    for brand, lines in MODEL_LINES.items():
        for line in lines:
            for scene in SCENES:
                rows.append({"brand": brand, "prompt": f"photo of a {brand} {line} luxury wristwatch, {scene}"})
    return rows


def protocol_document() -> dict:
    return {
        "protocol_id": PROTOCOL_ID,
        "question": "Do the Phase 69 and frozen production models detect images from generators outside the training folders?",
        "claim_scope": "detection of generated images only; real photos are not re-scored here",
        "generators": GENERATORS,
        "training_generator": "not recorded; every Label_1 file is 512x512",
        "prompt_template": "photo of a {brand} {model line} luxury wristwatch, {scene}",
        "model_lines": MODEL_LINES,
        "scenes": SCENES,
        "negative_prompt": NEGATIVE,
        "images_per_prompt": PER_PROMPT,
        "images_per_generator": len(prompts()) * PER_PROMPT,
        "seed_rule": f"seed = {SEED} * 100000 + prompt_index * {PER_PROMPT} + repeat",
        "normalization": "normalize_surface.normalize_image then save_jpeg (448x448, baseline JPEG q90, no EXIF)",
        "scorers": {
            "random_split": "Phase 69 random_split_cls_patch_attention.pt on normalized images (primary)",
            "held_out_fold": "Phase 69 fold model whose held-out brand is the image's prompt brand, normalized images",
            "frozen_raw": "frozen epoch_018.pt on the raw PNG with the production eval transform",
            "frozen_norm": "frozen epoch_018.pt on the normalized image",
        },
        "frozen_checkpoint_sha256": FROZEN_SHA256,
        "decision": "P(generated) >= 0.50, equivalently logit >= 0",
        "interval": "Clopper-Pearson 95% on the image-level detection rate",
        "not_claimed": [
            "authentic versus counterfeit watch",
            "generators other than the two named",
            "independence between images of the same prompt",
        ],
    }


def _write_json(path: Path, payload: dict) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(payload, indent=2, ensure_ascii=False) + "\n").encode()
    path.write_bytes(data)
    digest = hashlib.sha256(data).hexdigest()
    path.with_suffix(".sha256").write_text(f"{digest}  {path.name}\n")
    return digest


def _brand_dir(brand: str) -> str:
    return brand.replace(" ", "_").replace("&", "and").replace(".", "")


def generate(only: str | None) -> None:
    import torch
    from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline
    from PIL import Image

    _write_json(OUT / "protocol_v1.json", protocol_document())
    rows = prompts()
    manifest_path = OUT / "generation_manifest.csv"
    existing = {}
    if manifest_path.is_file():
        existing = {row["raw_path"]: row for row in csv.DictReader(manifest_path.open())}
    records = list(existing.values())
    for name, spec in GENERATORS.items():
        if only and name != only:
            continue
        cls = StableDiffusionXLPipeline if name.startswith("sdxl") else StableDiffusionPipeline
        kwargs = {"torch_dtype": torch.float16}
        if spec["variant"]:
            kwargs["variant"] = spec["variant"]
        if cls is StableDiffusionPipeline:
            kwargs["safety_checker"] = None
            kwargs["requires_safety_checker"] = False
        pipe = cls.from_pretrained(spec["model_id"], local_files_only=True, **kwargs).to("cuda")
        pipe.set_progress_bar_config(disable=True)
        started = time.time()
        made = 0
        for index, row in enumerate(rows):
            for repeat in range(PER_PROMPT):
                seed = SEED * 100000 + index * PER_PROMPT + repeat
                stem = f"{index:03d}_{repeat}"
                raw = DATA / name / "raw" / _brand_dir(row["brand"]) / f"{stem}.png"
                norm = DATA / name / "normalized" / _brand_dir(row["brand"]) / f"{stem}.jpg"
                if str(raw) in existing and raw.is_file() and norm.is_file():
                    continue
                generator = torch.Generator(device="cuda").manual_seed(seed)
                image = pipe(
                    prompt=row["prompt"],
                    negative_prompt=NEGATIVE,
                    num_inference_steps=spec["steps"],
                    guidance_scale=spec["guidance"],
                    width=spec["size"],
                    height=spec["size"],
                    generator=generator,
                ).images[0]
                raw.parent.mkdir(parents=True, exist_ok=True)
                image.save(raw, format="PNG")
                with Image.open(raw) as reopened:
                    square = normalize_image(reopened)
                save_jpeg(square, norm)
                record = {
                    "generator": name,
                    "brand": row["brand"],
                    "prompt_index": index,
                    "repeat": repeat,
                    "seed": seed,
                    "prompt": row["prompt"],
                    "raw_path": str(raw),
                    "normalized_path": str(norm),
                }
                existing[str(raw)] = record
                records.append(record)
                made += 1
                if made % 50 == 0:
                    rate = made / (time.time() - started)
                    print(f"{name}: {made} new images, {rate:.2f} img/s", flush=True)
                    _write_manifest(manifest_path, records)
        _write_manifest(manifest_path, records)
        print(f"{name}: done, {made} new images in {time.time() - started:.0f}s", flush=True)
        del pipe
        torch.cuda.empty_cache()


def _write_manifest(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["generator", "brand", "prompt_index", "repeat", "seed", "prompt", "raw_path", "normalized_path"]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in sorted(records, key=lambda r: (r["generator"], int(r["prompt_index"]), int(r["repeat"]))):
            writer.writerow({key: record[key] for key in fields})


def _binomial_cdf(k: int, n: int, p: float) -> float:
    """P(X <= k) for X ~ Binomial(n, p), summed in log space so large n does not overflow."""
    if p <= 0.0:
        return 1.0
    if p >= 1.0:
        return 1.0 if k >= n else 0.0
    log_p, log_q = math.log(p), math.log1p(-p)
    base = math.lgamma(n + 1)
    return min(1.0, sum(
        math.exp(base - math.lgamma(i + 1) - math.lgamma(n - i + 1) + i * log_p + (n - i) * log_q)
        for i in range(k + 1)
    ))


def _bisect(predicate, low: float = 0.0, high: float = 1.0) -> float:
    """Smallest p in [low, high] where ``predicate`` turns true; ``predicate`` must be monotone."""
    for _ in range(100):
        mid = (low + high) / 2.0
        if predicate(mid):
            high = mid
        else:
            low = mid
    return (low + high) / 2.0


def clopper_pearson(k: int, n: int) -> tuple[float, float]:
    """Exact two-sided 95% interval. Pure Python, so CI does not need scipy."""
    low = 0.0 if k == 0 else _bisect(lambda p: 1.0 - _binomial_cdf(k - 1, n, p) >= 0.025)
    high = 1.0 if k == n else _bisect(lambda p: _binomial_cdf(k, n, p) <= 0.025)
    return low, high


def _load(path: Path):
    import torch
    from classifier import build_authenticity_classifier

    model = build_authenticity_classifier("dinov3", "cls_patch_attention", image_size=512, pretrained=False, dropout=0.15)
    state = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model_state"])
    return model.to("cuda").eval()


def _logits(model, paths: list[str]) -> np.ndarray:
    import torch
    from PIL import Image
    from train import build_transforms

    transform = build_transforms("dinov3", train=False)
    out = []
    with torch.no_grad():
        for start in range(0, len(paths), 32):
            batch = []
            for path in paths[start : start + 32]:
                with Image.open(path) as image:
                    batch.append(transform(image.convert("RGB")))
            out.append(model(torch.stack(batch).to("cuda")).float().cpu().numpy().reshape(-1))
    return np.concatenate(out)


def _summary(values: np.ndarray) -> dict:
    detected = int((values >= 0).sum())
    low, high = clopper_pearson(detected, len(values))
    return {
        "n": int(len(values)),
        "detected": detected,
        "detection_rate": detected / len(values),
        "ci95": [low, high],
        "missed": int(len(values) - detected),
        "median_logit": float(np.median(values)),
    }


def score() -> None:
    import torch

    if hashlib.sha256(FROZEN.read_bytes()).hexdigest() != FROZEN_SHA256:
        raise RuntimeError("frozen checkpoint hash does not match")
    records = list(csv.DictReader((OUT / "generation_manifest.csv").open()))
    logits ={name: np.full(len(records), np.nan) for name in ("random_split", "held_out_fold", "frozen_raw", "frozen_norm")}
    norm_paths = [row["normalized_path"] for row in records]

    model = _load(PHASE69_WEIGHTS / "random_split_cls_patch_attention.pt")
    logits["random_split"] = _logits(model, norm_paths)
    del model
    for directory, brand in FOLDS:
        index = [i for i, row in enumerate(records) if row["brand"] == brand]
        model = _load(PHASE69_WEIGHTS / f"{directory}_cls_patch_attention.pt")
        logits["held_out_fold"][index] = _logits(model, [norm_paths[i] for i in index])
        del model
        torch.cuda.empty_cache()
    model = _load(FROZEN)
    logits["frozen_raw"] = _logits(model, [row["raw_path"] for row in records])
    logits["frozen_norm"] = _logits(model, norm_paths)
    del model

    with (OUT / "scores.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["generator", "brand", "prompt_index", "repeat", *logits])
        for i, row in enumerate(records):
            writer.writerow([row["generator"], row["brand"], row["prompt_index"], row["repeat"], *(f"{logits[k][i]:.6f}" for k in logits)])

    result = {"protocol_sha256": (OUT / "protocol_v1.sha256").read_text().split()[0], "threshold": THRESHOLD, "generators": {}}
    for generator in GENERATORS:
        mask = np.array([row["generator"] == generator for row in records])
        if not mask.any():
            continue
        block = {name: _summary(values[mask]) for name, values in logits.items()}
        block["by_brand"] = {}
        for _directory, brand in FOLDS:
            brand_mask = mask & np.array([row["brand"] == brand for row in records])
            block["by_brand"][brand] = {name: _summary(values[brand_mask]) for name, values in logits.items()}
        result["generators"][generator] = block
    _write_json(OUT / "detection_metrics.json", result)
    for generator, block in result["generators"].items():
        print(generator, {name: f"{block[name]['detected']}/{block[name]['n']}" for name in logits})


CONTROL_PER_BRAND = 100
FROZEN_SPLIT = _PKG / "experiments" / "dataset_audit" / "split_manifest_v2.json"
PHASE69_MANIFEST = _PKG / "experiments" / "normalized_cross_brand_v1" / "fold_manifest.json"


def _sample_real(rows: list[dict], key: str = "path") -> list[dict]:
    """Up to CONTROL_PER_BRAND Label_0 rows per brand, chosen by a seeded shuffle."""
    rng = np.random.default_rng(SEED)
    chosen = []
    for _directory, brand in FOLDS:
        pool = sorted((row for row in rows if row["brand"] == brand), key=lambda row: row[key])
        order = rng.permutation(len(pool))[:CONTROL_PER_BRAND]
        chosen += [pool[i] for i in sorted(order)]
    return chosen


def control() -> None:
    """Score real photos each scorer never trained on, through the same image path as the generated set."""
    import torch
    from normalize_surface import output_path, roc_auc

    phase69 = json.loads(PHASE69_MANIFEST.read_text())
    controls: dict[str, list[tuple[str, str]]] = {}
    random_rows = [r for r in phase69["random_split_control"]["rows"] if r["role"] == "random_test" and r["label"] == 0]
    controls["random_split"] = [(r["brand"], r["path"]) for r in _sample_real(random_rows)]
    held = []
    for directory, _brand in FOLDS:
        rows = [r for r in phase69["folds"][directory]["rows"] if r["role"] == "cross_brand_test" and r["label"] == 0]
        held += [(directory, r["brand"], r["path"]) for r in _sample_real(rows)]
    frozen_rows = [
        r for r in json.loads(FROZEN_SPLIT.read_text())["samples"]
        if r["split"] == "validation" and r["label"] == 0 and output_path(Path(r["path"]), 0).is_file()
    ]
    frozen_rows = _sample_real(frozen_rows)
    controls["frozen_raw"] = [(r["brand"], r["path"]) for r in frozen_rows]
    controls["frozen_norm"] = [(r["brand"], str(output_path(Path(r["path"]), 0))) for r in frozen_rows]

    real_logits: dict[str, np.ndarray] = {}
    model = _load(PHASE69_WEIGHTS / "random_split_cls_patch_attention.pt")
    real_logits["random_split"] = _logits(model, [p for _b, p in controls["random_split"]])
    del model
    parts = []
    for directory, _brand in FOLDS:
        model = _load(PHASE69_WEIGHTS / f"{directory}_cls_patch_attention.pt")
        parts.append(_logits(model, [p for d, _b, p in held if d == directory]))
        del model
        torch.cuda.empty_cache()
    real_logits["held_out_fold"] = np.concatenate(parts)
    controls["held_out_fold"] = [(b, p) for _d, b, p in held]
    model = _load(FROZEN)
    real_logits["frozen_raw"] = _logits(model, [p for _b, p in controls["frozen_raw"]])
    real_logits["frozen_norm"] = _logits(model, [p for _b, p in controls["frozen_norm"]])
    del model

    with (OUT / "control_scores.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["scorer", "brand", "path", "logit"])
        for name, pairs in controls.items():
            for (brand, path), value in zip(pairs, real_logits[name]):
                writer.writerow([name, brand, Path(path).relative_to(_REPO), f"{value:.6f}"])

    generated = list(csv.DictReader((OUT / "scores.csv").open()))
    result = {
        "protocol_sha256": (OUT / "protocol_v1.sha256").read_text().split()[0],
        "control_sources": {
            "random_split": "Phase 69 random_test Label_0, normalized",
            "held_out_fold": "each fold's cross_brand_test Label_0 (brand unseen by that model), normalized",
            "frozen_raw": "split_manifest_v2 validation Label_0, original files (used for epoch selection, not training)",
            "frozen_norm": "the same rows, normalized",
        },
        "scorers": {},
    }
    for name, values in real_logits.items():
        rejected = int((values >= 0).sum())
        low, high = clopper_pearson(rejected, len(values))
        block = {"real_n": int(len(values)), "real_called_generated": rejected, "false_fake_rate": rejected / len(values), "ffr_ci95": [low, high]}
        for generator in GENERATORS:
            fake = np.array([float(row[name]) for row in generated if row["generator"] == generator])
            scores = np.concatenate([values, fake])
            labels = np.concatenate([np.zeros(len(values), dtype=int), np.ones(len(fake), dtype=int)])
            block[generator] = {"detection_rate": float((fake >= 0).mean()), "roc_auc_vs_real_control": roc_auc(scores, labels)}
        result["scorers"][name] = block
    _write_json(OUT / "control_metrics.json", result)
    for name, block in result["scorers"].items():
        print(name, f"real called AI {block['real_called_generated']}/{block['real_n']}",
              {g: round(block[g]["roc_auc_vs_real_control"], 4) for g in GENERATORS})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    gen = sub.add_parser("generate")
    gen.add_argument("--only", choices=sorted(GENERATORS))
    sub.add_parser("score")
    sub.add_parser("control")
    args = parser.parse_args()
    if args.command == "generate":
        generate(args.only)
    elif args.command == "score":
        score()
    else:
        control()


if __name__ == "__main__":
    main()
