"""Watch-image transforms for the RTX 5080 authenticity experiments.

``conservative`` is the forensic training profile and the trainer default.
``current_legacy`` keeps the older aggressive pipeline for comparison only.
MixUp is not applied here; the trainer owns ``mixup_alpha`` of 0 or 0.05.

ImageNet normalization matches ``manifest.IMAGENET_MEAN`` / ``IMAGENET_STD``.
DINOv2 is always 504×504. DINOv3 is always 512×512.
"""

from __future__ import annotations

import io
import json
import random
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from manifest import IMAGENET_MEAN, IMAGENET_STD

CONSERVATIVE = "conservative"
CURRENT_LEGACY = "current_legacy"
PROFILES = (CONSERVATIVE, CURRENT_LEGACY)

FAMILY_IMAGE_SIZE = {
    "dinov2": 504,
    "dinov3": 512,
}

# Documented trainer settings. This module never mixes images.
MIXUP_ALPHA_DISABLED = 0.0
MIXUP_ALPHA_LIGHT = 0.05
MIXUP_ALPHAS = (MIXUP_ALPHA_DISABLED, MIXUP_ALPHA_LIGHT)

LEGACY_MARKER = "legacy"
TRAIN_TRANSFORM_VERSION = {
    CONSERVATIVE: "conservative_train_v1",
    CURRENT_LEGACY: "current_legacy_train_v1",
}
EVAL_TRANSFORM_VERSION = {
    CONSERVATIVE: "resize_pad_square_eval_v1",
    CURRENT_LEGACY: "resize_center_crop_eval_v1",
}

# Milder than torchvision's RandomPerspective default of 0.5, which warps
# bezel and bracelet geometry. The phase specifies probability only.
CONSERVATIVE_PERSPECTIVE_DISTORTION = 0.10
CONSERVATIVE_BLUR_PROBABILITY = 0.10
CONSERVATIVE_BLUR_SIGMA = (0.1, 0.8)


def imagenet_mean_fill() -> tuple[int, int, int]:
    """Neutral RGB fill. After ImageNet normalization these pixels are near zero."""
    return tuple(int(round(channel * 255.0)) for channel in IMAGENET_MEAN)


class ResizePadToSquare:
    """Resize the long side to ``size`` and pad the short side. Aspect ratio stays put."""

    def __init__(self, size: int, fill: tuple[int, int, int] | None = None) -> None:
        if size < 1:
            raise ValueError(f"size must be positive, got {size}")
        self.size = int(size)
        self.fill = imagenet_mean_fill() if fill is None else tuple(int(channel) for channel in fill)

    def __call__(self, image: Image.Image) -> Image.Image:
        rgb = image.convert("RGB")
        width, height = rgb.size
        scale = self.size / float(max(width, height))
        resized_w = max(1, min(self.size, int(round(width * scale))))
        resized_h = max(1, min(self.size, int(round(height * scale))))
        resized = rgb.resize((resized_w, resized_h), Image.Resampling.BICUBIC)
        canvas = Image.new("RGB", (self.size, self.size), self.fill)
        left = (self.size - resized_w) // 2
        top = (self.size - resized_h) // 2
        canvas.paste(resized, (left, top))
        return canvas

    def __repr__(self) -> str:
        return f"ResizePadToSquare(size={self.size}, fill={self.fill})"


class JPEGCompression:
    """Re-encode an RGB image as JPEG and decode it back to RGB."""

    def __init__(self, quality_range: tuple[int, int] = (60, 95)) -> None:
        low, high = quality_range
        if not 1 <= low <= high <= 100:
            raise ValueError(f"Invalid JPEG quality range {quality_range}")
        self.quality_range = (int(low), int(high))

    def __call__(self, image: Image.Image) -> Image.Image:
        rgb = image.convert("RGB")
        quality = random.randint(*self.quality_range)
        buffer = io.BytesIO()
        rgb.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        decoded = Image.open(buffer).convert("RGB")
        return decoded.copy()

    def __repr__(self) -> str:
        return f"JPEGCompression(quality_range={self.quality_range})"


def image_size_for_family(family: str) -> int:
    try:
        return FAMILY_IMAGE_SIZE[family]
    except KeyError as exc:
        raise ValueError(f"Unsupported family '{family}'. Expected dinov2 or dinov3.") from exc


def preprocessing_record(family: str, profile: str, classifier_arch: str) -> dict:
    """Fields stored in experiment ``config.json``. Both runs must match these."""
    profile = _validate_profile(profile)
    size = image_size_for_family(family)
    return {
        "model_family": family,
        "model_architecture": classifier_arch,
        "input_size": size,
        "augmentation_profile": profile,
        "normalization_mean": [float(channel) for channel in IMAGENET_MEAN],
        "normalization_std": [float(channel) for channel in IMAGENET_STD],
        "training_transform": TRAIN_TRANSFORM_VERSION[profile],
        "evaluation_transform": EVAL_TRANSFORM_VERSION[profile],
    }


def augmentation_report(family: str, profile: str, train: bool) -> dict:
    """JSON-serializable description of one transform configuration."""
    size = image_size_for_family(family)
    profile = _validate_profile(profile)
    if profile == CONSERVATIVE:
        steps = _conservative_steps(train)
    else:
        steps = _legacy_steps(train)
    return {
        "family": family,
        "profile": profile,
        "legacy": profile == CURRENT_LEGACY,
        "train": bool(train),
        "image_size": size,
        "interpolation": "bicubic",
        "normalization": {
            "mean": list(IMAGENET_MEAN),
            "std": list(IMAGENET_STD),
        },
        "mixup": {
            "applied_in_image_transform": False,
            "supported_alpha": list(MIXUP_ALPHAS),
        },
        "steps": steps,
    }


def build_transforms(family: str, *, train: bool, profile: str = CONSERVATIVE) -> transforms.Compose:
    """Build a train or eval transform. Eval is deterministic."""
    size = image_size_for_family(family)
    profile = _validate_profile(profile)
    if profile == CURRENT_LEGACY:
        pipeline = _legacy_pipeline(size, train=train)
    else:
        pipeline = _conservative_pipeline(size, train=train)
    return pipeline


def _validate_profile(profile: str) -> str:
    if profile not in PROFILES:
        raise ValueError(
            f"Unknown augmentation profile '{profile}'. Expected one of: {', '.join(PROFILES)}."
        )
    return profile


def _normalize() -> list:
    return [
        transforms.ToTensor(),
        transforms.Normalize(mean=list(IMAGENET_MEAN), std=list(IMAGENET_STD)),
    ]


def _conservative_pipeline(size: int, *, train: bool) -> transforms.Compose:
    fill = imagenet_mean_fill()
    if not train:
        return transforms.Compose(
            [
                ResizePadToSquare(size, fill=fill),
                *_normalize(),
            ]
        )
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(
                size,
                scale=(0.75, 1.0),
                ratio=(0.90, 1.10),
                interpolation=InterpolationMode.BICUBIC,
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(
                degrees=5,
                interpolation=InterpolationMode.BICUBIC,
                fill=fill,
            ),
            transforms.RandomPerspective(
                distortion_scale=CONSERVATIVE_PERSPECTIVE_DISTORTION,
                p=0.15,
                interpolation=InterpolationMode.BICUBIC,
                fill=fill,
            ),
            transforms.ColorJitter(brightness=0.20, contrast=0.20, saturation=0.10, hue=0.03),
            transforms.RandomApply(
                [transforms.GaussianBlur(kernel_size=3, sigma=CONSERVATIVE_BLUR_SIGMA)],
                p=CONSERVATIVE_BLUR_PROBABILITY,
            ),
            transforms.RandomApply([JPEGCompression((60, 95))], p=0.25),
            *_normalize(),
        ]
    )


def _legacy_pipeline(size: int, *, train: bool) -> transforms.Compose:
    """Older aggressive pipeline. Not the trainer default. Marked legacy."""
    if not train:
        return transforms.Compose(
            [
                transforms.Resize(size + 16, interpolation=InterpolationMode.BICUBIC),
                transforms.CenterCrop(size),
                *_normalize(),
            ]
        )
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(
                size,
                scale=(0.55, 1.0),
                ratio=(0.8, 1.25),
                interpolation=InterpolationMode.BICUBIC,
            ),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(degrees=15, interpolation=InterpolationMode.BICUBIC),
            transforms.RandomPerspective(distortion_scale=0.2, p=0.3, interpolation=InterpolationMode.BICUBIC),
            transforms.ColorJitter(brightness=0.35, contrast=0.35, saturation=0.25, hue=0.08),
            transforms.RandomGrayscale(p=0.03),
            transforms.RandomApply(
                [transforms.GaussianBlur(kernel_size=7, sigma=(0.1, 3.0))],
                p=0.2,
            ),
            transforms.RandomApply([JPEGCompression((40, 90))], p=0.3),
            *_normalize(),
        ]
    )


def _conservative_steps(train: bool) -> list[dict]:
    fill = list(imagenet_mean_fill())
    if not train:
        return [
            {
                "op": "resize_preserve_aspect",
                "size": "family_image_size",
                "random": False,
            },
            {
                "op": "pad_to_square",
                "size": "family_image_size",
                "fill_rgb": fill,
                "random": False,
            },
        ]
    return [
        {"op": "random_resized_crop", "scale": [0.75, 1.0], "ratio": [0.90, 1.10]},
        {"op": "horizontal_flip", "p": 0.5},
        {"op": "rotation", "degrees": 5, "fill_rgb": fill},
        {
            "op": "perspective",
            "p": 0.15,
            "distortion_scale": CONSERVATIVE_PERSPECTIVE_DISTORTION,
            "fill_rgb": fill,
        },
        {
            "op": "color_jitter",
            "brightness": 0.20,
            "contrast": 0.20,
            "saturation": 0.10,
            "hue": 0.03,
        },
        {
            "op": "gaussian_blur",
            "p": CONSERVATIVE_BLUR_PROBABILITY,
            "kernel_size": 3,
            "sigma": list(CONSERVATIVE_BLUR_SIGMA),
        },
        {"op": "jpeg", "p": 0.25, "quality": [60, 95]},
        {"op": "random_grayscale", "enabled": False},
    ]


def _legacy_steps(train: bool) -> list[dict]:
    if not train:
        return [
            {"op": "resize", "size": "family_image_size_plus_16", "random": False},
            {"op": "center_crop", "size": "family_image_size"},
        ]
    return [
        {"op": "random_resized_crop", "scale": [0.55, 1.0], "ratio": [0.8, 1.25], "marker": LEGACY_MARKER},
        {"op": "horizontal_flip", "p": 0.5, "marker": LEGACY_MARKER},
        {"op": "rotation", "degrees": 15, "marker": LEGACY_MARKER},
        {"op": "perspective", "p": 0.3, "distortion_scale": 0.2, "marker": LEGACY_MARKER},
        {
            "op": "color_jitter",
            "brightness": 0.35,
            "contrast": 0.35,
            "saturation": 0.25,
            "hue": 0.08,
            "marker": LEGACY_MARKER,
        },
        {"op": "random_grayscale", "enabled": True, "p": 0.03, "marker": LEGACY_MARKER},
        {
            "op": "gaussian_blur",
            "p": 0.2,
            "kernel_size": 7,
            "sigma": [0.1, 3.0],
            "marker": LEGACY_MARKER,
        },
        {"op": "jpeg", "p": 0.3, "quality": [40, 90], "marker": LEGACY_MARKER},
    ]


def iter_transforms(pipeline: transforms.Compose):
    for transform in pipeline.transforms:
        if isinstance(transform, transforms.RandomApply):
            yield transform
            yield from transform.transforms
        else:
            yield transform


def contains_random_grayscale(pipeline: transforms.Compose) -> bool:
    return any(isinstance(transform, transforms.RandomGrayscale) for transform in iter_transforms(pipeline))


def denormalize(tensor: torch.Tensor) -> Image.Image:
    """Inverse ImageNet normalization for preview images."""
    mean = torch.tensor(IMAGENET_MEAN, dtype=tensor.dtype).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD, dtype=tensor.dtype).view(3, 1, 1)
    image = (tensor.detach().cpu() * std + mean).clamp(0, 1)
    array = (image.permute(1, 2, 0).numpy() * 255).round().astype("uint8")
    return Image.fromarray(array, mode="RGB")


def write_preview_grid(
    image_paths: list[Path],
    out_dir: Path,
    *,
    family: str = "dinov2",
    train_samples: int = 4,
) -> Path:
    """Save a grid of original, deterministic eval, and conservative train views."""
    if not image_paths:
        raise ValueError("At least one image path is required")
    out_dir.mkdir(parents=True, exist_ok=True)
    eval_tf = build_transforms(family, train=False, profile=CONSERVATIVE)
    train_tf = build_transforms(family, train=True, profile=CONSERVATIVE)
    columns = 2 + train_samples
    tiles: list[list[Image.Image]] = []
    labels = ["original", "eval"] + [f"train {index + 1}" for index in range(train_samples)]
    for path in image_paths:
        source = Image.open(path).convert("RGB")
        row = [source.resize((FAMILY_IMAGE_SIZE[family], FAMILY_IMAGE_SIZE[family]), Image.Resampling.BICUBIC)]
        row.append(denormalize(eval_tf(source)))
        for _ in range(train_samples):
            row.append(denormalize(train_tf(source)))
        tiles.append(row)

    tile_w, tile_h = tiles[0][0].size
    label_h = 22
    caption_h = 22
    canvas = Image.new(
        "RGB",
        (columns * tile_w, label_h + len(tiles) * (tile_h + caption_h)),
        (16, 16, 16),
    )
    from PIL import ImageDraw

    draw = ImageDraw.Draw(canvas)
    for col_index, label in enumerate(labels):
        draw.text((col_index * tile_w + 8, 4), label, fill=(180, 210, 180))
    for row_index, row in enumerate(tiles):
        top = label_h + row_index * (tile_h + caption_h)
        name = image_paths[row_index].parent.name
        draw.text((8, top + 4), name[:64], fill=(240, 240, 240))
        for col_index, tile in enumerate(row):
            canvas.paste(tile, (col_index * tile_w, top + caption_h))
    destination = out_dir / f"{family}_{CONSERVATIVE}_preview.png"
    canvas.save(destination)
    report_path = out_dir / f"{family}_{CONSERVATIVE}_preview.json"
    report_path.write_text(
        json.dumps(
            {
                "images": [str(path) for path in image_paths],
                "train": augmentation_report(family, CONSERVATIVE, True),
                "eval": augmentation_report(family, CONSERVATIVE, False),
            },
            indent=2,
        )
        + "\n"
    )
    return destination
