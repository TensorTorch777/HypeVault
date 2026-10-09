"""Experiment settings for the RTX trainer.

Batch size is configurable and is not treated as VRAM-safe. The starting
point targets an effective batch near 32 via ``batch_size * accumulate_grad``
without locking either knob to a measured 16 GB training footprint.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import torch
import torch.nn as nn

from classifier import CLASSIFIER_ARCHS
from model import (
    DINOV2_IMAGE_SIZE,
    DINOV2_TIMM_NAME,
    DINOV3_IMAGE_SIZE,
    DINOV3_TIMM_NAME,
)

FAMILY_IMAGE_SIZE = {"dinov2": DINOV2_IMAGE_SIZE, "dinov3": DINOV3_IMAGE_SIZE}
FAMILY_TIMM_NAME = {"dinov2": DINOV2_TIMM_NAME, "dinov3": DINOV3_TIMM_NAME}

SUPPORTED_LABEL_SMOOTHING = (0.0, 0.02, 0.03)
SUPPORTED_MIXUP_ALPHAS = (0.0, 0.05)
# Below this minority/majority ratio the counts are reported as imbalanced.
# The default sampler stays shuffle unless the caller opts into weighting.
IMBALANCE_RATIO = 0.90
FORBIDDEN_OUTPUT_DIR_NAMES = frozenset({"checkpoints", "checkpoints_watches"})


def _match_allowed(value: float, allowed: tuple[float, ...], label: str) -> float:
    value = float(value)
    for candidate in allowed:
        if math.isclose(value, candidate, rel_tol=0.0, abs_tol=1e-8):
            return candidate
    rendered = ", ".join(f"{item:.2f}" for item in allowed)
    raise ValueError(f"{label} must be one of {rendered}, got {value}")


def validate_label_smoothing(value: float) -> float:
    return _match_allowed(value, SUPPORTED_LABEL_SMOOTHING, "label_smoothing")


def validate_mixup_alpha(value: float) -> float:
    return _match_allowed(value, SUPPORTED_MIXUP_ALPHAS, "mixup_alpha")


def validate_classifier_arch(arch: str) -> str:
    if arch not in CLASSIFIER_ARCHS:
        options = ", ".join(CLASSIFIER_ARCHS)
        raise ValueError(f"classifier_arch must be one of {options}, got {arch}")
    return arch


def resolve_family_image_size(family: str, img_size: int) -> int:
    if family not in FAMILY_IMAGE_SIZE:
        raise ValueError(f"family must be dinov2 or dinov3, got {family}")
    expected = FAMILY_IMAGE_SIZE[family]
    if int(img_size) not in (0, expected):
        raise ValueError(
            f"{family} trains at {expected}px, got {img_size}. Pass 0 to use that size."
        )
    return expected


def smooth_binary_targets(targets: torch.Tensor, label_smoothing: float) -> torch.Tensor:
    """``y' = y * (1 - s) + 0.5 * s``. The model still returns a raw logit."""
    smoothing = validate_label_smoothing(label_smoothing)
    if smoothing == 0.0:
        return targets
    return targets * (1.0 - smoothing) + 0.5 * smoothing


def build_bce_loss() -> nn.BCEWithLogitsLoss:
    """Binary loss on raw logits. Smoothing is applied to targets, not via sigmoid."""
    return nn.BCEWithLogitsLoss()


def resolve_amp(requested: str, *, cuda: bool, bf16_supported: bool) -> dict:
    """Prefer BF16 on CUDA when the device supports it. Otherwise fall back.

    CUDA without BF16 uses FP16 and a GradScaler. CPU uses FP32 with the
    scaler disabled, because the training autocast path is CUDA-only.
    """
    kind = str(requested).lower()
    if kind not in {"bf16", "fp16", "fp32"}:
        raise ValueError(f"amp_dtype must be bf16, fp16, or fp32, got {requested}")
    if kind == "bf16" and cuda and bf16_supported:
        return {"amp_dtype": "bf16", "amp_scaler": False, "amp_fallback": None}
    if kind == "bf16" and cuda:
        return {"amp_dtype": "fp16", "amp_scaler": True, "amp_fallback": "fp16"}
    if kind == "bf16":
        return {"amp_dtype": "fp32", "amp_scaler": False, "amp_fallback": "fp32"}
    if kind == "fp16" and cuda:
        return {"amp_dtype": "fp16", "amp_scaler": True, "amp_fallback": None}
    if kind == "fp16":
        return {"amp_dtype": "fp32", "amp_scaler": False, "amp_fallback": "fp32"}
    return {"amp_dtype": "fp32", "amp_scaler": False, "amp_fallback": None}


def class_balance(labels: list[int] | tuple[int, ...]) -> dict:
    counts = {0: 0, 1: 0}
    for label in labels:
        value = int(label)
        if value not in counts:
            raise ValueError(f"Expected binary label 0 or 1, got {value}")
        counts[value] += 1
    majority = max(counts.values()) if counts else 0
    minority = min(counts.values()) if counts else 0
    ratio = (minority / majority) if majority else 1.0
    return {
        "counts": counts,
        "minority_ratio": ratio,
        "materially_imbalanced": ratio < IMBALANCE_RATIO,
    }


def sampler_decision(labels: list[int] | tuple[int, ...], *, weighted_sampler: bool = False) -> dict:
    """Default is shuffle with no weighted sampler, including on balanced data.

    An imbalanced catalog is reported and still uses shuffle unless
    ``weighted_sampler`` is set.
    """
    balance = class_balance(labels)
    if weighted_sampler:
        return {**balance, "sampler": "weighted_random", "shuffle": False}
    return {**balance, "sampler": "none", "shuffle": True}


def assert_experiment_output_dir(path: Path) -> Path:
    """Refuse the existing checkpoint directories. Experiment folders are allowed."""
    resolved = Path(path).resolve()
    hits = [part for part in resolved.parts if part in FORBIDDEN_OUTPUT_DIR_NAMES]
    if hits:
        raise ValueError(
            f"Refusing to write training outputs into {resolved}. "
            "Use an experiment directory such as ml_rtx5080/experiments/<run>."
        )
    return resolved


def write_experiment_config(path: Path, cfg: dict) -> None:
    destination = Path(path)
    assert_experiment_output_dir(destination.parent)
    payload = dict(cfg)
    payload["mixup_alpha"] = validate_mixup_alpha(payload["mixup_alpha"])
    payload["label_smoothing"] = validate_label_smoothing(payload["label_smoothing"])
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2) + "\n")


def save_training_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    config: dict,
    global_step: int,
    val_acc: float | None = None,
    val_f1: float | None = None,
) -> None:
    destination = Path(path)
    assert_experiment_output_dir(destination.parent)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": int(epoch),
            "global_step": int(global_step),
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": None if scheduler is None else scheduler.state_dict(),
            "config": config,
            "val_acc": val_acc,
            "val_f1": val_f1,
        },
        destination,
    )


def load_training_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler=None,
    map_location: str | torch.device = "cpu",
) -> dict:
    checkpoint = torch.load(Path(path), map_location=map_location, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    if optimizer is not None and checkpoint.get("optimizer_state") is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
    if scheduler is not None and checkpoint.get("scheduler_state") is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state"])
    return checkpoint
