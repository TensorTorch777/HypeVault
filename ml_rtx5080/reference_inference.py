"""Canonical frozen-model inference used by the shadow path.

Preprocessing, the checkpoint load, and the forward pass live here so the
backend shadow wrapper does not reimplement them. This module does not train,
does not read the final test, and does not change a production verdict.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from policy_contract import (
    EXPECTED_CHECKPOINT_SHA256,
    INPUT_SIZE,
    MODEL_ARCHITECTURE,
    MODEL_FAMILY,
    MODEL_HEAD,
    PHASE23_CUTOFFS,
    PREPROCESSING_VERSION,
    PolicyConfigurationError,
)

REFERENCE_DTYPE_NOTE = "fp32 eval forward matching evaluation.collect_logits; no autocast and no temperature"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_reference_identity(
    *,
    family: str = MODEL_FAMILY,
    architecture: str = MODEL_ARCHITECTURE,
    head: str = MODEL_HEAD,
    input_size: int = INPUT_SIZE,
    preprocessing_version: str = PREPROCESSING_VERSION,
    checkpoint_sha256: str | None = None,
) -> None:
    """Refuse a shadow load that is not the frozen DINOv3 contract."""
    codes = ("POLICY_MODEL_MISMATCH", "POLICY_CONFIGURATION_ERROR")
    if family != MODEL_FAMILY:
        raise PolicyConfigurationError(f"model family {family!r} is not {MODEL_FAMILY}", codes)
    if architecture != MODEL_ARCHITECTURE:
        raise PolicyConfigurationError(f"architecture {architecture!r} is not {MODEL_ARCHITECTURE}", codes)
    if head != MODEL_HEAD:
        raise PolicyConfigurationError(f"head {head!r} is not {MODEL_HEAD}", codes)
    if int(input_size) != INPUT_SIZE:
        raise PolicyConfigurationError(f"input size {input_size} is not {INPUT_SIZE}", codes)
    if preprocessing_version != PREPROCESSING_VERSION:
        raise PolicyConfigurationError(
            f"preprocessing {preprocessing_version!r} is not {PREPROCESSING_VERSION}",
            ("POLICY_CONFIGURATION_ERROR",),
        )
    if checkpoint_sha256 is not None and checkpoint_sha256 != EXPECTED_CHECKPOINT_SHA256:
        raise PolicyConfigurationError("checkpoint hash does not match the frozen artifact", codes)


def load_reference_model(checkpoint: Path, device):
    """Load the frozen checkpoint. A hash or architecture mismatch raises before use."""
    from evaluation import build_eval_model

    path = Path(checkpoint)
    digest = file_sha256(path)
    validate_reference_identity(checkpoint_sha256=digest)
    model, image_size = build_eval_model(MODEL_FAMILY, MODEL_HEAD, path, device)
    if int(image_size) != INPUT_SIZE:
        raise PolicyConfigurationError(f"loaded image size {image_size} is not {INPUT_SIZE}")
    model.eval()
    return model, digest


def preprocess_image(image):
    """resize_pad_square_eval_v1 at 512. Returns a CHW float tensor."""
    import torch
    from augmentations import CONSERVATIVE, EVAL_TRANSFORM_VERSION, build_transforms

    if EVAL_TRANSFORM_VERSION[CONSERVATIVE] != PREPROCESSING_VERSION:
        raise PolicyConfigurationError("eval transform version drifted from resize_pad_square_eval_v1")
    tensor = build_transforms(MODEL_FAMILY, train=False, profile=CONSERVATIVE)(image)
    if tuple(tensor.shape) != (3, INPUT_SIZE, INPUT_SIZE):
        raise PolicyConfigurationError(f"preprocessed shape {tuple(tensor.shape)} is not 3x512x512")
    if not torch.isfinite(tensor).all():
        raise PolicyConfigurationError("preprocessed tensor is non-finite", ("POLICY_INVALID_INPUT",))
    return tensor


def forward_logits(model, batch, device):
    """Raw logits. Temperature is not applied here."""
    import torch

    if batch.ndim != 4 or tuple(batch.shape[1:]) != (3, INPUT_SIZE, INPUT_SIZE):
        raise PolicyConfigurationError(f"forward batch shape {tuple(batch.shape)} is not NCHW 512")
    moved = batch.to(device)
    with torch.inference_mode():
        output = model(moved)
    if isinstance(output, tuple):
        output = output[0]
    values = output.detach().float().reshape(-1).cpu()
    if not torch.isfinite(values).all():
        raise PolicyConfigurationError("model returned a non-finite logit", ("POLICY_INVALID_INPUT",))
    return values


def quality_flags_for_image(image) -> tuple[dict, tuple[str, ...]]:
    """Phase 23 flags from the original RGB image. They are not an authenticity score."""
    from image_quality import apply_quality_flags, extract_quality_features

    features = extract_quality_features(image)
    flags = tuple(apply_quality_flags(features, {"cutoffs": dict(PHASE23_CUTOFFS)}))
    return features, flags
