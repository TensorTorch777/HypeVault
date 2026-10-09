"""Allowlisted research router for DINOv2 and DINOv3.

Logical identifiers are the only values a client may send. Triton model
names, versions, and checkpoint paths are server-side. A missing model
returns an error. The other model is never used as a substitute.
"""

from __future__ import annotations

from typing import Any

from inference.triton_client import infer_named_model, named_model_ready

DINOV2_LEGACY = "dinov2_legacy"
DINOV3_EXPERIMENTAL = "dinov3_experimental"

FROZEN_DINOV3_SHA256 = "5a38c93fd442b03653c65d2a5ecc9c2687ef152f7c5c020763e4ce1fd9c7d28f"
FROZEN_DINOV3_TEMPERATURE = 0.24038200410185356
FROZEN_DINOV3_THRESHOLD = 0.5

# Fixed before any Triton parity measurement. These match the existing
# DINOv3 serving contract and are not revised after a failed comparison.
PARITY_LOGIT_ATOL = 1e-4
PARITY_PROBABILITY_ATOL = 1e-5

_MODELS: dict[str, dict[str, Any]] = {
    DINOV2_LEGACY: {
        "logical_id": DINOV2_LEGACY,
        "response_model": "LEGACY_DINOV2",
        "triton_name": "dinov2_classifier",
        "version": "1",
        "input_name": "input__0",
        "output_name": "output__0",
        "input_dims": [3, 518, 518],
        "input_dtype": "FP32",
        "output_meaning": "raw_logit",
        "temperature": None,
        "decision_threshold": None,
        "preprocessing": "legacy_square_resize_518_imagenet",
        "precision": "FP32",
        "research_only": False,
        "production_route": True,
    },
    DINOV3_EXPERIMENTAL: {
        "logical_id": DINOV3_EXPERIMENTAL,
        "response_model": "DINOV3_RESEARCH_PROTOTYPE",
        "triton_name": "dinov3_authenticity_candidate",
        "version": "1",
        "input_name": "input__0",
        "output_name": "output__0",
        "input_dims": [3, 512, 512],
        "input_dtype": "FP32",
        "output_meaning": "raw_logit",
        "temperature": FROZEN_DINOV3_TEMPERATURE,
        "decision_threshold": FROZEN_DINOV3_THRESHOLD,
        "preprocessing": "resize_pad_square_eval_v1",
        "precision": "FP32",
        "checkpoint_sha256": FROZEN_DINOV3_SHA256,
        "research_only": True,
        "production_route": False,
    },
}


class ModelRoutingError(Exception):
    def __init__(self, status: str, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.message = message


def allowlist() -> tuple[str, ...]:
    return tuple(_MODELS)


def resolve_model(model_id: str) -> dict[str, Any]:
    spec = _MODELS.get(model_id)
    if spec is None:
        raise ModelRoutingError("UNKNOWN_MODEL", "Unknown model identifier.")
    return spec


async def model_readiness(model_id: str) -> dict[str, Any]:
    spec = resolve_model(model_id)
    ready = await named_model_ready(spec["triton_name"])
    return {
        "logical_id": spec["logical_id"],
        "triton_name": spec["triton_name"],
        "version": spec["version"],
        "ready": ready,
        "research_only": spec["research_only"],
    }


async def infer_allowlisted_model(model_id: str, array_nchw) -> dict[str, Any]:
    """Run one allowlisted model. Unavailable and failed calls do not switch models."""
    spec = resolve_model(model_id)
    ready = await named_model_ready(spec["triton_name"])
    if not ready:
        raise ModelRoutingError(
            "MODEL_UNAVAILABLE",
            f"{spec['triton_name']} is not ready. No other model was used.",
        )
    try:
        output = await infer_named_model(
            spec["triton_name"],
            array_nchw,
            spec["input_name"],
            spec["output_name"],
        )
        logit = float(output.reshape(-1)[0])
    except Exception as exc:
        raise ModelRoutingError(
            "INFERENCE_ERROR",
            "The selected model did not return a result.",
        ) from exc
    return {
        "logit": logit,
        "logical_id": spec["logical_id"],
        "triton_name": spec["triton_name"],
        "version": spec["version"],
        "response_model": spec["response_model"],
        "temperature": spec["temperature"],
    }
