"""Allowlisted research router for DINOv2 and DINOv3.

Logical identifiers are the only values a client may send. Triton model
names, versions, and checkpoint paths are server-side. A missing model
returns an error. The other model is never used as a substitute.

A model is selectable only when Triton serves it on an instance kind, with
the CUDA execution-provider settings, that passed Triton parity.
`validated_serving` changes only in a reviewed commit that cites the result.
"""

from __future__ import annotations

import math
from typing import Any

from inference.triton_client import (
    infer_named_model,
    named_model_ready,
    named_model_serving_profile,
    triton_server_status,
)

# GPU parity failed with ONNX Runtime's default TF32 math and passed with exact FP32.
EXACT_FP32_GPU = {"use_tf32": "0"}

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
        "triton_name": "dinov2_vitb14_live",
        "version": "1",
        "input_name": "input__0",
        "output_name": "output__0",
        "input_dims": [3, 504, 504],
        "input_dtype": "FP32",
        "output_meaning": "raw_logit",
        "temperature": None,
        "decision_threshold": None,
        "decision_policy": "legacy_logit_v1_with_min_authentic_confidence",
        "preprocessing": "legacy_square_resize_504_imagenet",
        "precision": "FP32",
        "architecture": "DINOv2 ViT-B/14 (vit_base_patch14_dinov2.lvd142m), live listing-check checkpoint, 504 input",
        "source_checkpoint_sha256": "fe1daa0bf71c5e9b73267d40784442748b8fd1999a8d107979f1338c52f0fa66",
        "same_model_as_live_route": True,
        "validated_serving": {"KIND_GPU": EXACT_FP32_GPU},
        "validation_evidence": {
            "KIND_GPU": "dinov2_live_triton_gpu_parity_attempt2.json (PASS, protocol dinov2_live_triton_gpu_protocol_v2.json)",
        },
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
        "decision_policy": "shadow_v1",
        "preprocessing": "resize_pad_square_eval_v1",
        "precision": "FP32",
        "architecture": "DINOv3 ViT-B/16 cls_patch_attention ONNX export, 512 input",
        "checkpoint_sha256": FROZEN_DINOV3_SHA256,
        "same_model_as_live_route": False,
        "validated_serving": {"KIND_CPU": {}, "KIND_GPU": EXACT_FP32_GPU},
        "validation_evidence": {
            "KIND_CPU": "parity_results.json (CPU Triton parity PASS)",
            "KIND_GPU": "dinov3_triton_gpu_parity_attempt2.json (PASS, protocol dinov3_triton_gpu_protocol_v2.json)",
        },
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


async def _availability(spec: dict[str, Any]) -> tuple[bool, str | None, list[str]]:
    if not await named_model_ready(spec["triton_name"], spec["version"]):
        return False, "TRITON_MODEL_NOT_READY", []
    profile = await named_model_serving_profile(spec["triton_name"], spec["version"])
    if not profile or not profile["kinds"]:
        return False, "INSTANCE_KIND_UNKNOWN", []
    kinds = profile["kinds"]
    validated = spec["validated_serving"]
    if not kinds <= set(validated):
        return False, "PARITY_NOT_VALIDATED_FOR_SERVED_INSTANCE", sorted(kinds)
    for kind in kinds:
        required = validated[kind]
        if any(profile["cuda_parameters"].get(key) != value for key, value in required.items()):
            return False, "SERVING_PRECISION_NOT_VALIDATED", sorted(kinds)
    return True, None, sorted(kinds)


async def model_readiness(model_id: str) -> dict[str, Any]:
    spec = resolve_model(model_id)
    ready, reason, kinds = await _availability(spec)
    return {
        "logical_id": spec["logical_id"],
        "model": spec["response_model"],
        "triton_name": spec["triton_name"],
        "version": spec["version"],
        "architecture": spec["architecture"],
        "same_model_as_live_route": spec["same_model_as_live_route"],
        "served_instance_kinds": kinds,
        "validated_instance_kinds": sorted(spec["validated_serving"]),
        "ready": ready,
        "unavailable_reason": reason,
        "research_only": spec["research_only"],
    }


async def readiness_report() -> dict[str, Any]:
    server = await triton_server_status()
    models = [await model_readiness(model_id) for model_id in allowlist()]
    return {"server": server, "models": models, "publication_decision": "BLOCKED"}


async def infer_allowlisted_model(model_id: str, array_nchw) -> dict[str, Any]:
    """Run one allowlisted model. Unavailable, unvalidated, and failed calls do not switch models."""
    spec = resolve_model(model_id)
    ready, reason, _kinds = await _availability(spec)
    if not ready:
        raise ModelRoutingError(
            "MODEL_UNAVAILABLE",
            f"{spec['triton_name']} is not available ({reason}). No other model was used.",
        )
    try:
        output = await infer_named_model(
            spec["triton_name"],
            spec["version"],
            array_nchw,
            spec["input_name"],
            spec["output_name"],
        )
        values = output.reshape(-1)
        if values.size != array_nchw.shape[0]:
            raise ValueError(f"expected {array_nchw.shape[0]} logits, got {values.size}")
        logit = float(values[0])
    except Exception as exc:
        raise ModelRoutingError(
            "INFERENCE_ERROR",
            "The selected model did not return a result.",
        ) from exc
    if not math.isfinite(logit):
        raise ModelRoutingError("INFERENCE_ERROR", "The selected model returned a non-finite output.")
    return {
        "logit": logit,
        "logical_id": spec["logical_id"],
        "triton_name": spec["triton_name"],
        "version": spec["version"],
        "response_model": spec["response_model"],
        "temperature": spec["temperature"],
    }
