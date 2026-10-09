"""Shadow-only policy beside the current production verdict.

Default is off. This module does not change AuthenticateResponse, does not
call the production DINOv2 loader, and does not apply a second temperature
to a sigmoid probability. OOD stays unavailable.
"""

from __future__ import annotations

import io
import logging
import os
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
_ML = _REPO / "ml_rtx5080"
if str(_ML) not in sys.path:
    sys.path.insert(0, str(_ML))

from policy_contract import (  # noqa: E402
    EXPECTED_CHECKPOINT_SHA256,
    INPUT_SIZE,
    MODEL_ARCHITECTURE,
    MODEL_FAMILY,
    MODEL_HEAD,
    MODEL_VERSION,
    POLICY_VERSION,
    PREPROCESSING_VERSION,
    SYSTEM_STATUS_ERROR,
    PolicyConfigurationError,
    PolicyEngine,
    PolicyResult,
    frozen_shadow_configuration,
)
from quality_fastpath import fast_quality_gate  # noqa: E402
from inference.research_guard import (  # noqa: E402
    DINOV3_PRODUCTION_BLOCK_REASON,
    deployment_mode,
    dinov3_research_allowed,
)
from reference_inference import (  # noqa: E402
    forward_logits,
    load_reference_model,
    preprocess_image,
    validate_reference_identity,
)

_log = logging.getLogger(__name__)
SHADOW_ENV = "HYPEVAULT_SHADOW_POLICY"
SERVING_ENV = "HYPEVAULT_SHADOW_SERVING"
_MODEL = None
_MODEL_SHA = None
_DEVICE = None
_TEST_PATHS: set[str] | None = None


def shadow_mode_enabled() -> bool:
    """Customer traffic stays on the existing verdict unless this is set explicitly."""
    raw = os.environ.get(SHADOW_ENV, "0").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def shadow_serving_backend() -> str:
    """GPU and Triton serving stay off unless the environment selects them."""
    raw = os.environ.get(SERVING_ENV, "pytorch").strip().lower()
    if raw in {"", "pytorch", "reference"}:
        return "pytorch"
    if raw in {"onnx", "dinov3_onnx"}:
        return "onnx"
    if raw in {"onnx_cuda", "cuda"}:
        return "onnx_cuda"
    if raw == "triton":
        return "triton"
    raise PolicyConfigurationError(
        f"shadow serving backend {raw!r} is not pytorch, onnx, onnx_cuda, or triton",
        ("POLICY_CONFIGURATION_ERROR",),
    )


def reset_shadow_for_tests() -> None:
    global _MODEL, _MODEL_SHA, _DEVICE, _TEST_PATHS
    _MODEL = None
    _MODEL_SHA = None
    _DEVICE = None
    _TEST_PATHS = None


def bind_shadow_model(model, checkpoint_sha256: str, device) -> None:
    """Reuse a reference model already loaded for parity. This does not load a second copy."""
    global _MODEL, _MODEL_SHA, _DEVICE
    if checkpoint_sha256 != EXPECTED_CHECKPOINT_SHA256:
        raise PolicyConfigurationError("checkpoint hash does not match the frozen artifact")
    validate_reference_identity(checkpoint_sha256=checkpoint_sha256)
    _MODEL = model
    _MODEL_SHA = checkpoint_sha256
    _DEVICE = device


def compose_shadow_record(
    existing_verdict: str,
    existing_confidence: float,
    image=None,
    *,
    enabled: bool | None = None,
    sample_id: str | None = None,
    split: str | None = None,
) -> dict:
    """Keep the production verdict. Add shadow fields only when the mode is enabled."""
    record = {"verdict": existing_verdict, "confidence": float(existing_confidence)}
    if enabled is None:
        enabled = shadow_mode_enabled()
    if not enabled:
        return record
    try:
        result, _timings = run_shadow_image(image, sample_id=sample_id, split=split)
    except PolicyConfigurationError as exc:
        result = _configuration_error(str(exc), exc.reason_codes)
    record["shadow_decision"] = result.decision
    record["shadow_reason_codes"] = list(result.reason_codes)
    record["shadow_authenticity_probability"] = result.authenticity_probability
    record["shadow_quality_flags"] = list(result.quality_flags)
    record["shadow_ood_status"] = result.ood_status
    record["shadow_policy_version"] = result.policy_version
    record["shadow_model_version"] = result.model_version
    record["shadow_system_status"] = result.system_status
    record["shadow_model_role"] = "RESEARCH MODEL — NOT FOR PRODUCTION"
    record["verdict"] = existing_verdict
    _log.info(
        "dinov3_research_request mode=%s model_version=%s checkpoint_sha=%s policy_version=%s decision=%s",
        deployment_mode(),
        result.model_version,
        EXPECTED_CHECKPOINT_SHA256,
        result.policy_version,
        result.decision,
    )
    record["confidence"] = float(existing_confidence)
    return record


def run_shadow_image(image, *, sample_id: str | None = None, split: str | None = None, path: Path | None = None):
    """One image. Invalid input returns POLICY_ERROR and does not guess AUTHENTIC."""
    if split == "test":
        raise RuntimeError("shadow inference encountered a final-test sample")
    if path is not None:
        _reject_test_path(Path(path))
    if sample_id is not None and _sample_id_is_test(sample_id):
        raise RuntimeError("shadow inference encountered a final-test sample")
    opened, error = _coerce_image(image)
    engine = PolicyEngine(frozen_shadow_configuration())
    if error is not None:
        return engine._error(("POLICY_INVALID_INPUT",), error), {}
    import time

    started = time.perf_counter()
    features, flags = fast_quality_gate(opened)
    quality_s = time.perf_counter()
    tensor = preprocess_image(opened).unsqueeze(0)
    preprocess_s = time.perf_counter()
    logit, checkpoint_sha = _shadow_logit(tensor)
    model_s = time.perf_counter()
    result = engine.evaluate(
        logit=logit,
        quality_features=features,
        quality_flags=flags,
        preprocessing_version=PREPROCESSING_VERSION,
        input_size=INPUT_SIZE,
        checkpoint_sha256=checkpoint_sha,
        split=split,
        sample_id=sample_id,
    )
    policy_s = time.perf_counter()
    timings = {
        "quality_ms": (quality_s - started) * 1000,
        "preprocess_ms": (preprocess_s - quality_s) * 1000,
        "model_ms": (model_s - preprocess_s) * 1000,
        "policy_ms": (policy_s - model_s) * 1000,
        "total_ms": (policy_s - started) * 1000,
    }
    return result, timings


def run_shadow_batch(images, *, device=None) -> list[PolicyResult]:
    """Same preprocess and forward as the reference. Batch size does not change the policy."""
    if not images:
        return []
    if not dinov3_research_allowed():
        raise PolicyConfigurationError(
            DINOV3_PRODUCTION_BLOCK_REASON,
            ("POLICY_CONFIGURATION_ERROR",),
        )
    backend = shadow_serving_backend()
    if backend == "onnx_cuda":
        return _run_cuda_batch(images)
    if backend == "triton":
        raise PolicyConfigurationError(
            "Triton is not available for dinov3_authenticity_classifier. "
            "Shadow inference did not run and did not fall back to DINOv2.",
            ("POLICY_INFRASTRUCTURE_ERROR",),
        )
    if backend == "onnx":
        return _run_onnx_batch(images)
    model, resolved = _cached_model() if device is None else (_cached_model()[0], device)
    import torch

    tensors = [preprocess_image(image) for image in images]
    batch = torch.stack(tensors)
    logits = forward_logits(model, batch, resolved)
    engine = PolicyEngine(frozen_shadow_configuration())
    results = []
    for image, logit in zip(images, logits.tolist(), strict=True):
        features, flags = fast_quality_gate(image)
        results.append(
            engine.evaluate(
                logit=float(logit),
                quality_features=features,
                quality_flags=flags,
                preprocessing_version=PREPROCESSING_VERSION,
                input_size=INPUT_SIZE,
                checkpoint_sha256=_MODEL_SHA,
            )
        )
    return results


def _cached_model():
    """Bound models stay in place. Otherwise load the CUDA DINOv3 shadow backend."""
    global _MODEL, _MODEL_SHA, _DEVICE
    if _MODEL is None:
        from inference.dinov3_model import shared_backend

        backend = shared_backend()
        backend.load()
        _MODEL = backend.model
        _MODEL_SHA = backend.checkpoint_sha256
        _DEVICE = backend.device
    return _MODEL, _DEVICE


def _reject_test_path(path: Path) -> None:
    resolved = str(path.resolve())
    if resolved in _blocked_test_paths():
        raise RuntimeError("shadow inference encountered a final-test sample")


def _sample_id_is_test(sample_id: str) -> bool:
    catalog = _REPO / sample_id
    if catalog.is_file() and str(catalog.resolve()) in _blocked_test_paths():
        return True
    return False


def _blocked_test_paths() -> set[str]:
    global _TEST_PATHS
    if _TEST_PATHS is None:
        manifest = _REPO / "ml_rtx5080" / "experiments" / "dataset_audit" / "split_manifest_v2.json"
        if not manifest.is_file():
            _TEST_PATHS = set()
        else:
            import json

            payload = json.loads(manifest.read_text())
            _TEST_PATHS = {str(Path(path).resolve()) for path in (payload.get("membership") or {}).get("test") or []}
    return _TEST_PATHS


def _coerce_image(image):
    from PIL import Image, UnidentifiedImageError

    if image is None:
        return None, "missing image file"
    if isinstance(image, (bytes, bytearray)):
        try:
            with Image.open(io.BytesIO(image)) as handle:
                rgb = handle.convert("RGB")
                rgb.load()
                return rgb.copy(), None
        except (UnidentifiedImageError, OSError, Image.DecompressionBombError, ValueError) as exc:
            return None, f"corrupted image metadata: {exc}"
    if isinstance(image, Path) or isinstance(image, str):
        path = Path(image)
        _reject_test_path(path)
        if not path.is_file():
            return None, "missing image file"
        try:
            with Image.open(path) as handle:
                rgb = handle.convert("RGB")
                rgb.load()
                return rgb.copy(), None
        except (UnidentifiedImageError, OSError, Image.DecompressionBombError, ValueError) as exc:
            return None, f"corrupted image metadata: {exc}"
    if hasattr(image, "convert"):
        return image.convert("RGB"), None
    return None, "missing image file"


def _configuration_error(message: str, reason_codes: tuple[str, ...]) -> PolicyResult:
    engine = PolicyEngine(frozen_shadow_configuration())
    return engine._error(tuple(reason_codes) or ("POLICY_CONFIGURATION_ERROR",), message)


def shadow_contract_metadata() -> dict:
    validate_reference_identity()
    return {
        "model_family": MODEL_FAMILY,
        "backbone": MODEL_ARCHITECTURE,
        "head": MODEL_HEAD,
        "input_size": INPUT_SIZE,
        "preprocessing_version": PREPROCESSING_VERSION,
        "policy_version": POLICY_VERSION,
        "model_version": MODEL_VERSION,
        "shadow_mode_default": False,
        "shadow_serving_default": "pytorch",
        "system_error_status": SYSTEM_STATUS_ERROR,
    }


def _shadow_logit(tensor):
    """Raw logit from the selected shadow backend. GPU and Triton do not fall back."""
    if not dinov3_research_allowed():
        raise PolicyConfigurationError(
            DINOV3_PRODUCTION_BLOCK_REASON,
            ("POLICY_CONFIGURATION_ERROR",),
        )
    backend = shadow_serving_backend()
    if backend == "pytorch":
        model, device = _cached_model()
        return float(forward_logits(model, tensor, device)[0]), _MODEL_SHA
    if backend == "onnx":
        from dinov3_serving import forward_onnx_logits, serving_checkpoint_sha

        values = forward_onnx_logits(tensor)
        return float(values[0]), serving_checkpoint_sha()
    if backend == "onnx_cuda":
        from dinov3_serving import forward_cuda_logits

        values, sha = forward_cuda_logits(tensor)
        return float(values[0]), sha
    if backend == "triton":
        raise PolicyConfigurationError(
            "Triton is not available for dinov3_authenticity_classifier. "
            "Shadow inference did not run and did not fall back to DINOv2.",
            ("POLICY_INFRASTRUCTURE_ERROR",),
        )
    raise PolicyConfigurationError(
        f"shadow serving backend {backend!r} is not executable",
        ("POLICY_CONFIGURATION_ERROR",),
    )


def _run_cuda_batch(images) -> list[PolicyResult]:
    import torch
    from dinov3_serving import forward_cuda_logits

    tensors = [preprocess_image(image) for image in images]
    logits, sha = forward_cuda_logits(torch.stack(tensors))
    return _policy_batch(images, logits, sha)


def _run_onnx_batch(images) -> list[PolicyResult]:
    import torch
    from dinov3_serving import forward_onnx_logits, serving_checkpoint_sha

    tensors = [preprocess_image(image) for image in images]
    logits = forward_onnx_logits(torch.stack(tensors))
    return _policy_batch(images, logits, serving_checkpoint_sha())


def _policy_batch(images, logits, sha: str) -> list[PolicyResult]:
    engine = PolicyEngine(frozen_shadow_configuration())
    results = []
    for image, logit in zip(images, logits.tolist(), strict=True):
        features, flags = fast_quality_gate(image)
        results.append(
            engine.evaluate(
                logit=float(logit),
                quality_features=features,
                quality_flags=flags,
                preprocessing_version=PREPROCESSING_VERSION,
                input_size=INPUT_SIZE,
                checkpoint_sha256=sha,
            )
        )
    return results
