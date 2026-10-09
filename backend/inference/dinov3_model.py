"""GPU shadow inference for the frozen DINOv3 classifier.

This module does not serve customer traffic and does not load the DINOv2
production model. FP32 is the only authoritative forward. BF16 exists so a
benchmark can measure it. A machine without CUDA raises
POLICY_INFRASTRUCTURE_ERROR instead of falling back to CPU ONNX or DINOv2.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

_REPO = Path(__file__).resolve().parents[2]
_ML = _REPO / "ml_rtx5080"
if str(_ML) not in sys.path:
    sys.path.insert(0, str(_ML))

from policy import FROZEN_TEMPERATURE, calibrated_probability_from_logit  # noqa: E402
from policy_contract import (  # noqa: E402
    EXPECTED_CHECKPOINT_SHA256,
    INPUT_SIZE,
    MODEL_ARCHITECTURE,
    MODEL_FAMILY,
    MODEL_HEAD,
    PREPROCESSING_VERSION,
    PolicyConfigurationError,
)
from reference_inference import (  # noqa: E402
    forward_logits,
    load_reference_model,
    preprocess_image,
    quality_flags_for_image,
    validate_reference_identity,
)

CHECKPOINT = _ML / "experiments" / "v2_dinov3_cls_patch_attention" / "epoch_018.pt"
EXPECTED_GPU = "NVIDIA GeForce RTX 5080"

_SHARED = None


class DinoV3ShadowBackend:
    """Frozen DINOv3 weights on CUDA. The checkpoint file is only read."""

    def __init__(self) -> None:
        self.model = None
        self.device = None
        self.checkpoint_sha256 = None

    def load(self) -> "DinoV3ShadowBackend":
        """Load the frozen checkpoint onto CUDA. A mismatch raises before use."""
        self._require_cuda()
        if self.model is not None:
            return self
        validate_reference_identity()
        device = torch.device("cuda")
        try:
            model, digest = load_reference_model(CHECKPOINT, device)
        except PolicyConfigurationError:
            raise
        except Exception as exc:
            raise PolicyConfigurationError(
                f"DINOv3 shadow model failed to load. Inference did not fall back to CPU ONNX or DINOv2: {exc}",
                ("POLICY_INFRASTRUCTURE_ERROR",),
            ) from exc
        if digest != EXPECTED_CHECKPOINT_SHA256:
            raise PolicyConfigurationError(
                "checkpoint hash does not match the frozen artifact",
                ("POLICY_MODEL_MISMATCH", "POLICY_CONFIGURATION_ERROR"),
            )
        if model.training or any(module.training for module in model.modules() if isinstance(module, torch.nn.Dropout)):
            raise PolicyConfigurationError("DINOv3 shadow model is not in eval mode", ("POLICY_CONFIGURATION_ERROR",))
        if model.classifier_arch != MODEL_HEAD or model.backbone.backbone_name != MODEL_ARCHITECTURE:
            raise PolicyConfigurationError(
                "loaded DINOv3 module is not vit_base_patch16_dinov3.lvd1689m with cls_patch_attention",
                ("POLICY_MODEL_MISMATCH", "POLICY_CONFIGURATION_ERROR"),
            )
        if int(model.backbone.image_size) != INPUT_SIZE or int(model.backbone.num_register_tokens) != 4:
            raise PolicyConfigurationError("loaded DINOv3 input or register layout is not the frozen contract")
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        self.model = model
        self.device = device
        self.checkpoint_sha256 = digest
        return self

    def release(self) -> None:
        """Drop this process's model reference. The checkpoint file is not modified."""
        self.model = None
        self.device = None
        self.checkpoint_sha256 = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def preprocess(self, image):
        """resize_pad_square_eval_v1. This is the reference preprocess."""
        return preprocess_image(image)

    def predict_logits(self, batch: torch.Tensor) -> torch.Tensor:
        """Authoritative FP32 raw logits, shape [N, 1]. Temperature is not applied."""
        self._require_cuda()
        self.load()
        values = forward_logits(self.model, batch, self.device)
        return values.reshape(-1, 1)

    def predict_probability(self, batch: torch.Tensor) -> torch.Tensor:
        """sigmoid(logit / frozen temperature). The temperature is applied once, to the raw logit."""
        logits = self.predict_logits(batch).reshape(-1)
        probabilities = [calibrated_probability_from_logit(float(logit)) for logit in logits.tolist()]
        return torch.tensor(probabilities, dtype=torch.float32).reshape(-1, 1)

    def predict_logits_bf16(self, batch: torch.Tensor) -> torch.Tensor:
        """Measurement-only BF16 autocast. Shadow decisions do not call this."""
        self._require_cuda()
        self.load()
        if not torch.cuda.is_bf16_supported():
            raise PolicyConfigurationError(
                "BF16 autocast is not supported on this GPU",
                ("POLICY_INFRASTRUCTURE_ERROR",),
            )
        if batch.ndim != 4 or tuple(batch.shape[1:]) != (3, INPUT_SIZE, INPUT_SIZE):
            raise PolicyConfigurationError(f"forward batch shape {tuple(batch.shape)} is not NCHW 512")
        moved = batch.to(self.device)
        with torch.inference_mode():
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                output = self.model(moved)
        if isinstance(output, tuple):
            output = output[0]
        values = output.detach().float().reshape(-1, 1).cpu()
        if not torch.isfinite(values).all():
            raise PolicyConfigurationError("BF16 forward returned a non-finite logit", ("POLICY_INVALID_INPUT",))
        return values

    def quality(self, image):
        """Phase 23 features from the original image, not the padded tensor."""
        return quality_flags_for_image(image)

    def metadata(self) -> dict:
        validate_reference_identity()
        return {
            "model_family": MODEL_FAMILY,
            "backbone": MODEL_ARCHITECTURE,
            "head": MODEL_HEAD,
            "input_size": INPUT_SIZE,
            "preprocessing_version": PREPROCESSING_VERSION,
            "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
            "temperature": FROZEN_TEMPERATURE,
            "authoritative_precision": "fp32",
            "output": "raw_logit",
        }

    def _require_cuda(self) -> None:
        if not torch.cuda.is_available():
            raise PolicyConfigurationError(
                "CUDA is unavailable. DINOv3 shadow inference did not fall back to CPU ONNX or DINOv2.",
                ("POLICY_INFRASTRUCTURE_ERROR",),
            )
        name = torch.cuda.get_device_name(0)
        if EXPECTED_GPU not in name:
            raise PolicyConfigurationError(
                f"CUDA device {name!r} is not {EXPECTED_GPU}. Shadow inference did not fall back.",
                ("POLICY_INFRASTRUCTURE_ERROR",),
            )


def shared_backend() -> DinoV3ShadowBackend:
    global _SHARED
    if _SHARED is None:
        _SHARED = DinoV3ShadowBackend()
    return _SHARED


def reset_backend_for_tests() -> None:
    global _SHARED
    if _SHARED is not None:
        _SHARED.release()
    _SHARED = None
