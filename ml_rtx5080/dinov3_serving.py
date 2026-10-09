"""Parallel DINOv3 serving artifact.

This module exports and runs a new ONNX model. It does not write the DINOv2
repository, does not train, and does not apply temperature inside the graph.
The raw logit is the only network output. Calibration and the shadow policy
stay outside the graph.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch
import torch.nn as nn

from policy_contract import (
    EXPECTED_CHECKPOINT_SHA256,
    FROZEN_TEMPERATURE,
    INPUT_SIZE,
    MODEL_ARCHITECTURE,
    MODEL_FAMILY,
    MODEL_HEAD,
    POLICY_VERSION,
    PREPROCESSING_VERSION,
    PolicyConfigurationError,
)
from reference_inference import file_sha256, preprocess_image, validate_reference_identity

MODEL_NAME = "dinov3_authenticity_classifier"
INPUT_NAME = "input__0"
OUTPUT_NAME = "output__0"
MAX_BATCH_SIZE = 16
INSTANCE_COUNT = 1
CHECKPOINT = (
    Path(__file__).resolve().parent
    / "experiments"
    / "v2_dinov3_cls_patch_attention"
    / "epoch_018.pt"
)
REPO = Path(__file__).resolve().parents[1]
MODEL_ROOT = REPO / "models" / MODEL_NAME
ONNX_PATH = MODEL_ROOT / "1" / "model.onnx"
IDENTITY_PATH = MODEL_ROOT / "1" / "identity.json"
OLD_MODEL_ROOT = REPO / "models" / "dinov2_classifier"
LOGIT_ATOL = 1e-4
PROBABILITY_ATOL = 1e-5
TENSOR_ATOL = 1e-5
DETERMINISM_ATOL = 1e-5

_SESSION = None
_SESSION_SHA = None


class FrozenRawLogit(nn.Module):
    """Export wrapper. The graph returns one raw logit per image, shape [N, 1]."""

    def __init__(self, classifier: nn.Module) -> None:
        super().__init__()
        self.classifier = classifier

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        logits = self.classifier(images)
        if isinstance(logits, tuple):
            logits = logits[0]
        return logits.reshape(logits.shape[0], 1)


def preprocess_serving_image(image):
    """Authoritative eval preprocess. This is the frozen reference function."""
    tensor = preprocess_image(image)
    if tuple(tensor.shape) != (3, INPUT_SIZE, INPUT_SIZE):
        raise PolicyConfigurationError(f"serving preprocess shape {tuple(tensor.shape)} is not 3x512x512")
    return tensor


def architecture_record(model: nn.Module) -> dict:
    backbone = model.backbone
    return {
        "model_family": MODEL_FAMILY,
        "backbone": getattr(backbone, "backbone_name", MODEL_ARCHITECTURE),
        "head": model.classifier_arch,
        "num_register_tokens": int(backbone.num_register_tokens),
        "embed_dim": int(backbone.embed_dim),
        "patch_size": int(backbone.patch_size),
        "image_size": int(backbone.image_size),
        "attention_pool": model.attention_pool is not None,
        "fusion": model.fusion is not None,
        "trunk": model.trunk is not None,
        "cls_index": 0,
        "patch_token_start": 1 + int(backbone.num_register_tokens),
        "output": "raw_logit",
        "temperature_in_graph": False,
        "policy_in_graph": False,
    }


def export_frozen_onnx(destination: Path | None = None) -> dict:
    """Export the frozen checkpoint. Refuses the DINOv2 repository and a hash mismatch."""
    onnx_path = Path(destination) if destination is not None else ONNX_PATH
    if OLD_MODEL_ROOT in onnx_path.resolve().parents or onnx_path.resolve() == OLD_MODEL_ROOT:
        raise RuntimeError("refusing to write the existing DINOv2 model repository")
    if "dinov2_classifier" in onnx_path.parts:
        raise RuntimeError("refusing to overwrite dinov2_classifier")
    source_sha = file_sha256(CHECKPOINT)
    validate_reference_identity(checkpoint_sha256=source_sha)
    from evaluation import build_eval_model

    device = torch.device("cpu")
    model, image_size = build_eval_model(MODEL_FAMILY, MODEL_HEAD, CHECKPOINT, device)
    if int(image_size) != INPUT_SIZE or model.training:
        raise PolicyConfigurationError("export model is not the frozen eval contract")
    record = architecture_record(model)
    if record["num_register_tokens"] != 4 or record["head"] != MODEL_HEAD:
        raise PolicyConfigurationError("export architecture is not cls_patch_attention with 4 registers")
    wrapper = FrozenRawLogit(model).eval()
    dummy = torch.zeros(1, 3, INPUT_SIZE, INPUT_SIZE, dtype=torch.float32)
    with torch.no_grad():
        sample = wrapper(dummy)
    if tuple(sample.shape) != (1, 1):
        raise PolicyConfigurationError(f"export wrapper output {tuple(sample.shape)} is not [1, 1]")
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        wrapper,
        (dummy,),
        str(onnx_path),
        export_params=True,
        opset_version=18,
        do_constant_folding=True,
        dynamo=False,
        external_data=False,
        input_names=[INPUT_NAME],
        output_names=[OUTPUT_NAME],
        dynamic_axes={
            INPUT_NAME: {0: "batch"},
            OUTPUT_NAME: {0: "batch"},
        },
    )
    exported_sha = file_sha256(onnx_path)
    if file_sha256(CHECKPOINT) != source_sha:
        raise RuntimeError("export changed the frozen checkpoint")
    identity = {
        "model_name": MODEL_NAME,
        "source_checkpoint": str(CHECKPOINT),
        "source_checkpoint_sha256": source_sha,
        "export_artifact_sha256": exported_sha,
        "input_shape": [MAX_BATCH_SIZE, 3, INPUT_SIZE, INPUT_SIZE],
        "input_dims_without_batch": [3, INPUT_SIZE, INPUT_SIZE],
        "output": "raw_logit",
        "temperature": FROZEN_TEMPERATURE,
        "temperature_in_graph": False,
        "policy_version": POLICY_VERSION,
        "policy_in_graph": False,
        "architecture": record,
        "preprocessing_version": PREPROCESSING_VERSION,
    }
    identity_path = onnx_path.parent / "identity.json"
    identity_path.write_text(json.dumps(identity, indent=2, sort_keys=True) + "\n")
    config_path = onnx_path.parents[1] / "config.pbtxt"
    config_path.write_text(_triton_config())
    del wrapper, model
    return {
        "model_name": MODEL_NAME,
        "onnx_path": str(onnx_path),
        "config_path": str(config_path),
        "source_checkpoint_sha256": source_sha,
        "export_artifact_sha256": exported_sha,
        "identity_sha256": file_sha256(identity_path),
        "config_sha256": file_sha256(config_path),
        "input_dims": [3, INPUT_SIZE, INPUT_SIZE],
        "output_dims": [1],
        "max_batch_size": MAX_BATCH_SIZE,
        "instance_count": INSTANCE_COUNT,
        "backend": "onnxruntime",
        "temperature_in_graph": False,
        "tensorrt_status": "unavailable",
        "tensorrt_reason": "The tensorrt Python package is not installed. No engine was built.",
        "architecture": record,
        "artifact_bytes": onnx_path.stat().st_size,
    }


def _triton_config() -> str:
    return (
        f'name: "{MODEL_NAME}"\n'
        'backend: "onnxruntime"\n'
        f"max_batch_size: {MAX_BATCH_SIZE}\n"
        "\n"
        "input [\n"
        "  {\n"
        f'    name: "{INPUT_NAME}"\n'
        "    data_type: TYPE_FP32\n"
        f"    dims: [ 3, {INPUT_SIZE}, {INPUT_SIZE} ]\n"
        "  }\n"
        "]\n"
        "\n"
        "output [\n"
        "  {\n"
        f'    name: "{OUTPUT_NAME}"\n'
        "    data_type: TYPE_FP32\n"
        "    dims: [ 1 ]\n"
        "  }\n"
        "]\n"
        "\n"
        "instance_group [\n"
        "  {\n"
        f"    count: {INSTANCE_COUNT}\n"
        "    kind: KIND_CPU\n"
        "  }\n"
        "]\n"
        "\n"
        "dynamic_batching {\n"
        "  preferred_batch_size: [ 1, 2, 4, 8, 16 ]\n"
        "}\n"
    )


def load_identity() -> dict:
    if not IDENTITY_PATH.is_file() or not ONNX_PATH.is_file():
        raise PolicyConfigurationError(
            "DINOv3 serving artifact is missing",
            ("POLICY_CONFIGURATION_ERROR", "POLICY_MODEL_MISMATCH"),
        )
    identity = json.loads(IDENTITY_PATH.read_text())
    if identity.get("source_checkpoint_sha256") != EXPECTED_CHECKPOINT_SHA256:
        raise PolicyConfigurationError("serving artifact was not exported from the frozen checkpoint")
    if identity.get("temperature_in_graph") or identity.get("policy_in_graph"):
        raise PolicyConfigurationError("serving graph includes temperature or a policy decision")
    if file_sha256(ONNX_PATH) != identity.get("export_artifact_sha256"):
        raise PolicyConfigurationError("serving ONNX bytes do not match the export identity")
    dims = identity.get("input_dims_without_batch")
    if list(dims or []) != [3, INPUT_SIZE, INPUT_SIZE]:
        raise PolicyConfigurationError(f"serving input dims {dims} are not [3, 512, 512]")
    return identity


def serving_checkpoint_sha() -> str:
    return str(load_identity()["source_checkpoint_sha256"])


def onnx_session():
    """ONNX Runtime session for the new artifact. TensorRT is not selected."""
    global _SESSION, _SESSION_SHA
    identity = load_identity()
    digest = identity["export_artifact_sha256"]
    if _SESSION is not None and _SESSION_SHA == digest:
        return _SESSION
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    available = set(ort.get_available_providers())
    selected = [name for name in providers if name in available] or ["CPUExecutionProvider"]
    _SESSION = ort.InferenceSession(str(ONNX_PATH), sess_options=options, providers=selected)
    _SESSION_SHA = digest
    return _SESSION


def reset_onnx_session() -> None:
    global _SESSION, _SESSION_SHA, _CUDA_PROBE
    _SESSION = None
    _SESSION_SHA = None
    _CUDA_PROBE = None


_CUDA_PROBE: dict | None = None


def probe_cuda_execution_provider() -> dict:
    """Try the CUDA provider alone. A listed provider name is not a loaded library."""
    global _CUDA_PROBE
    if _CUDA_PROBE is not None:
        return _CUDA_PROBE
    import onnxruntime as ort

    listed = list(ort.get_available_providers())
    if "CUDAExecutionProvider" not in listed:
        _CUDA_PROBE = {
            "available": False,
            "listed": False,
            "providers_after_load": [],
            "error": "CUDAExecutionProvider is not in ort.get_available_providers()",
        }
        return _CUDA_PROBE
    if not ONNX_PATH.is_file():
        _CUDA_PROBE = {
            "available": False,
            "listed": True,
            "providers_after_load": [],
            "error": f"DINOv3 ONNX artifact is missing at {ONNX_PATH}",
        }
        return _CUDA_PROBE
    try:
        session = ort.InferenceSession(str(ONNX_PATH), providers=["CUDAExecutionProvider"])
        loaded = list(session.get_providers())
    except Exception as exc:
        _CUDA_PROBE = {
            "available": False,
            "listed": True,
            "providers_after_load": [],
            "error": f"{type(exc).__name__}: {exc}",
        }
        return _CUDA_PROBE
    available = "CUDAExecutionProvider" in loaded
    missing = _missing_cuda_libraries()
    detail = f"session providers were {loaded}, without CUDAExecutionProvider"
    if missing:
        detail = f"{detail}. Missing libraries: {', '.join(missing)}"
    _CUDA_PROBE = {
        "available": available,
        "listed": True,
        "providers_after_load": loaded,
        "error": None if available else detail,
    }
    return _CUDA_PROBE


def _missing_cuda_libraries() -> list[str]:
    import subprocess

    library = (
        Path(__file__).resolve().parents[1]
        / ".venv"
        / "lib"
        / "python3.12"
        / "site-packages"
        / "onnxruntime"
        / "capi"
        / "libonnxruntime_providers_cuda.so"
    )
    if not library.is_file():
        return ["libonnxruntime_providers_cuda.so missing"]
    completed = subprocess.run(["ldd", str(library)], check=False, capture_output=True, text=True)
    missing = []
    for line in completed.stdout.splitlines():
        if "not found" in line:
            missing.append(line.strip().split()[0])
    return missing


def forward_cuda_logits(batch: torch.Tensor):
    """GPU ONNX logits. A missing CUDA provider raises and does not use CPU or DINOv2."""
    probe = probe_cuda_execution_provider()
    if not probe["available"]:
        raise PolicyConfigurationError(
            "CUDA ONNX Runtime is unavailable. "
            f"{probe['error']} Shadow inference did not fall back to CPU ONNX or DINOv2.",
            ("POLICY_INFRASTRUCTURE_ERROR",),
        )
    if batch.ndim != 4 or tuple(batch.shape[1:]) != (3, INPUT_SIZE, INPUT_SIZE):
        raise PolicyConfigurationError(f"CUDA ONNX batch shape {tuple(batch.shape)} is not NCHW 512")
    import onnxruntime as ort

    session = ort.InferenceSession(str(ONNX_PATH), providers=["CUDAExecutionProvider"])
    feeds = {INPUT_NAME: batch.detach().cpu().float().numpy()}
    output = session.run([OUTPUT_NAME], feeds)[0]
    values = torch.from_numpy(output).float().reshape(-1)
    if not torch.isfinite(values).all():
        raise PolicyConfigurationError("CUDA ONNX returned a non-finite logit", ("POLICY_INVALID_INPUT",))
    return values, serving_checkpoint_sha()


def single_decode_inputs(path: Path):
    """One file decode. Quality uses the original image; the model uses the 512 pad."""
    from PIL import Image

    from image_quality import apply_quality_flags, extract_quality_features
    from policy_contract import PHASE23_CUTOFFS

    file_path = Path(path)
    with Image.open(file_path) as handle:
        image_format = handle.format
        tables = getattr(handle, "quantization", None)
        rgb = handle.convert("RGB")
        rgb.load()
        owned = rgb.copy()
    features = extract_quality_features(
        owned,
        file_bytes=file_path.stat().st_size,
        image_format=image_format,
        jpeg_quantization_tables_present=bool(tables),
    )
    flags = tuple(apply_quality_flags(features, {"cutoffs": dict(PHASE23_CUTOFFS)}))
    tensor = preprocess_serving_image(owned)
    return owned, features, flags, tensor


def forward_onnx_logits(batch: torch.Tensor) -> torch.Tensor:
    """Raw logits from the parallel artifact. A missing artifact raises and does not fall back."""
    if batch.ndim != 4 or tuple(batch.shape[1:]) != (3, INPUT_SIZE, INPUT_SIZE):
        raise PolicyConfigurationError(f"ONNX batch shape {tuple(batch.shape)} is not NCHW 512")
    session = onnx_session()
    feeds = {INPUT_NAME: batch.detach().cpu().float().numpy()}
    output = session.run([OUTPUT_NAME], feeds)[0]
    values = torch.from_numpy(output).float().reshape(-1)
    if not torch.isfinite(values).all():
        raise PolicyConfigurationError("ONNX returned a non-finite logit", ("POLICY_INVALID_INPUT",))
    return values


def old_repository_fingerprint() -> dict:
    """Hashes of the existing DINOv2 Triton files. Used to prove they were not rewritten."""
    config = OLD_MODEL_ROOT / "config.pbtxt"
    onnx = OLD_MODEL_ROOT / "1" / "model.onnx"
    weights = OLD_MODEL_ROOT / "1" / "model.onnx.data"
    return {
        "config_sha256": file_sha256(config) if config.is_file() else None,
        "onnx_sha256": file_sha256(onnx) if onnx.is_file() else None,
        "weights_sha256": file_sha256(weights) if weights.is_file() else None,
        "config_bytes": config.stat().st_size if config.is_file() else None,
        "onnx_bytes": onnx.stat().st_size if onnx.is_file() else None,
        "weights_bytes": weights.stat().st_size if weights.is_file() else None,
    }


def tensor_sha256(tensor: torch.Tensor) -> str:
    array = tensor.detach().cpu().float().numpy()
    return hashlib.sha256(array.tobytes()).hexdigest()
