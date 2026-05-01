"""Load trained `DINOv2Classifier` checkpoint and run inference (swap later for Triton)."""

from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path
from typing import Literal

import numpy as np
import torch

from config import REPO_ROOT, settings
from inference.dinov2_model import DINOv2Classifier
from inference.verdict import logits_to_verdict

_log = logging.getLogger(__name__)

_bundle: tuple[DINOv2Classifier, torch.device] | None = None
_load_lock = asyncio.Lock()


def _resolve_checkpoint_path() -> Path:
    raw = (settings.local_model_path or "").strip()
    if not raw:
        raise RuntimeError("LOCAL_MODEL_PATH is empty (required for INFERENCE_BACKEND=torch)")
    p = Path(raw)
    if not p.is_absolute():
        p = REPO_ROOT / p
    return p


def _pick_device() -> torch.device:
    explicit = (settings.torch_device or "").strip()
    if explicit:
        return torch.device(explicit)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _load_bundle_sync() -> tuple[DINOv2Classifier, torch.device]:
    ckpt_path = _resolve_checkpoint_path()
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    device = _pick_device()
    model_name = (settings.dinov2_model_name or "dinov2_vitg14_reg").strip()
    _log.info("local_torch_loading path=%s model=%s device=%s", ckpt_path, model_name, device)

    model = DINOv2Classifier(model_name)
    try:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt.get("model_state") if isinstance(ckpt, dict) else None
    if state is None and isinstance(ckpt, dict):
        state = ckpt.get("state_dict")
    if state is None:
        state = ckpt
    if not isinstance(state, dict):
        raise ValueError(f"Unexpected checkpoint format in {ckpt_path}")
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        _log.warning("load_state_dict partial missing=%s unexpected=%s", len(missing), len(unexpected))
    model.to(device)
    model.eval()
    return model, device


def _infer_sync(model: DINOv2Classifier, device: torch.device, array_nchw: np.ndarray) -> np.ndarray:
    x = torch.from_numpy(array_nchw).to(device, non_blocking=True)
    with torch.no_grad():
        if device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logit = model(x).float()
        else:
            logit = model(x)
    return logit.detach().cpu().numpy()


async def classify_image_torch(
    array_nchw: np.ndarray,
) -> tuple[Literal["AUTHENTIC", "FAKE"], float]:
    global _bundle
    async with _load_lock:
        if _bundle is None:
            _bundle = await asyncio.to_thread(_load_bundle_sync)

    model, device = _bundle
    started = time.perf_counter()
    out = await asyncio.to_thread(_infer_sync, model, device, array_nchw)
    verdict, confidence = logits_to_verdict(out)
    elapsed_ms = (time.perf_counter() - started) * 1000
    _log.info("local_torch_infer_ok ms=%.2f verdict=%s conf=%.4f", elapsed_ms, verdict, confidence)
    return verdict, confidence


def reset_local_model_for_tests() -> None:
    global _bundle
    _bundle = None
