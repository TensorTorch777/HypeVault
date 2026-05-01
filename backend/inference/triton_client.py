"""NVIDIA Triton gRPC client for dinov2_classifier."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Literal

import numpy as np
import tritonclient.grpc as grpcclient
from tritonclient.utils import InferenceServerException

from config import settings
from inference.verdict import logits_to_verdict

_log = logging.getLogger(__name__)

MODEL_NAME = settings.triton_model_name
INPUT_NAME = settings.triton_input_name
OUTPUT_NAME = settings.triton_output_name
TIMEOUT_S = 5.0
MAX_ATTEMPTS = 2
MODEL_INPUT_SIZE = 518  # DINOv2 native resolution


def preprocess_chw(image_rgb: np.ndarray) -> np.ndarray:
    """
    HWC RGB → NCHW float32 ImageNet normalized (batch 1).
    Resizes to MODEL_INPUT_SIZE (518 × 518) — critical for DINOv2 patch grid.
    """
    from PIL import Image as _PILImage

    try:
        pil = _PILImage.fromarray(image_rgb.astype(np.uint8)).resize(
            (MODEL_INPUT_SIZE, MODEL_INPUT_SIZE),
            _PILImage.BICUBIC,
        )
        img = np.array(pil).astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)
        img = (img - mean) / std
        chw = np.transpose(img, (2, 0, 1))
        batch = np.expand_dims(chw, axis=0).astype(np.float32)
        return np.ascontiguousarray(batch)
    except Exception as exc:
        _log.exception("preprocess_chw_failed: %s", exc)
        raise


def _infer_sync(array_nchw: np.ndarray) -> np.ndarray:
    url = f"{settings.triton_host}:{settings.triton_port}"
    client = grpcclient.InferenceServerClient(url=url)
    inputs = [grpcclient.InferInput(INPUT_NAME, array_nchw.shape, "FP32")]
    inputs[0].set_data_from_numpy(array_nchw)
    outputs = [grpcclient.InferRequestedOutput(OUTPUT_NAME)]
    res = client.infer(
        model_name=MODEL_NAME,
        inputs=inputs,
        outputs=outputs,
        client_timeout=TIMEOUT_S,
    )
    return res.as_numpy(OUTPUT_NAME)


async def classify_image_triton(array_nchw: np.ndarray) -> tuple[Literal["AUTHENTIC", "FAKE"], float]:
    """Run Triton inference with retries; returns verdict and confidence."""
    last_exc: Exception | None = None
    started = time.perf_counter()
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            out = await asyncio.to_thread(_infer_sync, array_nchw)
            verdict, confidence = logits_to_verdict(out)
            elapsed_ms = (time.perf_counter() - started) * 1000
            _log.info(
                "triton_infer_ok attempt=%s ms=%.2f verdict=%s conf=%.4f",
                attempt,
                elapsed_ms,
                verdict,
                confidence,
            )
            return verdict, confidence
        except InferenceServerException as exc:
            last_exc = exc
            _log.warning("triton_infer_attempt_failed attempt=%s err=%s", attempt, exc)
        except Exception as exc:
            last_exc = exc
            _log.warning("triton_infer_attempt_failed attempt=%s err=%s", attempt, exc)
        try:
            await asyncio.sleep(0.25 * attempt)
        except Exception:
            pass
    _log.error("triton_infer_exhausted: %s", last_exc)
    if last_exc:
        raise last_exc
    raise RuntimeError("Triton inference failed")


async def triton_ready() -> bool:
    def _check() -> bool:
        client = grpcclient.InferenceServerClient(url=f"{settings.triton_host}:{settings.triton_port}")
        return bool(client.is_server_ready() and client.is_model_ready(MODEL_NAME))

    try:
        return await asyncio.to_thread(_check)
    except Exception:
        return False
