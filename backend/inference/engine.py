"""
Inference entrypoint — local PyTorch checkpoint or NVIDIA Triton.

Set `INFERENCE_BACKEND=torch` + `LOCAL_MODEL_PATH` for RunPod-free local verify.
Later: `INFERENCE_BACKEND=triton` (default) with Triton + TensorRT.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from fastapi import HTTPException, status

from config import settings


async def classify_image(
    array_nchw: np.ndarray,
) -> tuple[Literal["AUTHENTIC", "FAKE"], float]:
    backend = (settings.inference_backend or "triton").strip().lower()
    if settings.report_enforce_triton and backend != "triton":
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Report mode requires Triton backend. Set INFERENCE_BACKEND=triton.",
        )
    if backend == "torch":
        from inference.local_torch import classify_image_torch

        return await classify_image_torch(array_nchw)

    from inference.triton_client import classify_image_triton

    return await classify_image_triton(array_nchw)
