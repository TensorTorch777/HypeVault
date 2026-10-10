"""Deterministic region-occlusion sensitivity for an already-trained classifier.

Masking a rectangle and re-running the frozen model measures how that
model's raw logit changes. It does not name watch parts, prove authenticity,
or validate a counterfeit indicator.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from typing import Any

from PIL import Image

METHOD_NAME = "region_occlusion_v1"
METHOD_VERSION = "1"
GRID_ROWS = 4
GRID_COLS = 4
# Integer ImageNet mean used as a deterministic, documented fill.
FILL_RGB = (124, 116, 104)
# Provisional display heuristic only. It was not tuned on the locked final test set
# and is not a validated counterfeit or authenticity threshold.
WEAK_ABS_DELTA = 1e-3
WEAK_ABS_DELTA_ROLE = "provisional_heuristic"


def occlusion_boxes(width: int, height: int, rows: int = GRID_ROWS, cols: int = GRID_COLS) -> list[dict[str, int]]:
    """Cover the image with a fixed grid. The last row and column absorb remainders."""
    if width < 1 or height < 1:
        raise ValueError("Image dimensions must be positive")
    if rows < 1 or cols < 1:
        raise ValueError("Grid size must be positive")
    boxes: list[dict[str, int]] = []
    for row in range(rows):
        y0 = (row * height) // rows
        y1 = ((row + 1) * height) // rows
        for col in range(cols):
            x0 = (col * width) // cols
            x1 = ((col + 1) * width) // cols
            boxes.append(
                {
                    "row": row,
                    "col": col,
                    "x": x0,
                    "y": y0,
                    "width": x1 - x0,
                    "height": y1 - y0,
                }
            )
    return boxes


def mask_region(image: Image.Image, box: dict[str, int], fill: tuple[int, int, int] = FILL_RGB) -> Image.Image:
    """Return a copy with one rectangle replaced by the documented fill."""
    masked = image.convert("RGB").copy()
    overlay = Image.new("RGB", (box["width"], box["height"]), fill)
    masked.paste(overlay, (box["x"], box["y"]))
    return masked


async def measure_regions(
    image: Image.Image,
    logit_fn: Callable[[Image.Image], Awaitable[float]],
    timeout_s: float,
) -> tuple[float, list[float], list[dict[str, int]], float]:
    """Run baseline plus one masked inference per cell.

    The deadline covers the whole operation. A call that starts inside the
    budget and finishes after it does not return a successful map.
    """
    boxes = occlusion_boxes(image.width, image.height)
    if not boxes:
        raise ValueError("Occlusion evidence is empty or misaligned")
    started = time.perf_counter()

    def _expired() -> bool:
        return time.perf_counter() - started > timeout_s

    async def _bounded(sample: Image.Image) -> float:
        if _expired():
            raise TimeoutError("Explanation computation exceeded the time limit")
        value = float(await logit_fn(sample))
        if _expired():
            raise TimeoutError("Explanation computation exceeded the time limit")
        return value

    async def _run() -> tuple[float, list[float], list[dict[str, int]], float]:
        baseline = await _bounded(image)
        masked: list[float] = []
        for box in boxes:
            masked.append(await _bounded(mask_region(image, box)))
        if _expired():
            raise TimeoutError("Explanation computation exceeded the time limit")
        elapsed_ms = round((time.perf_counter() - started) * 1000.0, 3)
        return baseline, masked, boxes, elapsed_ms

    try:
        return await asyncio.wait_for(_run(), timeout=timeout_s)
    except asyncio.TimeoutError as exc:
        raise TimeoutError("Explanation computation exceeded the time limit") from exc


def sensitivity_record(
    *,
    baseline_logit: float,
    masked_logits: list[float],
    boxes: list[dict[str, int]],
    model_id: str,
    model_version: str,
    preprocessing_id: str,
    decision: str,
) -> dict[str, Any]:
    """Map logit changes back to image coordinates. Does not invent component names."""
    if not boxes or len(boxes) != len(masked_logits):
        raise ValueError("Occlusion evidence is empty or misaligned")
    if decision not in {"AUTHENTIC", "FAKE", "REVIEW"}:
        raise ValueError("Baseline decision is not a known research class")
    patches = []
    for box, masked in zip(boxes, masked_logits, strict=True):
        delta = float(masked) - float(baseline_logit)
        patches.append({**box, "masked_logit": float(masked), "delta_logit": delta})
    strongest = max(abs(item["delta_logit"]) for item in patches)
    return {
        "method": {
            "name": METHOD_NAME,
            "version": METHOD_VERSION,
            "grid_rows": GRID_ROWS,
            "grid_cols": GRID_COLS,
            "fill_rgb": list(FILL_RGB),
            "score": "raw_logit",
            "weak_abs_delta": WEAK_ABS_DELTA,
            "weak_abs_delta_role": WEAK_ABS_DELTA_ROLE,
            "weak_abs_delta_note": (
                "0.001 is a provisional display heuristic. It was not tuned on the locked final test set "
                "and is not a validated authenticity or counterfeit threshold."
            ),
            "component_labels": False,
        },
        "model_id": model_id,
        "model_version": model_version,
        "preprocessing_id": preprocessing_id,
        "baseline_logit": float(baseline_logit),
        "baseline_decision": decision,
        "patches": patches,
        "max_abs_delta_logit": strongest,
        "evidence_supports_visual_summary": strongest >= WEAK_ABS_DELTA,
    }
