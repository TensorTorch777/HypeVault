"""Serving quality gate for the two shadow decision flags.

The full Phase 23 analyzer stays the diagnostic reference. This gate computes
only LOW_RESOLUTION and LOW_CONTRAST, on the original RGB image, with the same
BT.601 luminance and the same strict cutoffs.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError

from image_quality import STD_DDOF, QualityReadError, luminance_image
from policy import LOW_CONTRAST_CUTOFF, LOW_RESOLUTION_CUTOFF

DECISION_FLAGS = ("LOW_RESOLUTION", "LOW_CONTRAST")
STD_SNAP = 1e-12
# Closest uint8 constructions found for a 512×512 image. Exact equality with
# the cutoff is not representable at this resolution.
BOUNDARY_BELOW_WHITE_PIXELS = 10197
BOUNDARY_BELOW_GRAY = 249
BOUNDARY_ABOVE_WHITE_PIXELS = 10198


def strict_low_resolution(min_dimension: float) -> bool:
    """True only when the shorter original side is strictly below 512."""
    return float(min_dimension) < float(LOW_RESOLUTION_CUTOFF)


def strict_low_contrast(luminance_std: float) -> bool:
    """True only when population luminance std is strictly below the cutoff."""
    value = float(luminance_std)
    if value < STD_SNAP:
        value = 0.0
    return value < float(LOW_CONTRAST_CUTOFF)


def fast_quality_gate(image: Image.Image) -> tuple[dict, tuple[str, ...]]:
    """Decision features from the original image. The padded model tensor is not an input."""
    rgb = image.convert("RGB")
    width, height = rgb.size
    if width < 1 or height < 1:
        raise QualityReadError("image has no pixels")
    min_dimension = int(min(width, height))
    luminance_std = _population_std(luminance_image(rgb))
    flags = []
    if strict_low_resolution(min_dimension):
        flags.append("LOW_RESOLUTION")
    if strict_low_contrast(luminance_std):
        flags.append("LOW_CONTRAST")
    features = {
        "width": int(width),
        "height": int(height),
        "min_dimension": min_dimension,
        "luminance_std": luminance_std,
    }
    return features, tuple(flags)


def decision_flags(flags) -> tuple[str, ...]:
    """The flags that can change a shadow decision. Diagnostic flags are ignored."""
    present = set(flags)
    return tuple(flag for flag in DECISION_FLAGS if flag in present)


def load_original_rgb(path: Path, blocked: set[str] | None = None) -> Image.Image:
    """Decode once. A final-test path is rejected before the file is opened."""
    file_path = Path(path)
    if blocked is not None and str(file_path.resolve()) in blocked:
        raise RuntimeError("quality gate encountered a final-test sample")
    if not file_path.is_file():
        raise QualityReadError("missing image file")
    try:
        with Image.open(file_path) as handle:
            rgb = handle.convert("RGB")
            rgb.load()
            return rgb.copy()
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError, ValueError) as exc:
        raise QualityReadError(str(exc)) from exc


def resolution_probe(width: int, height: int) -> Image.Image:
    """High-contrast probe so the resolution flag is not confused with contrast."""
    array = np.zeros((height, width, 3), dtype=np.uint8)
    array[::2, ::2] = 255
    array[1::2, 1::2] = 255
    return Image.fromarray(array, mode="RGB")


def contrast_boundary_images(size: int = 512) -> dict:
    """Nearest uint8 images below and above the frozen contrast cutoff."""
    below = _binary_image(size, BOUNDARY_BELOW_WHITE_PIXELS)
    below_array = np.array(below, copy=True)
    flat = np.where(below_array[:, :, 0] == 0)
    below_array[int(flat[0][0]), int(flat[1][0])] = BOUNDARY_BELOW_GRAY
    below = Image.fromarray(below_array, mode="RGB")
    above = _binary_image(size, BOUNDARY_ABOVE_WHITE_PIXELS)
    return {"below": below, "above": above}


def _population_std(luminance: np.ndarray) -> float:
    value = float(np.std(luminance, ddof=STD_DDOF))
    if value < STD_SNAP:
        return 0.0
    return value


def _binary_image(size: int, white_pixels: int) -> Image.Image:
    array = np.zeros((size, size), dtype=np.uint8)
    array.ravel()[:white_pixels] = 255
    gray = Image.fromarray(array, mode="L")
    return gray.convert("RGB")
