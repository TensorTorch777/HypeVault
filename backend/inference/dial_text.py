"""Can the declared brand's name be read on the dial? A hint for the seller, never a verdict.

RapidOCR (literal: it does not correct spelling) reads the whole photo and the central 60% at
double resolution. FOUND when either view has a word containing one of the declared brand's
words. The result is reported to the seller and stored with the listing; it does not change
whether the listing goes live. When OCR is not installed or fails, the result is UNKNOWN.
"""

from __future__ import annotations

import logging
import sys
import threading
from pathlib import Path
from typing import Literal

from PIL import Image

_ML = Path(__file__).resolve().parents[2] / "ml_rtx5080"
if str(_ML) not in sys.path:
    sys.path.insert(0, str(_ML))

from watch_forensics import BRAND_WORDS, words  # noqa: E402

_log = logging.getLogger(__name__)

DialText = Literal["FOUND", "NOT_FOUND", "UNKNOWN"]
MIN_CONFIDENCE = 0.6
CROP_FRACTION = 0.6
CROP_SIDE = 1024
NOT_FOUND_HINT = (
    "We couldn't read the brand name on the dial. A sharp, straight-on photo of the dial "
    "helps buyers and makes screening more reliable."
)

_reader = None
_lock = threading.Lock()


def _ocr():
    global _reader
    if _reader is None:
        from rapidocr import RapidOCR

        _reader = RapidOCR()
    return _reader


def _views(pil: Image.Image) -> list[Image.Image]:
    w, h = pil.size
    cw, ch = int(w * CROP_FRACTION), int(h * CROP_FRACTION)
    crop = pil.crop(((w - cw) // 2, (h - ch) // 2, (w + cw) // 2, (h + ch) // 2)).resize((CROP_SIDE, CROP_SIDE), Image.LANCZOS)
    return [pil, crop]


def brand_in_lines(brand: str, lines: list[tuple[str, float]]) -> bool:
    own = BRAND_WORDS.get(brand, ())
    found = {w for text, score in lines if score >= MIN_CONFIDENCE for w in words(text)}
    return any(o in w for w in found for o in own)


def read_dial_lines(pil: Image.Image) -> list[tuple[str, float]]:
    """Every confident line of text in the photo and its centre crop. Empty when OCR is unavailable."""
    try:
        import numpy as np

        lines: list[tuple[str, float]] = []
        with _lock:
            reader = _ocr()
            for view in _views(pil.convert("RGB")):
                res = reader(np.asarray(view))
                lines += [(t, float(s)) for t, s in zip(res.txts or (), res.scores or ()) if s >= MIN_CONFIDENCE]
        return lines
    except Exception as exc:
        _log.warning("dial_text_unavailable: %s", exc)
        return []


def read_dial_brand(pil: Image.Image, brand: str | None) -> DialText:
    if not brand or brand not in BRAND_WORDS:
        return "UNKNOWN"
    try:
        import numpy as np

        with _lock:
            reader = _ocr()
            for view in _views(pil.convert("RGB")):
                res = reader(np.asarray(view))
                if brand_in_lines(brand, list(zip(res.txts or (), res.scores or ()))):
                    return "FOUND"
        return "NOT_FOUND"
    except Exception as exc:  # OCR missing or failed: the listing is not affected
        _log.warning("dial_text_unavailable: %s", exc)
        return "UNKNOWN"
