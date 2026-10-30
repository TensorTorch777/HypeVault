"""Known AI images: is an upload a copy of an AI image we have already seen, even cropped, rescaled or screenshotted?

A screenshot or crop of an AI image can slip past the pixel detector (2026-10-30: a screenshot of the
Grandmaster image went live). Copy detection does not ask "does this look generated?" but "is this the
same picture as one we know is generated?", which edits barely change.

SSCD (Pizzi et al., "A Self-Supervised Descriptor for Image Copy Detection", CVPR 2022; Meta, MIT licence)
gives one 512-d descriptor per image; cosine similarity at or above `known_ai_min_similarity` is a copy.
The library is the seed set (ml_rtx5080/experiments/known_ai/library_seed.npz: AI watch images from
benchmarks v1-v6 and the Grandmaster image) plus every upload screening rejects as AI, kept under
`known_ai_store`. Measured before use (experiments/known_ai/measure.json): at 0.65, 99.8% of edited copies
matched and 0 of 1,224 real watch photos did.
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from config import settings

_ML = Path(__file__).resolve().parents[2] / "ml_rtx5080"
_DIR = _ML / "experiments" / "known_ai"
SEED = _DIR / "library_seed.npz"
MODEL = _DIR / "sscd_disc_mixup.torchscript.pt"
MODEL_URL = "https://dl.fbaipublicfiles.com/sscd-copy-detection/sscd_disc_mixup.torchscript.pt"
MODEL_SHA256 = "9f26bd4c848cc19b73d2ae92eea6e04886f61a7b764ceb7a13aeee62e6a6db56"
SHORT_SIDE = 288

_log = logging.getLogger(__name__)
_lock = threading.Lock()
_state: dict = {}


@dataclass(frozen=True)
class KnownMatch:
    similarity: float
    source: str | None
    vector: np.ndarray

    @property
    def matched(self) -> bool:
        return self.similarity >= settings.known_ai_min_similarity


def _store() -> Path:
    path = Path(settings.known_ai_store)
    return path if path.is_absolute() else Path(__file__).resolve().parents[1] / path


def _model():
    import torch

    if "model" not in _state:
        if not MODEL.exists():
            _log.info("known_ai_model_download")
            MODEL.parent.mkdir(parents=True, exist_ok=True)
            tmp = MODEL.with_suffix(".part")
            urllib.request.urlretrieve(MODEL_URL, tmp)
            tmp.rename(MODEL)
        if hashlib.sha256(MODEL.read_bytes()).hexdigest() != MODEL_SHA256:
            raise RuntimeError("SSCD model does not match its recorded SHA-256")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _state["device"] = device
        _state["model"] = torch.jit.load(str(MODEL), map_location=device).eval()
    return _state["model"]


def _library() -> tuple[np.ndarray, list[str]]:
    if "vectors" not in _state:
        z = np.load(SEED)
        vectors, sources = [z["vectors"].astype(np.float32)], z["source"].tolist()
        store = _store()
        if (store / "vectors.npy").exists():
            vectors.append(np.load(store / "vectors.npy").astype(np.float32))
            sources += [json.loads(x)["source"] for x in (store / "sources.jsonl").read_text().splitlines()]
        _state["vectors"], _state["sources"] = np.concatenate(vectors), sources
    return _state["vectors"], _state["sources"]


def describe(pil: Image.Image) -> np.ndarray:
    """L2-normalised SSCD descriptor of an RGB image."""
    import torch

    img = pil.convert("RGB")
    scale = SHORT_SIDE / min(img.size)
    img = img.resize((max(1, round(img.width * scale)), max(1, round(img.height * scale))), Image.BILINEAR)
    x = torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0).permute(2, 0, 1)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    with _lock, torch.inference_mode():
        model = _model()
        v = model(((x - mean) / std).unsqueeze(0).to(_state["device"])).float().cpu().numpy()[0]
    return v / np.linalg.norm(v)


def check(pil: Image.Image) -> KnownMatch:
    """Closest known AI image to this upload."""
    vector = describe(pil)
    with _lock:
        vectors, sources = _library()
        sims = vectors @ vector
    best = int(np.argmax(sims))
    return KnownMatch(similarity=float(sims[best]), source=sources[best], vector=vector)


def remember(vector: np.ndarray, source: str) -> None:
    """Add an upload that screening rejected as AI, so later copies of it are caught too."""
    with _lock:
        vectors, sources = _library()
        if len(vectors) and float((vectors @ vector).max()) >= 0.98:
            return  # already known
        store = _store()
        store.mkdir(parents=True, exist_ok=True)
        added = np.load(store / "vectors.npy") if (store / "vectors.npy").exists() else np.zeros((0, len(vector)), np.float16)
        np.save(store / "vectors.npy", np.concatenate([added, vector[None].astype(np.float16)]))
        with (store / "sources.jsonl").open("a") as f:
            f.write(json.dumps({"source": source}) + "\n")
        _state["vectors"] = np.concatenate([vectors, vector[None].astype(np.float32)])
        _state["sources"] = sources + [source]


def preload() -> None:
    """Load the model and library at startup. Errors are logged only; screening then holds the listing."""
    try:
        with _lock:
            _model()
            _library()
        _log.info("known_ai_loaded")
    except Exception as exc:
        _log.exception("known_ai_preload_failed: %s", exc)


def reset_for_tests() -> None:
    _state.clear()
