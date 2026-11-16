"""Image fingerprints from the pretrained DINOv3 ViT-B backbone (public weights, no fine-tuning).

This is not the authenticity classifier. That checkpoint was trained to answer "real photo or
generated?", which throws away the details that tell one watch model from another. The plain
self-supervised backbone keeps them, so nearest neighbours in its feature space look alike.

The CLS token is the fingerprint. On the eBay gallery (leave-one-out, 369 labelled photos) it named
the model 56.6% of the time against 15.4% for the mean of the patch tokens, whose similarities
are dominated by background (see ml_rtx5080/experiments/watch_identify_v1/eval_report.json).
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time

import numpy as np
from PIL import Image, ImageOps

_log = logging.getLogger(__name__)

BACKBONE = "vit_base_patch16_dinov3.lvd1689m"
SIDE = 256
PREFIX_TOKENS = 5  # CLS + 4 registers
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
_PAD = tuple(int(round(m * 255)) for m in MEAN)


def preprocess(image: Image.Image) -> np.ndarray:
    """Letterbox to SIDE x SIDE (the whole watch stays in frame), normalize, CHW float32."""
    square = ImageOps.pad(image.convert("RGB"), (SIDE, SIDE), method=Image.BICUBIC, color=_PAD)
    arr = (np.asarray(square, dtype=np.float32) / 255.0 - MEAN) / STD
    return arr.transpose(2, 0, 1)


class Embedder:
    """Lazily loaded backbone. Uses CUDA when present, otherwise CPU."""

    def __init__(self, device: str | None = None) -> None:
        self._lock = threading.Lock()
        self._model = None
        self._device = device

    def _load(self):
        import timm
        import torch

        device = self._device or ("cuda" if torch.cuda.is_available() else "cpu")
        model = timm.create_model(BACKBONE, pretrained=True, num_classes=0).to(device).eval()
        for p in model.parameters():
            p.requires_grad_(False)
        self.torch, self._model, self.device = torch, model, device
        _log.info("identify_embedder_loaded backbone=%s device=%s", BACKBONE, device)

    @property
    def dim(self) -> int:
        return 768

    def embed_pools(self, images: list[Image.Image]) -> dict[str, np.ndarray]:
        """L2-normalised vectors [N, 768] float32 for both poolings: "cls" and "patch_mean"."""
        with self._lock:
            if self._model is None:
                self._load()
            torch = self.torch
            batch = torch.from_numpy(np.stack([preprocess(im) for im in images])).to(self.device)
            with torch.inference_mode():
                tokens = self._model.forward_features(batch)
            out = {}
            for pool, vec in (("cls", tokens[:, 0]), ("patch_mean", tokens[:, PREFIX_TOKENS:].mean(dim=1))):
                vec = vec.float()
                out[pool] = (vec / vec.norm(dim=-1, keepdim=True).clamp_min(1e-12)).cpu().numpy()
            return out

    def embed(self, images: list[Image.Image], pool: str = "cls") -> np.ndarray:
        return self.embed_pools(images)[pool]

    async def embed_one(self, image: Image.Image, pool: str = "cls") -> np.ndarray:
        started = time.perf_counter()
        vec = (await asyncio.to_thread(self.embed, [image], pool))[0]
        _log.info("identify_embed_ms=%.1f", (time.perf_counter() - started) * 1000)
        return vec


_shared: Embedder | None = None


def shared_embedder() -> Embedder:
    global _shared
    if _shared is None:
        _shared = Embedder()
    return _shared
