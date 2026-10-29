"""Second origin check: an open-source detector trained on frontier image generators.

Our own models (DINOv2 live, DINOv3 epoch_018, the Phase 74 head) were trained on a handful of
diffusion generators. A ChatGPT/Gemini-style image of a Patek Philippe Grandmaster Chime passed
all three as a "real photo" (DINOv2 99.4% original). This detector was fine-tuned on 40,101
frontier-generator images (GPT Image, DALL-E 3, FLUX, Imagen, Seedream, Midjourney and others)
on top of Community Forensics (CVPR 2025, 4,803 generators).

  Thermostatic/community-forensics-frontier-detector-2026-08, MIT licence, ViT-S/16 at 384 px.
  Preprocess: short edge -> 440 (bicubic), centre crop 384, ImageNet mean/std.
  Decision: raw logit >= 1.359375 (the published frozen boundary) = AI-generated.
  Probability: sigmoid(logit - 0.7403), the published bias calibrator (0.65 at the boundary).

Weights are pinned by revision and SHA-256 and loaded into a plain timm ViT by our own code;
nothing from the model repository is executed. The model card warns it is a screening signal,
weak on very small or very low-resolution images, and not proof of provenance.
"""

from __future__ import annotations

import hashlib
import logging
import math
import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

_log = logging.getLogger(__name__)
_REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class AiScore:
    is_ai: bool
    probability: float  # calibrated P(AI-generated)
    logit: float


class DetectorUnavailable(RuntimeError):
    """The weights are missing or do not match the pinned hash."""


class FrontierDetector:
    REPO = "Thermostatic/community-forensics-frontier-detector-2026-08"
    REVISION = "16db135220b318d811b207db576d90368980b595"
    WEIGHTS = "model.safetensors"
    WEIGHTS_SHA256 = "2f445aecc310caee9b55d3ae0aeebc404a32d72e5836e30523ce0a9a1720cd47"
    ARCH = "vit_small_patch16_384"
    SHORT_EDGE = 440
    SIDE = 384
    MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    LOGIT_BOUNDARY = 1.359375
    CALIBRATION_INTERCEPT = -0.7403357915937764

    def __init__(self, weights_dir: Path | None = None, device: str | None = None) -> None:
        self.weights_dir = weights_dir or _REPO_ROOT / "models" / "external" / "commfor_frontier_2026_08"
        self._device = device
        self._model = None
        self._lock = threading.Lock()

    def _weights(self) -> Path:
        path = self.weights_dir / self.WEIGHTS
        if not path.exists():
            # Free download from the Hub, pinned to the reviewed revision.
            from huggingface_hub import hf_hub_download

            hf_hub_download(self.REPO, self.WEIGHTS, revision=self.REVISION, local_dir=self.weights_dir)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != self.WEIGHTS_SHA256:
            raise DetectorUnavailable(f"{path} does not match the pinned SHA-256")
        return path

    def _load(self) -> None:
        import timm
        import torch
        from safetensors.torch import load_file

        device = self._device or ("cuda" if torch.cuda.is_available() else "cpu")
        model = timm.create_model(self.ARCH, pretrained=False, num_classes=1)
        model.load_state_dict(load_file(self._weights()), strict=True)
        self.torch, self._model, self.device = torch, model.to(device).eval(), device
        _log.info("ai_image_detector_loaded repo=%s device=%s", self.REPO, device)

    def preprocess(self, image: Image.Image) -> np.ndarray:
        im = image.convert("RGB")
        scale = self.SHORT_EDGE / min(im.size)
        im = im.resize((max(self.SIDE, round(im.width * scale)), max(self.SIDE, round(im.height * scale))), Image.BICUBIC)
        left, top = (im.width - self.SIDE) // 2, (im.height - self.SIDE) // 2
        im = im.crop((left, top, left + self.SIDE, top + self.SIDE))
        arr = (np.asarray(im, dtype=np.float32) / 255.0 - self.MEAN) / self.STD
        return arr.transpose(2, 0, 1)

    def score_batch(self, images: list[Image.Image]) -> list[AiScore]:
        with self._lock:
            if self._model is None:
                self._load()
            torch = self.torch
            x = torch.from_numpy(np.stack([self.preprocess(im) for im in images])).to(self.device)
            with torch.inference_mode():
                logits = self._model(x).reshape(-1).float().cpu().tolist()
        return [
            AiScore(is_ai=z >= self.LOGIT_BOUNDARY, probability=1.0 / (1.0 + math.exp(-(z + self.CALIBRATION_INTERCEPT))), logit=z)
            for z in logits
        ]

    def score(self, image: Image.Image) -> AiScore:
        return self.score_batch([image])[0]


_shared: FrontierDetector | None = None


def shared_detector() -> FrontierDetector:
    global _shared
    if _shared is None:
        _shared = FrontierDetector()
    return _shared
