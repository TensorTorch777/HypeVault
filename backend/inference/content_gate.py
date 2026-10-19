"""Is there a wristwatch in this photo? Zero-shot CLIP check run before any authenticity model.

The DINOv2 listing model only answers "real photograph or AI-generated?". A real photo of
a person passes that question, so without this gate a selfie could be listed as a watch.

CLIP scores the image against short descriptions of watch photos and of common
non-watch subjects. The softmax mass on the watch descriptions is the watch score; when
the photo is rejected, the strongest non-watch description becomes the seller's reason.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass

from PIL import Image

from config import settings

_log = logging.getLogger(__name__)

WATCH_PROMPTS = (
    "a photo of a wristwatch",
    "a close-up photo of a watch dial",
    "a product photo of a luxury watch",
    "a photo of a watch worn on a wrist",
    "a photo of the back of a mechanical watch showing the movement",
    "a photo of a watch lying on a table",
    "a photo of a watch in its box",
    "a photo of a rectangular tonneau-shaped watch on a wrist",
    "a photo of a skeleton watch with a visible movement",
)

# (prompt, what we tell the seller the photo looks like)
OTHER_PROMPTS = (
    ("a photo of a person", "a person"),
    ("a selfie", "a selfie or portrait"),
    ("a portrait photo of a face", "a selfie or portrait"),
    ("a screenshot of a website or app", "a screenshot"),
    ("a document or a page of text", "a document or text"),
    ("a photo of shoes or sneakers", "footwear"),
    ("a photo of clothing", "clothing"),
    ("a photo of a handbag", "a bag"),
    ("a photo of jewellery such as a ring or a necklace", "jewellery"),
    ("a photo of a wall clock", "a wall or desk clock"),
    ("a photo of a mobile phone", "a phone"),
    ("a photo of an animal", "an animal"),
    ("a photo of food", "food"),
    ("a photo of a car", "a vehicle"),
    ("a photo of a landscape or a building", "a landscape or building"),
    ("a cartoon or a drawing", "a drawing or cartoon"),
    ("a blank, dark or blurry image", "a blank or unreadable image"),
)


@dataclass(frozen=True)
class ContentCheck:
    is_watch: bool
    watch_score: float
    looks_like: str | None  # strongest non-watch subject, when the photo is not a watch
    model: str


class _Clip:
    """Lazily loaded CLIP model with cached text embeddings."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._ready = False

    def _load(self) -> None:
        import torch
        from transformers import CLIPModel, CLIPProcessor

        device = settings.content_gate_device
        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
        model = CLIPModel.from_pretrained(settings.content_gate_model).to(device).eval()
        processor = CLIPProcessor.from_pretrained(settings.content_gate_model)
        prompts = list(WATCH_PROMPTS) + [p for p, _ in OTHER_PROMPTS]
        with torch.no_grad():
            tokens = processor(text=prompts, return_tensors="pt", padding=True).to(device)
            text = _as_tensor(model.get_text_features(**tokens))
            text = text / text.norm(dim=-1, keepdim=True)
        self.torch = torch
        self.model = model
        self.processor = processor
        self.device = device
        self.text = text
        self.scale = float(model.logit_scale.exp().item())
        self._ready = True

    def probabilities(self, image: Image.Image) -> list[float]:
        with self._lock:
            if not self._ready:
                self._load()
            torch = self.torch
            with torch.no_grad():
                pixels = self.processor(images=image.convert("RGB"), return_tensors="pt").to(self.device)
                img = _as_tensor(self.model.get_image_features(**pixels))
                img = img / img.norm(dim=-1, keepdim=True)
                logits = self.scale * img @ self.text.T
                return logits.softmax(dim=-1)[0].tolist()


def _as_tensor(out):  # transformers 5 may wrap features in an output object
    if hasattr(out, "norm"):
        return out
    for name in ("image_embeds", "text_embeds", "pooler_output"):
        value = getattr(out, name, None)
        if value is not None:
            return value
    raise TypeError(f"unexpected CLIP feature output: {type(out).__name__}")


_clip = _Clip()


def score_watch(image: Image.Image) -> ContentCheck:
    probs = _clip.probabilities(image)
    n = len(WATCH_PROMPTS)
    watch_score = float(sum(probs[:n]))
    others = probs[n:]
    best = max(range(len(others)), key=lambda i: others[i])
    is_watch = watch_score >= settings.content_gate_min_watch_score
    return ContentCheck(
        is_watch=is_watch,
        watch_score=watch_score,
        looks_like=None if is_watch else OTHER_PROMPTS[best][1],
        model=settings.content_gate_model,
    )
