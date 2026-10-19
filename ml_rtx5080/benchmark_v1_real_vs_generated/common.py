"""Shared helpers for benchmark_v1_real_vs_generated: paths, normalisation, hashing, manifests.

Nothing here opens a training or locked-test image: overlap checks use the SHA-256 and dHash
values already stored in split_manifest_v2.json.
"""

from __future__ import annotations

import hashlib
import io
import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Iterator

from PIL import Image, ImageOps

HERE = Path(__file__).resolve().parent
ML_ROOT = HERE.parent
REPO = ML_ROOT.parent
DATA = HERE / "data"
RESULTS = HERE / "results"
PROTOCOL_PATH = HERE / "protocol.json"
TRAINING_MANIFEST = ML_ROOT / "experiments" / "dataset_audit" / "split_manifest_v2.json"

for _p in (ML_ROOT, REPO / "backend"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from split_v2 import dhash_image  # noqa: E402  (same hash the training manifest stores)

NORMALISED_SIDE = 768
JPEG_QUALITY = 90
MIN_SOURCE_SIDE = 768
DHASH_EXCLUDE_BITS = 2
CLASSES = ("REAL_PHOTO", "GENERATED")
BRANDS = ("A. Lange & Söhne", "Audemars Piguet", "Patek Philippe", "Richard Mille", "Vacheron Constantin")


@dataclass
class Sample:
    sample_id: str
    label: str  # REAL_PHOTO | GENERATED
    group: str  # real_ebay | heldout_kandinsky22 | sdxl_base10 | smoke_*
    brand: str
    unit_id: str  # independence unit: listing for real photos, prompt template for generated
    path: str  # normalised JPEG, relative to HERE
    original_sha256: str
    original_dhash: str
    normalised_sha256: str
    source_width: int
    source_height: int
    excluded: str | None = None
    meta: dict = field(default_factory=dict)


def load_protocol() -> dict:
    return json.loads(PROTOCOL_PATH.read_text())


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def normalise(raw: bytes) -> tuple[bytes, Image.Image, tuple[int, int]]:
    """Apply the pre-registered normalisation. Returns (jpeg bytes, decoded original, original size)."""
    with Image.open(io.BytesIO(raw)) as im:
        im.load()
        original = ImageOps.exif_transpose(im).convert("RGB")
    w, h = original.size
    side = min(w, h)
    left, top = (w - side) // 2, (h - side) // 2
    square = original.crop((left, top, left + side, top + side))
    if side != NORMALISED_SIDE:
        # Sources below the minimum are excluded by admit(); this keeps every admitted image the same size.
        square = square.resize((NORMALISED_SIDE, NORMALISED_SIDE), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    square.save(buf, format="JPEG", quality=JPEG_QUALITY, subsampling=2, optimize=True)
    return buf.getvalue(), original, (w, h)


def hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()


class TrainingIndex:
    """Stored hashes of every image in the training manifest (train, calibration, validation, test)."""

    def __init__(self, sha256s: set[str], dhashes: list[int]) -> None:
        self.sha256s = sha256s
        self.dhashes = dhashes

    @classmethod
    def load(cls, path: Path = TRAINING_MANIFEST) -> "TrainingIndex":
        samples = json.loads(path.read_text())["samples"]
        return cls(
            {s["sha256"] for s in samples if s.get("sha256")},
            [int(s["perceptual_hash"], 16) for s in samples if s.get("perceptual_hash")],
        )

    def overlap(self, sha256: str, dhash: int) -> str | None:
        if sha256 in self.sha256s:
            return "training_sha256_match"
        if any(hamming(dhash, d) <= DHASH_EXCLUDE_BITS for d in self.dhashes):
            return "training_dhash_near_duplicate"
        return None


class BenchmarkDedup:
    def __init__(self) -> None:
        self.sha256s: set[str] = set()
        self.dhashes: list[int] = []

    def check_and_add(self, sha256: str, dhash: int) -> str | None:
        if sha256 in self.sha256s:
            return "benchmark_duplicate_sha256"
        if any(hamming(dhash, d) <= DHASH_EXCLUDE_BITS for d in self.dhashes):
            return "benchmark_near_duplicate"
        self.sha256s.add(sha256)
        self.dhashes.append(dhash)
        return None


def admit(
    raw: bytes,
    *,
    sample_id: str,
    label: str,
    group: str,
    brand: str,
    unit_id: str,
    out_dir: Path,
    training: TrainingIndex | None,
    dedup: BenchmarkDedup,
    meta: dict,
) -> Sample:
    """Normalise one image, apply the exclusion rules and write it if admitted."""
    jpeg, original, (w, h) = normalise(raw)
    orig_sha = sha256_bytes(raw)
    dh = dhash_image(original)
    excluded = None
    if min(w, h) < MIN_SOURCE_SIDE:
        excluded = "source_short_side_below_768"
    elif training is not None and (reason := training.overlap(orig_sha, dh)):
        excluded = reason
    elif reason := dedup.check_and_add(orig_sha, dh):
        excluded = reason
    rel = ""
    if excluded is None:
        out_dir.mkdir(parents=True, exist_ok=True)
        dest = out_dir / f"{sample_id}.jpg"
        dest.write_bytes(jpeg)
        rel = str(dest.relative_to(HERE)) if dest.is_relative_to(HERE) else str(dest)
    return Sample(
        sample_id=sample_id, label=label, group=group, brand=brand, unit_id=unit_id, path=rel,
        original_sha256=orig_sha, original_dhash=f"{dh:016x}", normalised_sha256=sha256_bytes(jpeg) if not excluded else "",
        source_width=w, source_height=h, excluded=excluded, meta=meta,
    )


def append_manifest(path: Path, samples: Iterable[Sample]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(asdict(s), ensure_ascii=False) + "\n")


def read_manifest(path: Path) -> Iterator[Sample]:
    if not path.exists():
        return
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield Sample(**json.loads(line))
