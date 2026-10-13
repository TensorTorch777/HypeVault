"""Shared scraper utilities for research-only image collection.

These helpers never assign VERIFIED_AUTHENTIC or VERIFIED_COUNTERFEIT.
Scraped images are UNVERIFIED_WEB_IMAGE; generated images are
SYNTHETIC_AI_GENERATED. Authenticity benchmarking eligibility is always false
until a separate evidence-backed human review process changes a record.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import tempfile
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Any, Mapping

from PIL import Image, UnidentifiedImageError

SCHEMA_VERSION = "scraper_manifest_v1"
RESUME_SCHEMA_VERSION = "scraper_resume_v1"

STATUS_UNVERIFIED_WEB = "UNVERIFIED_WEB_IMAGE"
STATUS_SYNTHETIC = "SYNTHETIC_AI_GENERATED"
STATUS_VERIFIED_AUTHENTIC = "VERIFIED_AUTHENTIC"
STATUS_VERIFIED_COUNTERFEIT = "VERIFIED_COUNTERFEIT"

AUTO_ASSIGNABLE_STATUSES = frozenset({STATUS_UNVERIFIED_WEB, STATUS_SYNTHETIC})
VERIFIED_STATUSES = frozenset({STATUS_VERIFIED_AUTHENTIC, STATUS_VERIFIED_COUNTERFEIT})

SUPPORTED_FORMATS = frozenset({"JPEG", "PNG", "WEBP"})
SUPPORTED_CONTENT_TYPES = frozenset({
    "image/jpeg",
    "image/jpg",
    "image/png",
    "image/webp",
})
SUPPORTED_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".webp"})

DEFAULT_MAX_BYTES = 15 * 1024 * 1024
DEFAULT_MIN_DIMENSION = 64
DEFAULT_MAX_DIMENSION = 8192
DEFAULT_TIMEOUT_SECONDS = 15
NEAR_DUP_HAMMING_THRESHOLD = 6

ARTIFACT_ROOT_REL = Path("local_data") / "scraper_artifacts"
UNVERIFIED_WEB_REL = ARTIFACT_ROOT_REL / "unverified_web_images"
SYNTHETIC_REL = ARTIFACT_ROOT_REL / "synthetic_research_only"
STATE_REL = ARTIFACT_ROOT_REL / "state"
MANIFEST_NAME = "manifest.jsonl"
INDEX_NAME = "content_index.json"
CORPUS_INDEX_NAME = "corpus_content_index.json"
EXACT_DUP_LOG_NAME = "exact_duplicate_events.jsonl"

LOGGER = logging.getLogger("hypevault.scraper")


class ScraperError(Exception):
    """Base error for scraper integrity failures."""


class ValidationError(ScraperError):
    """Raised when bytes are not a usable research image."""


class ResumeStateError(ScraperError):
    """Raised when on-disk resume state is malformed."""


def repo_root() -> Path:
    """Return the repository root regardless of the caller's cwd."""
    return Path(__file__).resolve().parents[1]


def resolve_under_repo(path: str | Path | None, default_relative: str | Path) -> Path:
    """Resolve a path; relative paths are taken from the repository root."""
    if path is None or path == "":
        return (repo_root() / Path(default_relative)).resolve()
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate.resolve()
    return (repo_root() / candidate).resolve()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def safe_brand_name(brand: str) -> str:
    cleaned = brand.strip().replace("/", "_").replace("\\", "_")
    cleaned = re.sub(r"\s+", " ", cleaned)
    if not cleaned:
        raise ValueError("brand name is empty")
    return cleaned


def family_output_dir(kind: str, family: str, brand: str, *, root: Path | None = None) -> Path:
    """Return the isolated artifact directory for a brand.

    kind is 'unverified_web' or 'synthetic'. family is 'watches' or 'sneakers'.
    """
    base = root or repo_root()
    if kind == "unverified_web":
        rel = UNVERIFIED_WEB_REL / family / safe_brand_name(brand)
    elif kind == "synthetic":
        rel = SYNTHETIC_REL / family / safe_brand_name(brand)
    else:
        raise ValueError(f"unsupported artifact kind: {kind}")
    return ensure_dir((base / rel).resolve())


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def is_supported_extension(path: Path) -> bool:
    return path.suffix.lower() in SUPPORTED_EXTENSIONS


def _looks_like_html(data: bytes) -> bool:
    head = data[:256].lstrip().lower()
    return head.startswith(b"<!doctype html") or head.startswith(b"<html") or b"<html" in head[:64]


def decode_image(
    data: bytes,
    *,
    max_bytes: int = DEFAULT_MAX_BYTES,
    min_dimension: int = DEFAULT_MIN_DIMENSION,
    max_dimension: int = DEFAULT_MAX_DIMENSION,
    content_type: str | None = None,
) -> tuple[Image.Image, str]:
    """Decode and validate image bytes. Returns (image, format_name)."""
    if not data:
        raise ValidationError("empty response body")
    if len(data) > max_bytes:
        raise ValidationError(f"response exceeds max_bytes={max_bytes}")
    if _looks_like_html(data):
        raise ValidationError("response looks like an HTML error page")
    if content_type:
        mime = content_type.split(";", 1)[0].strip().lower()
        if mime and mime not in SUPPORTED_CONTENT_TYPES and not mime.startswith("image/"):
            raise ValidationError(f"unsupported content type: {content_type}")
        if mime and mime.startswith("image/") and mime not in SUPPORTED_CONTENT_TYPES:
            raise ValidationError(f"unsupported image content type: {content_type}")
    try:
        image = Image.open(BytesIO(data))
        image.load()
    except UnidentifiedImageError as exc:
        raise ValidationError("bytes are not a decodable image") from exc
    except OSError as exc:
        raise ValidationError(f"truncated or unreadable image: {exc}") from exc
    fmt = (image.format or "").upper()
    if fmt not in SUPPORTED_FORMATS:
        raise ValidationError(f"unsupported image format: {fmt or 'UNKNOWN'}")
    width, height = image.size
    if width < min_dimension or height < min_dimension:
        raise ValidationError(f"image too small: {width}x{height}")
    if width > max_dimension or height > max_dimension:
        raise ValidationError(f"image too large: {width}x{height}")
    return image, fmt


def normalize_image_bytes(image: Image.Image, fmt: str = "JPEG") -> bytes:
    """Normalize a valid image to a documented research format (JPEG RGB)."""
    working = image
    if working.mode not in {"RGB", "L"}:
        working = working.convert("RGB")
    elif working.mode == "L":
        working = working.convert("RGB")
    buffer = BytesIO()
    working.save(buffer, format=fmt, quality=95, optimize=True)
    return buffer.getvalue()


def average_hash(image: Image.Image, hash_size: int = 8) -> str:
    """Return a hex average-hash for near-duplicate review (not identity)."""
    gray = image.convert("L").resize((hash_size, hash_size), Image.Resampling.LANCZOS)
    pixels = list(gray.getdata())
    mean = sum(pixels) / len(pixels)
    bits = "".join("1" if pixel >= mean else "0" for pixel in pixels)
    value = int(bits, 2)
    return f"{value:0{hash_size * hash_size // 4}x}"


def hamming_distance(hash_a: str, hash_b: str) -> int:
    if len(hash_a) != len(hash_b):
        raise ValueError("hash lengths differ")
    return sum(bin(int(a, 16) ^ int(b, 16)).count("1") for a, b in zip(hash_a, hash_b))


def atomic_write_bytes(path: Path, data: bytes) -> None:
    ensure_dir(path.parent)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".partial", dir=str(path.parent))
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)


def atomic_write_text(path: Path, text: str) -> None:
    atomic_write_bytes(path, text.encode("utf-8"))


def append_jsonl(path: Path, record: Mapping[str, Any]) -> None:
    ensure_dir(path.parent)
    line = json.dumps(dict(record), sort_keys=True, ensure_ascii=False) + "\n"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line)
        handle.flush()
        os.fsync(handle.fileno())


def build_manifest_record(
    *,
    relative_image_path: str,
    stored_sha256: str,
    original_sha256: str | None,
    width: int,
    height: int,
    image_format: str,
    byte_size: int,
    family: str,
    claimed_brand: str,
    claimed_model: str | None,
    search_query: str | None,
    search_provider: str | None,
    source_image_url: str | None,
    source_page_url: str | None,
    label_status: str,
    perceptual_hash: str | None = None,
    duplicate_status: str = "UNIQUE",
    near_duplicate_of: str | None = None,
    generation: Mapping[str, Any] | None = None,
    provenance_note: str | None = None,
) -> dict[str, Any]:
    if label_status not in AUTO_ASSIGNABLE_STATUSES:
        raise ValueError(
            f"scraper tools may only assign {sorted(AUTO_ASSIGNABLE_STATUSES)}; "
            f"got {label_status}"
        )
    missing_source = not source_image_url or source_image_url in {"", "UNKNOWN", "NOT_RECORDED"}
    provenance_status = "INCOMPLETE" if missing_source else "PARTIAL"
    if provenance_note:
        note = provenance_note
    elif missing_source:
        note = "source URL unavailable from collector; recorded as UNKNOWN"
    else:
        note = "source URL recorded; rights and authenticity remain unreviewed"
    record = {
        "schema_version": SCHEMA_VERSION,
        "record_id": str(uuid.uuid4()),
        "created_at_utc": utc_now_iso(),
        "image_path": relative_image_path,
        "stored_sha256": stored_sha256,
        "original_sha256": original_sha256 or "NOT_RECORDED",
        "width": width,
        "height": height,
        "format": image_format,
        "byte_size": byte_size,
        "family": family,
        "claimed_brand": claimed_brand,
        "claimed_model": claimed_model or "UNKNOWN",
        "search_query": search_query or "NOT_RECORDED",
        "retrieved_at_utc": utc_now_iso(),
        "search_provider": search_provider or "UNKNOWN",
        "source_image_url": source_image_url or "UNKNOWN",
        "source_page_url": source_page_url or "UNKNOWN",
        "provenance_completeness": provenance_status,
        "provenance_note": note,
        "label_status": label_status,
        "target_label": None,
        "evidence_status": "UNREVIEWED",
        "rights_review_status": "UNREVIEWED",
        "intended_use_eligibility": {
            "authenticity_benchmark": False,
            "model_training_as_verified_class": False,
            "reason": (
                "Automatic scraper/generator output is not independently verified "
                "authentic or counterfeit evidence."
            ),
        },
        "duplicate_status": duplicate_status,
        "near_duplicate_of": near_duplicate_of or "NOT_APPLICABLE",
        "perceptual_hash": perceptual_hash or "NOT_RECORDED",
        "generation": dict(generation) if generation else "NOT_APPLICABLE",
    }
    return record


@dataclass
class StoreStats:
    attempted: int = 0
    stored: int = 0
    exact_duplicates: int = 0
    near_duplicate_candidates: int = 0
    invalid: int = 0
    errors: int = 0

    def as_dict(self) -> dict[str, int]:
        return {
            "attempted": self.attempted,
            "stored": self.stored,
            "exact_duplicates": self.exact_duplicates,
            "near_duplicate_candidates": self.near_duplicate_candidates,
            "invalid": self.invalid,
            "errors": self.errors,
        }


@dataclass
class ContentIndex:
    path: Path
    sha256_to_path: dict[str, str] = field(default_factory=dict)
    perceptual_hashes: dict[str, str] = field(default_factory=dict)

    @classmethod
    def load(cls, path: Path) -> "ContentIndex":
        if not path.exists():
            return cls(path=path)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ResumeStateError(f"malformed content index at {path}: {exc}") from exc
        if not isinstance(payload, dict):
            raise ResumeStateError(f"malformed content index at {path}: root must be object")
        sha_map = payload.get("sha256_to_path", {})
        phash_map = payload.get("perceptual_hashes", {})
        if not isinstance(sha_map, dict) or not isinstance(phash_map, dict):
            raise ResumeStateError(f"malformed content index maps at {path}")
        return cls(
            path=path,
            sha256_to_path={str(k): str(v) for k, v in sha_map.items()},
            perceptual_hashes={str(k): str(v) for k, v in phash_map.items()},
        )

    def save(self) -> None:
        payload = {
            "schema_version": "scraper_content_index_v1",
            "sha256_to_path": self.sha256_to_path,
            "perceptual_hashes": self.perceptual_hashes,
        }
        atomic_write_text(self.path, json.dumps(payload, indent=2, sort_keys=True) + "\n")

    def find_near_duplicate(self, phash: str) -> str | None:
        for other_hash, rel_path in self.perceptual_hashes.items():
            if hamming_distance(phash, other_hash) <= NEAR_DUP_HAMMING_THRESHOLD:
                return rel_path
        return None


def count_valid_images(directory: Path) -> int:
    """Count only decodable supported images under directory (non-recursive)."""
    if not directory.exists():
        return 0
    count = 0
    for path in directory.iterdir():
        if not path.is_file() or not is_supported_extension(path):
            continue
        try:
            with path.open("rb") as handle:
                decode_image(handle.read())
            count += 1
        except (OSError, ValidationError):
            continue
    return count


def relative_to_artifact_root(path: Path, artifact_root: Path) -> str:
    return path.resolve().relative_to(artifact_root.resolve()).as_posix()


def corpus_index_path(artifact_root: Path) -> Path:
    """Shared exact/near-duplicate index for the entire artifact root."""
    return artifact_root / CORPUS_INDEX_NAME


def load_corpus_index(artifact_root: Path) -> ContentIndex:
    return ContentIndex.load(corpus_index_path(artifact_root))


def _safe_relpath(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path)


def _record_exact_duplicate_event(
    artifact_root: Path,
    *,
    stored_sha: str,
    original_sha: str,
    existing_path: str,
    claimed_brand: str,
    family: str,
    dest_dir: Path,
) -> dict[str, Any]:
    event = {
        "schema_version": SCHEMA_VERSION,
        "outcome": "EXACT_DUPLICATE",
        "created_at_utc": utc_now_iso(),
        "stored_sha256": stored_sha,
        "original_sha256": original_sha,
        "exact_duplicate_of": existing_path,
        "claimed_brand": claimed_brand,
        "family": family,
        "attempted_dest_dir": _safe_relpath(dest_dir, artifact_root),
        "newly_stored": False,
    }
    append_jsonl(artifact_root / "state" / EXACT_DUP_LOG_NAME, event)
    return event


def store_validated_image(
    data: bytes,
    *,
    dest_dir: Path,
    artifact_root: Path,
    family: str,
    claimed_brand: str,
    claimed_model: str | None,
    label_status: str,
    search_query: str | None = None,
    search_provider: str | None = None,
    source_image_url: str | None = None,
    source_page_url: str | None = None,
    generation: Mapping[str, Any] | None = None,
    provenance_note: str | None = None,
    content_type: str | None = None,
    max_bytes: int = DEFAULT_MAX_BYTES,
    index: ContentIndex | None = None,
    stats: StoreStats | None = None,
) -> dict[str, Any] | None:
    """Validate, normalize, dedupe corpus-wide, and store one image with a manifest row.

    Exact duplicates anywhere under the artifact root are not written again. The
    returned event remains traceable via ``exact_duplicate_of``. Near-duplicates
    are still stored with their own provenance and flagged for review.
    """
    stats = stats or StoreStats()
    stats.attempted += 1
    try:
        image, fmt = decode_image(data, max_bytes=max_bytes, content_type=content_type)
        del fmt
        original_sha = sha256_bytes(data)
        normalized = normalize_image_bytes(image)
        stored_sha = sha256_bytes(normalized)
        phash = average_hash(image)
        ensure_dir(dest_dir)
        ensure_dir(artifact_root)
        index = index or load_corpus_index(artifact_root)
        existing = index.sha256_to_path.get(stored_sha) or index.sha256_to_path.get(original_sha)
        if existing:
            stats.exact_duplicates += 1
            LOGGER.info(
                "exact duplicate skipped sha256=%s existing=%s attempted_brand=%s",
                stored_sha[:12],
                existing,
                claimed_brand,
            )
            return _record_exact_duplicate_event(
                artifact_root,
                stored_sha=stored_sha,
                original_sha=original_sha,
                existing_path=existing,
                claimed_brand=claimed_brand,
                family=family,
                dest_dir=dest_dir,
            )
        near_of = index.find_near_duplicate(phash)
        duplicate_status = "NEAR_DUPLICATE_CANDIDATE" if near_of else "UNIQUE"
        if near_of:
            stats.near_duplicate_candidates += 1
        filename = f"{stored_sha[:16]}.jpg"
        dest_path = dest_dir / filename
        if dest_path.exists():
            stats.exact_duplicates += 1
            rel_existing = relative_to_artifact_root(dest_path, artifact_root)
            return _record_exact_duplicate_event(
                artifact_root,
                stored_sha=stored_sha,
                original_sha=original_sha,
                existing_path=rel_existing,
                claimed_brand=claimed_brand,
                family=family,
                dest_dir=dest_dir,
            )
        atomic_write_bytes(dest_path, normalized)
        # Re-decode saved bytes so a partial write cannot count as success.
        with dest_path.open("rb") as handle:
            decode_image(handle.read())
        rel_path = relative_to_artifact_root(dest_path, artifact_root)
        record = build_manifest_record(
            relative_image_path=rel_path,
            stored_sha256=stored_sha,
            original_sha256=original_sha,
            width=image.width,
            height=image.height,
            image_format="JPEG",
            byte_size=len(normalized),
            family=family,
            claimed_brand=claimed_brand,
            claimed_model=claimed_model,
            search_query=search_query,
            search_provider=search_provider,
            source_image_url=source_image_url,
            source_page_url=source_page_url,
            label_status=label_status,
            perceptual_hash=phash,
            duplicate_status=duplicate_status,
            near_duplicate_of=near_of,
            generation=generation,
            provenance_note=provenance_note,
        )
        record["outcome"] = "STORED"
        record["newly_stored"] = True
        append_jsonl(dest_dir / MANIFEST_NAME, record)
        index.sha256_to_path[stored_sha] = rel_path
        index.sha256_to_path[original_sha] = rel_path
        index.perceptual_hashes[phash] = rel_path
        index.save()
        stats.stored += 1
        return record
    except ValidationError as exc:
        stats.invalid += 1
        LOGGER.warning("rejected invalid image: %s", exc)
        return None
    except Exception as exc:  # noqa: BLE001 - counted and surfaced to caller logs
        stats.errors += 1
        LOGGER.error("store failed: %s", exc)
        return None


def load_resume_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "schema_version": RESUME_SCHEMA_VERSION,
            "completed_queries": [],
            "failed_queries": [],
            "stats": StoreStats().as_dict(),
        }
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ResumeStateError(f"malformed resume state at {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ResumeStateError(f"malformed resume state at {path}: root must be object")
    if payload.get("schema_version") != RESUME_SCHEMA_VERSION:
        raise ResumeStateError(
            f"unsupported resume schema {payload.get('schema_version')!r} at {path}"
        )
    if not isinstance(payload.get("completed_queries"), list):
        raise ResumeStateError("completed_queries must be a list")
    return payload


def save_resume_state(path: Path, state: Mapping[str, Any]) -> None:
    payload = dict(state)
    payload["schema_version"] = RESUME_SCHEMA_VERSION
    payload["updated_at_utc"] = utc_now_iso()
    atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def mark_query_completed(state: dict[str, Any], query: str, *, success: bool) -> None:
    """Record query outcome. Failed queries are not permanently completed."""
    completed = set(state.get("completed_queries") or [])
    failed = set(state.get("failed_queries") or [])
    if success:
        completed.add(query)
        failed.discard(query)
    else:
        failed.add(query)
        completed.discard(query)
    state["completed_queries"] = sorted(completed)
    state["failed_queries"] = sorted(failed)


def download_bounded(
    session: Any,
    url: str,
    *,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    max_bytes: int = DEFAULT_MAX_BYTES,
) -> tuple[bytes, str | None]:
    """Download with size bound. Always closes the streaming response.

    Rejects non-200 responses and oversized bodies. Timeouts and read failures
    propagate after the response is closed.
    """
    response = session.get(url, timeout=timeout, stream=True)
    try:
        status = getattr(response, "status_code", None)
        if status is not None and status != 200:
            raise ValidationError(f"HTTP status {status} for url={url}")
        headers = getattr(response, "headers", {}) or {}
        content_type = headers.get("Content-Type") or headers.get("content-type")
        length = headers.get("Content-Length") or headers.get("content-length")
        if length is not None:
            try:
                if int(length) > max_bytes:
                    raise ValidationError(
                        f"Content-Length {length} exceeds max_bytes={max_bytes} for url={url}"
                    )
            except ValueError:
                pass
        chunks: list[bytes] = []
        total = 0
        try:
            for chunk in response.iter_content(chunk_size=64 * 1024):
                if not chunk:
                    continue
                total += len(chunk)
                if total > max_bytes:
                    raise ValidationError(
                        f"download exceeded max_bytes={max_bytes} for url={url}"
                    )
                chunks.append(chunk)
        except ValidationError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise ValidationError(f"download read failed for url={url}: {exc}") from exc
        return b"".join(chunks), content_type
    finally:
        close = getattr(response, "close", None)
        if callable(close):
            close()


def require_network_opt_in(allow_network: bool) -> None:
    if not allow_network:
        raise SystemExit(
            "Refusing to contact external services. Re-run with --allow-network "
            "after confirming permitted access and reuse rights for the target source."
        )


def require_generation_opt_in(allow_generation: bool) -> None:
    if not allow_generation:
        raise SystemExit(
            "Refusing to load a generative model or write synthetic images. "
            "Re-run with --allow-generation after reviewing the synthetic-research-only policy."
        )


def assert_not_legacy_training_dir(path: Path) -> None:
    parts = {part.lower() for part in path.resolve().parts}
    banned = {
        "label_0_watches",
        "label_1_watches",
        "label_0_sneakers",
        "label_1_sneakers",
    }
    if parts & banned:
        raise ScraperError(
            f"refusing to write into legacy training directory: {path}. "
            "Use local_data/scraper_artifacts/ instead."
        )


def make_tiny_jpeg_bytes(color: tuple[int, int, int] = (12, 34, 56), size: tuple[int, int] = (96, 96)) -> bytes:
    """Helper for offline tests: synthesize a tiny valid JPEG."""
    image = Image.new("RGB", size, color)
    return normalize_image_bytes(image)


def clear_cuda_cache() -> None:
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def is_cuda_oom(exc: BaseException) -> bool:
    name = type(exc).__name__
    text = str(exc).lower()
    return name == "OutOfMemoryError" or "out of memory" in text or "cuda" in text and "memory" in text


@dataclass
class OOMRetryController:
    max_retries: int = 3
    attempts: int = 0

    def should_retry(self, exc: BaseException) -> bool:
        if not is_cuda_oom(exc):
            return False
        self.attempts += 1
        if self.attempts > self.max_retries:
            return False
        clear_cuda_cache()
        return True

    def record_failure(self, failures_path: Path, *, prompt: str, seed: int, error: str) -> None:
        ensure_dir(failures_path.parent)
        entry = {
            "timestamp_utc": utc_now_iso(),
            "error_category": "CUDA_OOM",
            "prompt": prompt,
            "seed": seed,
            "error": error,
            "attempts": self.attempts,
        }
        append_jsonl(failures_path, entry)
