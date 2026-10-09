"""Authoritative model-manifest generation for RTX 5080 training runs.

Thresholds stay null until calibration writes them. This module does not read
backend environment defaults.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

SCHEMA_PATH = Path(__file__).with_name("model_manifest.schema.json")

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

REQUIRED_FIELDS = (
    "model_version",
    "model_family",
    "backbone",
    "backbone_revision",
    "architecture_version",
    "classifier_version",
    "image_size",
    "patch_size",
    "normalization",
    "calibration_version",
    "fake_threshold",
    "review_thresholds",
    "git_sha",
    "dataset_hash",
    "training_config_hash",
    "checkpoint_sha256",
)


def load_schema() -> dict:
    return json.loads(SCHEMA_PATH.read_text())


def infer_model_family(backbone: str) -> str:
    name = backbone.lower()
    if "dinov3" in name:
        return "dinov3"
    if "dinov2" in name:
        return "dinov2"
    raise ValueError(f"Cannot infer model family from backbone '{backbone}'")


def infer_patch_size(backbone: str) -> int:
    marker = "patch"
    name = backbone.lower()
    start = name.find(marker)
    if start < 0:
        raise ValueError(f"Cannot infer patch size from backbone '{backbone}'")
    digits = []
    for char in name[start + len(marker) :]:
        if char.isdigit():
            digits.append(char)
        else:
            break
    if not digits:
        raise ValueError(f"Cannot infer patch size from backbone '{backbone}'")
    return int("".join(digits))


def infer_backbone_revision(backbone: str) -> str | None:
    if "." not in backbone:
        return None
    revision = backbone.rsplit(".", 1)[-1].strip()
    return revision or None


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def training_config_hash(config: dict) -> str:
    payload = json.dumps(config, sort_keys=True, separators=(",", ":"), default=str)
    return sha256_bytes(payload.encode("utf-8"))


def dataset_hash_from_samples(samples: list[tuple[Path, int]]) -> str:
    """Stable identity from path, byte size, and label. Does not invent metadata."""
    digest = hashlib.sha256()
    rows = sorted((str(path.resolve()), int(label), path.stat().st_size) for path, label in samples)
    for path, label, size in rows:
        digest.update(f"{path}\t{label}\t{size}\n".encode())
    return digest.hexdigest()


def git_sha(repo_root: Path) -> str | None:
    try:
        output = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    sha = output.strip()
    return sha or None


def validate_model_manifest(manifest: dict) -> None:
    schema = load_schema()
    required = schema["required"]
    missing = [key for key in required if key not in manifest]
    if missing:
        raise ValueError(f"Manifest missing fields: {missing}")
    extra = [key for key in manifest if key not in schema["properties"]]
    if extra:
        raise ValueError(f"Manifest has unknown fields: {extra}")

    family = manifest["model_family"]
    if family not in schema["properties"]["model_family"]["enum"]:
        raise ValueError(f"Unsupported model_family '{family}'")
    patch = int(manifest["patch_size"])
    if patch not in schema["properties"]["patch_size"]["enum"]:
        raise ValueError(f"Unsupported patch_size {patch}")
    image_size = int(manifest["image_size"])
    if image_size % patch != 0:
        raise ValueError(f"image_size {image_size} is not divisible by patch_size {patch}")
    if family == "dinov3" and image_size == 504:
        raise ValueError("DINOv3 must not be trained or served at 504×504")
    if family == "dinov2" and patch != 14:
        raise ValueError("DINOv2 manifest requires patch_size 14")
    if family == "dinov3" and patch != 16:
        raise ValueError("DINOv3 manifest requires patch_size 16")

    normalization = manifest["normalization"]
    if list(normalization["mean"]) != list(IMAGENET_MEAN) or list(normalization["std"]) != list(IMAGENET_STD):
        raise ValueError("Normalization must match the ImageNet stats used in training")

    review = manifest["review_thresholds"]
    if review is not None:
        if float(review["authentic_below"]) >= float(review["fake_at_or_above"]):
            raise ValueError("review_thresholds require authentic_below < fake_at_or_above")

    config_hash = manifest["training_config_hash"]
    if not isinstance(config_hash, str) or len(config_hash) != 64:
        raise ValueError("training_config_hash must be a sha256 hex digest")
    checkpoint_hash = manifest["checkpoint_sha256"]
    if checkpoint_hash is not None and (not isinstance(checkpoint_hash, str) or len(checkpoint_hash) != 64):
        raise ValueError("checkpoint_sha256 must be a sha256 hex digest or null")


def build_model_manifest(
    *,
    backbone: str,
    image_size: int,
    training_config: dict,
    checkpoint_path: Path | None = None,
    dataset_samples: list[tuple[Path, int]] | None = None,
    architecture_version: str = "cls_only.v1",
    classifier_version: str = "ln-dropout-linear512-gelu-dropout-linear1",
    calibration_version: str | None = None,
    fake_threshold: float | None = None,
    review_thresholds: dict | None = None,
    repo_root: Path | None = None,
    model_version: str | None = None,
) -> dict:
    family = infer_model_family(backbone)
    patch_size = infer_patch_size(backbone)
    revision = infer_backbone_revision(backbone)
    config_hash = training_config_hash(training_config)
    checkpoint_sha = sha256_file(checkpoint_path) if checkpoint_path is not None and checkpoint_path.is_file() else None
    data_hash = dataset_hash_from_samples(dataset_samples) if dataset_samples else None
    root = repo_root if repo_root is not None else Path(__file__).resolve().parents[1]
    version = model_version or f"hv-{family}-{architecture_version}-img{int(image_size)}-{config_hash[:12]}"
    manifest = {
        "model_version": version,
        "model_family": family,
        "backbone": backbone,
        "backbone_revision": revision,
        "architecture_version": architecture_version,
        "classifier_version": classifier_version,
        "image_size": int(image_size),
        "patch_size": patch_size,
        "normalization": {"mean": list(IMAGENET_MEAN), "std": list(IMAGENET_STD)},
        "calibration_version": calibration_version,
        "fake_threshold": fake_threshold,
        "review_thresholds": review_thresholds,
        "git_sha": git_sha(root),
        "dataset_hash": data_hash,
        "training_config_hash": config_hash,
        "checkpoint_sha256": checkpoint_sha,
    }
    validate_model_manifest(manifest)
    return manifest


def write_training_manifest(out_dir: Path, **kwargs) -> dict:
    """Write model_manifest.json. Does not modify checkpoint files."""
    manifest = build_model_manifest(**kwargs)
    destination = out_dir / "model_manifest.json"
    destination.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest
