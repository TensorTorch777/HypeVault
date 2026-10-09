"""Startup-verified identity of the frozen DINOv3 checkpoint.

The SHA-256 of the ~1 GB checkpoint is computed at initialization (and again
on an explicit reload). It is cached only after it matches the frozen digest.
Request-path inference must not re-hash the file.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from inference.scope_gate import CHECKPOINT_SHA256

FROZEN_CHECKPOINT_SHA256 = CHECKPOINT_SHA256
CHECKPOINT = (
    Path(__file__).resolve().parents[2]
    / "ml_rtx5080"
    / "experiments"
    / "v2_dinov3_cls_patch_attention"
    / "epoch_018.pt"
)

_log = logging.getLogger(__name__)
_verified_digest: str | None = None
_hash_calls = 0
_last_error: CheckpointIdentityError | None = None


class CheckpointIdentityError(Exception):
    """The checkpoint is missing, unreadable, or does not match the frozen SHA-256."""

    def __init__(self, message: str, *, status: str = "POLICY_ERROR") -> None:
        super().__init__(message)
        self.status = status


def hash_call_count() -> int:
    return _hash_calls


def checkpoint_identity_ready() -> bool:
    return _verified_digest == FROZEN_CHECKPOINT_SHA256


def reset_checkpoint_identity_for_tests() -> None:
    global _verified_digest, _hash_calls, _last_error
    _verified_digest = None
    _hash_calls = 0
    _last_error = None


def checkpoint_identity_status() -> dict[str, str | int | bool | None]:
    ready = checkpoint_identity_ready()
    return {
        "verified": ready,
        "sha256": _verified_digest if ready else None,
        "hash_calls": _hash_calls,
    }


def verified_checkpoint_digest() -> str:
    """Return the cached digest. Does not read the checkpoint file."""
    if _verified_digest == FROZEN_CHECKPOINT_SHA256:
        return _verified_digest
    if _last_error is not None:
        raise CheckpointIdentityError(str(_last_error), status=_last_error.status)
    raise CheckpointIdentityError(
        "DINOv3 checkpoint identity is not verified.",
        status="MODEL_ERROR",
    )


def _compute_digest(path: Path) -> str:
    ml = Path(__file__).resolve().parents[2] / "ml_rtx5080"
    if str(ml) not in sys.path:
        sys.path.insert(0, str(ml))
    from reference_inference import file_sha256

    return file_sha256(path)


def _hash_file(path: Path) -> str:
    global _hash_calls
    _hash_calls += 1
    return _compute_digest(path)


def verify_frozen_checkpoint(*, path: Path | None = None, force: bool = False) -> str:
    """Hash the checkpoint, compare it with the frozen SHA-256, and cache only on match."""
    global _verified_digest, _last_error
    if not force and _verified_digest == FROZEN_CHECKPOINT_SHA256:
        return _verified_digest
    target = Path(path) if path is not None else CHECKPOINT
    try:
        digest = _hash_file(target)
    except CheckpointIdentityError:
        raise
    except Exception as exc:
        _verified_digest = None
        _last_error = CheckpointIdentityError(
            "The DINOv3 checkpoint could not be verified.",
            status="MODEL_ERROR",
        )
        raise _last_error from exc
    if digest != FROZEN_CHECKPOINT_SHA256:
        _verified_digest = None
        _last_error = CheckpointIdentityError(
            "DINOv3 checkpoint does not match the frozen artifact.",
            status="POLICY_ERROR",
        )
        raise _last_error
    _verified_digest = digest
    _last_error = None
    _log.info("dinov3_checkpoint_verified sha256=%s hash_calls=%s", digest, _hash_calls)
    return digest


def reload_checkpoint_identity(*, path: Path | None = None) -> str:
    """Clear the cache and re-hash after an explicit reload or replacement."""
    global _verified_digest
    _verified_digest = None
    return verify_frozen_checkpoint(path=path, force=True)
