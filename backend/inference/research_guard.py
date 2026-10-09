"""Deployment states for the legacy DINOv2 route and the DINOv3 candidate.

Missing mode keeps the existing safe default: legacy DINOv2 may run, and
DINOv3 stays blocked. An unknown or malformed mode fails closed. It is never
treated as research.
"""

from __future__ import annotations

import os

DEPLOYMENT_ENV = "HYPEVAULT_DEPLOYMENT_MODE"
DINOV3_PRODUCTION_BLOCK_REASON = "DINOv3 candidate is blocked outside research mode."

RESEARCH = "RESEARCH"
SHADOW = "SHADOW"
LEGACY_PRODUCTION = "LEGACY_PRODUCTION"
PRODUCTION_BLOCKED = "PRODUCTION_BLOCKED"

_DINOV3_ALLOWED = frozenset({RESEARCH, SHADOW})


def deployment_state() -> str:
    """Return one explicit state. Unknown values do not become research."""
    if DEPLOYMENT_ENV not in os.environ:
        return LEGACY_PRODUCTION
    raw = os.environ.get(DEPLOYMENT_ENV, "")
    value = raw.strip().lower()
    if value == "":
        return PRODUCTION_BLOCKED
    if value == "research":
        return RESEARCH
    if value == "shadow":
        return SHADOW
    if value == "production":
        return LEGACY_PRODUCTION
    return PRODUCTION_BLOCKED


def deployment_mode() -> str:
    return deployment_state()


def dinov3_research_allowed() -> bool:
    return deployment_state() in _DINOV3_ALLOWED


def live_path_targets_dinov3(
    local_model_path: str,
    triton_model_name: str,
    dinov2_model_name: str,
) -> bool:
    """True when live settings point at the DINOv3 candidate instead of DINOv2."""
    blob = " ".join((local_model_path, triton_model_name, dinov2_model_name)).lower()
    return "dinov3" in blob
