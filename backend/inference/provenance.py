"""Does the file carry a signed Content Credentials (C2PA) label saying an AI made it?

ChatGPT and Gemini sign every image they create. The label lives in the file itself, so it is
read from the uploaded bytes, before any decoding. A label is decisive when present; its absence
says nothing, because screenshots and re-saves drop it.

AI when either:
  1. any manifest in the file records an AI digital source type (IPTC trainedAlgorithmicMedia etc.;
     Gemini puts it on an inner manifest), or
  2. the label was made by a known AI image service and records no camera capture
     (older Gemini images only say "Added visible watermark").

A certificate that fails trust checks still counts: forging an AI label onto a real photo only
harms the forger. When the library is missing or the file has no label, the result is None.
"""

from __future__ import annotations

import io
import json
import logging
from dataclasses import dataclass

_log = logging.getLogger(__name__)

AI_SOURCE_TYPES = {"trainedAlgorithmicMedia", "compositeWithTrainedAlgorithmicMedia", "algorithmicMedia", "compositeSynthetic"}
CAPTURE_SOURCE_TYPES = {"digitalCapture", "computationalCapture", "negativeFilm", "positiveFilm", "print"}
AI_SERVICES = ("OpenAI Media Service", "Google C2PA Core Generator Library")


@dataclass(frozen=True)
class AiLabel:
    maker: str  # who signed the label, e.g. "OpenAI OpCo, LLC"
    tool: str | None  # e.g. "ChatGPT gpt-image"
    source_type: str | None  # the AI source type found, or None under rule 2
    validation_state: str | None


def _actions(manifest: dict) -> list[dict]:
    return [a for asr in manifest.get("assertions", []) if "actions" in asr.get("label", "")
            for a in (asr.get("data") or {}).get("actions", [])]


def _source_type(action: dict) -> str:
    return (action.get("digitalSourceType") or "").rsplit("/", 1)[-1]


def _tool(action: dict) -> str | None:
    agent = action.get("softwareAgent")
    if isinstance(agent, dict):
        return " ".join(x for x in (agent.get("name"), agent.get("version")) if x) or None
    return agent or None


def ai_label_from_store(store: dict) -> AiLabel | None:
    """Decide from a parsed manifest store (c2pa Reader JSON)."""
    manifests = store.get("manifests") or {}
    active = manifests.get(store.get("active_manifest") or "", {})
    signer = active.get("signature_info") or {}
    maker = signer.get("issuer") or signer.get("common_name") or "an unnamed signer"
    actions = [a for m in manifests.values() for a in _actions(m)]
    state = store.get("validation_state")
    for a in actions:
        if _source_type(a) in AI_SOURCE_TYPES:
            return AiLabel(maker, _tool(a), _source_type(a), state)
    names = [i.get("name") or "" for m in manifests.values() for i in m.get("claim_generator_info") or []]
    names += [(m.get("signature_info") or {}).get("common_name") or "" for m in manifests.values()]
    by_ai_service = any(n.startswith(s) for n in names for s in AI_SERVICES)
    captured = any(_source_type(a) in CAPTURE_SOURCE_TYPES for a in actions)
    if by_ai_service and not captured:
        return AiLabel(maker, next((t for a in actions if (t := _tool(a))), None), None, state)
    return None


def read_ai_label(raw: bytes, mime: str) -> AiLabel | None:
    try:
        import c2pa
    except ImportError:
        _log.warning("provenance_unavailable: c2pa-python is not installed")
        return None
    try:
        with c2pa.Reader(mime, io.BytesIO(raw)) as reader:
            store = json.loads(reader.json())
    except Exception as exc:  # no label (the usual case) or an unreadable one
        if "ManifestNotFound" not in type(exc).__name__:
            _log.info("provenance_unreadable: %s", exc)
        return None
    return ai_label_from_store(store)
