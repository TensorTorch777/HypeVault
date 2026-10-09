"""Fail-closed shadow policy contract. This is not a live verdict.

Precedence, documented and unchanged from the Phase 26 shadow policy:

1. LOW_RESOLUTION or LOW_CONTRAST -> REVIEW
2. calibrated P(fake) >= 0.50 -> FAKE
3. otherwise -> AUTHENTIC

OOD status ``unavailable`` is not a score, not a false flag, and not an
in-distribution claim. POLICY_ERROR is an internal system status. It is not
an authenticity verdict. The policy version is ``shadow_v1`` and remains
``not_promoted``.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any

from policy import (
    AUTHENTICITY_THRESHOLD,
    EXPECTED_CHECKPOINT_SHA256,
    FROZEN_TEMPERATURE,
    LOW_CONTRAST_CUTOFF,
    LOW_RESOLUTION_CUTOFF,
    OOD_UNAVAILABLE,
    PRODUCTION_POLICY_STATUS,
    QUALITY_RULE,
    QUALITY_RULE_STATUS,
    THRESHOLD_STATUS,
    calibrated_probability_from_logit,
    parse_flags,
)

POLICY_VERSION = "shadow_v1"
POLICY_STATUS = "not_promoted"
QUALITY_POLICY_VERSION = "phase23_low_resolution_or_low_contrast_v1"
PREPROCESSING_VERSION = "resize_pad_square_eval_v1"
MODEL_FAMILY = "dinov3"
MODEL_ARCHITECTURE = "vit_base_patch16_dinov3.lvd1689m"
MODEL_HEAD = "cls_patch_attention"
INPUT_SIZE = 512
MODEL_VERSION = "dinov3/vit_base_patch16_dinov3.lvd1689m/cls_patch_attention/512"
BATCH_ABSOLUTE_TOLERANCE = 1e-5
DECISIONS = ("AUTHENTIC", "REVIEW", "FAKE")
SYSTEM_STATUS_OK = "ok"
SYSTEM_STATUS_ERROR = "POLICY_ERROR"
PRECEDENCE = (
    "quality review (LOW_RESOLUTION or LOW_CONTRAST) -> REVIEW",
    "else calibrated P(fake) >= 0.50 -> FAKE",
    "else AUTHENTIC",
    "OOD is not a decision input while ood_status is unavailable",
    "invalid critical input -> POLICY_ERROR, which is not an authenticity verdict",
)
AUTHENTICITY_REASON_CODES = (
    "AUTHENTICITY_THRESHOLD",
    "AUTHENTICITY_BELOW_THRESHOLD",
)
QUALITY_REASON_CODES = (
    "QUALITY_LOW_RESOLUTION",
    "QUALITY_LOW_CONTRAST",
    "QUALITY_MULTIPLE_FLAGS",
)
SYSTEM_REASON_CODES = (
    "POLICY_INVALID_INPUT",
    "POLICY_MODEL_MISMATCH",
    "POLICY_CONFIGURATION_ERROR",
    "POLICY_INFRASTRUCTURE_ERROR",
)
DIAGNOSTIC_FLAGS = (
    "LOW_RESOLUTION",
    "BLURRY",
    "TOO_DARK",
    "TOO_BRIGHT",
    "LOW_CONTRAST",
    "EXTREME_ASPECT_RATIO",
    "EXCESSIVE_CLIPPING",
    "SUSPICIOUS_FRAMING",
)
DECISION_QUALITY_FLAGS = ("LOW_RESOLUTION", "LOW_CONTRAST")
PHASE23_CUTOFFS = {
    "min_dimension_p5": 512.0,
    "sharpness_p5": 0.0017845827011386734,
    "mean_luminance_p5": 0.2075038491796002,
    "mean_luminance_p95": 0.8228318202094357,
    "luminance_std_p5": 0.1933616647502347,
    "aspect_ratio_p5": 0.75,
    "aspect_ratio_p95": 1.5009380863039399,
    "near_black_fraction_p95": 0.5198648164223238,
    "near_white_fraction_p95": 0.6175287543402778,
    "foreground_occupancy_p5": 0.2854902433757662,
}
REJECTED_PREPROCESSING = (
    "conservative_train_v1",
    "current_legacy_train_v1",
    "resize_center_crop_eval_v1",
)


class PolicyConfigurationError(Exception):
    """Initialization failure. The engine does not guess a verdict."""

    def __init__(self, message: str, reason_codes: tuple[str, ...] = ("POLICY_CONFIGURATION_ERROR",)) -> None:
        super().__init__(message)
        self.reason_codes = tuple(reason_codes)
        if not self.reason_codes or any(code not in SYSTEM_REASON_CODES for code in self.reason_codes):
            raise ValueError("configuration failures must use system reason codes")


@dataclass(frozen=True)
class PolicyConfiguration:
    """Versioned shadow configuration. Production promotion fields stay null."""

    policy_version: str = POLICY_VERSION
    policy_status: str = POLICY_STATUS
    model_family: str = MODEL_FAMILY
    architecture: str = MODEL_ARCHITECTURE
    head: str = MODEL_HEAD
    input_size: int = INPUT_SIZE
    preprocessing_version: str = PREPROCESSING_VERSION
    checkpoint_sha256: str = EXPECTED_CHECKPOINT_SHA256
    temperature: float = FROZEN_TEMPERATURE
    authenticity_threshold: float = AUTHENTICITY_THRESHOLD
    threshold_status: str = THRESHOLD_STATUS
    quality_policy_version: str = QUALITY_POLICY_VERSION
    quality_rule: str = QUALITY_RULE
    quality_rule_status: str = QUALITY_RULE_STATUS
    low_resolution_cutoff: float = LOW_RESOLUTION_CUTOFF
    low_contrast_cutoff: float = LOW_CONTRAST_CUTOFF
    ood_status: str = OOD_UNAVAILABLE
    production_authenticity_threshold: float | None = None
    production_quality_gate: str | None = None
    production_ood_threshold: float | None = None
    production_policy_status: str = PRODUCTION_POLICY_STATUS

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PolicyResult:
    """Serializable decision. ``decision`` is null when the system status is POLICY_ERROR."""

    decision: str | None
    authenticity_probability: float | None
    authenticity_threshold: float
    quality_flags: tuple[str, ...]
    quality_review: bool | None
    ood_status: str
    ood_score: float | None
    ood_flag: bool | None
    reason_codes: tuple[str, ...]
    policy_version: str
    model_version: str
    checkpoint_sha256: str
    temperature: float
    quality_policy_version: str
    system_status: str
    error_reason: str | None = None
    policy_status: str = POLICY_STATUS
    production_policy_status: str = PRODUCTION_POLICY_STATUS
    in_distribution_assumed: bool = False
    authenticity_logit: float | None = None

    def __post_init__(self) -> None:
        if self.decision not in DECISIONS and self.decision is not None:
            raise PolicyConfigurationError(f"decision {self.decision!r} is not an authenticity verdict")
        if self.system_status == SYSTEM_STATUS_OK:
            if self.decision not in DECISIONS:
                raise PolicyConfigurationError("a successful result needs AUTHENTIC, REVIEW, or FAKE")
            if self.error_reason is not None:
                raise PolicyConfigurationError("a successful result cannot carry an error reason")
        elif self.system_status == SYSTEM_STATUS_ERROR:
            if self.decision is not None:
                raise PolicyConfigurationError("POLICY_ERROR must not be exposed as an authenticity verdict")
        else:
            raise PolicyConfigurationError(f"unknown system status {self.system_status!r}")
        if not self.reason_codes:
            raise PolicyConfigurationError("every policy result needs at least one reason code")
        if self.ood_status == OOD_UNAVAILABLE and (self.ood_score is not None or self.ood_flag is not None):
            raise PolicyConfigurationError("OOD unavailable cannot be stored as a score or a false flag")
        if self.in_distribution_assumed:
            raise PolicyConfigurationError("shadow_v1 does not assume an image is in-distribution")
        if self.policy_status != POLICY_STATUS or self.production_policy_status != PRODUCTION_POLICY_STATUS:
            raise PolicyConfigurationError("shadow_v1 cannot be marked promoted")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["quality_flags"] = list(self.quality_flags)
        payload["reason_codes"] = list(self.reason_codes)
        return payload


@dataclass
class PolicyCounters:
    """In-process counters. These are not production traffic metrics."""

    authentic: int = 0
    review: int = 0
    fake: int = 0
    policy_error: int = 0
    quality_review: int = 0
    low_resolution: int = 0
    low_contrast: int = 0
    ood_unavailable: int = 0

    def observe(self, result: PolicyResult) -> None:
        if result.system_status == SYSTEM_STATUS_ERROR:
            self.policy_error += 1
        elif result.decision == "AUTHENTIC":
            self.authentic += 1
        elif result.decision == "REVIEW":
            self.review += 1
        elif result.decision == "FAKE":
            self.fake += 1
        if result.quality_review:
            self.quality_review += 1
        if "LOW_RESOLUTION" in result.quality_flags:
            self.low_resolution += 1
        if "LOW_CONTRAST" in result.quality_flags:
            self.low_contrast += 1
        if result.ood_status == OOD_UNAVAILABLE:
            self.ood_unavailable += 1

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


def frozen_shadow_configuration(**overrides: Any) -> PolicyConfiguration:
    """Build the shadow configuration and fail unless it matches the frozen contract."""
    config = PolicyConfiguration(**overrides)
    validate_policy_configuration(config)
    return config


def validate_policy_configuration(config: PolicyConfiguration) -> None:
    """Reject a config that would change the frozen shadow contract."""
    if config.policy_version != POLICY_VERSION:
        raise PolicyConfigurationError(
            f"policy version {config.policy_version!r} is not the shadow contract",
            ("POLICY_CONFIGURATION_ERROR",),
        )
    if config.policy_status != POLICY_STATUS or config.production_policy_status != PRODUCTION_POLICY_STATUS:
        raise PolicyConfigurationError("production policy status must stay not_promoted")
    if config.production_authenticity_threshold is not None or config.production_quality_gate is not None:
        raise PolicyConfigurationError("production authenticity and quality gates stay null")
    if config.production_ood_threshold is not None:
        raise PolicyConfigurationError("production OOD threshold stays null")
    _model_identity(config)
    _temperature(config.temperature)
    _threshold(config.authenticity_threshold)
    if config.threshold_status != THRESHOLD_STATUS:
        raise PolicyConfigurationError("threshold status must stay research_policy_candidate")
    if config.quality_policy_version != QUALITY_POLICY_VERSION or config.quality_rule != QUALITY_RULE:
        raise PolicyConfigurationError("quality policy version is not the Phase 23 research candidate")
    if config.quality_rule_status != QUALITY_RULE_STATUS:
        raise PolicyConfigurationError("quality rule status must stay a research candidate")
    _finite_cutoff("low_resolution_cutoff", config.low_resolution_cutoff, LOW_RESOLUTION_CUTOFF)
    _finite_cutoff("low_contrast_cutoff", config.low_contrast_cutoff, LOW_CONTRAST_CUTOFF)
    if float(PHASE23_CUTOFFS["min_dimension_p5"]) != LOW_RESOLUTION_CUTOFF:
        raise PolicyConfigurationError("Phase 23 low-resolution cutoff drifted")
    if float(PHASE23_CUTOFFS["luminance_std_p5"]) != LOW_CONTRAST_CUTOFF:
        raise PolicyConfigurationError("Phase 23 low-contrast cutoff drifted")
    if any(not math.isfinite(float(value)) for value in PHASE23_CUTOFFS.values()):
        raise PolicyConfigurationError("quality thresholds must be finite")
    if config.ood_status != OOD_UNAVAILABLE:
        raise PolicyConfigurationError(
            "OOD status must be unavailable; that is not an in-distribution claim and not a false flag",
            ("POLICY_CONFIGURATION_ERROR",),
        )
    if config.preprocessing_version != PREPROCESSING_VERSION:
        raise PolicyConfigurationError(
            f"preprocessing {config.preprocessing_version!r} is not {PREPROCESSING_VERSION}",
            ("POLICY_CONFIGURATION_ERROR",),
        )


def build_policy_event(result: PolicyResult, *, request_id: str | None = None, timestamp: str | None = None) -> dict:
    """Structured event. Image pixels and raw bytes are not fields."""
    return {
        "timestamp": timestamp if timestamp is not None else datetime.now(timezone.utc).isoformat(),
        "request_id": request_id,
        "model_version": result.model_version,
        "policy_version": result.policy_version,
        "decision": result.decision,
        "system_status": result.system_status,
        "reason_codes": list(result.reason_codes),
        "authenticity_probability": result.authenticity_probability,
        "quality_flags": list(result.quality_flags),
        "ood_status": result.ood_status,
    }


def policy_result_schema() -> dict:
    """JSON schema for the public result. POLICY_ERROR is a system status, not a decision."""
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "hypevault/policy/shadow_v1",
        "title": "ShadowPolicyResult",
        "type": "object",
        "additionalProperties": False,
        "required": [
            "decision",
            "authenticity_probability",
            "authenticity_threshold",
            "quality_flags",
            "quality_review",
            "ood_status",
            "ood_score",
            "ood_flag",
            "reason_codes",
            "policy_version",
            "model_version",
            "checkpoint_sha256",
            "temperature",
            "quality_policy_version",
            "system_status",
            "policy_status",
            "production_policy_status",
            "in_distribution_assumed",
        ],
        "properties": {
            "decision": {"enum": ["AUTHENTIC", "REVIEW", "FAKE", None]},
            "authenticity_probability": {"type": ["number", "null"]},
            "authenticity_threshold": {"const": AUTHENTICITY_THRESHOLD},
            "quality_flags": {"type": "array", "items": {"enum": list(DIAGNOSTIC_FLAGS)}},
            "quality_review": {"type": ["boolean", "null"]},
            "ood_status": {"const": OOD_UNAVAILABLE},
            "ood_score": {"type": "null"},
            "ood_flag": {"type": "null"},
            "reason_codes": {
                "type": "array",
                "minItems": 1,
                "items": {"enum": list(AUTHENTICITY_REASON_CODES + QUALITY_REASON_CODES + SYSTEM_REASON_CODES)},
            },
            "policy_version": {"const": POLICY_VERSION},
            "model_version": {"const": MODEL_VERSION},
            "checkpoint_sha256": {"const": EXPECTED_CHECKPOINT_SHA256},
            "temperature": {"const": FROZEN_TEMPERATURE},
            "quality_policy_version": {"const": QUALITY_POLICY_VERSION},
            "system_status": {"enum": [SYSTEM_STATUS_OK, SYSTEM_STATUS_ERROR]},
            "error_reason": {"type": ["string", "null"]},
            "policy_status": {"const": POLICY_STATUS},
            "production_policy_status": {"const": PRODUCTION_POLICY_STATUS},
            "in_distribution_assumed": {"const": False},
            "authenticity_logit": {"type": ["number", "null"]},
        },
    }


class PolicyEngine:
    """Deterministic shadow engine. Initialization fails closed. Calls do not train."""

    def __init__(self, configuration: PolicyConfiguration | None = None, *, event_logger: Callable[[dict], None] | None = None) -> None:
        self.configuration = frozen_shadow_configuration() if configuration is None else configuration
        validate_policy_configuration(self.configuration)
        self.event_logger = event_logger
        self.counters = PolicyCounters()

    def evaluate(
        self,
        *,
        logit: float | None = None,
        probability: float | None = None,
        recorded_probability_side: str | None = None,
        quality_flags: Sequence[str] | str | None = None,
        quality_features: dict | None = None,
        preprocessing_version: str | None = None,
        input_size: int | None = None,
        checkpoint_sha256: str | None = None,
        ood_status: str | None = None,
        ood_score: float | None = None,
        ood_flag: bool | None = None,
        split: str | None = None,
        sample_id: str | None = None,
        test_sample_ids: set[str] | None = None,
        request_id: str | None = None,
        temperature: float | None = None,
    ) -> PolicyResult:
        """One sample. A missing or corrupt critical input becomes POLICY_ERROR, not AUTHENTIC."""
        if split == "test" or (sample_id is not None and test_sample_ids is not None and sample_id in test_sample_ids):
            raise RuntimeError("policy contract encountered a final-test sample")
        failure = self._request_failure(
            preprocessing_version=preprocessing_version,
            input_size=input_size,
            checkpoint_sha256=checkpoint_sha256,
            ood_status=ood_status,
            ood_score=ood_score,
            ood_flag=ood_flag,
            temperature=temperature,
        )
        flags: tuple[str, ...] = ()
        if failure is None:
            try:
                flags = _resolve_quality(quality_flags, quality_features, self.configuration)
            except ValueError as exc:
                failure = (("POLICY_INVALID_INPUT",), str(exc))
        resolved_probability = None
        resolved_logit = None
        side = None
        if failure is None:
            try:
                resolved_probability, resolved_logit, side = _resolve_signal(
                    logit=logit,
                    probability=probability,
                    recorded_probability_side=recorded_probability_side,
                    threshold=self.configuration.authenticity_threshold,
                )
            except ValueError as exc:
                failure = (("POLICY_INVALID_INPUT",), str(exc))
        if failure is not None:
            result = self._error(failure[0], failure[1])
        else:
            result = self._decide(flags, resolved_probability, resolved_logit, side)
        return self._emit(result, request_id=request_id)

    def evaluate_images(
        self,
        images: Sequence[Any],
        logit_fn: Callable[[Any], Sequence[float]],
        *,
        batch_size: int = 1,
        request_id: str | None = None,
    ) -> list[PolicyResult]:
        """Quality from the original image, logit from the contracted 512 eval batch."""
        if int(batch_size) < 1:
            raise PolicyConfigurationError("batch size must be positive")
        if not images:
            return []
        batch = preprocess_eval_images(list(images))
        features = [_features_and_flags(image) for image in images]
        logits = _batched_logits(batch, logit_fn, int(batch_size))
        if len(logits) != len(images):
            return [
                self.evaluate(
                    quality_features=feature,
                    quality_flags=flags,
                    preprocessing_version=PREPROCESSING_VERSION,
                    request_id=request_id,
                    logit=float("nan"),
                )
                for feature, flags in features
            ]
        results = []
        for (feature, flags), logit in zip(features, logits, strict=True):
            results.append(
                self.evaluate(
                    logit=logit,
                    quality_features=feature,
                    quality_flags=flags,
                    preprocessing_version=PREPROCESSING_VERSION,
                    request_id=request_id,
                )
            )
        return results

    def evaluate_paths(
        self,
        paths: Sequence[Any],
        *,
        logits: dict[str, float] | None = None,
        batch_size: int = 1,
        test_paths: set[str] | None = None,
    ) -> list[dict]:
        """Read-only dry run. Missing files and corrupt images fail closed. Source files are not written."""
        from pathlib import Path

        selected = [Path(path) for path in paths]
        blocked = test_paths or set()
        for path in selected:
            if str(path.resolve()) in blocked:
                raise RuntimeError("policy contract encountered a final-test sample")
        if int(batch_size) < 1:
            raise PolicyConfigurationError("batch size must be positive")
        ordered: list[tuple[Path, PolicyResult | None]] = []
        ready: list[tuple[Path, Any, float | None]] = []
        supplied = logits or {}
        for path in selected:
            image, error = _open_policy_image(path)
            if error is not None:
                ordered.append((path, self._emit(self._error(("POLICY_INVALID_INPUT",), error))))
                continue
            key = str(path.resolve())
            ready.append((path, image, supplied.get(key)))
            ordered.append((path, None))
        if ready:
            images = [image for _path, image, _logit in ready]
            preprocess_eval_images(images)
            values = [logit for _path, _image, logit in ready]

            def logit_fn(batch: Any) -> list[float]:
                start = logit_fn.offset
                count = int(batch.shape[0])
                chunk = []
                for logit in values[start : start + count]:
                    chunk.append(float("nan") if logit is None else float(logit))
                logit_fn.offset = start + count
                return chunk

            logit_fn.offset = 0  # type: ignore[attr-defined]
            decided = iter(self.evaluate_images(images, logit_fn, batch_size=int(batch_size)))
            ordered = [(path, next(decided) if current is None else current) for path, current in ordered]
        return [_dry_run_row(path, result) for path, result in ordered if result is not None]

    def _decide(
        self,
        flags: tuple[str, ...],
        probability: float | None,
        logit: float | None,
        side: str | None,
    ) -> PolicyResult:
        quality_review = "LOW_RESOLUTION" in flags or "LOW_CONTRAST" in flags
        if quality_review:
            decision = "REVIEW"
            reasons = _review_reasons(flags)
        elif side == "above":
            decision = "FAKE"
            reasons = ("AUTHENTICITY_THRESHOLD",)
        elif side == "below":
            decision = "AUTHENTIC"
            reasons = ("AUTHENTICITY_BELOW_THRESHOLD",)
        else:
            return self._error(("POLICY_INVALID_INPUT",), "missing calibrated authenticity signal; refusing to guess AUTHENTIC")
        return self._ok(decision, reasons, flags, quality_review, probability, logit)

    def _emit(self, result: PolicyResult, request_id: str | None = None) -> PolicyResult:
        self.counters.observe(result)
        if self.event_logger is not None:
            self.event_logger(build_policy_event(result, request_id=request_id))
        return result

    def _request_failure(
        self,
        *,
        preprocessing_version: str | None,
        input_size: int | None,
        checkpoint_sha256: str | None,
        ood_status: str | None,
        ood_score: float | None,
        ood_flag: bool | None,
        temperature: float | None,
    ) -> tuple[tuple[str, ...], str] | None:
        if temperature is not None and float(temperature) != self.configuration.temperature:
            return (("POLICY_CONFIGURATION_ERROR",), "runtime code cannot replace the frozen temperature")
        if preprocessing_version is None:
            return (("POLICY_INVALID_INPUT",), "missing required preprocessing metadata")
        if preprocessing_version != PREPROCESSING_VERSION:
            return (("POLICY_CONFIGURATION_ERROR",), f"preprocessing {preprocessing_version!r} is not {PREPROCESSING_VERSION}")
        if input_size is not None and int(input_size) != INPUT_SIZE:
            return (("POLICY_MODEL_MISMATCH", "POLICY_CONFIGURATION_ERROR"), f"input size {input_size} is not {INPUT_SIZE}")
        if checkpoint_sha256 is not None and checkpoint_sha256 != self.configuration.checkpoint_sha256:
            return (("POLICY_MODEL_MISMATCH", "POLICY_CONFIGURATION_ERROR"), "checkpoint hash does not match the frozen artifact")
        if ood_status not in (None, OOD_UNAVAILABLE):
            return (("POLICY_CONFIGURATION_ERROR",), "OOD unavailable is not an in-distribution claim and not a false flag")
        if ood_score is not None or ood_flag is not None:
            return (("POLICY_INVALID_INPUT",), "OOD unavailable cannot be stored as a score or a false flag")
        return None

    def _ok(
        self,
        decision: str,
        reasons: tuple[str, ...],
        flags: tuple[str, ...],
        quality_review: bool,
        probability: float | None,
        logit: float | None,
    ) -> PolicyResult:
        config = self.configuration
        return PolicyResult(
            decision=decision,
            authenticity_probability=probability,
            authenticity_threshold=config.authenticity_threshold,
            quality_flags=flags,
            quality_review=quality_review,
            ood_status=OOD_UNAVAILABLE,
            ood_score=None,
            ood_flag=None,
            reason_codes=reasons,
            policy_version=config.policy_version,
            model_version=MODEL_VERSION,
            checkpoint_sha256=config.checkpoint_sha256,
            temperature=config.temperature,
            quality_policy_version=config.quality_policy_version,
            system_status=SYSTEM_STATUS_OK,
            error_reason=None,
            authenticity_logit=logit,
        )

    def _error(self, reason_codes: tuple[str, ...], message: str) -> PolicyResult:
        config = self.configuration
        return PolicyResult(
            decision=None,
            authenticity_probability=None,
            authenticity_threshold=config.authenticity_threshold,
            quality_flags=(),
            quality_review=None,
            ood_status=OOD_UNAVAILABLE,
            ood_score=None,
            ood_flag=None,
            reason_codes=reason_codes,
            policy_version=config.policy_version,
            model_version=MODEL_VERSION,
            checkpoint_sha256=config.checkpoint_sha256,
            temperature=config.temperature,
            quality_policy_version=config.quality_policy_version,
            system_status=SYSTEM_STATUS_ERROR,
            error_reason=message,
            authenticity_logit=None,
        )


def preprocess_eval_images(images: Sequence[Any], *, family: str = MODEL_FAMILY, train: bool = False, profile: str = "conservative") -> Any:
    """Contracted eval preprocess: resize_pad_square_eval_v1 at 512. Train and DINOv2 sizes are rejected."""
    if train:
        raise PolicyConfigurationError("training transforms are not the policy preprocess")
    if family != MODEL_FAMILY:
        raise PolicyConfigurationError(
            f"model family {family!r} is not {MODEL_FAMILY}",
            ("POLICY_MODEL_MISMATCH", "POLICY_CONFIGURATION_ERROR"),
        )
    if profile != "conservative":
        raise PolicyConfigurationError(f"preprocessing profile {profile!r} is not resize_pad_square_eval_v1")
    import torch
    from augmentations import CONSERVATIVE, EVAL_TRANSFORM_VERSION, build_transforms

    if EVAL_TRANSFORM_VERSION[CONSERVATIVE] != PREPROCESSING_VERSION:
        raise PolicyConfigurationError("eval transform version drifted from resize_pad_square_eval_v1")
    transform = build_transforms(MODEL_FAMILY, train=False, profile=CONSERVATIVE)
    tensors = [transform(image) for image in images]
    batch = torch.stack(tensors)
    if tuple(batch.shape[1:]) != (3, INPUT_SIZE, INPUT_SIZE):
        raise PolicyConfigurationError(f"preprocessed batch shape {tuple(batch.shape)} is not NCHW 512")
    return batch


def results_equivalent(left: PolicyResult, right: PolicyResult, *, tolerance: float = BATCH_ABSOLUTE_TOLERANCE) -> bool:
    """Decision and quality flags must match. Logit and probability may differ by ``tolerance``."""
    if left.decision != right.decision or left.quality_flags != right.quality_flags or left.system_status != right.system_status:
        return False
    if left.reason_codes != right.reason_codes or left.ood_status != right.ood_status:
        return False
    return _close(left.authenticity_probability, right.authenticity_probability, tolerance) and _close(
        left.authenticity_logit, right.authenticity_logit, tolerance
    )


def _model_identity(config: PolicyConfiguration) -> None:
    codes = ("POLICY_MODEL_MISMATCH", "POLICY_CONFIGURATION_ERROR")
    if config.model_family != MODEL_FAMILY:
        raise PolicyConfigurationError(f"model family {config.model_family!r} is not {MODEL_FAMILY}", codes)
    if config.architecture != MODEL_ARCHITECTURE:
        raise PolicyConfigurationError(f"architecture {config.architecture!r} is not {MODEL_ARCHITECTURE}", codes)
    if config.head != MODEL_HEAD:
        raise PolicyConfigurationError(f"head {config.head!r} is not {MODEL_HEAD}", codes)
    if isinstance(config.input_size, bool) or int(config.input_size) != INPUT_SIZE:
        raise PolicyConfigurationError(f"input size {config.input_size!r} is not {INPUT_SIZE}", codes)
    if config.checkpoint_sha256 != EXPECTED_CHECKPOINT_SHA256:
        raise PolicyConfigurationError("checkpoint hash does not match the frozen artifact", codes)


def _temperature(value: float) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)) or float(value) <= 0.0:
        raise PolicyConfigurationError("temperature must be finite and positive")
    if float(value) != FROZEN_TEMPERATURE:
        raise PolicyConfigurationError("temperature does not match the frozen calibrated temperature")


def _threshold(value: float) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)) or not 0.0 < float(value) < 1.0:
        raise PolicyConfigurationError("authenticity threshold must be finite and strictly between 0 and 1")
    if float(value) != AUTHENTICITY_THRESHOLD:
        raise PolicyConfigurationError("authenticity threshold candidate is 0.50 and is not promoted")


def _finite_cutoff(name: str, value: float, expected: float) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise PolicyConfigurationError(f"{name} must be finite")
    if float(value) != float(expected):
        raise PolicyConfigurationError(f"{name} does not match the Phase 23 cutoff")


def _resolve_quality(flags, features: dict | None, config: PolicyConfiguration) -> tuple[str, ...]:
    if flags is None and features is None:
        raise ValueError("missing quality feature")
    parsed = _normalize_flags(flags) if flags is not None else None
    if features is not None:
        implied = _flags_from_features(features, config)
        if parsed is None:
            parsed = implied
        elif set(implied) != {flag for flag in parsed if flag in DECISION_QUALITY_FLAGS}:
            raise ValueError("quality flags disagree with the quality features")
    if parsed is None:
        raise ValueError("missing quality feature")
    return parsed


def _normalize_flags(flags) -> tuple[str, ...]:
    parsed = parse_flags(flags)
    unknown = [flag for flag in parsed if flag not in DIAGNOSTIC_FLAGS]
    if unknown:
        raise ValueError(f"malformed quality feature {unknown[0]}")
    return tuple(flag for flag in DIAGNOSTIC_FLAGS if flag in parsed)


def _flags_from_features(features: dict, config: PolicyConfiguration) -> tuple[str, ...]:
    if "min_dimension" not in features or "luminance_std" not in features:
        raise ValueError("missing quality feature")
    try:
        min_dimension = float(features["min_dimension"])
        luminance_std = float(features["luminance_std"])
    except (TypeError, ValueError) as exc:
        raise ValueError("malformed quality feature") from exc
    if not math.isfinite(min_dimension) or not math.isfinite(luminance_std) or min_dimension < 0.0:
        raise ValueError("malformed quality feature")
    implied = []
    if min_dimension < float(config.low_resolution_cutoff):
        implied.append("LOW_RESOLUTION")
    if luminance_std < float(config.low_contrast_cutoff):
        implied.append("LOW_CONTRAST")
    return tuple(implied)


def _resolve_signal(
    *,
    logit: float | None,
    probability: float | None,
    recorded_probability_side: str | None,
    threshold: float,
) -> tuple[float | None, float | None, str | None]:
    resolved_logit = None if logit is None else _finite_number(logit, "authenticity logit")
    computed = None if resolved_logit is None else calibrated_probability_from_logit(resolved_logit)
    resolved_probability = None if probability is None else _probability(probability)
    if computed is not None and resolved_probability is not None and abs(computed - resolved_probability) > BATCH_ABSOLUTE_TOLERANCE:
        raise ValueError("authenticity probability does not match sigmoid(logit / T)")
    if resolved_probability is None:
        resolved_probability = computed
    side = None
    if resolved_probability is not None:
        side = "above" if resolved_probability >= float(threshold) else "below"
    if recorded_probability_side is not None:
        if recorded_probability_side not in {"above", "below"}:
            raise ValueError("recorded probability side must be above or below")
        if side is not None and side != recorded_probability_side:
            raise ValueError("recorded probability side contradicts the calibrated probability")
        if side is None:
            side = recorded_probability_side
    return resolved_probability, resolved_logit, side


def _finite_number(value: float, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid {label}") from exc
    if not math.isfinite(number):
        raise ValueError(f"non-finite {label}")
    return number


def _probability(value: float) -> float:
    number = _finite_number(value, "authenticity probability")
    if number < 0.0 or number > 1.0:
        raise ValueError("authenticity probability must be between 0 and 1")
    return number


def _review_reasons(flags: tuple[str, ...]) -> tuple[str, ...]:
    low_resolution = "LOW_RESOLUTION" in flags
    low_contrast = "LOW_CONTRAST" in flags
    reasons = []
    if low_resolution and low_contrast:
        reasons.append("QUALITY_MULTIPLE_FLAGS")
    if low_resolution:
        reasons.append("QUALITY_LOW_RESOLUTION")
    if low_contrast:
        reasons.append("QUALITY_LOW_CONTRAST")
    if not reasons:
        raise ValueError("REVIEW requires a low-resolution or low-contrast flag")
    return tuple(reasons)


def _features_and_flags(image: Any) -> tuple[dict, tuple[str, ...]]:
    from image_quality import apply_quality_flags, extract_quality_features

    features = extract_quality_features(image)
    flags = tuple(apply_quality_flags(features, {"cutoffs": dict(PHASE23_CUTOFFS)}))
    return features, flags


def _batched_logits(batch: Any, logit_fn: Callable[[Any], Sequence[float]], batch_size: int) -> list[float]:
    logits: list[float] = []
    total = int(batch.shape[0])
    for start in range(0, total, batch_size):
        chunk = batch[start : start + batch_size]
        try:
            values = list(logit_fn(chunk))
        except (TypeError, ValueError, RuntimeError):
            return []
        if len(values) != int(chunk.shape[0]):
            return []
        try:
            logits.extend(float(value) for value in values)
        except (TypeError, ValueError):
            return []
    return logits


def _open_policy_image(path: Any) -> tuple[Any, str | None]:
    from pathlib import Path

    from PIL import Image, UnidentifiedImageError

    file_path = Path(path)
    if not file_path.is_file():
        return None, "missing image file"
    try:
        with Image.open(file_path) as handle:
            rgb = handle.convert("RGB")
            rgb.load()
            return rgb.copy(), None
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError, ValueError) as exc:
        return None, f"corrupted image metadata: {exc}"


def _dry_run_row(path: Any, result: PolicyResult) -> dict:
    return {
        "path": str(path),
        "decision": result.decision,
        "system_status": result.system_status,
        "authenticity_probability": result.authenticity_probability,
        "quality_flags": list(result.quality_flags),
        "ood_status": result.ood_status,
        "ood_score": result.ood_score,
        "ood_flag": result.ood_flag,
        "reason_codes": list(result.reason_codes),
        "error_reason": result.error_reason,
    }


def _close(left: float | None, right: float | None, tolerance: float) -> bool:
    if left is None or right is None:
        return left is None and right is None
    return abs(float(left) - float(right)) <= float(tolerance)
