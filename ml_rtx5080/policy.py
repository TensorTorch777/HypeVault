"""Shadow authenticity / review policy. This is not a live backend verdict.

Precedence is quality review, then the frozen calibrated probability at 0.50.
OOD is an explicit availability state. Unavailable OOD is not recorded as
in-distribution, and it is not given a score of zero or a false flag.
"""

from __future__ import annotations

import math

FROZEN_TEMPERATURE = 0.24038200410185356
EXPECTED_CHECKPOINT_SHA256 = "5a38c93fd442b03653c65d2a5ecc9c2687ef152f7c5c020763e4ce1fd9c7d28f"
AUTHENTICITY_THRESHOLD = 0.50
THRESHOLD_STATUS = "research_policy_candidate"
QUALITY_RULE = "low_resolution_or_low_contrast"
QUALITY_RULE_STATUS = "research_quality_gate_candidate"
LOW_RESOLUTION_CUTOFF = 512.0
LOW_CONTRAST_CUTOFF = 0.1933616647502347
STATES = ("AUTHENTIC", "FAKE", "REVIEW")
OOD_UNAVAILABLE = "unavailable"
OOD_AVAILABLE = "available"
OOD_STATUSES = (OOD_AVAILABLE, OOD_UNAVAILABLE)
POSITIVE_LABEL = 1
PRODUCTION_POLICY_STATUS = "not_promoted"


def calibrated_probability_from_logit(logit: float) -> float:
    """P(fake) = sigmoid(logit / T). Temperature is not fit here."""
    scaled = float(logit) / FROZEN_TEMPERATURE
    if scaled >= 0.0:
        return 1.0 / (1.0 + math.exp(-scaled))
    exponent = math.exp(scaled)
    return exponent / (1.0 + exponent)


def parse_flags(value) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, (list, tuple)):
        return tuple(str(flag) for flag in value if str(flag))
    return tuple(part for part in str(value).split("|") if part)


def evaluate_quality_gate(flags) -> bool:
    """Research candidate: LOW_RESOLUTION or LOW_CONTRAST. Definition is fixed."""
    present = set(parse_flags(flags))
    return "LOW_RESOLUTION" in present or "LOW_CONTRAST" in present


def reason_codes(decision: str, flags) -> tuple[str, ...]:
    """Stable reason list. REVIEW names the quality flags that forced it."""
    present = set(parse_flags(flags))
    if decision == "REVIEW":
        codes = []
        low_resolution = "LOW_RESOLUTION" in present
        low_contrast = "LOW_CONTRAST" in present
        if low_resolution and low_contrast:
            codes.append("QUALITY_MULTIPLE_FLAGS")
        if low_resolution:
            codes.append("QUALITY_LOW_RESOLUTION")
        if low_contrast:
            codes.append("QUALITY_LOW_CONTRAST")
        if not codes:
            raise ValueError("REVIEW requires a low-resolution or low-contrast flag")
        return tuple(codes)
    if decision == "FAKE":
        return ("AUTHENTICITY_THRESHOLD",)
    if decision == "AUTHENTIC":
        return ("AUTHENTICITY_BELOW_THRESHOLD", "QUALITY_OK")
    raise ValueError(f"unknown shadow decision {decision!r}")


def evaluate_shadow_policy(
    *,
    probability: float | None,
    quality_flags,
    probability_side: str | None = None,
    ood_status: str = OOD_UNAVAILABLE,
    ood_score: float | None = None,
    ood_flag: bool | None = None,
    threshold: float = AUTHENTICITY_THRESHOLD,
) -> dict:
    """Deterministic shadow label. OOD unavailable stays null, not false."""
    status = _ood_status(ood_status, ood_score, ood_flag)
    flags = parse_flags(quality_flags)
    quality_review = evaluate_quality_gate(flags)
    cutoff = float(threshold)
    if quality_review:
        decision = "REVIEW"
        side = _side(probability, probability_side, required=False)
    else:
        side = _side(probability, probability_side, required=True)
        decision = "FAKE" if side == "above" else "AUTHENTIC"
    if decision not in STATES:
        raise RuntimeError(f"shadow policy produced {decision!r}")
    numeric = None if probability is None else float(probability)
    return {
        "authenticity_probability": numeric,
        "authenticity_threshold": cutoff,
        "threshold_status": THRESHOLD_STATUS,
        "quality_flags": list(flags),
        "quality_review": quality_review,
        "quality_rule": QUALITY_RULE,
        "quality_rule_status": QUALITY_RULE_STATUS,
        "ood_status": status,
        "ood_score": None if status == OOD_UNAVAILABLE else ood_score,
        "ood_flag": None if status == OOD_UNAVAILABLE else ood_flag,
        "ood_checked": status == OOD_AVAILABLE,
        "in_distribution_assumed": False,
        "decision": decision,
        "reason_codes": list(reason_codes(decision, flags)),
        "probability_side": side,
        "production_policy_status": PRODUCTION_POLICY_STATUS,
    }


def build_policy_trace(
    sample: dict,
    *,
    probability: float | None,
    probability_side: str | None = None,
    ood_status: str = OOD_UNAVAILABLE,
) -> dict:
    """One serializable row. Image tensors are not accepted."""
    if "pixel_values" in sample or "image_tensor" in sample:
        raise ValueError("policy traces must not store image tensors")
    decision = evaluate_shadow_policy(
        probability=probability,
        quality_flags=sample.get("quality_flags"),
        probability_side=probability_side,
        ood_status=ood_status,
        ood_score=None,
        ood_flag=None,
    )
    return {
        "sample_id": sample["sample_id"],
        "split": sample.get("source_split", sample.get("split")),
        "brand": sample.get("brand"),
        "label": int(sample["label"]),
        "calibrated_probability": decision["authenticity_probability"],
        "probability_source": sample.get("probability_source"),
        "authenticity_threshold": decision["authenticity_threshold"],
        "quality_flags": decision["quality_flags"],
        "quality_review": decision["quality_review"],
        "ood_status": decision["ood_status"],
        "ood_score": decision["ood_score"],
        "ood_flag": decision["ood_flag"],
        "decision": decision["decision"],
        "reason_codes": decision["reason_codes"],
        "production_policy_status": PRODUCTION_POLICY_STATUS,
    }


def authenticity_only_decision(probability: float | None, probability_side: str | None) -> str:
    """Threshold-only shadow label. Quality does not move the row to REVIEW."""
    side = _side(probability, probability_side, required=True)
    return "FAKE" if side == "above" else "AUTHENTIC"


def summarize_policy(rows: list[dict]) -> dict:
    """2×3 policy matrix. REVIEW is neither a binary success nor a binary error."""
    matrix = {
        "authentic": {state: 0 for state in STATES},
        "fake": {state: 0 for state in STATES},
    }
    for row in rows:
        key = "fake" if int(row["label"]) == POSITIVE_LABEL else "authentic"
        decision = row["decision"]
        if decision not in STATES:
            raise ValueError(f"unknown shadow decision {decision!r}")
        matrix[key][decision] += 1
    authentic_count = sum(matrix["authentic"].values())
    fake_count = sum(matrix["fake"].values())
    total = authentic_count + fake_count
    review = matrix["authentic"]["REVIEW"] + matrix["fake"]["REVIEW"]
    false_authentic = matrix["fake"]["AUTHENTIC"]
    false_fake = matrix["authentic"]["FAKE"]
    hard = [
        row
        for row in rows
        if row["decision"] in {"AUTHENTIC", "FAKE"}
    ]
    hard_correct = sum(
        (row["decision"] == "FAKE") == (int(row["label"]) == POSITIVE_LABEL) for row in hard
    )
    return {
        "sample_count": total,
        "actual_authentic_count": authentic_count,
        "actual_fake_count": fake_count,
        "authentic_count": matrix["authentic"]["AUTHENTIC"] + matrix["fake"]["AUTHENTIC"],
        "fake_count": matrix["authentic"]["FAKE"] + matrix["fake"]["FAKE"],
        "review_count": review,
        "authentic_rate": _rate(matrix["authentic"]["AUTHENTIC"] + matrix["fake"]["AUTHENTIC"], total),
        "fake_rate": _rate(matrix["authentic"]["FAKE"] + matrix["fake"]["FAKE"], total),
        "review_rate": _rate(review, total),
        "authentic_review_rate": _rate(matrix["authentic"]["REVIEW"], authentic_count),
        "fake_review_rate": _rate(matrix["fake"]["REVIEW"], fake_count),
        "false_authentic_count": false_authentic,
        "false_authentic_rate": _rate(false_authentic, fake_count),
        "false_fake_count": false_fake,
        "false_fake_rate": _rate(false_fake, authentic_count),
        "authentic_to_review_count": matrix["authentic"]["REVIEW"],
        "confusion": {
            "actual_authentic": dict(matrix["authentic"]),
            "actual_fake": dict(matrix["fake"]),
            "review_excluded_from_binary_metrics": True,
        },
        "binary_decision_count": len(hard),
        "binary_accuracy_review_excluded": _rate(hard_correct, len(hard)),
    }


def screen_splits(rows: list[dict], test_sample_ids: set[str]) -> None:
    """Final-test membership cannot enter the shadow simulation."""
    for row in rows:
        split = row.get("source_split", row.get("split"))
        if split == "test":
            raise RuntimeError("shadow policy encountered a final-test sample")
        if row.get("sample_id") in test_sample_ids:
            raise RuntimeError("shadow policy encountered a final-test sample")


def _ood_status(status: str, score, flag) -> str:
    if status not in OOD_STATUSES:
        raise ValueError(f"OOD status must be one of {OOD_STATUSES}, got {status!r}")
    if status == OOD_UNAVAILABLE:
        if score is not None or flag is not None:
            raise ValueError("OOD unavailable cannot be stored as a score or a false flag")
        return OOD_UNAVAILABLE
    return OOD_AVAILABLE


def _side(probability: float | None, recorded: str | None, *, required: bool) -> str | None:
    if probability is not None:
        if not math.isfinite(float(probability)):
            raise ValueError("calibrated probability must be finite")
        return "above" if float(probability) >= AUTHENTICITY_THRESHOLD else "below"
    if recorded is None:
        if required:
            raise ValueError("shadow policy requires a calibrated probability or a recorded side of 0.50")
        return None
    if recorded not in {"above", "below"}:
        raise ValueError(f"recorded probability side must be above or below, got {recorded!r}")
    return recorded


def _rate(numerator: int, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return numerator / denominator
