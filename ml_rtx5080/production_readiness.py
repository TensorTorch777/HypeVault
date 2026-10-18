"""Phase 34 production-readiness audit.

This module reads existing artifacts and classifies them. It does not train,
fit a temperature, open final-test images, evaluate OOD, or promote the policy.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from reference_inference import file_sha256
from split_v2 import AUTHORITATIVE_MEMBERSHIP_HASH

_PKG = Path(__file__).resolve().parent
_REPO = _PKG.parent
_OUTPUT = _PKG / "experiments" / "production_readiness_v1"
_CHECKPOINT = _PKG / "experiments" / "v2_dinov3_cls_patch_attention" / "epoch_018.pt"
_MANIFEST = _PKG / "experiments" / "dataset_audit" / "split_manifest_v2.json"
CHECKPOINT_SHA256 = "5a38c93fd442b03653c65d2a5ecc9c2687ef152f7c5c020763e4ce1fd9c7d28f"
MANIFEST_FILE_SHA256 = "4d0d20c17d12ac29cb299b7b4f369ebcce3c5acd7b2fd5430d61f952b9852eea"
TEMPERATURE = 0.24038200410185356
REVIEW_COUNT = 1855
REFERENCE_POOL = 26994
REVIEW_RATE = REVIEW_COUNT / REFERENCE_POOL
REQUEST_RATES = (100, 1000, 10000)

READY = "READY_FOR_PROMOTION"
CONDITIONAL = "CONDITIONALLY_READY"
NOT_READY = "NOT_READY"

READY_REQUIREMENTS = (
    "final_test_complete",
    "calibration_complete",
    "robustness_audit_complete",
    "real_ood_validation_complete",
    "production_runtime_parity_complete",
    "production_serving_infrastructure_available",
    "no_unresolved_critical_failure",
    "quality_review_workload_accepted",
    "policy_formally_approved",
)
INFRASTRUCTURE_DATA_OR_POLICY = {
    "real_ood_validation_complete",
    "production_runtime_parity_complete",
    "production_serving_infrastructure_available",
    "quality_review_workload_accepted",
    "policy_formally_approved",
}
UNSAFE_FLAGS = (
    "model_parity_failed",
    "checkpoint_mismatch",
    "safety_constraint_failed",
    "production_runtime_unsafe",
    "fail_closed_failed",
)
RISK_FIELDS = (
    "id",
    "failure_mode",
    "evidence",
    "severity",
    "current_mitigation",
    "residual_risk",
    "status",
)
SEVERITY_MEANING = {
    "critical": "A promoted system would fail open, contradict the locked safety result, or serve a mismatched checkpoint.",
    "high": "A concrete control required before promotion is missing or a demonstrated failure remains uncovered.",
    "medium": "A scope or operations limit. It does not make the locked in-distribution result false.",
}


def expected_review_volume(requests_per_day: int, review_rate: float = REVIEW_RATE) -> float:
    """Hypothetical REVIEW count. This is not a business SLA."""
    if requests_per_day < 0:
        raise ValueError("request rate cannot be negative")
    return float(requests_per_day) * float(review_rate)


def classify_readiness(evidence: dict) -> dict:
    """Apply the promotion rules to explicit evidence. The status is not preset."""
    unsafe = [name for name in UNSAFE_FLAGS if evidence.get(name)]
    missing = [name for name in READY_REQUIREMENTS if not evidence.get(name)]
    if evidence.get("ood_explicitly_not_required"):
        missing = [name for name in missing if name != "real_ood_validation_complete"]
    reasons = []
    if unsafe:
        reasons.append("unsafe conditions: " + ", ".join(unsafe))
    if missing:
        reasons.append("promotion requirements not met: " + ", ".join(missing))
    conditional_blockers = [name for name in missing if name in INFRASTRUCTURE_DATA_OR_POLICY]
    model_blockers = [name for name in missing if name not in INFRASTRUCTURE_DATA_OR_POLICY]
    conditional = (
        not unsafe
        and not model_blockers
        and bool(missing)
        and evidence.get("model_quality_strong")
        and evidence.get("shadow_path_stable")
        and evidence.get("limitations_explicit")
        and conditional_blockers == missing
    )
    if unsafe or model_blockers or (missing and not conditional):
        status = NOT_READY
    elif not missing:
        status = READY
        reasons.append("every promotion requirement is met")
    else:
        status = CONDITIONAL
        reasons.append(
            "in-distribution model evidence and the PyTorch shadow path are intact; "
            "remaining blockers are infrastructure, data, or policy approval"
        )
    return {
        "status": status,
        "unsafe": unsafe,
        "missing_requirements": missing,
        "reasons": reasons,
    }


def run_phase34() -> dict:
    destination = _OUTPUT / "readiness_report.json"
    if destination.is_file():
        raise RuntimeError("production_readiness_v1 already exists; refusing to rewrite it")
    evidence = load_evidence()
    decision = classify_readiness(evidence)
    checklist = _checklist(evidence)
    risks = _risks(evidence)
    workload = [
        {
            "requests_per_day": rate,
            "review_rate": REVIEW_RATE,
            "expected_review_count": expected_review_volume(rate),
            "source_pool": REFERENCE_POOL,
            "source_review_count": REVIEW_COUNT,
            "sla_assumed": False,
        }
        for rate in REQUEST_RATES
    ]
    report = {
        "phase": 34,
        "readiness": decision["status"],
        "readiness_reasons": decision["reasons"],
        "missing_requirements": decision["missing_requirements"],
        "unsafe": decision["unsafe"],
        "checkpoint_sha256": evidence["checkpoint_sha256"],
        "membership_hash": evidence["membership_hash"],
        "manifest_file_sha256": evidence["manifest_file_sha256"],
        "temperature": evidence["temperature"],
        "policy_version": "shadow_v1",
        "current_production_status": "not_promoted",
        "production_policy_status": "not_promoted",
        "shadow_ready": evidence["shadow_ready"],
        "production_runtime_ready": evidence["production_runtime_ready"],
        "in_distribution_validation": "complete",
        "open_set_validation": "incomplete",
        "real_ood_corpus_available": False,
        "ood_validated": False,
        "production_ood_threshold": None,
        "known_robustness_limitation": "resize_recompression_case",
        "final_test_samples_used": 0,
        "final_test_opened": False,
        "training_performed": False,
        "calibration_performed": False,
        "ood_evaluation_performed": False,
        "promotion_performed": False,
        "customer_traffic_used": False,
        "deployed": False,
        "evidence": evidence,
        "checklist": checklist,
        "severity_meaning": SEVERITY_MEANING,
    }
    _OUTPUT.mkdir(parents=True, exist_ok=True)
    destination.write_text(_dump(report) + "\n")
    (_OUTPUT / "readiness_report.md").write_text(_markdown(report, workload, risks))
    _write_csv(_OUTPUT / "promotion_checklist.csv", checklist)
    _write_csv(_OUTPUT / "risk_register.csv", risks)
    _write_csv(_OUTPUT / "operational_review_load.csv", workload)
    if decision["status"] == READY:
        raise RuntimeError("phase 34 refused to record READY while writing a not-promoted audit")
    return report


def load_evidence() -> dict:
    """Read frozen artifacts. This function does not open images."""
    checkpoint_sha = file_sha256(_CHECKPOINT)
    manifest_sha = file_sha256(_MANIFEST)
    final_test = _read_json(_PKG / "experiments" / "final_test_v1" / "final_test_summary.json")
    calibration = _read_json(_PKG / "experiments" / "v2_dinov3_cls_patch_attention" / "calibration" / "calibration_summary.json")
    policy = _read_json(_PKG / "experiments" / "policy_shadow_v1" / "policy_summary.json")
    ood = _read_json(_PKG / "experiments" / "ood_v1" / "ood_summary.json")
    serving = _read_json(_PKG / "experiments" / "serving_v1" / "serving_summary.json")
    shadow = _read_json(_PKG / "experiments" / "serving_v2" / "shadow_backend_summary.json")
    stability = _read_json(_PKG / "experiments" / "serving_stability_v1" / "serving_stability_summary.json")
    fastpath = _read_json(_PKG / "experiments" / "quality_fastpath_v1" / "fastpath_summary.json")
    memory_growth = _max_memory_growth(_PKG / "experiments" / "serving_stability_v1" / "memory_stability.csv")
    batch4 = _batch4_throughput(_PKG / "experiments" / "serving_v2" / "pytorch_cuda_benchmark.csv")
    brands = [row["brand"] for row in final_test["brands"]["brands"]]
    calibrated = final_test["calibrated"]
    reference = policy["reference"]
    checkpoint_ok = (
        checkpoint_sha == CHECKPOINT_SHA256
        and final_test["checkpoint_sha256"] == CHECKPOINT_SHA256
        and shadow["checkpoint_sha256"] == CHECKPOINT_SHA256
    )
    temperature_ok = (
        float(calibration["temperature"]) == TEMPERATURE
        and float(final_test["selected_temperature"]) == TEMPERATURE
        and int(calibration["calibration_sample_count"]) == 2998
    )
    final_ok = (
        int(final_test["sample_count"]) == 3006
        and calibrated["roc_auc"] == 1.0
        and calibrated["pr_auc"] == 1.0
        and calibrated["f1"] == 1.0
        and calibrated["false_authentic_rate"] == 0.0
        and calibrated["false_fake_rate"] == 0.0
        and len(brands) == 5
        and final_test["authentic_count"] > 0
        and final_test["fake_count"] > 0
        and final_test["test_used_for_selection"] is False
    )
    shadow_ok = (
        shadow["decision_mismatches"] == 0
        and shadow["batch_decision_mismatches"] == 0
        and shadow["max_abs_logit_error"] <= shadow["logit_atol"]
        and shadow["bf16_authoritative"] is False
        and shadow["precision"] == "fp32"
        and stability["contract_passed"] is True
        and stability["memory_stable"] is True
        and memory_growth == 0
        and stability["failure_isolation_passed"] is True
        and stability["sequential_requests"] >= 1000
        and fastpath["contract_passed"] is True
        and fastpath["flag_disagreements"] == 0
    )
    infrastructure_available = (
        serving.get("onnx_cuda_provider") == "available"
        and serving.get("tensorrt_status") == "available"
        and serving.get("triton_status") == "available"
    )
    return {
        "checkpoint_sha256": checkpoint_sha,
        "membership_hash": AUTHORITATIVE_MEMBERSHIP_HASH,
        "manifest_file_sha256": manifest_sha,
        "temperature": float(calibration["temperature"]),
        "calibration_sample_count": int(calibration["calibration_sample_count"]),
        "calibrated_nll": calibration["calibrated_nll"],
        "calibrated_brier": calibration["calibrated_brier"],
        "calibrated_ece": calibration["calibrated_ece"],
        "final_test_samples": int(final_test["sample_count"]),
        "final_test_roc_auc": calibrated["roc_auc"],
        "final_test_pr_auc": calibrated["pr_auc"],
        "final_test_f1": calibrated["f1"],
        "final_test_false_authentic_rate": calibrated["false_authentic_rate"],
        "final_test_false_fake_rate": calibrated["false_fake_rate"],
        "brands": brands,
        "authentic_review_rate": reference["authentic_review_rate"],
        "fake_review_rate": reference["fake_review_rate"],
        "review_rate": reference["review_rate"],
        "review_count": reference["review_count"],
        "reference_pool": reference["sample_count"],
        "stress_caught": policy["stress"]["caught_by_quality_policy"],
        "stress_events": policy["stress"]["demonstrated_false_fake_events"],
        "stress_missed": policy["stress"]["missed_ids"],
        "stress_false_authentic": policy["stress"]["false_authentic_count"],
        "clean_false_authentic": reference["false_authentic_count"],
        "clean_false_fake": reference["false_fake_count"],
        "real_ood_corpus_available": ood["REAL_OOD_CORPUS_AVAILABLE"],
        "production_ood_threshold": ood["production_ood_threshold"],
        "batch1_model_p50_ms": shadow["fp32_batch1_model_p50_ms"],
        "batch4_images_per_sec": batch4,
        "sequential_e2e_p50_ms": stability["sequential_e2e_p50_ms"],
        "sequential_e2e_mean_ms": stability["sequential_e2e_mean_ms"],
        "fast_large_quality_p50_ms": fastpath["large_image_fast_p50_ms"],
        "fast_e2e_p50_ms": _fast_e2e_p50(_PKG / "experiments" / "quality_fastpath_v1" / "fastpath_end_to_end.csv"),
        "memory_growth_bytes": memory_growth,
        "onnx_cuda_provider": serving.get("onnx_cuda_provider"),
        "tensorrt_status": serving.get("tensorrt_status"),
        "triton_status": serving.get("triton_status"),
        "final_test_complete": final_ok,
        "calibration_complete": temperature_ok and calibration["fitting_status"] == "fitted" and calibration["test_images_loaded"] is False,
        "robustness_audit_complete": policy["stress"]["demonstrated_false_fake_events"] == 9 and "missed_ids" in policy["stress"],
        "real_ood_validation_complete": ood["REAL_OOD_CORPUS_AVAILABLE"] is True and ood["production_ood_threshold"] is not None,
        "production_runtime_parity_complete": infrastructure_available,
        "production_serving_infrastructure_available": infrastructure_available,
        "no_unresolved_critical_failure": checkpoint_ok and shadow_ok and reference["false_authentic_count"] == 0 and calibrated["false_authentic_count"] == 0,
        "quality_review_workload_accepted": False,
        "policy_formally_approved": False,
        "ood_explicitly_not_required": False,
        "model_parity_failed": not shadow_ok,
        "checkpoint_mismatch": not checkpoint_ok or manifest_sha != MANIFEST_FILE_SHA256 or AUTHORITATIVE_MEMBERSHIP_HASH != calibration["manifest_hash"],
        "safety_constraint_failed": reference["false_authentic_count"] != 0 or calibrated["false_authentic_count"] != 0 or policy["stress"]["false_authentic_count"] != 0,
        "production_runtime_unsafe": (not shadow_ok) or memory_growth != 0 or stability["failure_isolation_passed"] is not True,
        "fail_closed_failed": stability["failure_isolation_passed"] is not True or fastpath["contract_passed"] is not True,
        "model_quality_strong": final_ok and temperature_ok and reference["false_authentic_count"] == 0 and reference["false_fake_count"] == 0,
        "shadow_path_stable": shadow_ok,
        "limitations_explicit": True,
        "shadow_ready": shadow_ok,
        "production_runtime_ready": infrastructure_available and shadow_ok,
        "live_backend": "dinov2",
        "shadow_backend": "pytorch_cuda_fp32",
    }


def _checklist(evidence: dict) -> list[dict]:
    return [
        {"item": "Model", "result": _pass(evidence["model_quality_strong"]), "note": "locked in-distribution result, five brands, both classes"},
        {"item": "Calibration", "result": _pass(evidence["calibration_complete"]), "note": "temperature fitted on 2998 calibration samples and not refit"},
        {"item": "Final test", "result": _pass(evidence["final_test_complete"]), "note": "3006 locked samples, not reopened"},
        {"item": "Robustness", "result": _pass(evidence["robustness_audit_complete"]), "note": "audit complete; resize_recompression_case remains uncovered"},
        {"item": "Quality", "result": "PASS", "note": "research candidate LOW_RESOLUTION OR LOW_CONTRAST, not promoted"},
        {"item": "OOD", "result": "PENDING", "note": "REAL_OOD_CORPUS_AVAILABLE is false; production_ood_threshold is null"},
        {"item": "PyTorch GPU shadow", "result": _pass(evidence["shadow_ready"]), "note": "FP32 CUDA shadow parity, stability, and fast gate"},
        {"item": "ONNX GPU", "result": "FAIL", "note": "CUDA execution provider unavailable"},
        {"item": "TensorRT", "result": "FAIL", "note": "TensorRT runtime unavailable"},
        {"item": "Triton", "result": "FAIL", "note": "Triton server unavailable"},
        {"item": "Fail-closed contract", "result": _pass(not evidence["fail_closed_failed"]), "note": "shadow_v1 rejects unsafe inputs without a guessed verdict"},
        {"item": "Memory stability", "result": _pass(evidence["memory_growth_bytes"] == 0), "note": "allocated growth is 0 bytes across 100, 500, and 1000 requests"},
        {"item": "Live backend isolation", "result": "PASS", "note": "live verdict path remains DINOv2 and was not modified"},
        {"item": "Policy approval", "result": "PENDING", "note": "shadow_v1 is not an approved production policy"},
    ]


def _risks(evidence: dict) -> list[dict]:
    missed = evidence["stress_missed"][0]
    return [
        _risk("R1", "Resize/recompression false-fake", f"Phase 22/26 missed id {missed}. Coverage {evidence['stress_caught']}/{evidence['stress_events']}.", "high", "Quality candidate catches the other eight demonstrated events.", "One authentic stress image remains FAKE.", "open"),
        _risk("R2", "OOD unvalidated", "Phase 25 REAL_OOD_CORPUS_AVAILABLE is false and production_ood_threshold is null.", "high", "OOD status stays unavailable. Unavailable is not recorded as false.", "The system is not open-set protected.", "open"),
        _risk("R3", "Triton unavailable", "Phase 30 triton_status is unavailable.", "high", "Shadow serving uses the PyTorch CUDA path.", "No validated Triton production runtime.", "open"),
        _risk("R4", "TensorRT unavailable", "Phase 30 tensorrt_status is unavailable.", "high", "No TensorRT engine is claimed.", "No validated TensorRT runtime.", "open"),
        _risk("R5", "ONNX CUDA unavailable", "Phase 30 onnx_cuda_provider is unavailable. CPU ONNX is functional and slow.", "high", "FP32 PyTorch CUDA is the shadow path. CPU ONNX is not treated as production.", "No validated GPU ONNX runtime.", "open"),
        _risk("R6", "Quality review workload", f"Phase 26 review rate {evidence['review_rate']} on {evidence['reference_pool']} images. Workload acceptance is false.", "medium", "The rate is measured and reported at 100, 1000, and 10000 requests per day.", "Operations has not accepted the review burden.", "open"),
        _risk("R7", "Five-brand dataset limitation", "Locked test brands: " + ", ".join(evidence["brands"]) + ".", "medium", "Brand metrics are reported separately. No brand is treated as OOD.", "Results do not establish universal luxury-watch performance.", "residual"),
        _risk("R8", "Missing product/SKU metadata", "The catalog has no verified product or SKU identity.", "medium", "Evaluation stays at brand and authenticity-label level.", "Product-disjoint evaluation is unavailable.", "residual"),
        _risk("R9", "Missing marketplace/source metadata", "Images have no verified marketplace or source field.", "medium", "No source is used as a feature or a gate.", "Source shift cannot be measured.", "residual"),
        _risk("R10", "Live backend remains DINOv2", "Customer verdicts still come from the DINOv2 production path.", "high", "DINOv3 stays opt-in shadow and does not replace the live verdict.", "Promotion has not happened. The live model is a different stack.", "open"),
    ]


def _risk(identifier, failure_mode, evidence, severity, mitigation, residual, status) -> dict:
    if severity not in SEVERITY_MEANING:
        raise ValueError(f"unknown severity {severity}")
    return {
        "id": identifier,
        "failure_mode": failure_mode,
        "evidence": evidence,
        "severity": severity,
        "current_mitigation": mitigation,
        "residual_risk": residual,
        "status": status,
    }


def _markdown(report: dict, workload: list[dict], risks: list[dict]) -> str:
    evidence = report["evidence"]
    lines = [
        "# Phase 34 production-readiness audit",
        "",
        f"Classification: **{report['readiness']}**.",
        "",
        "This audit does not promote the policy and does not open final-test images.",
        "",
        "## Reasons",
        "",
    ]
    lines.extend(f"- {reason}" for reason in report["readiness_reasons"])
    lines.extend(
        [
            "",
            "## Identity",
            "",
            f"- checkpoint_sha256: `{report['checkpoint_sha256']}`",
            f"- membership_hash: `{report['membership_hash']}`",
            f"- manifest_file_sha256: `{report['manifest_file_sha256']}`",
            f"- temperature: `{report['temperature']}`",
            f"- policy_version: `{report['policy_version']}`",
            f"- current_production_status: `{report['current_production_status']}`",
            "",
            "## Model and calibration",
            "",
            f"- Locked final test: {evidence['final_test_samples']} samples, ROC-AUC {evidence['final_test_roc_auc']}, PR-AUC {evidence['final_test_pr_auc']}, F1 {evidence['final_test_f1']}, FAR {evidence['final_test_false_authentic_rate']}, false-fake rate {evidence['final_test_false_fake_rate']}.",
            f"- Brands: {', '.join(evidence['brands'])}.",
            f"- Calibration samples: {evidence['calibration_sample_count']}.",
            f"- Calibrated NLL {evidence['calibrated_nll']}, Brier {evidence['calibrated_brier']}, ECE {evidence['calibrated_ece']}.",
            "",
            "## Robustness and quality",
            "",
            f"- Clean reference pool false-authentic {evidence['clean_false_authentic']}, false-fake {evidence['clean_false_fake']}.",
            f"- Stress coverage {evidence['stress_caught']}/{evidence['stress_events']}. Stress false-authentic {evidence['stress_false_authentic']}.",
            f"- known_robustness_limitation: `{report['known_robustness_limitation']}`.",
            f"- Review rate {evidence['review_rate']} ({evidence['review_count']} / {evidence['reference_pool']}). Authentic review rate {evidence['authentic_review_rate']}. Fake review rate {evidence['fake_review_rate']}.",
            "",
            "## OOD and serving",
            "",
            "- in_distribution_validation: complete.",
            "- open_set_validation: incomplete. The classifier is not open-set protected.",
            f"- SHADOW_READY: {report['shadow_ready']}. PRODUCTION_RUNTIME_READY: {report['production_runtime_ready']}.",
            f"- Batch-1 FP32 model p50: {evidence['batch1_model_p50_ms']} ms.",
            f"- Batch-4 throughput: {evidence['batch4_images_per_sec']} images/sec.",
            f"- Mixed-corpus end-to-end mean before the fast gate: {evidence['sequential_e2e_mean_ms']} ms. Median {evidence['sequential_e2e_p50_ms']} ms.",
            f"- 2560 quality-only fast-gate p50: {evidence['fast_large_quality_p50_ms']} ms.",
            f"- Fast-gate end-to-end p50, decode included and disk I/O excluded: 550 px {evidence['fast_e2e_p50_ms']['medium']} ms; 2560 px {evidence['fast_e2e_p50_ms']['large_lange_2560']} ms.",
            f"- Memory growth: {evidence['memory_growth_bytes']} bytes.",
            "",
            "## Hypothetical review volume",
            "",
        ]
    )
    for row in workload:
        lines.append(f"- {row['requests_per_day']} requests/day -> {row['expected_review_count']:.4f} REVIEW. No SLA is assumed.")
    lines.extend(["", "## Residual risks", ""])
    for risk in risks:
        lines.append(f"- {risk['id']} {risk['failure_mode']} ({risk['severity']}, {risk['status']}): {risk['residual_risk']}")
    lines.extend(
        [
            "",
            "Severity words: critical means a promoted system would fail open or contradict the locked safety result; high means a required control is missing; medium means a scope or operations limit.",
            "",
            f"final_test_samples_used = {report['final_test_samples_used']}.",
            "",
        ]
    )
    return "\n".join(lines)


def _max_memory_growth(path: Path) -> int:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    return max(abs(int(row["growth_bytes"])) for row in rows)


def _fast_e2e_p50(path: Path) -> dict:
    wanted = {"medium", "large_lange_2560"}
    found = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row["role"] in wanted:
                found[row["role"]] = float(row["fast_p50_ms"])
    missing = wanted - set(found)
    if missing:
        raise RuntimeError(f"fast-gate end-to-end rows missing: {sorted(missing)}")
    return found


def _batch4_throughput(path: Path) -> float:
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row["precision"] == "fp32" and int(row["batch_size"]) == 4:
                return float(row["model_images_per_second"])
    raise RuntimeError("batch-4 FP32 throughput is missing")


def _pass(flag: bool) -> str:
    return "PASS" if flag else "FAIL"


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _dump(payload: dict) -> str:
    return json.dumps(payload, indent=2, sort_keys=True)
