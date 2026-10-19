"""Phase 33: fast shadow quality gate versus the full Phase 23 analyzer.

The checkpoint, temperature, cutoffs, and live DINOv2 path stay unchanged.
Final-test images are rejected before open.
"""

from __future__ import annotations

import csv
import io
import json
import math
import statistics
import sys
import time
from pathlib import Path

import torch
from PIL import Image

_PKG = Path(__file__).resolve().parent
_REPO = _PKG.parent
if str(_REPO / "backend") not in sys.path:
    sys.path.insert(0, str(_REPO / "backend"))

from image_quality import apply_quality_flags, extract_quality_features  # noqa: E402
from policy import FROZEN_TEMPERATURE, LOW_CONTRAST_CUTOFF, LOW_RESOLUTION_CUTOFF  # noqa: E402
from policy_contract import (  # noqa: E402
    INPUT_SIZE,
    PHASE23_CUTOFFS,
    POLICY_VERSION,
    PREPROCESSING_VERSION,
    PolicyEngine,
    frozen_shadow_configuration,
)
from quality_fastpath import (  # noqa: E402
    contrast_boundary_images,
    decision_flags,
    fast_quality_gate,
    load_original_rgb,
    resolution_probe,
)
from reference_inference import file_sha256, preprocess_image  # noqa: E402
from split_v2 import AUTHORITATIVE_MEMBERSHIP_HASH  # noqa: E402

_OUTPUT = _PKG / "experiments" / "quality_fastpath_v1"
_QUALITY = _PKG / "experiments" / "image_quality_v1" / "quality_features_nontest.csv"
_PARITY_31 = _PKG / "experiments" / "serving_v2" / "pytorch_cuda_parity.csv"
_PARITY_29 = _PKG / "experiments" / "backend_parity_v2" / "onnx_parity.csv"
_MANIFEST = _PKG / "experiments" / "dataset_audit" / "split_manifest_v2.json"
_FINAL_TEST = _PKG / "experiments" / "final_test_v1" / "final_test_summary.json"
_CHECKPOINT = _PKG / "experiments" / "v2_dinov3_cls_patch_attention" / "epoch_018.pt"
_CHECKPOINT_SHA = "5a38c93fd442b03653c65d2a5ecc9c2687ef152f7c5c020763e4ce1fd9c7d28f"
_PRODUCTION = (
    "backend/inference/verdict.py",
    "backend/inference/dinov2_model.py",
    "backend/inference/local_torch.py",
    "backend/inference/triton_client.py",
    "backend/config.py",
)
_STRESS = (
    ("Label_0_Watches/Audemars Piguet/b1a40ed1b5b9.jpg", "mild_crop", "REVIEW"),
    ("Label_0_Watches/Audemars Piguet/b1a40ed1b5b9.jpg", "partial_crop", "REVIEW"),
    ("Label_0_Watches/Patek Philippe/9ba50fea3805.jpg", "lower_resolution", "REVIEW"),
    ("Label_0_Watches/Patek Philippe/c2f778c66d15.jpg", "lower_resolution", "REVIEW"),
    ("Label_0_Watches/A. Lange & Söhne/63d2c86b29d3.jpg", "lower_resolution", "REVIEW"),
    ("Label_0_Watches/Audemars Piguet/99fcd3b9051c.jpg", "lower_resolution", "REVIEW"),
    ("Label_0_Watches/Richard Mille/a11094843532.jpg", "lower_resolution", "REVIEW"),
    ("Label_0_Watches/Richard Mille/9eda705bd8c9.jpg", "darker", "REVIEW"),
    ("Label_0_Watches/Richard Mille/9eda705bd8c9.jpg", "resize_recompression", "FAKE"),
)
_BENCH = (
    ("tiny", "Label_0_Watches/Audemars Piguet/7ab91150974a.jpg"),
    ("medium", "Label_0_Watches/Audemars Piguet/b1a40ed1b5b9.jpg"),
    ("wide", "Label_0_Watches/A. Lange & Söhne/0105a0230857.jpg"),
    ("tall", "Label_0_Watches/A. Lange & Söhne/00ef95346cfe.jpg"),
    ("low_contrast", "Label_0_Watches/A. Lange & Söhne/015fec2a8afd.jpg"),
    ("large_lange_2560", "Label_0_Watches/A. Lange & Söhne/0003c0bd6ebe.jpg"),
    ("large_rm_2000", "Label_0_Watches/Richard Mille/9eda705bd8c9.jpg"),
)


def run_phase33() -> dict:
    summary_path = _OUTPUT / "fastpath_summary.json"
    if summary_path.is_file():
        raise RuntimeError("quality_fastpath_v1 already exists; refusing to rewrite it")
    checkpoint_before = _stat(_CHECKPOINT)
    final_before = _stat(_FINAL_TEST)
    production_before = {name: file_sha256(_REPO / name) for name in _PRODUCTION}
    if file_sha256(_CHECKPOINT) != _CHECKPOINT_SHA:
        raise RuntimeError("frozen checkpoint hash mismatch")
    blocked = _blocked_test_paths()
    catalog = _catalog(blocked)
    print(f"phase33 parity images {len(catalog)}", flush=True)
    parity_rows = [_parity_row(item, blocked) for item in catalog]
    edge_rows = _edge_rows()
    print("phase33 quality benchmark", flush=True)
    bench_rows, bench_images = _quality_benchmark(blocked)
    print("phase33 end to end", flush=True)
    e2e_rows = _end_to_end(bench_images)
    print("phase33 stress", flush=True)
    stress_rows = _stress(blocked)
    disagreements = sum(not row["flags_match"] for row in parity_rows) + sum(not row["flags_match"] for row in edge_rows)
    decision_mismatches = sum(not row["decision_match"] for row in parity_rows) + sum(not row["decision_match"] for row in stress_rows)
    reference_p50 = _percentile([row["reference_p50_ms"] for row in bench_rows], 0.50)
    fast_p50 = _percentile([row["fast_p50_ms"] for row in bench_rows], 0.50)
    large = next(row for row in bench_rows if row["role"] == "large_lange_2560")
    summary = {
        "phase": 33,
        "checkpoint_sha256": _CHECKPOINT_SHA,
        "membership_hash": AUTHORITATIVE_MEMBERSHIP_HASH,
        "manifest_file_sha256": file_sha256(_MANIFEST),
        "temperature": FROZEN_TEMPERATURE,
        "low_resolution_cutoff": LOW_RESOLUTION_CUTOFF,
        "low_contrast_cutoff": LOW_CONTRAST_CUTOFF,
        "quality_threshold_comparison": "strictly_less_than",
        "policy_version": POLICY_VERSION,
        "precision": "fp32",
        "production_policy_status": "not_promoted",
        "full_analyzer_retained": True,
        "images": len(parity_rows),
        "flag_disagreements": disagreements,
        "decision_mismatches": decision_mismatches,
        "max_abs_luminance_std_delta": max(row["abs_luminance_std_delta"] for row in parity_rows),
        "reference_quality_p50_ms": reference_p50,
        "fast_quality_p50_ms": fast_p50,
        "quality_speedup_p50": reference_p50 / fast_p50 if fast_p50 else None,
        "large_image_reference_p50_ms": large["reference_p50_ms"],
        "large_image_fast_p50_ms": large["fast_p50_ms"],
        "large_image_speedup": large["speedup_p50"],
        "e2e_reference_mean_ms": statistics.fmean(row["reference_mean_ms"] for row in e2e_rows),
        "e2e_fast_mean_ms": statistics.fmean(row["fast_mean_ms"] for row in e2e_rows),
        "stress_matches_expected": all(row["matches_expected"] for row in stress_rows),
        "final_test_samples_used": 0,
        "final_test_opened": False,
        "training_performed": False,
        "calibration_performed": False,
        "ood_evaluation_performed": False,
        "checkpoint_changed": False,
        "live_backend_modified": False,
        "customer_traffic_used": False,
        "ood_status": "unavailable",
        "exact_cutoff_constructible_uint8_512": False,
        "boundary_note": (
            "A 512x512 uint8 image cannot land on the cutoff exactly. "
            "The closest measured standard deviation below the cutoff is flagged. "
            "The closest measured value above the cutoff is not flagged. "
            "A value equal to the cutoff is not flagged."
        ),
    }
    if disagreements or decision_mismatches or not summary["stress_matches_expected"]:
        summary["contract_passed"] = False
    else:
        summary["contract_passed"] = True
    if _stat(_CHECKPOINT) != checkpoint_before or file_sha256(_CHECKPOINT) != _CHECKPOINT_SHA:
        raise RuntimeError("phase 33 changed the frozen checkpoint")
    if _stat(_FINAL_TEST) != final_before:
        raise RuntimeError("phase 33 changed the final-test summary")
    if {name: file_sha256(_REPO / name) for name in production_before} != production_before:
        raise RuntimeError("phase 33 changed a production inference file")
    _OUTPUT.mkdir(parents=True, exist_ok=True)
    (_OUTPUT / "fastpath_config.json").write_text(_dump(_config()) + "\n")
    _write_csv(_OUTPUT / "fastpath_parity.csv", parity_rows)
    _write_csv(_OUTPUT / "fastpath_edge_cases.csv", edge_rows)
    _write_csv(_OUTPUT / "fastpath_benchmark.csv", bench_rows)
    _write_csv(_OUTPUT / "fastpath_end_to_end.csv", e2e_rows)
    _write_csv(_OUTPUT / "fastpath_stress.csv", stress_rows)
    summary_path.write_text(_dump(summary) + "\n")
    print("phase33 artifacts written", flush=True)
    if not summary["contract_passed"]:
        raise RuntimeError("phase 33 fast-path contract failed; artifacts were kept for diagnosis")
    return summary


def _catalog(blocked: set[str]) -> list[dict]:
    chosen = []
    seen = set()

    def add(sample_id: str, split: str, role: str) -> None:
        if sample_id in seen:
            return
        path = (_REPO / sample_id).resolve()
        if str(path) in blocked or split == "test":
            raise RuntimeError("final-test sample reached the fast-path catalog")
        seen.add(sample_id)
        chosen.append({"sample_id": sample_id, "split": split, "role": role})

    for path in (_PARITY_31, _PARITY_29):
        with path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                sample_id = row["sample_id"].split("|", 1)[0]
                add(sample_id, row.get("split") or "unknown", row.get("role") or "golden")
    for sample_id, _name, _expected in _STRESS:
        add(sample_id, "challenge_source", "phase22_source")
    quotas = {"train": 12, "calibration": 12, "validation": 12}
    with _QUALITY.open(newline="") as handle:
        for row in csv.DictReader(handle):
            split = row["source_split"]
            if split not in quotas or quotas[split] <= 0:
                continue
            if row["sample_id"] in seen:
                continue
            add(row["sample_id"], split, "representative")
            quotas[split] -= 1
    return chosen


def _parity_row(item: dict, blocked: set[str]) -> dict:
    image = load_original_rgb(_REPO / item["sample_id"], blocked)
    reference_features, reference_flags, fast_features, fast_flags = _both(image)
    reference_decision = _decision(reference_features, reference_flags, logit=-1.0)
    fast_decision = _decision(fast_features, fast_flags, logit=-1.0)
    return {
        "sample_id": item["sample_id"],
        "role": item["role"],
        "split": item["split"],
        "width": fast_features["width"],
        "height": fast_features["height"],
        "reference_min_dimension": reference_features["min_dimension"],
        "fast_min_dimension": fast_features["min_dimension"],
        "abs_luminance_std_delta": abs(reference_features["luminance_std"] - fast_features["luminance_std"]),
        "reference_decision_flags": "|".join(reference_flags),
        "fast_flags": "|".join(fast_flags),
        "flags_match": reference_flags == fast_flags,
        "reference_decision": reference_decision.decision,
        "fast_decision": fast_decision.decision,
        "decision_match": reference_decision.decision == fast_decision.decision,
        "ood_status": reference_decision.ood_status,
    }


def _edge_rows() -> list[dict]:
    rows = []
    probes = [
        ("511x511", resolution_probe(511, 511), ("LOW_RESOLUTION",)),
        ("512x512", resolution_probe(512, 512), ()),
        ("513x513", resolution_probe(513, 513), ()),
        ("511x2000", resolution_probe(2000, 511), ("LOW_RESOLUTION",)),
        ("512x2000", resolution_probe(2000, 512), ()),
        ("rgb", resolution_probe(600, 600), ()),
        ("grayscale", resolution_probe(600, 600).convert("L"), ()),
        ("uniform_black", Image.new("RGB", (600, 600), (0, 0, 0)), ("LOW_CONTRAST",)),
        ("uniform_white", Image.new("RGB", (600, 600), (255, 255, 255)), ("LOW_CONTRAST",)),
    ]
    boundary = contrast_boundary_images()
    probes.append(("std_below_cutoff", boundary["below"], ("LOW_CONTRAST",)))
    probes.append(("std_above_cutoff", boundary["above"], ()))
    for name, image, expected in probes:
        reference_features, reference_flags, fast_features, fast_flags = _both(image)
        rows.append(
            {
                "case": name,
                "width": fast_features["width"],
                "height": fast_features["height"],
                "min_dimension": fast_features["min_dimension"],
                "luminance_std": fast_features["luminance_std"],
                "reference_decision_flags": "|".join(reference_flags),
                "fast_flags": "|".join(fast_flags),
                "expected_flags": "|".join(expected),
                "flags_match": reference_flags == fast_flags == expected,
                "low_resolution": "LOW_RESOLUTION" in fast_flags,
                "low_contrast": "LOW_CONTRAST" in fast_flags,
            }
        )
    equal = LOW_CONTRAST_CUTOFF
    rows.append(
        {
            "case": "std_equal_to_cutoff_operator",
            "width": "",
            "height": "",
            "min_dimension": 512,
            "luminance_std": equal,
            "reference_decision_flags": "",
            "fast_flags": "",
            "expected_flags": "",
            "flags_match": not (equal < LOW_CONTRAST_CUTOFF),
            "low_resolution": False,
            "low_contrast": False,
        }
    )
    return rows


def _quality_benchmark(blocked: set[str]) -> tuple[list[dict], list[dict]]:
    rows = []
    loaded = []
    for role, sample_id in _BENCH:
        image = load_original_rgb(_REPO / sample_id, blocked)
        payload = image.tobytes()
        size = image.size
        reference_times = _time_quality(lambda: _reference(image), warmup=5, timed=30)
        fast_times = _time_quality(lambda: fast_quality_gate(image), warmup=5, timed=30)
        loaded.append({"role": role, "sample_id": sample_id, "image": image, "png_size": size, "raw_bytes": payload})
        rows.append(
            {
                "role": role,
                "sample_id": sample_id,
                "width": size[0],
                "height": size[1],
                "reference_p50_ms": _percentile(reference_times, 0.50),
                "reference_p95_ms": _percentile(reference_times, 0.95),
                "reference_p99_ms": _percentile(reference_times, 0.99),
                "reference_mean_ms": statistics.fmean(reference_times),
                "reference_max_ms": max(reference_times),
                "fast_p50_ms": _percentile(fast_times, 0.50),
                "fast_p95_ms": _percentile(fast_times, 0.95),
                "fast_p99_ms": _percentile(fast_times, 0.99),
                "fast_mean_ms": statistics.fmean(fast_times),
                "fast_max_ms": max(fast_times),
                "speedup_p50": _percentile(reference_times, 0.50) / _percentile(fast_times, 0.50),
                "temporary_float64_bytes": int(size[0] * size[1] * 4 * 8),
                "model_inference_included": False,
            }
        )
    return rows, loaded


def _end_to_end(loaded: list[dict]) -> list[dict]:
    from inference.dinov3_model import shared_backend

    backend = shared_backend()
    backend.load()
    torch.cuda.synchronize()
    engine = PolicyEngine(frozen_shadow_configuration())
    rows = []
    for item in loaded:
        encoded = _png_bytes(item["image"])
        reference_times = _time_e2e(encoded, backend, engine, fast=False)
        fast_times = _time_e2e(encoded, backend, engine, fast=True)
        rows.append(
            {
                "role": item["role"],
                "sample_id": item["sample_id"],
                "reference_p50_ms": _percentile(reference_times, 0.50),
                "reference_p95_ms": _percentile(reference_times, 0.95),
                "reference_p99_ms": _percentile(reference_times, 0.99),
                "reference_mean_ms": statistics.fmean(reference_times),
                "fast_p50_ms": _percentile(fast_times, 0.50),
                "fast_p95_ms": _percentile(fast_times, 0.95),
                "fast_p99_ms": _percentile(fast_times, 0.99),
                "fast_mean_ms": statistics.fmean(fast_times),
                "reference_images_per_sec": 1000.0 / statistics.fmean(reference_times),
                "fast_images_per_sec": 1000.0 / statistics.fmean(fast_times),
                "disk_io_included": False,
                "decode_included": True,
                "batch_size": 1,
            }
        )
    return rows


def _stress(blocked: set[str]) -> list[dict]:
    from hard_negative import apply_challenge_transform
    from inference.dinov3_model import shared_backend

    backend = shared_backend()
    backend.load()
    rows = []
    opened = {}
    for sample_id, transform, expected in _STRESS:
        if sample_id not in opened:
            opened[sample_id] = load_original_rgb(_REPO / sample_id, blocked)
        image = apply_challenge_transform(opened[sample_id], transform)
        reference_features, reference_flags, fast_features, fast_flags = _both(image)
        tensor = preprocess_image(image).unsqueeze(0)
        torch.cuda.synchronize()
        logit = float(backend.predict_logits(tensor)[0])
        torch.cuda.synchronize()
        reference_decision = _decision(reference_features, reference_flags, logit)
        fast_decision = _decision(fast_features, fast_flags, logit)
        rows.append(
            {
                "sample_id": sample_id,
                "transform": transform,
                "expected_decision": expected,
                "reference_decision_flags": "|".join(reference_flags),
                "fast_flags": "|".join(fast_flags),
                "flags_match": reference_flags == fast_flags,
                "reference_decision": reference_decision.decision,
                "fast_decision": fast_decision.decision,
                "decision_match": reference_decision.decision == fast_decision.decision,
                "matches_expected": reference_decision.decision == fast_decision.decision == expected,
                "logit": logit,
            }
        )
    return rows


def _time_e2e(encoded: bytes, backend, engine, *, fast: bool) -> list[float]:
    def once() -> None:
        torch.cuda.synchronize()
        with Image.open(io.BytesIO(encoded)) as handle:
            image = handle.convert("RGB")
            image.load()
        if fast:
            features, flags = fast_quality_gate(image)
        else:
            features, flags = _reference_decision(image)
        tensor = preprocess_image(image).unsqueeze(0)
        logit = float(backend.predict_logits(tensor)[0])
        torch.cuda.synchronize()
        engine.evaluate(
            logit=logit,
            quality_features=features,
            quality_flags=flags,
            preprocessing_version=PREPROCESSING_VERSION,
            input_size=INPUT_SIZE,
            checkpoint_sha256=_CHECKPOINT_SHA,
        )

    for _ in range(3):
        once()
    samples = []
    for _ in range(20):
        torch.cuda.synchronize()
        started = time.perf_counter()
        once()
        samples.append((time.perf_counter() - started) * 1000.0)
    return samples


def _both(image):
    reference_features = extract_quality_features(image)
    reference_all = apply_quality_flags(reference_features, {"cutoffs": dict(PHASE23_CUTOFFS)})
    fast_features, fast_flags = fast_quality_gate(image)
    return reference_features, decision_flags(reference_all), fast_features, fast_flags


def _reference_decision(image):
    features = extract_quality_features(image)
    flags = decision_flags(apply_quality_flags(features, {"cutoffs": dict(PHASE23_CUTOFFS)}))
    return features, flags


def _reference(image):
    return _reference_decision(image)


def _decision(features: dict, flags: tuple[str, ...], logit: float):
    engine = PolicyEngine(frozen_shadow_configuration())
    return engine.evaluate(
        logit=logit,
        quality_features={"min_dimension": features["min_dimension"], "luminance_std": features["luminance_std"]},
        quality_flags=flags,
        preprocessing_version=PREPROCESSING_VERSION,
        input_size=INPUT_SIZE,
        checkpoint_sha256=_CHECKPOINT_SHA,
    )


def _time_quality(function, *, warmup: int, timed: int) -> list[float]:
    for _ in range(warmup):
        function()
    samples = []
    for _ in range(timed):
        started = time.perf_counter()
        function()
        samples.append((time.perf_counter() - started) * 1000.0)
    return samples


def _png_bytes(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _config() -> dict:
    return {
        "low_resolution": "min(original width, original height) < 512",
        "low_contrast": "population std of BT.601 luminance < 0.1933616647502347",
        "equality_is_not_flagged": True,
        "luminance": "BT.601 Y = (0.299 R + 0.587 G + 0.114 B) / 255 on the original RGB image",
        "dtype": "float64",
        "std_ddof": 0,
        "std_snap_below": 1e-12,
        "reference_function": "image_quality.luminance_image",
        "full_analyzer": "image_quality.extract_quality_features",
        "fast_gate": "quality_fastpath.fast_quality_gate",
        "model_tensor_used_for_quality": False,
        "low_resolution_cutoff": LOW_RESOLUTION_CUTOFF,
        "low_contrast_cutoff": LOW_CONTRAST_CUTOFF,
        "temperature": FROZEN_TEMPERATURE,
        "checkpoint_sha256": _CHECKPOINT_SHA,
        "policy_version": POLICY_VERSION,
    }


def _blocked_test_paths() -> set[str]:
    payload = json.loads(_MANIFEST.read_text())
    return {str(Path(item).resolve()) for item in (payload.get("membership") or {}).get("test") or []}


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    low = math.floor(position)
    high = min(low + 1, len(ordered) - 1)
    weight = position - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


def _stat(path: Path) -> tuple[int, int]:
    info = path.stat()
    return info.st_size, info.st_mtime_ns


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _dump(payload: dict) -> str:
    return json.dumps(payload, indent=2, sort_keys=True)
