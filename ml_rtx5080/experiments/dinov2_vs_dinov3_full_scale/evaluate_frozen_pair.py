"""Paired GPU evaluation of the frozen DINOv2 and DINOv3 checkpoints.

Reads the committed protocol, refuses a hash or cohort mismatch, checks batched
Triton logits against the reference forwards, then scores the watch corpus.
Locked final-test paths are never opened. This script does not train or write
model weights.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import io
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

_CORE = Path(__file__).resolve().parent
_PKG = _CORE.parents[1]
_REPO = _PKG.parent
for path in (_CORE, _PKG, _REPO / "backend"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import compare_core as core

PROTOCOL_PATH = _CORE / "protocol_v1.json"
LOCAL_RESULTS = _CORE / "local_results"
PARITY_PROTOCOL = _PKG / "experiments" / "dual_model_triton_v1" / "dinov2_live_triton_gpu_protocol_v2.json"
TRITON_URL = "localhost:18001"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def _load_protocol() -> dict:
    protocol = json.loads(PROTOCOL_PATH.read_text())
    if protocol["protocol"] != core.PROTOCOL_NAME:
        raise core.ComparisonGuardError("protocol name drifted")
    if protocol["thresholds_calibration_or_metrics_may_change_after_results"] is not False:
        raise core.ComparisonGuardError("protocol allows changing metrics after results")
    return protocol


def _verify_identities(protocol: dict) -> None:
    dinov2 = protocol["models"]["dinov2_legacy"]
    dinov3 = protocol["models"]["dinov3_experimental"]
    checks = [
        (dinov2["checkpoint_path"], dinov2["checkpoint_sha256"], "DINOv2 checkpoint"),
        (dinov2["training_config_path"], dinov2["training_config_sha256"], "DINOv2 config"),
        (dinov2["split_manifest_path"], dinov2["split_manifest_sha256"], "DINOv2 split manifest"),
        (dinov2["triton_config_path"], dinov2["triton_config_sha256"], "DINOv2 Triton config"),
        (dinov3["checkpoint_path"], dinov3["checkpoint_sha256"], "DINOv3 checkpoint"),
        (dinov3["training_config_path"], dinov3["training_config_sha256"], "DINOv3 config"),
        (dinov3["split_manifest_path"], dinov3["split_manifest_sha256"], "DINOv3 split manifest"),
        (dinov3["temperature_file"], dinov3["temperature_file_sha256"], "DINOv3 temperature"),
        (dinov3["triton_config_path"], dinov3["triton_config_sha256"], "DINOv3 Triton config"),
    ]
    for relative, expected, label in checks:
        core.require_sha256(_REPO / relative, expected, label)
    for logical, installed in (
        ("dinov2_legacy", _REPO / "models" / "dinov2_vitb14_live" / "config.pbtxt"),
        ("dinov3_experimental", _REPO / "models" / "dinov3_authenticity_candidate" / "config.pbtxt"),
    ):
        pinned = (_REPO / protocol["models"][logical]["triton_config_path"]).read_bytes()
        if not installed.is_file() or installed.read_bytes() != pinned:
            raise core.ComparisonGuardError(f"installed Triton config for {logical} is not the pinned GPU config")


def _runtime_guard(protocol: dict) -> dict:
    import torch

    pinned = protocol["runtime_pinned_before_inference"]
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    observed = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": gpu,
    }
    for key in ("python", "torch", "cuda", "gpu"):
        if observed[key] != pinned[key]:
            raise core.ComparisonGuardError(f"{key} {observed[key]!r} != pinned {pinned[key]!r}")
    return observed


def _membership(protocol: dict) -> tuple[list[dict], set[str]]:
    dinov2 = json.loads((_REPO / protocol["models"]["dinov2_legacy"]["split_manifest_path"]).read_text())
    dinov3 = json.loads((_REPO / protocol["models"]["dinov3_experimental"]["split_manifest_path"]).read_text())
    if dinov3.get("membership_hash") != protocol["models"]["dinov3_experimental"]["membership_hash"]:
        raise core.ComparisonGuardError("DINOv3 membership hash drifted")
    train_folders = {str(Path(path).resolve()) for path in dinov2["train_product_folders"]}
    val_folders = {str(Path(path).resolve()) for path in dinov2["val_product_folders"]}
    blocked = {str(Path(path).resolve()) for path in dinov3["membership"]["test"]}
    dinov3_split = {
        str(Path(path).resolve()): split
        for split, paths in dinov3["membership"].items()
        for path in paths
    }
    records = []
    for label, dirname in ((0, "Label_0_Watches"), (1, "Label_1_Watches")):
        folder = _REPO / dirname
        if not folder.is_dir():
            raise core.ComparisonGuardError(f"missing watch corpus folder {dirname}")
        for brand_dir in sorted(path for path in folder.iterdir() if path.is_dir()):
            for image in sorted(brand_dir.iterdir()):
                if image.suffix.lower() not in IMAGE_SUFFIXES or not image.is_file():
                    continue
                resolved = image.resolve()
                resolved_s = str(resolved)
                split = dinov3_split.get(resolved_s, "unassigned")
                role = core.assign_role(
                    parent_folder=str(resolved.parent),
                    resolved_path=resolved_s,
                    dinov2_train_folders=train_folders,
                    dinov2_val_folders=val_folders,
                    dinov3_split=split,
                    blocked=blocked,
                )
                if str(resolved.parent) in val_folders:
                    d2_split = "validation"
                elif str(resolved.parent) in train_folders:
                    d2_split = "train"
                else:
                    d2_split = "outside_manifest"
                records.append(
                    {
                        "sample_id": resolved.relative_to(_REPO).as_posix(),
                        "resolved_path": resolved_s,
                        "label": label,
                        "brand": brand_dir.name,
                        "role": role,
                        "dinov2_split": d2_split,
                        "dinov3_split": split,
                    }
                )
    if len(records) != core.CORPUS_COUNT:
        raise core.ComparisonGuardError(f"watch corpus has {len(records)} images, not {core.CORPUS_COUNT}")
    if sum(1 for row in records if row["resolved_path"] in blocked) != core.LOCKED_TEST_COUNT:
        raise core.ComparisonGuardError("locked final-test exclusion count drifted")
    for row in records:
        if row["role"] == "excluded":
            blocked.add(row["resolved_path"])
    grouped = core.select_catalog(records, blocked)
    if len(grouped["excluded"]) != core.LOCKED_TEST_COUNT:
        raise core.ComparisonGuardError("excluded count drifted")
    if len(grouped["primary_validation"]) + len(grouped["diagnostic"]) != core.SWEEP_COUNT:
        raise core.ComparisonGuardError("sweep count drifted")
    core.assert_primary_cohort(grouped["primary_validation"])
    return grouped["primary_validation"] + grouped["diagnostic"], blocked


def _read_image(path: str):
    from PIL import Image

    raw = Path(path).read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    with Image.open(io.BytesIO(raw)) as handle:
        rgb = handle.convert("RGB")
        rgb.load()
        return rgb.copy(), digest


def _triton_logits(client, model_name: str, version: str, batch) -> tuple[list[float], float]:
    import tritonclient.grpc as grpcclient

    started = time.perf_counter()
    inputs = [grpcclient.InferInput("input__0", list(batch.shape), "FP32")]
    inputs[0].set_data_from_numpy(batch)
    result = client.infer(model_name, inputs, model_version=version, client_timeout=120.0)
    elapsed = time.perf_counter() - started
    return result.as_numpy("output__0").reshape(-1).astype("float64").tolist(), elapsed


def _require_gpu_models() -> None:
    import tritonclient.grpc as grpcclient

    client = grpcclient.InferenceServerClient(url=TRITON_URL)
    if not client.is_server_live():
        raise core.ComparisonGuardError("Triton is not live")
    for name, version, batch_size in (
        ("dinov2_vitb14_live", "1", 8),
        ("dinov3_authenticity_candidate", "1", 16),
    ):
        if not client.is_model_ready(name, version):
            raise core.ComparisonGuardError(f"{name} is not ready")
        config = client.get_model_config(name, version, as_json=True)["config"]
        kinds = {group.get("kind", "KIND_AUTO") for group in config.get("instance_group", [])}
        if "KIND_GPU" not in kinds:
            raise core.ComparisonGuardError(f"{name} is not on KIND_GPU: {sorted(kinds)}")
        if int(config.get("max_batch_size", 0)) < batch_size:
            raise core.ComparisonGuardError(f"{name} max batch is below the protocol batch")
    return client


def _parity(blocked: set[str]) -> dict:
    import numpy as np
    import torch
    from PIL import Image

    from inference.dinov2_model import DINOv2Classifier
    from inference.triton_client import preprocess_chw
    from reference_inference import forward_logits, load_reference_model, preprocess_image

    payload = json.loads(PARITY_PROTOCOL.read_text())
    cases = payload["inputs"]
    if any(case.get("split") == "test" for case in cases):
        raise core.ComparisonGuardError("parity protocol contains a final-test split")
    images = []
    for case in cases:
        path = str((_REPO / case["sample_id"]).resolve())
        images.append(core.open_if_allowed(path, blocked, _read_image)[0])
    client = _require_gpu_models()

    dinov2_batch = np.ascontiguousarray(
        np.stack([preprocess_chw(np.asarray(image), side=504)[0] for image in images]).astype(np.float32)
    )
    model = DINOv2Classifier("vit_base_patch14_dinov2.lvd142m", dropout=0.2)
    state = torch.load(_PKG / "checkpoints" / "best_model.pt", map_location="cpu", weights_only=False)["model_state"]
    load = model.load_state_dict(state, strict=True)
    if load.missing_keys or load.unexpected_keys:
        raise core.ComparisonGuardError("DINOv2 checkpoint did not load strictly")
    model.eval()
    with torch.inference_mode():
        reference = model(torch.from_numpy(dinov2_batch)).detach().float().reshape(-1).numpy()
    served = []
    for start in range(0, len(dinov2_batch), 8):
        values, _elapsed = _triton_logits(client, "dinov2_vitb14_live", "1", dinov2_batch[start : start + 8])
        served.extend(values)
    dinov2_error = max(abs(float(a) - float(b)) for a, b in zip(served, reference.tolist(), strict=True))
    del model, state
    gc.collect()

    dinov3_model, digest = load_reference_model(
        _PKG / "experiments" / "v2_dinov3_cls_patch_attention" / "epoch_018.pt",
        torch.device("cpu"),
    )
    if digest != core.DINOV3_SHA256:
        raise core.ComparisonGuardError("DINOv3 reference hash drifted during parity")
    dinov3_batch = torch.stack([preprocess_image(image) for image in images])
    reference3 = forward_logits(dinov3_model, dinov3_batch, torch.device("cpu")).tolist()
    served3 = []
    array = np.ascontiguousarray(dinov3_batch.numpy().astype(np.float32))
    for start in range(0, len(array), 16):
        values, _elapsed = _triton_logits(client, "dinov3_authenticity_candidate", "1", array[start : start + 16])
        served3.extend(values)
    dinov3_error = max(abs(float(a) - float(b)) for a, b in zip(served3, reference3, strict=True))
    del dinov3_model
    gc.collect()
    if dinov2_error > core.LOGIT_ATOL or dinov3_error > core.LOGIT_ATOL:
        raise core.ComparisonGuardError(
            f"parity failed dinov2={dinov2_error} dinov3={dinov3_error}; sweep not started"
        )
    return {
        "status": "PASS",
        "samples": len(cases),
        "final_test_images_opened": 0,
        "dinov2_max_abs_logit_error": dinov2_error,
        "dinov3_max_abs_logit_error": dinov3_error,
        "tolerance": core.LOGIT_ATOL,
        "gpu_inference": True,
    }


def _gpu_mib() -> float | None:
    try:
        text = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            text=True,
        ).strip().splitlines()[0]
        return float(text)
    except (OSError, subprocess.CalledProcessError, ValueError, IndexError):
        return None


def _error_row(record: dict, model_id: str, message: str) -> dict:
    return {
        "sample_id": record["sample_id"],
        "image_sha256": None,
        "role": record["role"],
        "inherited_label": "fake_labeled" if int(record["label"]) == 1 else "authentic_labeled",
        "label": int(record["label"]),
        "brand": record["brand"],
        "dinov2_split": record["dinov2_split"],
        "dinov3_split": record["dinov3_split"],
        "model_id": model_id,
        "checkpoint_sha256": core.DINOV2_SHA256 if model_id == core.DINOV2_ID else core.DINOV3_SHA256,
        "preprocessing_id": core.DINOV2_PREPROCESS if model_id == core.DINOV2_ID else core.DINOV3_PREPROCESS,
        "raw_logit": None,
        "fake_score": None,
        "fake_score_kind": None,
        "decision": None,
        "error": message,
        "latency_ms": None,
    }


def _ok_row(record, model_id, digest, logit, score, kind, decision, latency_ms) -> dict:
    return {
        "sample_id": record["sample_id"],
        "image_sha256": digest,
        "role": record["role"],
        "inherited_label": "fake_labeled" if int(record["label"]) == 1 else "authentic_labeled",
        "label": int(record["label"]),
        "brand": record["brand"],
        "dinov2_split": record["dinov2_split"],
        "dinov3_split": record["dinov3_split"],
        "model_id": model_id,
        "checkpoint_sha256": core.DINOV2_SHA256 if model_id == core.DINOV2_ID else core.DINOV3_SHA256,
        "preprocessing_id": core.DINOV2_PREPROCESS if model_id == core.DINOV2_ID else core.DINOV3_PREPROCESS,
        "raw_logit": float(logit),
        "fake_score": float(score),
        "fake_score_kind": kind,
        "decision": decision,
        "error": None,
        "latency_ms": latency_ms,
    }


def _flush(client, pending: list[dict]) -> list[dict]:
    import numpy as np

    from inference.triton_client import preprocess_chw
    from quality_fastpath import fast_quality_gate
    from reference_inference import preprocess_image

    rows = []
    dinov2_ready = []
    dinov3_ready = []
    for item in pending:
        try:
            array = preprocess_chw(np.asarray(item["image"]), side=504)[0]
            dinov2_ready.append((item, array))
        except Exception as exc:
            rows.append(_error_row(item["record"], core.DINOV2_ID, f"preprocess:{type(exc).__name__}"))
        try:
            flags = fast_quality_gate(item["image"])[1]
            tensor = preprocess_image(item["image"])
            dinov3_ready.append((item, tensor, flags))
        except Exception as exc:
            rows.append(_error_row(item["record"], core.DINOV3_ID, f"preprocess:{type(exc).__name__}"))

    for start in range(0, len(dinov2_ready), 8):
        chunk = dinov2_ready[start : start + 8]
        batch = np.ascontiguousarray(np.stack([array for _item, array in chunk]).astype(np.float32))
        try:
            logits, elapsed = _triton_logits(client, "dinov2_vitb14_live", "1", batch)
        except Exception as exc:
            for item, _array in chunk:
                rows.append(_error_row(item["record"], core.DINOV2_ID, f"inference:{type(exc).__name__}"))
            continue
        per_image = (elapsed * 1000.0) / len(chunk)
        for (item, _array), logit in zip(chunk, logits, strict=True):
            try:
                decision, score = core.dinov2_decision(logit)
            except core.ComparisonGuardError as exc:
                rows.append(_error_row(item["record"], core.DINOV2_ID, str(exc)))
                continue
            rows.append(
                _ok_row(item["record"], core.DINOV2_ID, item["sha256"], logit, score, "uncalibrated_sigmoid", decision, per_image)
            )

    for start in range(0, len(dinov3_ready), 16):
        chunk = dinov3_ready[start : start + 16]
        batch = np.ascontiguousarray(np.stack([tensor.numpy() for _item, tensor, _flags in chunk]).astype(np.float32))
        try:
            logits, elapsed = _triton_logits(client, "dinov3_authenticity_candidate", "1", batch)
        except Exception as exc:
            for item, _tensor, _flags in chunk:
                rows.append(_error_row(item["record"], core.DINOV3_ID, f"inference:{type(exc).__name__}"))
            continue
        per_image = (elapsed * 1000.0) / len(chunk)
        for (item, _tensor, flags), logit in zip(chunk, logits, strict=True):
            try:
                decision, score = core.dinov3_decision(logit, flags)
            except core.ComparisonGuardError as exc:
                rows.append(_error_row(item["record"], core.DINOV3_ID, str(exc)))
                continue
            rows.append(
                _ok_row(
                    item["record"],
                    core.DINOV3_ID,
                    item["sha256"],
                    logit,
                    score,
                    "temperature_scaled_sigmoid",
                    decision,
                    per_image,
                )
            )
    return rows


def _sweep(records: list[dict], blocked: set[str]) -> tuple[list[dict], dict]:
    ordered = sorted(records, key=lambda row: row["sample_id"])
    client = _require_gpu_models()
    rows: list[dict] = []
    pending: list[dict] = []
    memory = []
    started = time.perf_counter()
    sample = _gpu_mib()
    if sample is not None:
        memory.append(sample)

    def flush() -> None:
        nonlocal pending
        if pending:
            rows.extend(_flush(client, pending))
            pending = []

    for index, record in enumerate(ordered, start=1):
        try:
            image, digest = core.open_if_allowed(record["resolved_path"], blocked, _read_image)
        except core.ComparisonGuardError:
            raise
        except Exception as exc:
            rows.append(_error_row(record, core.DINOV2_ID, f"read:{type(exc).__name__}"))
            rows.append(_error_row(record, core.DINOV3_ID, f"read:{type(exc).__name__}"))
            continue
        pending.append({"record": record, "image": image, "sha256": digest})
        if len(pending) == 16:
            flush()
        if index % 512 == 0:
            sample = _gpu_mib()
            if sample is not None:
                memory.append(sample)
            elapsed = time.perf_counter() - started
            print(f"sweep {index}/{len(ordered)} elapsed_s={elapsed:.1f}", flush=True)
    flush()
    sample = _gpu_mib()
    if sample is not None:
        memory.append(sample)
    core.assert_complete_pairs(rows, [record["sample_id"] for record in ordered])
    return rows, {"gpu_memory_mib_max": max(memory) if memory else None, "wall_seconds": time.perf_counter() - started}


def _subset_metrics(rows: list[dict], predicate) -> dict:
    chosen = [row for row in rows if predicate(row)]
    by_model = {}
    for model_id in (core.DINOV2_ID, core.DINOV3_ID):
        model_rows = [row for row in chosen if row["model_id"] == model_id]
        by_model[model_id] = core.score_rows(model_rows) if model_rows else None
        if model_rows:
            by_model[model_id]["brands"] = core.brand_breakdown(model_rows)
            times = [float(row["latency_ms"]) for row in model_rows if row.get("latency_ms") is not None]
            infer_seconds = sum(times) / 1000.0
            by_model[model_id]["latency"] = core.latency_summary(
                times,
                infer_seconds,
                sum(1 for row in model_rows if not row.get("error")),
            )
    left = [row for row in chosen if row["model_id"] == core.DINOV2_ID]
    right = [row for row in chosen if row["model_id"] == core.DINOV3_ID]
    agreement = core.paired_agreement(left, right) if left and right else None
    return {"samples": len(left), "models": by_model, "agreement": agreement}


def _metrics(rows: list[dict], parity: dict, resources: dict) -> dict:
    primary = _subset_metrics(rows, lambda row: row["role"] == "primary_validation")
    diagnostic = _subset_metrics(rows, lambda row: row["role"] == "diagnostic")
    dinov2_splits = {}
    dinov3_splits = {}
    for split in ("train", "validation"):
        dinov2_splits[split] = _subset_metrics(
            rows, lambda row, split=split: row["role"] == "diagnostic" and row["dinov2_split"] == split or (
                row["role"] == "primary_validation" and row["dinov2_split"] == split
            )
        )
    for split in ("train", "calibration", "validation"):
        dinov3_splits[split] = _subset_metrics(rows, lambda row, split=split: row["dinov3_split"] == split)
    leakage = sum(
        1
        for row in rows
        if row["model_id"] == core.DINOV2_ID and row["dinov3_split"] == "validation" and row["dinov2_split"] == "train"
    )
    return {
        "protocol": core.PROTOCOL_NAME,
        "data_readiness": "NO_GO_FOR_AUTHENTICITY_CLAIMS",
        "winner": None,
        "metrics_status": "exploratory",
        "directory_labels_are_verified_authenticity": False,
        "final_test_images_opened": 0,
        "parity": parity,
        "gpu_memory_mib_max": resources["gpu_memory_mib_max"],
        "sweep_wall_seconds": resources["wall_seconds"],
        "primary_validation": primary,
        "diagnostic": diagnostic,
        "descriptive_by_model_split": {
            "note": "Includes each model's own train, calibration, and validation rows. Train and calibration are diagnostic. A model's own validation is not the paired held-out cohort when the other model used those images for training or calibration.",
            "dinov2": dinov2_splits,
            "dinov3": dinov3_splits,
            "dinov3_validation_images_also_in_dinov2_train": leakage,
        },
    }


def _fmt(value) -> str:
    if value is None:
        return "undefined"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _model_table(metrics: dict) -> str:
    lines = [
        "| Model | Successful | Failed | AUTHENTIC-labeled | FAKE-labeled | REVIEW | Coverage | Accuracy | Precision | Recall | F1 | Balanced accuracy | False-authentic rate | False-fake rate | ROC-AUC | PR-AUC |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for model_id, body in metrics["models"].items():
        if body is None:
            continue
        lines.append(
            "| {model} | {successful} | {failed} | {authentic} | {fake} | {review} | {coverage} | {accuracy} | {precision} | {recall} | {f1} | {balanced} | {far} | {ffr} | {roc} | {pr} |".format(
                model=model_id,
                successful=body["successful"],
                failed=body["failed"],
                authentic=body["authentic_labeled"],
                fake=body["fake_labeled"],
                review=body["review_count"],
                coverage=_fmt(body["decision_coverage"]),
                accuracy=_fmt(body["accuracy"]),
                precision=_fmt(body["precision_fake"]),
                recall=_fmt(body["recall_fake"]),
                f1=_fmt(body["f1_fake"]),
                balanced=_fmt(body["balanced_accuracy"]),
                far=_fmt(body["false_authentic_rate"]),
                ffr=_fmt(body["false_fake_rate"]),
                roc=_fmt(body["roc_auc_raw_logit"]),
                pr=_fmt(body["pr_auc_raw_logit"]),
            )
        )
    return "\n".join(lines)


def _report(metrics: dict) -> str:
    primary = metrics["primary_validation"]
    agreement = primary["agreement"]
    return f"""# DINOv2 vs DINOv3 full-scale frozen comparison

DATA_READINESS = NO_GO_FOR_AUTHENTICITY_CLAIMS

No model is selected. Directory labels are historical dataset labels, not verified authenticity. The common held-out validation cohort is 315 authentic-labeled A. Lange & Söhne images and 0 fake-labeled images. Ranking metrics on that cohort are undefined. The stored manifests do not match a Vacheron Constantin versus Richard Mille validation split, and no other subset was promoted to validation.

## What was compared

The live DINOv2 checkpoint `fe1daa0bf71c5e9b73267d40784442748b8fd1999a8d107979f1338c52f0fa66` uses 504-square ImageNet preprocessing and the 0.50 / 0.88 operational policy. Its sigmoid is uncalibrated. The experimental DINOv3 checkpoint `5a38c93fd442b03653c65d2a5ecc9c2687ef152f7c5c020763e4ce1fd9c7d28f` uses `resize_pad_square_eval_v1` at 512, temperature 0.24038200410185356, threshold 0.50, and the shadow quality review rule. Temperature-scaled scores are not on the DINOv2 scale. Both checkpoints were hashed before and after the sweep. Neither weight file was written.

The models were not trained on the same images. DINOv2's stored split is a product-folder split that also contains sneakers. DINOv3 uses the brand-stratified watch manifest. Images in either training or calibration membership are diagnostic only.

## Primary paired cohort

{agreement["samples"]} images, both models successful on {agreement["both_successful"]}, agreement {agreement["agreement"]}, disagreement {agreement["disagreement"]}, both binary and both wrong {agreement["both_binary_and_both_wrong"]}.

{_model_table(primary)}

False-authentic and false-fake rates use every successful image of that inherited label as the denominator. REVIEW stays in that denominator and out of the binary accuracy denominator. DINOv2 has no REVIEW state.

## Diagnostic sweep

Train and calibration outputs are not validation. DINOv3's own validation still contains {metrics["descriptive_by_model_split"]["dinov3_validation_images_also_in_dinov2_train"]} images that are in the live DINOv2 training folders, so that split is not a paired held-out comparison.

{_model_table(metrics["diagnostic"])}

## Serving

Parity against the CPU FP32 reference on {metrics["parity"]["samples"]} non-test images passed with max absolute logit error {metrics["parity"]["dinov2_max_abs_logit_error"]:.3e} for DINOv2 and {metrics["parity"]["dinov3_max_abs_logit_error"]:.3e} for DINOv3. The tolerance was 1e-4. Triton batches stayed at 8 and 16 on KIND_GPU. Peak sampled GPU memory was {metrics["gpu_memory_mib_max"]} MiB. Locked final-test images opened: {metrics["final_test_images_opened"]}.

Per-image latency is the Triton request wall time divided by the images in that request. Primary-cohort latency is inside the machine-readable metrics. This run does not change production approval, publication, temperature, or the 0.88 floor.
"""


def _sha_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _write_outputs(rows: list[dict], metrics: dict, observed: dict) -> None:
    LOCAL_RESULTS.mkdir(parents=True, exist_ok=True)
    prediction_lines = [json.dumps(row, sort_keys=True, ensure_ascii=False) for row in sorted(rows, key=lambda item: (item["sample_id"], item["model_id"]))]
    prediction_bytes = ("\n".join(prediction_lines) + "\n").encode()
    (LOCAL_RESULTS / "predictions.jsonl").write_bytes(prediction_bytes)
    sample_ids = sorted({row["sample_id"] for row in rows})
    sample_bytes = ("\n".join(sample_ids) + "\n").encode()
    (LOCAL_RESULTS / "sample_ids.txt").write_bytes(sample_bytes)
    metrics_bytes = (json.dumps(metrics, indent=2, sort_keys=True) + "\n").encode()
    (_CORE / "metrics.json").write_bytes(metrics_bytes)
    environment = {
        "protocol": core.PROTOCOL_NAME,
        "code_commit": "800b3e8c2b0ea93b18dd82ff1e60b3eebce30678",
        "evaluator_commit_base": "888d97ebf0e9a41d1cadc873d2ed34f1254dc4d0",
        "runtime": observed,
        "checkpoints": {
            "dinov2_sha256": core.DINOV2_SHA256,
            "dinov3_sha256": core.DINOV3_SHA256,
        },
        "final_test_images_opened": 0,
        "gpu_inference_completed": True,
    }
    environment_bytes = (json.dumps(environment, indent=2, sort_keys=True) + "\n").encode()
    (_CORE / "environment.json").write_bytes(environment_bytes)
    report_bytes = _report(metrics).encode()
    (_CORE / "comparison_report.md").write_bytes(report_bytes)
    checksums = {
        "sample_list_sha256": _sha_bytes(sample_bytes),
        "predictions_sha256": _sha_bytes(prediction_bytes),
        "metrics_sha256": _sha_bytes(metrics_bytes),
        "environment_sha256": _sha_bytes(environment_bytes),
        "report_sha256": _sha_bytes(report_bytes),
        "prediction_rows": len(rows),
        "sweep_images": len(sample_ids),
        "final_test_images_opened": 0,
    }
    (_CORE / "checksums.json").write_text(json.dumps(checksums, indent=2, sort_keys=True) + "\n")
    recomputed = _subset_metrics(rows, lambda row: row["role"] == "primary_validation")
    if recomputed["models"][core.DINOV2_ID]["successful"] != metrics["primary_validation"]["models"][core.DINOV2_ID]["successful"]:
        raise core.ComparisonGuardError("report totals do not match the prediction manifest")
    if recomputed["agreement"] != metrics["primary_validation"]["agreement"]:
        raise core.ComparisonGuardError("agreement totals do not match the prediction manifest")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parity-only", action="store_true")
    args = parser.parse_args()
    protocol = _load_protocol()
    _verify_identities(protocol)
    observed = _runtime_guard(protocol)
    _records, blocked = _membership(protocol)
    parity = _parity(blocked)
    print(json.dumps({"parity": parity}), flush=True)
    if args.parity_only:
        return 0
    rows, resources = _sweep(_records, blocked)
    _verify_identities(protocol)
    metrics = _metrics(rows, parity, resources)
    _write_outputs(rows, metrics, observed)
    print(json.dumps({"data_readiness": metrics["data_readiness"], "winner": metrics["winner"], "rows": len(rows)}), flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except core.ComparisonGuardError as exc:
        print(f"STOP {exc}", file=sys.stderr)
        raise SystemExit(2)
