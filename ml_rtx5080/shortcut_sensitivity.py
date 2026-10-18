"""Phase 52: sensitivity of the frozen DINOv3 Triton path to encoding, compression, resize, and padding.

Register the protocol and its SHA-256 before sampling or inference. The runner
refuses to open an image until that hash matches. Folder names are strata, not
verified authenticity labels. This module does not train, does not edit labels,
and does not open the locked final test.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

_PKG = Path(__file__).resolve().parent
_REPO = _PKG.parent
for path in (_PKG, _REPO / "backend"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

OUT = _PKG / "experiments" / "shortcut_sensitivity_v1"
PROTOCOL_PATH = OUT / "shortcut_sensitivity_protocol_v1.json"
PROTOCOL_SHA_PATH = OUT / "shortcut_sensitivity_protocol_v1.sha256"
MANIFEST_PATH = OUT / "sample_manifest.json"
RESULTS_CSV = OUT / "sensitivity_results.csv"
SUMMARY_PATH = OUT / "sensitivity_summary.json"
REPORT_PATH = OUT / "phase52_report.md"
SPLIT_MANIFEST = _PKG / "experiments" / "dataset_audit" / "split_manifest_v2.json"
CHECKPOINT = _PKG / "experiments" / "v2_dinov3_cls_patch_attention" / "epoch_018.pt"

FROZEN_SHA = "5a38c93fd442b03653c65d2a5ecc9c2687ef152f7c5c020763e4ce1fd9c7d28f"
FROZEN_TEMPERATURE = 0.24038200410185356
THRESHOLD = 0.5
MODEL_NAME = "dinov3_authenticity_candidate"
MODEL_VERSION = "1"
ELIGIBLE_SPLITS = ("train", "calibration", "validation")
BRANDS = (
    "A. Lange & Söhne",
    "Audemars Piguet",
    "Patek Philippe",
    "Richard Mille",
    "Vacheron Constantin",
)
MEAN_FILL = (124, 116, 104)
PER_CELL = 4
SEED = 52

VARIANTS = (
    {"id": "jpeg_q95", "family": "encoding", "jpeg_quality": 95},
    {"id": "png", "family": "encoding"},
    {"id": "webp_q90", "family": "encoding", "webp_quality": 90},
    {"id": "jpeg_q75", "family": "compression", "jpeg_quality": 75},
    {"id": "jpeg_q50", "family": "compression", "jpeg_quality": 50},
    {"id": "jpeg_q30", "family": "compression", "jpeg_quality": 30},
    {"id": "long_side_256", "family": "resizing", "long_side": 256},
    {"id": "stretch_512", "family": "resizing", "size": 512},
    {"id": "pad_square_black", "family": "padding", "fill": [0, 0, 0]},
    {"id": "pad_square_white", "family": "padding", "fill": [255, 255, 255]},
    {"id": "pad_border_32_imagenet_mean", "family": "padding", "border": 32, "fill": list(MEAN_FILL)},
)


def protocol_document() -> dict:
    return {
        "phase": 52,
        "name": "dinov3_shortcut_sensitivity_v1",
        "question": (
            "On eligible non-test catalog images, how much do the frozen DINOv3 raw logit, "
            "calibrated P(fake), and threshold decision change when only encoding, JPEG compression, "
            "resize, or padding is changed before the frozen resize_pad_square_eval_v1 preprocess?"
        ),
        "not_a_claim": (
            "A measured shift is sensitivity of this serving path. It is not evidence that the network "
            "learned JPEG tables, 512-square geometry, or padding as an authenticity feature, and it is "
            "not a cross-brand or real-world authenticity result. Folder labels stay unverified."
        ),
        "frozen": {
            "checkpoint_sha256": FROZEN_SHA,
            "temperature": FROZEN_TEMPERATURE,
            "threshold": THRESHOLD,
            "probability": "P(fake) = sigmoid(logit / temperature). Threshold 0.50: P(fake) >= 0.50 is FAKE, otherwise AUTHENTIC.",
            "primary_decision": "threshold only. Quality flags do not move the primary decision to REVIEW.",
            "secondary_decision": "shadow policy, where LOW_RESOLUTION or LOW_CONTRAST forces REVIEW. Downscales can trip that rule by definition.",
            "preprocessing": "resize_pad_square_eval_v1 via preprocess_serving_image after the variant is applied",
            "triton_model": MODEL_NAME,
            "triton_version": MODEL_VERSION,
            "precision": "KIND_GPU on GPU 0 with use_tf32=0. Any other serving config aborts before images are opened.",
            "no_checkpoint_change": True,
            "no_temperature_change": True,
            "no_threshold_change": True,
            "no_label_change": True,
            "no_production_route_change": True,
            "final_test_unopened": True,
            "no_training": True,
        },
        "population": {
            "manifest": "ml_rtx5080/experiments/dataset_audit/split_manifest_v2.json",
            "eligible_splits": list(ELIGIBLE_SPLITS),
            "excluded_splits": ["test"],
            "seed": SEED,
            "per_cell": PER_CELL,
            "brands": list(BRANDS),
            "folder_label_rule": "Label_0_Watches -> folder_authentic; Label_1_Watches -> folder_fake. This is the directory name, not a verified class.",
            "draw": (
                "Sort eligible paths within each brand x folder_label cell. Shuffle with random.Random(52) "
                "in brand-then-label order, one generator. Keep the first 4 files that decode as RGB. "
                "A decode failure takes the next file in that cell only. Do not borrow from the test split. "
                "A short cell is recorded and not filled."
            ),
            "secondary_stratum": "source_file_is_512_square, measured from the file before any variant. Descriptive only.",
        },
        "variants": list(VARIANTS),
        "variant_rules": {
            "decode": "PIL convert RGB. Variants start from that image, not from a second catalog file.",
            "jpeg": "quality as specified, subsampling 0, progressive false, optimize false, then decode RGB",
            "png": "compress_level 6, then decode RGB",
            "webp": "quality 90, method 4, then decode RGB",
            "long_side_256": "bicubic; the long side is forced to 256; no codec round-trip",
            "stretch_512": "bicubic resize to 512x512, aspect not preserved, no codec round-trip",
            "pad_square": "center the image on a square canvas of side max(width, height). Already-square images are unchanged.",
            "pad_border_32": "32 px ImageNet-mean border (124, 116, 104) on every side",
            "original_repeat": "second Triton call on the same preprocessed original tensor. Not a content variant.",
        },
        "endpoints": {
            "delta_logit": "variant logit minus original logit",
            "delta_p_fake": "variant P(fake) minus original P(fake)",
            "threshold_flip": "primary decision differs from the original",
            "policy_flip": "secondary shadow decision differs from the original",
            "near_threshold_slice": "original abs(P(fake) - 0.50) < 0.05, pre-specified, not a filter that drops rows from the main table",
            "pixel_changed": "size differs, or max absolute RGB difference is greater than 0",
            "largest_changes": "15 content-variant rows with the largest absolute delta_p_fake. The repeat row is excluded.",
        },
        "abort_if": [
            "protocol file hash does not match the registered sha256",
            "checkpoint sha256 differs from the frozen digest",
            "Triton model is not READY, not KIND_GPU, not GPU 0, or use_tf32 is not 0",
            "a selected path resolves into the test membership",
        ],
    }


def canonical_protocol_bytes() -> bytes:
    return (json.dumps(protocol_document(), indent=2, sort_keys=True) + "\n").encode("utf-8")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def freeze_protocol() -> str:
    """Write the protocol once. Refuses to replace a hash that already exists."""
    OUT.mkdir(parents=True, exist_ok=True)
    payload = canonical_protocol_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    if PROTOCOL_SHA_PATH.is_file():
        registered = PROTOCOL_SHA_PATH.read_text().split()[0]
        if registered != digest or PROTOCOL_PATH.read_bytes() != payload:
            raise RuntimeError("protocol is already registered and does not match this document")
        return digest
    if PROTOCOL_PATH.is_file() and PROTOCOL_PATH.read_bytes() != payload:
        raise RuntimeError("protocol file exists without a matching registration")
    PROTOCOL_PATH.write_bytes(payload)
    PROTOCOL_SHA_PATH.write_text(f"{digest}  {PROTOCOL_PATH.name}\n")
    return digest


def require_registered_protocol() -> str:
    if not PROTOCOL_SHA_PATH.is_file() or not PROTOCOL_PATH.is_file():
        raise RuntimeError("protocol is not registered; freeze it before sampling or inference")
    digest = hashlib.sha256(PROTOCOL_PATH.read_bytes()).hexdigest()
    registered = PROTOCOL_SHA_PATH.read_text().split()[0]
    if digest != registered or PROTOCOL_PATH.read_bytes() != canonical_protocol_bytes():
        raise RuntimeError("protocol hash does not match the registered document")
    return digest


def _folder_label(path: Path) -> str | None:
    parts = set(path.parts)
    if "Label_0_Watches" in parts:
        return "folder_authentic"
    if "Label_1_Watches" in parts:
        return "folder_fake"
    return None


def eligible_records(membership: dict) -> list[dict]:
    test = {str(Path(path).resolve()) for path in membership.get("test") or []}
    records = []
    for split in ELIGIBLE_SPLITS:
        for raw in membership.get(split) or []:
            path = Path(raw).resolve()
            if str(path) in test:
                raise RuntimeError(f"eligible split contains a final-test path: {path}")
            label = _folder_label(path)
            brand = path.parent.name
            if label is None or brand not in BRANDS:
                continue
            records.append(
                {
                    "path": path,
                    "split": split,
                    "brand": brand,
                    "folder_label": label,
                }
            )
    return records


def select_sample(membership: dict, *, opener=None) -> list[dict]:
    """Draw the pre-registered sample. opener(path) -> PIL image or raises. Default is a real decode."""
    require_registered_protocol()
    grouped: dict[tuple[str, str], list[dict]] = {}
    for record in eligible_records(membership):
        grouped.setdefault((record["brand"], record["folder_label"]), []).append(record)
    rng = random.Random(SEED)
    selected = []
    for brand in BRANDS:
        for label in ("folder_authentic", "folder_fake"):
            cell = sorted(grouped.get((brand, label), []), key=lambda row: str(row["path"]))
            rng.shuffle(cell)
            kept = 0
            skipped = 0
            for record in cell:
                if kept >= PER_CELL:
                    break
                if opener is None:
                    try:
                        with Image.open(record["path"]) as handle:
                            handle.convert("RGB").load()
                    except Exception:
                        skipped += 1
                        continue
                else:
                    try:
                        opener(record["path"])
                    except Exception:
                        skipped += 1
                        continue
                selected.append({**record, "path": str(record["path"]), "decode_skips_before_keep": skipped})
                kept += 1
                skipped = 0
            if kept < PER_CELL:
                selected.append(
                    {
                        "path": None,
                        "split": None,
                        "brand": brand,
                        "folder_label": label,
                        "shortfall": PER_CELL - kept,
                    }
                )
    return selected


def _codec_roundtrip(image: Image.Image, *, fmt: str, **save_kwargs) -> Image.Image:
    buffer = io.BytesIO()
    image.save(buffer, format=fmt, **save_kwargs)
    buffer.seek(0)
    with Image.open(buffer) as handle:
        rgb = handle.convert("RGB")
        rgb.load()
        return rgb.copy()


def _long_side(image: Image.Image, side: int) -> Image.Image:
    width, height = image.size
    if width >= height:
        resized_w = side
        resized_h = max(1, int(round(height * (side / float(width)))))
    else:
        resized_h = side
        resized_w = max(1, int(round(width * (side / float(height)))))
    return image.resize((resized_w, resized_h), Image.Resampling.BICUBIC)


def apply_variant(image: Image.Image, variant_id: str) -> Image.Image:
    rgb = image.convert("RGB")
    if variant_id == "jpeg_q95":
        return _codec_roundtrip(rgb, fmt="JPEG", quality=95, subsampling=0, progressive=False, optimize=False)
    if variant_id == "png":
        return _codec_roundtrip(rgb, fmt="PNG", compress_level=6)
    if variant_id == "webp_q90":
        return _codec_roundtrip(rgb, fmt="WEBP", quality=90, method=4)
    if variant_id == "jpeg_q75":
        return _codec_roundtrip(rgb, fmt="JPEG", quality=75, subsampling=0, progressive=False, optimize=False)
    if variant_id == "jpeg_q50":
        return _codec_roundtrip(rgb, fmt="JPEG", quality=50, subsampling=0, progressive=False, optimize=False)
    if variant_id == "jpeg_q30":
        return _codec_roundtrip(rgb, fmt="JPEG", quality=30, subsampling=0, progressive=False, optimize=False)
    if variant_id == "long_side_256":
        return _long_side(rgb, 256)
    if variant_id == "stretch_512":
        return rgb.resize((512, 512), Image.Resampling.BICUBIC)
    if variant_id == "pad_square_black":
        return _pad_square(rgb, (0, 0, 0))
    if variant_id == "pad_square_white":
        return _pad_square(rgb, (255, 255, 255))
    if variant_id == "pad_border_32_imagenet_mean":
        width, height = rgb.size
        canvas = Image.new("RGB", (width + 64, height + 64), MEAN_FILL)
        canvas.paste(rgb, (32, 32))
        return canvas
    raise KeyError(variant_id)


def _pad_square(image: Image.Image, fill: tuple[int, int, int]) -> Image.Image:
    width, height = image.size
    side = max(width, height)
    if (width, height) == (side, side):
        return image.copy()
    canvas = Image.new("RGB", (side, side), fill)
    canvas.paste(image, ((side - width) // 2, (side - height) // 2))
    return canvas


def pixels_changed(original: Image.Image, variant: Image.Image) -> bool:
    if original.size != variant.size:
        return True
    left = np.asarray(original.convert("RGB"))
    right = np.asarray(variant.convert("RGB"))
    return bool(np.any(left != right))


def _serving_config(client) -> dict:
    cfg = client.get_model_config(MODEL_NAME, MODEL_VERSION, as_json=True)["config"]
    groups = cfg.get("instance_group") or []
    kinds = sorted({group.get("kind") for group in groups})
    gpus = sorted({int(gpu) for group in groups for gpu in (group.get("gpus") or [0])})
    cuda: dict[str, str] = {}
    accelerators = (cfg.get("optimization") or {}).get("execution_accelerators") or {}
    for accelerator in accelerators.get("gpu_execution_accelerator") or []:
        if accelerator.get("name") == "cuda":
            cuda.update({str(key): str(value) for key, value in (accelerator.get("parameters") or {}).items()})
    return {"kinds": kinds, "gpus": gpus, "use_tf32": cuda.get("use_tf32"), "backend": cfg.get("backend")}


def require_gpu_triton(url: str) -> dict:
    import tritonclient.grpc as grpcclient

    client = grpcclient.InferenceServerClient(url=url)
    if not client.is_model_ready(MODEL_NAME, MODEL_VERSION):
        raise RuntimeError(f"{MODEL_NAME} is not READY")
    profile = _serving_config(client)
    if profile["kinds"] != ["KIND_GPU"] or profile["gpus"] != [0] or profile["use_tf32"] != "0":
        raise RuntimeError(f"refusing to score without exact-FP32 GPU 0: {profile}")
    return {"client": client, "grpc": grpcclient, "profile": profile}


def infer_logit(client, grpc, array: np.ndarray) -> float:
    inputs = [grpc.InferInput("input__0", list(array.shape), "FP32")]
    inputs[0].set_data_from_numpy(array)
    output = client.infer(MODEL_NAME, inputs, model_version=MODEL_VERSION, client_timeout=60.0).as_numpy("output__0")
    value = float(np.asarray(output).reshape(-1)[0])
    if not np.isfinite(value):
        raise RuntimeError("Triton returned a non-finite logit")
    return value


def _score_pair(original_logit: float, original_decision: str, original_policy: str, logit: float) -> dict:
    from policy import authenticity_only_decision, calibrated_probability_from_logit, evaluate_shadow_policy

    probability = calibrated_probability_from_logit(logit)
    decision = authenticity_only_decision(probability, None)
    return {
        "logit": logit,
        "p_fake": probability,
        "threshold_decision": decision,
        "delta_logit": logit - original_logit,
        "delta_p_fake": probability - calibrated_probability_from_logit(original_logit),
        "threshold_flip": decision != original_decision,
        "policy_decision": original_policy,
        "policy_flip": False,
    }


def _policy_fields(image: Image.Image, probability: float) -> dict:
    from policy import evaluate_shadow_policy
    from reference_inference import quality_flags_for_image

    _features, flags = quality_flags_for_image(image)
    decision = evaluate_shadow_policy(probability=probability, quality_flags=flags)
    return {"policy_decision": decision["decision"], "quality_flags": list(flags)}


def run(url: str) -> dict:
    digest = require_registered_protocol()
    if RESULTS_CSV.is_file() or SUMMARY_PATH.is_file():
        raise RuntimeError("results already exist; refusing to rerun and replace them")
    checkpoint = file_sha256(CHECKPOINT)
    if checkpoint != FROZEN_SHA:
        raise RuntimeError("checkpoint hash does not match the frozen artifact; no images were scored")
    triton = require_gpu_triton(url)
    membership = json.loads(SPLIT_MANIFEST.read_text())["membership"]
    test_paths = {str(Path(path).resolve()) for path in membership.get("test") or []}
    drawn = select_sample(membership)
    images = [row for row in drawn if row.get("path")]
    shortfalls = [row for row in drawn if row.get("shortfall")]
    for row in images:
        if str(Path(row["path"]).resolve()) in test_paths:
            raise RuntimeError("refusing to open a final-test image")
    MANIFEST_PATH.write_text(
        json.dumps(
            {
                "protocol_sha256": digest,
                "checkpoint_sha256": checkpoint,
                "triton": triton["profile"],
                "selected": images,
                "shortfalls": shortfalls,
            },
            indent=2,
        )
        + "\n"
    )

    from dinov3_serving import preprocess_serving_image
    from policy import authenticity_only_decision, calibrated_probability_from_logit

    rows: list[dict] = []
    for index, record in enumerate(images, start=1):
        with Image.open(record["path"]) as handle:
            original = handle.convert("RGB")
            original.load()
            original = original.copy()
        width, height = original.size
        tensor = preprocess_serving_image(original).unsqueeze(0).numpy().astype(np.float32)
        logit = infer_logit(triton["client"], triton["grpc"], tensor)
        probability = calibrated_probability_from_logit(logit)
        decision = authenticity_only_decision(probability, None)
        policy = _policy_fields(original, probability)
        base = {
            "sample_id": str(Path(record["path"]).relative_to(_REPO)) if str(record["path"]).startswith(str(_REPO)) else record["path"],
            "split": record["split"],
            "brand": record["brand"],
            "folder_label": record["folder_label"],
            "source_file_is_512_square": width == 512 and height == 512,
            "source_width": width,
            "source_height": height,
        }
        original_row = {
            **base,
            "variant": "original",
            "family": "reference",
            "pixel_changed": False,
            "logit": logit,
            "p_fake": probability,
            "threshold_decision": decision,
            "delta_logit": 0.0,
            "delta_p_fake": 0.0,
            "threshold_flip": False,
            **policy,
            "policy_flip": False,
            "near_threshold": abs(probability - THRESHOLD) < 0.05,
        }
        rows.append(original_row)
        repeat = infer_logit(triton["client"], triton["grpc"], tensor)
        repeat_score = _score_pair(logit, decision, policy["policy_decision"], repeat)
        repeat_score["policy_decision"] = policy["policy_decision"]
        rows.append({**base, "variant": "original_repeat", "family": "determinism", "pixel_changed": False, "quality_flags": policy["quality_flags"], "near_threshold": original_row["near_threshold"], **repeat_score})
        for spec in VARIANTS:
            variant_image = apply_variant(original, spec["id"])
            changed = pixels_changed(original, variant_image)
            variant_tensor = preprocess_serving_image(variant_image).unsqueeze(0).numpy().astype(np.float32)
            variant_logit = infer_logit(triton["client"], triton["grpc"], variant_tensor)
            scored = _score_pair(logit, decision, policy["policy_decision"], variant_logit)
            variant_policy = _policy_fields(variant_image, scored["p_fake"])
            scored["policy_decision"] = variant_policy["policy_decision"]
            scored["policy_flip"] = variant_policy["policy_decision"] != policy["policy_decision"]
            rows.append(
                {
                    **base,
                    "variant": spec["id"],
                    "family": spec["family"],
                    "pixel_changed": changed,
                    "quality_flags": variant_policy["quality_flags"],
                    "near_threshold": original_row["near_threshold"],
                    **scored,
                }
            )
        print(f"scored {index}/{len(images)} {base['sample_id']}", flush=True)

    _write_csv(rows)
    summary = _summarize(rows, digest, checkpoint, triton["profile"], len(images), shortfalls)
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2) + "\n")
    REPORT_PATH.write_text(_report(summary))
    return summary


def _write_csv(rows: list[dict]) -> None:
    fields = [
        "sample_id",
        "split",
        "brand",
        "folder_label",
        "source_file_is_512_square",
        "source_width",
        "source_height",
        "variant",
        "family",
        "pixel_changed",
        "logit",
        "p_fake",
        "threshold_decision",
        "delta_logit",
        "delta_p_fake",
        "threshold_flip",
        "policy_decision",
        "policy_flip",
        "near_threshold",
        "quality_flags",
    ]
    with RESULTS_CSV.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            payload = {key: row.get(key) for key in fields}
            payload["quality_flags"] = "|".join(row.get("quality_flags") or [])
            writer.writerow(payload)


def _variant_summary(rows: list[dict]) -> dict:
    deltas = [abs(float(row["delta_p_fake"])) for row in rows]
    flips = [row for row in rows if row["threshold_flip"]]
    near = [row for row in rows if row["near_threshold"]]
    changed = [row for row in rows if row["pixel_changed"]]
    ordered = sorted(deltas)
    return {
        "n": len(rows),
        "pixel_changed_n": len(changed),
        "mean_abs_delta_p_fake": float(np.mean(deltas)) if deltas else None,
        "median_abs_delta_p_fake": float(np.median(deltas)) if deltas else None,
        "max_abs_delta_p_fake": float(max(deltas)) if deltas else None,
        "max_abs_delta_logit": float(max(abs(float(row["delta_logit"])) for row in rows)) if rows else None,
        "threshold_flips": len(flips),
        "threshold_flip_rate": len(flips) / len(rows) if rows else None,
        "threshold_flips_near_0.50": sum(1 for row in flips if row["near_threshold"]),
        "near_threshold_n": len(near),
        "policy_flips": sum(1 for row in rows if row["policy_flip"]),
        "mean_abs_delta_p_fake_when_pixels_changed": float(np.mean([abs(float(row["delta_p_fake"])) for row in changed])) if changed else None,
    }


def _summarize(rows, digest, checkpoint, profile, n_images, shortfalls) -> dict:
    content = [row for row in rows if row["family"] not in {"reference", "determinism"}]
    by_variant = {}
    for spec in VARIANTS:
        by_variant[spec["id"]] = _variant_summary([row for row in content if row["variant"] == spec["id"]])
        by_variant[spec["id"]]["family"] = spec["family"]
    repeat = [row for row in rows if row["variant"] == "original_repeat"]
    largest = sorted(content, key=lambda row: abs(float(row["delta_p_fake"])), reverse=True)[:15]
    return {
        "protocol_sha256": digest,
        "checkpoint_sha256": checkpoint,
        "temperature": FROZEN_TEMPERATURE,
        "threshold": THRESHOLD,
        "triton": profile,
        "images": n_images,
        "shortfalls": shortfalls,
        "original_repeat": _variant_summary(repeat),
        "by_variant": by_variant,
        "by_family": {
            family: _variant_summary([row for row in content if row["family"] == family])
            for family in ("encoding", "compression", "resizing", "padding")
        },
        "by_folder_label": {
            label: {
                variant: _variant_summary(
                    [row for row in content if row["variant"] == variant and row["folder_label"] == label]
                )
                for variant in [spec["id"] for spec in VARIANTS]
            }
            for label in ("folder_authentic", "folder_fake")
        },
        "by_512_square": {
            str(flag).lower(): {
                "stretch_512": _variant_summary(
                    [row for row in content if row["variant"] == "stretch_512" and row["source_file_is_512_square"] is flag]
                ),
                "jpeg_q30": _variant_summary(
                    [row for row in content if row["variant"] == "jpeg_q30" and row["source_file_is_512_square"] is flag]
                ),
            }
            for flag in (True, False)
        },
        "largest_abs_delta_p_fake": [
            {
                "sample_id": row["sample_id"],
                "brand": row["brand"],
                "folder_label": row["folder_label"],
                "variant": row["variant"],
                "family": row["family"],
                "delta_p_fake": row["delta_p_fake"],
                "delta_logit": row["delta_logit"],
                "original_near_threshold": row["near_threshold"],
                "threshold_flip": row["threshold_flip"],
                "policy_flip": row["policy_flip"],
                "source_file_is_512_square": row["source_file_is_512_square"],
            }
            for row in largest
        ],
        "interpretation_limit": protocol_document()["not_a_claim"],
    }


def _fmt(value, digits=4) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _report(summary: dict) -> str:
    lines = [
        "# Phase 52 — DINOv3 shortcut sensitivity",
        "",
        "Measured sensitivity of the frozen GPU Triton DINOv3 path to controlled transforms.",
        "Folder labels are directory names. They are not verified authenticity.",
        "",
        f"Protocol SHA-256 `{summary['protocol_sha256']}`.",
        "",
        f"Checkpoint `{summary['checkpoint_sha256']}`. Temperature `{summary['temperature']}`. Threshold `{summary['threshold']}`.",
        "",
        f"Triton `{MODEL_NAME}` v1: `{json.dumps(summary['triton'])}`.",
        "",
        f"Images scored: {summary['images']}. Short cells: {len(summary['shortfalls'])}.",
        "",
        "Primary decision is threshold-only P(fake) at 0.50. Policy flips include the quality gate and are secondary.",
        "A shift is not evidence of what the training procedure learned.",
        "",
        "## Original repeat",
        "",
        f"Same tensor scored twice. Max abs delta P(fake) `{_fmt(summary['original_repeat']['max_abs_delta_p_fake'], 8)}`. Threshold flips `{summary['original_repeat']['threshold_flips']}`.",
        "",
        "## Variants",
        "",
        "| Variant | Family | n | Mean abs ΔP | Median abs ΔP | Max abs ΔP | Threshold flips | Policy flips | Pixels changed |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for spec in VARIANTS:
        row = summary["by_variant"][spec["id"]]
        lines.append(
            f"| `{spec['id']}` | {spec['family']} | {row['n']} | {_fmt(row['mean_abs_delta_p_fake'])} | {_fmt(row['median_abs_delta_p_fake'])} | {_fmt(row['max_abs_delta_p_fake'])} | {row['threshold_flips']} | {row['policy_flips']} | {row['pixel_changed_n']} |"
        )
    lines.extend(["", "## Largest absolute P(fake) changes", ""])
    for row in summary["largest_abs_delta_p_fake"]:
        lines.append(
            f"- `{row['variant']}` on `{row['sample_id']}` ({row['brand']}, {row['folder_label']}): "
            f"ΔP(fake) `{row['delta_p_fake']:.6f}`, Δlogit `{row['delta_logit']:.6f}`, "
            f"threshold flip `{row['threshold_flip']}`, near 0.50 `{row['original_near_threshold']}`, "
            f"512-square file `{row['source_file_is_512_square']}`."
        )
    lines.extend(
        [
            "",
            "## What this does not show",
            "",
            summary["interpretation_limit"],
            "",
            "`MODEL_CHANGED = false`",
            "",
            "`CALIBRATION_CHANGED = false`",
            "",
            "`FINAL_TEST_TOUCHED = false`",
            "",
            "`DATASET_READY_FOR_TRAINING = false`",
            "",
            "`PRODUCTION_PROMOTION_ALLOWED = false`",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze-protocol", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--url", default="localhost:18001")
    args = parser.parse_args()
    if args.freeze_protocol:
        print(freeze_protocol())
        return 0
    if args.execute:
        started = time.perf_counter()
        summary = run(args.url)
        print(json.dumps({"images": summary["images"], "seconds": time.perf_counter() - started, "repeat_max_abs_dp": summary["original_repeat"]["max_abs_delta_p_fake"]}, indent=2))
        return 0
    parser.error("choose --freeze-protocol or --execute")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
