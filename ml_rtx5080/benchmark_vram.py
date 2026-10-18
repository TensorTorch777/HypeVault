"""RTX training-step VRAM and throughput sweep.

This is not a training run. Each case loads the trainer's pretrained model,
runs the conservative training transform on a few catalog images, and times
forward, BCE-with-logits, backward, and one AdamW step.

Results are written only under ``ml_rtx5080/experiments/benchmarks``.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torch.amp import GradScaler

_PKG = Path(__file__).resolve().parent
if str(_PKG) not in sys.path:
    sys.path.insert(0, str(_PKG))

from augmentations import CONSERVATIVE, build_transforms, preprocessing_record
from classifier import build_authenticity_classifier
from dataset import load_records
from train import (
    DEFAULTS,
    apply_accumulated_step,
    autocast_for,
    build_optimizer,
    cosine_schedule_with_warmup,
)
from training_config import (
    FAMILY_TIMM_NAME,
    assert_experiment_output_dir,
    build_bce_loss,
    resolve_amp,
    smooth_binary_targets,
)

BENCHMARK_VERSION = "phase10-vram-v2"
DEFAULT_BATCH_SIZES = (2, 4, 6, 8, 10, 12, 16)
TARGET_EFFECTIVE_BATCH = 32
PREFERRED_HEADROOM = 0.15
MINIMUM_HEADROOM = 0.10
MEASURED_STEPS = 2
CASES = (
    ("dinov2", "cls_only"),
    ("dinov2", "cls_patch_attention"),
    ("dinov3", "cls_only"),
    ("dinov3", "cls_patch_attention"),
)
OPTIMIZER_KEYS = ("backbone_lr", "head_lr", "layer_decay", "weight_decay")


def select_representative_records(records: list, count: int) -> list:
    """Even stride across a read-only catalog. Does not write a split."""
    if count < 1:
        raise ValueError(f"count must be >= 1, got {count}")
    if len(records) < count:
        raise ValueError(f"Need at least {count} catalog images, found {len(records)}")
    return [records[int(index * len(records) / count)] for index in range(count)]


def effective_batch(microbatch: int, accumulate_grad: int) -> int:
    if microbatch < 1 or accumulate_grad < 1:
        raise ValueError("microbatch and accumulate_grad must be >= 1")
    return int(microbatch) * int(accumulate_grad)


def accumulation_for_target(microbatch: int, target: int = TARGET_EFFECTIVE_BATCH) -> dict:
    """Pick an accumulation count whose product is closest to ``target``."""
    if microbatch < 1 or target < 1:
        raise ValueError("microbatch and target must be >= 1")
    accumulate_grad = min(
        range(1, target + 1),
        key=lambda steps: (abs(microbatch * steps - target), steps),
    )
    return {
        "microbatch": int(microbatch),
        "accumulate_grad": int(accumulate_grad),
        "effective_batch": effective_batch(microbatch, accumulate_grad),
        "target_effective_batch": int(target),
    }


def validate_measurement(result: dict) -> dict:
    """Reject a row that cannot be turned into a benchmark table."""
    status = result.get("status")
    if status not in {"FIT", "OOM"}:
        raise ValueError(f"status must be FIT or OOM, got {status!r}")
    batch_size = result.get("batch_size")
    if not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError(f"batch_size must be a positive int, got {batch_size!r}")
    if status == "OOM":
        if not str(result.get("error") or "").strip():
            raise ValueError("OOM result requires an error string")
        return result
    for key in ("peak_allocated_bytes", "peak_reserved_bytes", "step_time_seconds", "images_per_sec"):
        value = result.get(key)
        if not isinstance(value, (int, float)) or value < 0:
            raise ValueError(f"FIT result needs a non-negative {key}")
    return result


def _within_headroom(result: dict, total_vram_bytes: int, baseline_free_bytes: int, headroom: float) -> bool:
    limit = 1.0 - headroom
    if result["peak_allocated_bytes"] > total_vram_bytes * limit:
        return False
    if result["peak_reserved_bytes"] > baseline_free_bytes * limit:
        return False
    return True


def recommend_microbatch(
    results: list[dict],
    *,
    total_vram_bytes: int,
    baseline_free_bytes: int,
    preferred_headroom: float = PREFERRED_HEADROOM,
    minimum_headroom: float = MINIMUM_HEADROOM,
) -> dict | None:
    """Largest FIT batch that keeps the preferred headroom, then the minimum headroom.

    A batch that merely fits in whatever memory happens to be free is not selected.
    Peak reserved memory must also leave that fraction of the pre-run free memory.
    """
    if total_vram_bytes < 1 or baseline_free_bytes < 1:
        raise ValueError("total and baseline free memory must be positive")
    fits = []
    for result in results:
        checked = validate_measurement(result)
        if checked["status"] == "FIT":
            fits.append(checked)
    for headroom in (preferred_headroom, minimum_headroom):
        eligible = [
            result
            for result in fits
            if _within_headroom(result, total_vram_bytes, baseline_free_bytes, headroom)
        ]
        if not eligible:
            continue
        chosen = max(eligible, key=lambda result: result["batch_size"])
        return {
            "batch_size": chosen["batch_size"],
            "headroom_rule": headroom,
            "peak_allocated_bytes": chosen["peak_allocated_bytes"],
            "peak_reserved_bytes": chosen["peak_reserved_bytes"],
            "headroom_bytes": int(total_vram_bytes - chosen["peak_allocated_bytes"]),
            "headroom_fraction": (total_vram_bytes - chosen["peak_allocated_bytes"]) / total_vram_bytes,
        }
    return None


def recommend_training_config(
    case_results: dict[tuple[str, str], list[dict]],
    *,
    total_vram_bytes: int,
    baseline_free_bytes: int,
    target_effective_batch: int = TARGET_EFFECTIVE_BATCH,
) -> dict:
    """One microbatch that is safe for every measured architecture, plus accumulation."""
    per_case = {}
    safe_batches = []
    for key, results in case_results.items():
        choice = recommend_microbatch(
            results,
            total_vram_bytes=total_vram_bytes,
            baseline_free_bytes=baseline_free_bytes,
        )
        family, head = key
        per_case[f"{family}:{head}"] = choice
        if choice is not None:
            safe_batches.append(choice["batch_size"])
    if not case_results or len(safe_batches) != len(case_results):
        return {
            "microbatch": None,
            "accumulate_grad": None,
            "effective_batch": None,
            "target_effective_batch": target_effective_batch,
            "per_case": per_case,
            "reason": "every architecture needs a batch inside the VRAM headroom",
        }
    microbatch = min(safe_batches)
    schedule = accumulation_for_target(microbatch, target_effective_batch)
    return {**schedule, "per_case": per_case, "reason": "minimum safe microbatch across architectures"}


def is_cuda_oom(exc: BaseException) -> bool:
    if not isinstance(exc, RuntimeError):
        return False
    return "out of memory" in str(exc).lower()


def measurement_row(result: dict, recommendation: dict | None) -> dict:
    validate_measurement(result)
    row = {
        "model_family": result.get("model_family"),
        "model_architecture": result.get("model_architecture"),
        "input_size": result.get("input_size"),
        "batch_size": result["batch_size"],
        "status": result["status"],
        "peak_allocated_bytes": result.get("peak_allocated_bytes"),
        "peak_reserved_bytes": result.get("peak_reserved_bytes"),
        "step_time_seconds": result.get("step_time_seconds"),
        "images_per_sec": result.get("images_per_sec"),
        "error": result.get("error"),
        "accumulate_grad": result.get("accumulate_grad", 1),
        "recommended_microbatch": None if not recommendation else recommendation.get("microbatch"),
        "recommended_accumulate_grad": None if not recommendation else recommendation.get("accumulate_grad"),
        "recommended_effective_batch": None if not recommendation else recommendation.get("effective_batch"),
    }
    return row


def write_benchmark_report(directory: Path, payload: dict, rows: list[dict]) -> tuple[Path, Path]:
    root = assert_experiment_output_dir(directory)
    root.mkdir(parents=True, exist_ok=True)
    stamp = payload["timestamp"].replace(":", "").replace("-", "").replace("+", "p")
    json_path = root / f"vram_benchmark_{stamp}.json"
    csv_path = root / f"vram_benchmark_{stamp}.csv"
    json_path.write_text(json.dumps(payload, indent=2) + "\n")
    fieldnames = list(rows[0].keys()) if rows else ["status"]
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return json_path, csv_path


def _gib(num_bytes: float | None) -> str:
    if num_bytes is None:
        return "-"
    return f"{num_bytes / (1024 ** 3):.2f} GiB"


def _release_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def _cuda_healthy() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        torch.cuda.synchronize()
        probe = torch.empty(1, device="cuda")
        del probe
        _release_cuda()
        return True
    except RuntimeError:
        return False


def _preprocess(records, family: str, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    transform = build_transforms(family, train=True, profile=CONSERVATIVE)
    random.seed(seed)
    torch.manual_seed(seed)
    images = []
    labels = []
    for record in records:
        with Image.open(record.image_path) as handle:
            image = handle.convert("RGB")
        images.append(transform(image))
        labels.append(int(record.label))
    batch = torch.stack(images)
    expected = batch.shape[-1]
    if batch.shape[-2] != expected:
        raise RuntimeError(f"{family} preprocess produced non-square tensors {tuple(batch.shape)}")
    return batch, torch.tensor(labels, dtype=torch.float32)


def _optimizer_cfg() -> dict:
    return {key: DEFAULTS[key] for key in OPTIMIZER_KEYS}


def _training_step(
    model: nn.Module,
    optimizer,
    scaler,
    scheduler,
    images: torch.Tensor,
    labels: torch.Tensor,
    *,
    amp_dtype: str,
    accumulate_grad: int,
) -> None:
    criterion = build_bce_loss()
    targets = smooth_binary_targets(labels, float(DEFAULTS["label_smoothing"]))
    for _ in range(accumulate_grad):
        with autocast_for(images.device, amp_dtype):
            logits = model(images)
            loss = criterion(logits, targets)
            loss = loss / accumulate_grad
        scaler.scale(loss).backward()
    apply_accumulated_step(model, optimizer, scaler, scheduler, float(DEFAULTS["grad_clip"]))


def _measure_loaded(
    model: nn.Module,
    optimizer,
    scaler,
    scheduler,
    images: torch.Tensor,
    labels: torch.Tensor,
    *,
    amp_dtype: str,
    batch_size: int,
    accumulate_grad: int,
    family: str,
    head: str,
    input_size: int,
) -> dict:
    device = images.device
    model.train()
    micro_images = images[:batch_size]
    micro_labels = labels[:batch_size]
    _training_step(
        model, optimizer, scaler, scheduler, micro_images, micro_labels,
        amp_dtype=amp_dtype, accumulate_grad=accumulate_grad,
    )
    torch.cuda.synchronize()
    # Drop cached blocks from the warmup step so reserved memory is this step's pool.
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    durations = []
    for _ in range(MEASURED_STEPS):
        torch.cuda.synchronize()
        started = time.perf_counter()
        _training_step(
            model, optimizer, scaler, scheduler, micro_images, micro_labels,
            amp_dtype=amp_dtype, accumulate_grad=accumulate_grad,
        )
        torch.cuda.synchronize()
        durations.append(time.perf_counter() - started)
    step_time = sum(durations) / len(durations)
    images_in_step = batch_size * accumulate_grad
    return validate_measurement(
        {
            "model_family": family,
            "model_architecture": head,
            "input_size": input_size,
            "batch_size": batch_size,
            "accumulate_grad": accumulate_grad,
            "status": "FIT",
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
            "step_time_seconds": step_time,
            "images_per_sec": images_in_step / step_time,
            "error": None,
        }
    )


def _oom_result(family: str, head: str, input_size: int, batch_size: int, exc: BaseException) -> dict:
    return validate_measurement(
        {
            "model_family": family,
            "model_architecture": head,
            "input_size": input_size,
            "batch_size": batch_size,
            "accumulate_grad": 1,
            "status": "OOM",
            "peak_allocated_bytes": None,
            "peak_reserved_bytes": None,
            "step_time_seconds": None,
            "images_per_sec": None,
            "error": str(exc).splitlines()[0][:500],
        }
    )


def _sweep_case(
    family: str,
    head: str,
    records,
    *,
    batch_sizes: tuple[int, ...],
    amp: dict,
    seed: int,
    device: torch.device,
) -> tuple[list[dict], dict]:
    input_size = preprocessing_record(family, CONSERVATIVE, head)["input_size"]
    cpu_images, cpu_labels = _preprocess(records, family, seed)
    if cpu_images.shape[-1] != input_size:
        raise RuntimeError(f"{family} tensors are {cpu_images.shape[-1]}px, expected {input_size}")
    try:
        model = build_authenticity_classifier(
            family, head, image_size=input_size, pretrained=True
        )
    except RuntimeError as exc:
        if is_cuda_oom(exc):
            raise
        return [], {"input_size": input_size, "free_after_bytes": None, "pretrained_error": str(exc)}
    except Exception as exc:
        print(f"  pretrained load failed for {family} {head}: {exc}", flush=True)
        return [], {"input_size": input_size, "free_after_bytes": None, "pretrained_error": str(exc)}
    model = model.to(device)
    optimizer, _groups, _table = build_optimizer(model, _optimizer_cfg())
    scheduler = cosine_schedule_with_warmup(optimizer, warmup_steps=0, total_steps=10_000)
    scaler = GradScaler("cuda", enabled=bool(amp["amp_scaler"]))
    gpu_images = cpu_images.to(device, non_blocking=True)
    gpu_labels = cpu_labels.to(device, non_blocking=True)
    del cpu_images
    results = []
    stopped = False
    for batch_size in batch_sizes:
        if stopped:
            break
        print(f"  {family} {head} microbatch {batch_size}", flush=True)
        try:
            measured = _measure_loaded(
                model, optimizer, scaler, scheduler, gpu_images, gpu_labels,
                amp_dtype=amp["amp_dtype"], batch_size=batch_size, accumulate_grad=1,
                family=family, head=head, input_size=input_size,
            )
        except RuntimeError as exc:
            if not is_cuda_oom(exc):
                raise
            print(f"    OOM at microbatch {batch_size}", flush=True)
            results.append(_oom_result(family, head, input_size, batch_size, exc))
            _release_cuda()
            stopped = True
            break
        results.append(measured)
        print(
            f"    FIT peak={_gib(measured['peak_allocated_bytes'])} "
            f"step={measured['step_time_seconds']:.3f}s "
            f"{measured['images_per_sec']:.1f} img/s",
            flush=True,
        )
    free_after = None
    del gpu_images, gpu_labels, scaler, scheduler, optimizer, model
    _release_cuda()
    if torch.cuda.is_available():
        free_after, _total = torch.cuda.mem_get_info()
    return results, {"input_size": input_size, "free_after_bytes": free_after}


def run_benchmark(
    data_root: Path,
    output_dir: Path,
    batch_sizes: tuple[int, ...] = DEFAULT_BATCH_SIZES,
    seed: int = 42,
) -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Refusing an FP32 stand-in for the RTX benchmark.")
    device = torch.device("cuda")
    bf16_supported = bool(torch.cuda.is_bf16_supported())
    amp = resolve_amp("bf16", cuda=True, bf16_supported=bf16_supported)
    if amp["amp_dtype"] == "fp32":
        raise RuntimeError("BF16 and FP16 are unavailable. Refusing an FP32 production measurement.")
    properties = torch.cuda.get_device_properties(0)
    free_before, driver_total = torch.cuda.mem_get_info()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    image_count = max(batch_sizes)
    records = select_representative_records(load_records(data_root), image_count)
    catalog_sample = [
        {"path": str(record.image_path), "label": int(record.label)} for record in records
    ]
    case_results: dict[tuple[str, str], list[dict]] = {}
    case_meta = []
    cuda_ok = True
    for family, head in CASES:
        if not cuda_ok:
            break
        print(f"\nCase {family} {head}", flush=True)
        try:
            results, meta = _sweep_case(
                family, head, records, batch_sizes=batch_sizes, amp=amp, seed=seed, device=device
            )
        except RuntimeError as exc:
            if not is_cuda_oom(exc):
                raise
            results = [_oom_result(family, head, preprocessing_record(family, CONSERVATIVE, head)["input_size"], batch_sizes[0], exc)]
            meta = {"input_size": results[0]["input_size"], "free_after_bytes": None, "pretrained_error": None}
            _release_cuda()
        case_results[(family, head)] = results
        case_meta.append({"model_family": family, "model_architecture": head, **meta, "results": results})
        if not _cuda_healthy():
            cuda_ok = False
            print("CUDA remained unhealthy after cleanup; stopping further cases.", flush=True)
    recommendation = recommend_training_config(
        case_results,
        total_vram_bytes=int(properties.total_memory),
        baseline_free_bytes=int(free_before),
    )
    confirmation = None
    if cuda_ok and recommendation.get("microbatch"):
        confirmation = _confirm_accumulation(
            records,
            recommendation,
            case_results,
            amp=amp,
            seed=seed,
            device=device,
        )
        recommendation["accumulation_confirmation"] = confirmation
    payload = {
        "benchmark_version": BENCHMARK_VERSION,
        "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
        "seed": seed,
        "gpu_name": properties.name,
        "total_vram_bytes": int(properties.total_memory),
        "driver_total_vram_bytes": int(driver_total),
        "baseline_free_bytes": int(free_before),
        "cuda_version": torch.version.cuda,
        "pytorch_version": torch.__version__,
        "cuda_available": True,
        "bf16_supported": bf16_supported,
        "precision": amp["amp_dtype"],
        "amp_scaler": amp["amp_scaler"],
        "amp_fallback": amp["amp_fallback"],
        "batch_sizes": list(batch_sizes),
        "measured_steps": MEASURED_STEPS,
        "loss": "BCEWithLogitsLoss",
        "label_smoothing": float(DEFAULTS["label_smoothing"]),
        "optimizer": {"name": "AdamW", **_optimizer_cfg()},
        "preprocessing_profile": CONSERVATIVE,
        "model_names": FAMILY_TIMM_NAME,
        "catalog_sample": catalog_sample,
        "cases": case_meta,
        "recommendation": recommendation,
        "cuda_healthy_at_end": cuda_ok,
    }
    rows = []
    for results in case_results.values():
        for result in results:
            rows.append(measurement_row(result, recommendation))
    if confirmation is not None:
        rows.append(measurement_row(confirmation, recommendation))
    json_path, csv_path = write_benchmark_report(output_dir, payload, rows)
    payload["json_path"] = str(json_path)
    payload["csv_path"] = str(csv_path)
    return payload


def _confirm_accumulation(records, recommendation, case_results, *, amp, seed, device) -> dict:
    """One extra step at the safe microbatch with accumulate_grad=2."""
    microbatch = int(recommendation["microbatch"])
    heaviest = None
    for (family, head), results in case_results.items():
        match = [
            result for result in results
            if result["status"] == "FIT" and result["batch_size"] == microbatch
        ]
        if not match:
            continue
        candidate = match[0]
        if heaviest is None or candidate["peak_allocated_bytes"] > heaviest["peak_allocated_bytes"]:
            heaviest = {**candidate, "model_family": family, "model_architecture": head}
    if heaviest is None:
        raise RuntimeError("accumulation confirmation could not find the recommended microbatch")
    family = heaviest["model_family"]
    head = heaviest["model_architecture"]
    print(f"\nAccumulation check {family} {head} microbatch {microbatch} x 2", flush=True)
    input_size = preprocessing_record(family, CONSERVATIVE, head)["input_size"]
    cpu_images, cpu_labels = _preprocess(records, family, seed)
    model = build_authenticity_classifier(family, head, image_size=input_size, pretrained=True).to(device)
    optimizer, _groups, _table = build_optimizer(model, _optimizer_cfg())
    scheduler = cosine_schedule_with_warmup(optimizer, warmup_steps=0, total_steps=10_000)
    scaler = GradScaler("cuda", enabled=bool(amp["amp_scaler"]))
    gpu_images = cpu_images.to(device)
    gpu_labels = cpu_labels.to(device)
    del cpu_images, cpu_labels
    measured = _measure_loaded(
        model, optimizer, scaler, scheduler, gpu_images, gpu_labels,
        amp_dtype=amp["amp_dtype"], batch_size=microbatch, accumulate_grad=2,
        family=family, head=head, input_size=input_size,
    )
    single_peak = heaviest["peak_allocated_bytes"]
    measured["comparison"] = "accumulate_grad_2"
    measured["single_step_peak_allocated_bytes"] = single_peak
    measured["allocated_ratio_vs_single_step"] = measured["peak_allocated_bytes"] / single_peak
    del gpu_images, gpu_labels, model, optimizer, scheduler, scaler
    _release_cuda()
    return measured


def _print_summary(payload: dict) -> None:
    print("\nModel | Head | Input | Microbatch | Peak VRAM | Step | Img/s | Status")
    for case in payload["cases"]:
        for result in case["results"]:
            print(
                f"{result['model_family']} | {result['model_architecture']} | "
                f"{result['input_size']} | {result['batch_size']} | "
                f"{_gib(result.get('peak_allocated_bytes'))} | "
                f"{result.get('step_time_seconds')} | {result.get('images_per_sec')} | "
                f"{result['status']}"
            )
    recommendation = payload["recommendation"]
    print(
        "Recommendation: "
        f"microbatch={recommendation.get('microbatch')} "
        f"accumulate={recommendation.get('accumulate_grad')} "
        f"effective={recommendation.get('effective_batch')}"
    )
    print(f"Wrote {payload['json_path']}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="RTX 5080 training-step VRAM benchmark")
    parser.add_argument("--data-root", type=Path, default=Path(DEFAULTS["data_root"]))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "experiments" / "benchmarks",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=list(DEFAULT_BATCH_SIZES))
    args = parser.parse_args(argv)
    payload = run_benchmark(
        args.data_root,
        args.output_dir,
        batch_sizes=tuple(args.batch_sizes),
        seed=args.seed,
    )
    _print_summary(payload)


if __name__ == "__main__":
    main()
