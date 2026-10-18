"""Train one leave-one-brand-out DINOv3 run.

Weights stay on the memory filesystem. Metrics are written under the experiment
directory. The frozen epoch_018 checkpoint is not loaded, and the locked final
test is not read.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.amp import GradScaler
from torch.utils.data import DataLoader

_PKG = Path(__file__).resolve().parent
if str(_PKG) not in sys.path:
    sys.path.insert(0, str(_PKG))

from calibration import DEFAULT_BINS, fit_temperature, scaled_logits, score_temperature
from checkpoint_selection import (
    candidate_from_validation,
    new_selection_state,
    should_save_production_checkpoint,
    update_selection,
)
from classifier import build_authenticity_classifier
from cross_brand import INITIALIZATION, SEED
from evaluation import evaluate_logits
from ood import cosine_distance, fit_reference, mahalanobis_distance
from train import (
    HypeVaultDataset,
    apply_accumulated_step,
    autocast_for,
    build_optimizer,
    build_train_loader,
    build_transforms,
    cosine_schedule_with_warmup,
    microbatch_loss_divisor,
    optimizer_step_due,
    optimizer_updates_per_epoch,
    smooth_binary_targets,
)
from training_config import build_bce_loss, resolve_amp

SHM_ROOT = Path("/dev/shm/hypevault_v3")
HYPER = {
    "family": "dinov3",
    "img_size": 512,
    "batch_size": 16,
    "accumulate_grad": 2,
    "num_epochs": 30,
    "backbone_lr": 1e-5,
    "head_lr": 1e-4,
    "layer_decay": 0.75,
    "weight_decay": 0.05,
    "warmup_epochs": 2,
    "label_smoothing": 0.02,
    "mixup_alpha": 0.0,
    "dropout": 0.15,
    "grad_clip": 1.0,
    "amp_dtype": "bf16",
    "early_stop_patience": 6,
    "max_false_authentic_rate": 0.02,
    "metric_tolerance": 1e-6,
    "validation_threshold": 0.5,
    "weighted_sampler": 0,
    "num_workers": 2,
    "seed": SEED,
}


def _pairs(rows: list[dict], role: str) -> list[tuple[Path, int]]:
    return [(Path(row["path"]), int(row["label"])) for row in rows if row["role"] == role]


def _loader(pairs: list[tuple[Path, int]], transform, *, shuffle: bool, batch_size: int) -> DataLoader:
    dataset = HypeVaultDataset(Path("/"), transform=transform, samples=pairs, shuffle=False)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=int(HYPER["num_workers"]),
        pin_memory=True,
        persistent_workers=int(HYPER["num_workers"]) > 0,
    )


def _train_one_epoch(model, loader, optimizer, criterion, scaler, scheduler, device, step_offset, epoch) -> tuple[float, int]:
    model.train()
    total = 0.0
    step = step_offset
    pending = 0
    pending_divisor = 1
    num_batches = len(loader)
    accumulate = int(HYPER["accumulate_grad"])
    for batch_idx, (images, labels) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        targets = smooth_binary_targets(labels, float(HYPER["label_smoothing"]))
        divisor = microbatch_loss_divisor(batch_idx, num_batches, accumulate)
        with autocast_for(device, "bf16"):
            loss = criterion(model(images), targets) / divisor
        scaler.scale(loss).backward()
        pending += 1
        pending_divisor = divisor
        if optimizer_step_due(batch_idx, num_batches, accumulate):
            apply_accumulated_step(model, optimizer, scaler, scheduler, float(HYPER["grad_clip"]))
            pending = 0
            step += 1
        total += float(loss.detach()) * divisor
    if pending:
        from train import rescale_accumulated_gradients

        rescale_accumulated_gradients(model, pending_divisor, pending)
        apply_accumulated_step(model, optimizer, scaler, scheduler, float(HYPER["grad_clip"]))
        step += 1
    return total / max(num_batches, 1), step


@torch.no_grad()
def collect_outputs(model, pairs: list[tuple[Path, int]], transform, device, *, features: bool) -> dict:
    loader = _loader(pairs, transform, shuffle=False, batch_size=32)
    model.eval()
    logits = []
    labels = []
    cls_rows = []
    for images, target in loader:
        images = images.to(device, non_blocking=True)
        if features:
            token_map = model.backbone.forward_features(images)
            cls_rows.append(model.backbone.get_cls_token(token_map).float().cpu())
        logits.append(model(images).float().cpu())
        labels.append(target.float().cpu())
    payload = {"logits": torch.cat(logits), "labels": torch.cat(labels).round().long()}
    if features:
        payload["cls"] = torch.cat(cls_rows)
    return payload


def _percentiles(values: np.ndarray) -> dict:
    points = (0, 5, 25, 50, 75, 95, 100)
    names = ("minimum", "p5", "p25", "median", "p75", "p95", "maximum")
    chosen = np.percentile(values, points)
    return {name: float(value) for name, value in zip(names, chosen)}


def _save_weights(path: Path, model, epoch: int, arch: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": int(epoch),
            "classifier_arch": arch,
            "initialization": INITIALIZATION,
            "model_state": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        },
        path,
    )


def _score_split(logits: torch.Tensor, labels: torch.Tensor, temperature: float) -> dict:
    scored = evaluate_logits(scaled_logits(logits, temperature), labels, threshold=0.5)
    calibration = score_temperature(logits, labels, temperature, DEFAULT_BINS)
    return {
        "roc_auc": scored["roc_auc"],
        "pr_auc": scored["pr_auc"],
        "f1": scored["f1"],
        "false_authentic_rate": scored["false_authentic_rate"],
        "false_fake_rate": scored["false_fake_rate"],
        "nll": calibration["nll"],
        "brier": calibration["brier"],
        "ece": calibration["ece"],
        "sample_count": scored["sample_count"],
        "authentic_count": scored["authentic_count"],
        "fake_count": scored["fake_count"],
        "temperature": temperature,
    }


def _completion_path(weight_path: Path) -> Path:
    return weight_path.with_suffix(".complete.json")


def _evaluate_saved(
    rows: list[dict],
    *,
    arch: str,
    weight_path: Path,
    eval_roles: tuple[str, ...],
    marker: dict,
) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    eval_tf = build_transforms("dinov3", train=False)
    model = build_authenticity_classifier("dinov3", arch, image_size=512, pretrained=False, dropout=0.15)
    model = model.to(device)
    state = torch.load(weight_path, map_location=device, weights_only=False)
    model.load_state_dict(state["model_state"])
    calibration = collect_outputs(model, _pairs(rows, "calibration"), eval_tf, device, features=False)
    fit = fit_temperature(calibration["logits"], calibration["labels"])
    temperature = float(fit["temperature"])
    calibration_score = _score_split(calibration["logits"], calibration["labels"], temperature)
    reports = {}
    feature_report = None
    train_features = None
    for role in eval_roles:
        if role == "train":
            train_features = collect_outputs(model, _pairs(rows, "train"), eval_tf, device, features=True)
            continue
        want_features = role in {"cross_brand_test", "random_test"}
        outputs = collect_outputs(model, _pairs(rows, role), eval_tf, device, features=want_features)
        if role == "validation":
            reports["in_domain"] = _score_split(outputs["logits"], outputs["labels"], temperature)
        elif role in {"cross_brand_test", "random_test"}:
            reports["unseen" if role == "cross_brand_test" else "random_test"] = _score_split(
                outputs["logits"], outputs["labels"], temperature
            )
            reports[role + "_probabilities"] = torch.sigmoid(scaled_logits(outputs["logits"], temperature)).numpy()
            reports[role + "_labels"] = outputs["labels"].numpy()
            reports[role + "_paths"] = [str(path) for path, _label in _pairs(rows, role)]
            if "cls" in outputs:
                reports[role + "_cls"] = outputs["cls"]
    if train_features is not None and "cross_brand_test_cls" in reports:
        reference = fit_reference(train_features["cls"], train_features["labels"])
        held_distance = mahalanobis_distance(reports["cross_brand_test_cls"], reference["gaussian"]).numpy()
        train_distance = mahalanobis_distance(train_features["cls"], reference["gaussian"]).numpy()
        held_cosine = cosine_distance(reports["cross_brand_test_cls"], reference["l2_centroid"]).numpy()
        train_cosine = cosine_distance(train_features["cls"], reference["l2_centroid"]).numpy()
        held_probability = reports["cross_brand_test_probabilities"]
        above = held_distance > np.percentile(train_distance, 95)
        below = held_distance <= np.median(train_distance)
        feature_report = {
            "production_ood_threshold": None,
            "train_mahalanobis": _percentiles(train_distance),
            "held_out_mahalanobis": _percentiles(held_distance),
            "train_cosine_distance": _percentiles(train_cosine),
            "held_out_cosine_distance": _percentiles(held_cosine),
            "held_out_fraction_above_train_mahalanobis_p95": float(np.mean(above)),
            "mean_p_fake_above_train_mahalanobis_p95": float(held_probability[above].mean()) if np.any(above) else None,
            "mean_p_fake_below_train_mahalanobis_median": float(held_probability[below].mean()) if np.any(below) else None,
        }
    del model
    torch.cuda.empty_cache()
    return {
        "initialization": INITIALIZATION,
        "loaded_frozen_checkpoint": False,
        "classifier_arch": arch,
        "selected_epoch": marker["selected_epoch"],
        "selected_checkpoint": marker["selected_checkpoint"],
        "temperature": temperature,
        "calibration": calibration_score,
        "history": marker["history"],
        "reports": {key: value for key, value in reports.items() if not key.endswith(("_probabilities", "_labels", "_paths", "_cls"))},
        "bundles": reports,
        "features": feature_report,
    }


def train_fold(rows: list[dict], *, arch: str, weight_path: Path, eval_roles: tuple[str, ...]) -> dict:
    if "epoch_018" in str(weight_path):
        raise RuntimeError("refusing to write the frozen checkpoint")
    completion = _completion_path(weight_path)
    if weight_path.is_file() and completion.is_file():
        print(f"scoring saved checkpoint {weight_path.name}", flush=True)
        return _evaluate_saved(
            rows,
            arch=arch,
            weight_path=weight_path,
            eval_roles=eval_roles,
            marker=json.loads(completion.read_text()),
        )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = dict(HYPER)
    cfg["classifier_arch"] = arch
    bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    amp = resolve_amp("bf16", cuda=device.type == "cuda", bf16_supported=bf16)
    if amp["amp_dtype"] != "bf16":
        raise RuntimeError(f"this protocol trains in bf16, resolved {amp['amp_dtype']}")
    train_tf = build_transforms("dinov3", train=True)
    eval_tf = build_transforms("dinov3", train=False)
    train_pairs = _pairs(rows, "train")
    val_pairs = _pairs(rows, "validation")
    train_ds = HypeVaultDataset(Path("/"), transform=train_tf, samples=train_pairs)
    train_loader, _decision = build_train_loader(train_ds, cfg, train_ds.labels)
    model = build_authenticity_classifier("dinov3", arch, image_size=512, pretrained=True, dropout=0.15)
    model = model.to(device)
    optimizer, _groups, _table = build_optimizer(model, cfg)
    steps_per_epoch = optimizer_updates_per_epoch(len(train_loader), cfg["accumulate_grad"])
    scheduler = cosine_schedule_with_warmup(
        optimizer,
        steps_per_epoch * int(cfg["warmup_epochs"]),
        steps_per_epoch * int(cfg["num_epochs"]),
    )
    criterion = build_bce_loss()
    scaler = GradScaler("cuda", enabled=False)
    selection = new_selection_state(
        max_false_authentic_rate=0.02,
        metric_tolerance=1e-6,
        validation_threshold=0.5,
        model_family="dinov3",
        classifier_arch=arch,
    )
    best_epoch = None
    global_step = 0
    history = []
    for epoch in range(1, int(cfg["num_epochs"]) + 1):
        train_loss, global_step = _train_one_epoch(
            model, train_loader, optimizer, criterion, scaler, scheduler, device, global_step, epoch
        )
        validation = collect_outputs(model, val_pairs, eval_tf, device, features=False)
        scored = evaluate_logits(validation["logits"], validation["labels"], threshold=0.5)
        val_loss = float(
            torch.nn.functional.binary_cross_entropy_with_logits(
                validation["logits"], validation["labels"].float()
            )
        )
        epoch_path = weight_path.with_name(f"epoch_{epoch:03d}.pt")
        _save_weights(epoch_path, model, epoch, arch)
        selection = update_selection(
            selection,
            candidate_from_validation(
                checkpoint_id=f"epoch_{epoch:03d}",
                epoch=epoch,
                global_step=global_step,
                validation_loss=val_loss,
                model_family="dinov3",
                classifier_arch=arch,
                metrics=scored,
                checkpoint_path=str(epoch_path),
                checkpoint_available=True,
            ),
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_pr_auc": scored["pr_auc"],
                "val_f1": scored["f1"],
                "val_false_authentic_rate": scored["false_authentic_rate"],
                "action": selection["action"],
            }
        )
        if should_save_production_checkpoint(selection):
            _save_weights(weight_path, model, epoch, arch)
            best_epoch = epoch
        selected_id = selection["selected_checkpoint"]
        retained = []
        for row in selection["candidates"]:
            path = Path(row.get("checkpoint_path") or "")
            if row["checkpoint_id"] == selected_id:
                retained.append(row)
                continue
            if path.is_file() and path != weight_path:
                path.unlink()
        selection["candidates"] = retained
        print(
            f"epoch {epoch} train {train_loss:.4f} val {val_loss:.4f} pr {scored['pr_auc']:.4f} "
            f"far {scored['false_authentic_rate']} action {selection['action']}",
            flush=True,
        )
        if (
            selection["action"] == "increment"
            and selection["epochs_without_improvement"] >= int(cfg["early_stop_patience"])
        ):
            print("early stopping", flush=True)
            break
    if best_epoch is None or not weight_path.is_file():
        raise RuntimeError("no safety-validated checkpoint was selected from the four-brand validation split")
    marker = {
        "selected_epoch": best_epoch,
        "selected_checkpoint": selection["selected_checkpoint"],
        "history": history,
        "history_source": "trainer",
    }
    completion.write_text(json.dumps(marker) + "\n")
    del model
    torch.cuda.empty_cache()
    return _evaluate_saved(
        rows,
        arch=arch,
        weight_path=weight_path,
        eval_roles=eval_roles,
        marker=marker,
    )
