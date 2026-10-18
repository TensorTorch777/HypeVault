"""
HypeVault — DINOv2 local training (RTX 5080 profile)
=====================================================
Binary classification: Authentic (0) vs Deepfake (1)

Smaller backbone (DINOv2-Base via timm), **504×504** input (multiple of patch 14; 512² is invalid for ViT-P/14), batch settings sized
for ~16 GB VRAM. Original `ml/train.py` remains the DINOv2-Giant / 518px path.

Dataset layout expected (watches only):
  Label_0_Watches/<brand>/*.jpg    → label 0 (Authentic)
  Label_1_Watches/<brand>/*.jpg    → label 1 (Deepfake)
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from collections import Counter
from contextlib import nullcontext
from pathlib import Path

_PKG = Path(__file__).resolve().parent
if str(_PKG) not in sys.path:
    sys.path.insert(0, str(_PKG))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torch.amp import GradScaler, autocast
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from PIL import Image, ImageFilter
import numpy as np
from tqdm import tqdm

try:
    import timm
    HAS_TIMM = True
except ImportError:
    HAS_TIMM = False

from dataset import (
    LABEL_DIRS,
    load_hypevault_samples,
    load_records,
    product_group_key,
    split_indices_by_product,
    carve_calibration_records,
    split_records,
    verify_no_image_overlap,
    verify_no_product_overlap,
    verify_no_source_leakage,
    verify_product_split,
    write_split_manifest,
)
from split_v2 import (
    AUTHORITATIVE_MEMBERSHIP_HASH,
    records_from_fixed_manifest,
    write_fixed_membership_copy,
)
from augmentations import CONSERVATIVE, build_transforms as build_shared_transforms, preprocessing_record
from checkpoint_selection import (
    DEFAULT_MAX_FALSE_AUTHENTIC_RATE,
    DEFAULT_METRIC_TOLERANCE,
    DEFAULT_VALIDATION_THRESHOLD,
    THRESHOLD_POLICY,
    UNQUALIFIED_MESSAGE,
    candidate_from_validation,
    epoch_artifact_path,
    load_selection_state,
    new_selection_state,
    should_save_production_checkpoint,
    update_selection,
    validate_max_false_authentic_rate,
    validate_metric_tolerance,
    validate_validation_threshold,
    write_selection_report,
)
from classifier import build_authenticity_classifier
from evaluation import evaluate_logits
from layerwise_lr import (
    build_layerwise_param_groups,
    format_decay_summary,
    format_param_group_table,
)
from manifest import IMAGENET_MEAN, IMAGENET_STD, write_training_manifest
from training_config import (
    FAMILY_TIMM_NAME,
    assert_experiment_output_dir,
    build_bce_loss,
    load_training_checkpoint,
    resolve_amp,
    resolve_family_image_size,
    sampler_decision,
    save_training_checkpoint,
    smooth_binary_targets,
    validate_classifier_arch,
    validate_label_smoothing,
    validate_mixup_alpha,
    write_experiment_config,
)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from sklearn.metrics import roc_curve, auc, precision_recall_curve, confusion_matrix
    HAS_PLOT = True
except ImportError:
    HAS_PLOT = False

# ─────────────────────────────────────────────
# Hyper-parameters (all overridable via CLI)
# ─────────────────────────────────────────────
DEFAULTS = dict(
    data_root="/home/tensortorch26/Desktop/HypeVault",
    # Experiment directory. checkpoints/ and checkpoints_watches/ are refused.
    output_dir="/home/tensortorch26/Desktop/HypeVault/ml_rtx5080/experiments/dinov2_cls_only",
    family="dinov2",
    classifier_arch="cls_only",
    # Filled from the family (504 for DINOv2, 512 for DINOv3). 0 means "use that default".
    model_name="vit_base_patch14_dinov2.lvd142m",
    img_size=0,
    # Phase 10 BF16 measurement on the RTX 5080: microbatch 16, accumulation 2.
    batch_size=16,
    accumulate_grad=2,
    num_epochs=30,
    backbone_lr=1e-5,
    head_lr=1e-4,
    layer_decay=0.75,
    weight_decay=0.05,
    warmup_epochs=2,
    label_smoothing=0.02,
    dropout=0.15,
    val_split=0.1,
    num_workers=6,
    seed=42,
    grad_clip=1.0,
    amp_dtype="bf16",
    save_every=2,
    resume_from=None,                   # path to a checkpoint to resume from
    start_epoch=1,                      # epoch to start from when resuming
    early_stop_patience=6,
    # Production selection constraint. At most this fraction of actual fakes
    # may be predicted authentic. This is not a calibrated threshold.
    max_false_authentic_rate=DEFAULT_MAX_FALSE_AUTHENTIC_RATE,
    metric_tolerance=DEFAULT_METRIC_TOLERANCE,
    validation_threshold=DEFAULT_VALIDATION_THRESHOLD,
    mixup_alpha=0.0,
    augmentation_profile="conservative",
    weighted_sampler=0,                 # 0 = shuffle, 1 = WeightedRandomSampler
    split_strategy="product_disjoint",
    test_fraction=0.1,
    # Fraction of training-side parent directories reserved from weight updates.
    calibration_fraction=0.10,
    test_root=None,
    # Phase 16 manifest. When set, membership is read from this file and the
    # old directory split / calibration carve is not called.
    fixed_split_manifest=None,
    # Confidence threshold for AUTHENTIC verdict (raise if too many false-negatives)
    # Default 0.5 = 50% sigmoid. Raise to 0.65–0.75 if real items get flagged as fake.
    confidence_threshold=0.5,
)

# ─────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────

class HypeVaultDataset(Dataset):
    """
    Loads images from the 4 label folders:
      Label_0_Watches → 0 (Authentic)
      Label_1_Watches → 1 (Deepfake)
    """

    LABEL_DIRS = LABEL_DIRS

    def __init__(
        self,
        root: Path,
        transform=None,
        indices: list[int] | None = None,
        samples: list[tuple[Path, int]] | None = None,
        shuffle: bool = False,
        seed: int | None = None,
        shuffle_brands: bool | None = None,
    ):
        self.root = root
        self.transform = transform
        if samples is not None:
            data = list(samples)
        else:
            data = load_hypevault_samples(root)
            if shuffle_brands:
                by_brand: dict[str, list[tuple[Path, int]]] = {}
                for path, label in data:
                    label_folder = next(part for part in path.parts if part.startswith("Label_"))
                    brand_key = f"{label_folder}/{path.parent.name}"
                    by_brand.setdefault(brand_key, []).append((path, label))
                rng = np.random.default_rng(seed)
                brand_keys = list(by_brand.keys())
                rng.shuffle(brand_keys)
                data = []
                for brand_key in brand_keys:
                    group = by_brand[brand_key][:]
                    rng.shuffle(group)
                    data.extend(group)

        if indices is not None:
            data = [data[i] for i in indices]

        if shuffle:
            rng = np.random.default_rng(seed)
            rng.shuffle(data)

        self.samples = data
        # WeightedRandomSampler must see labels only for the filtered subset.
        self.labels = [s[1] for s in self.samples]

    def __len__(self):
        return len(self.samples)

    def product_groups(self) -> dict[str, list[int]]:
        groups: dict[str, list[int]] = {}
        for idx, (path, _) in enumerate(self.samples):
            groups.setdefault(product_group_key(path), []).append(idx)
        return groups

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            # return a blank image instead of crashing the worker
            img = Image.new("RGB", (224, 224), (128, 128, 128))
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(label, dtype=torch.float32)


def build_transforms(family: str, *, train: bool, profile: str = CONSERVATIVE) -> transforms.Compose:
    """Use the shared pipeline in ``augmentations.py``. This function does not define one."""
    return build_shared_transforms(family, train=train, profile=profile)


def make_weighted_sampler(dataset: HypeVaultDataset) -> WeightedRandomSampler:
    """Balance classes during training (same # authentic vs deepfake)."""
    counts = np.bincount(dataset.labels)
    class_weights = 1.0 / counts
    sample_weights = [class_weights[l] for l in dataset.labels]
    return WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)


# ─────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────

class DINOv2Classifier(nn.Module):
    """
    DINOv2 backbone (Giant / Large / Base — set `model_name`) + binary head.
    Full fine-tune of the backbone by default.
    """

    def __init__(self, model_name: str, dropout: float = 0.2):
        super().__init__()

        # Load backbone via torch.hub (official Meta weights) or timm
        self.backbone = self._load_backbone(model_name)
        embed_dim = self._get_embed_dim(self.backbone)

        # Classification head
        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Dropout(p=dropout),
            nn.Linear(embed_dim, 512),
            nn.GELU(),
            nn.Dropout(p=dropout / 2),
            nn.Linear(512, 1),   # binary logit
        )

    def _load_backbone(self, model_name: str):
        # Try torch.hub first (official Meta DINOv2 weights)
        hub_names = {
            "dinov2_vitg14_reg": "dinov2_vitg14_reg",
            "dinov2_vitg14":     "dinov2_vitg14",
            "dinov2_vitl14_reg": "dinov2_vitl14_reg",
            "dinov2_vitl14":     "dinov2_vitl14",
            "dinov2_vitb14_reg": "dinov2_vitb14_reg",
            "dinov2_vitb14":     "dinov2_vitb14",
        }
        if model_name in hub_names:
            try:
                print(f"Loading {model_name} from torch.hub (facebookresearch/dinov2)...")
                model = torch.hub.load(
                    "facebookresearch/dinov2",
                    hub_names[model_name],
                    pretrained=True,
                    force_reload=False,
                )
                print(f"  Loaded via torch.hub ✓")
                self._enable_grad_checkpointing(model)
                return model
            except Exception as e:
                print(f"  torch.hub failed ({e}), trying timm...")

        # Fallback: timm
        if HAS_TIMM:
            timm_names = {
                "dinov2_vitg14_reg": "vit_giant_patch14_reg4_dinov2.lvd142m",
                "dinov2_vitg14":     "vit_giant_patch14_dinov2.lvd142m",
                "dinov2_vitl14_reg": "vit_large_patch14_reg4_dinov2.lvd142m",
                "dinov2_vitl14":     "vit_large_patch14_dinov2.lvd142m",
                "dinov2_vitb14_reg": "vit_base_patch14_reg4_dinov2.lvd142m",
                "dinov2_vitb14":     "vit_base_patch14_dinov2.lvd142m",
            }
            timm_name = timm_names.get(model_name, model_name)
            print(f"Loading {timm_name} via timm...")
            model = timm.create_model(
                timm_name,
                pretrained=True,
                num_classes=0,
                dynamic_img_size=True,
            )
            print(f"  Loaded via timm ✓")
            self._enable_grad_checkpointing(model)
            return model

        raise RuntimeError(
            "Could not load DINOv2. Install timm or ensure torch.hub can reach github."
        )

    def _enable_grad_checkpointing(self, model) -> None:
        """Enable gradient checkpointing to reduce VRAM ~40% at cost of ~20% speed."""
        # timm models
        if hasattr(model, "set_grad_checkpointing"):
            model.set_grad_checkpointing(True)
            print("  Gradient checkpointing enabled (timm) ✓")
            return
        # torch.hub DINOv2: patch each transformer block
        enabled = 0
        for module in model.modules():
            if hasattr(module, "grad_checkpointing"):
                module.grad_checkpointing = True
                enabled += 1
        if enabled:
            print(f"  Gradient checkpointing enabled ({enabled} blocks) ✓")
            return
        # Manual patch using torch.utils.checkpoint on block list
        import torch.utils.checkpoint as ckpt_util
        blocks = None
        if hasattr(model, "blocks"):
            blocks = model.blocks
        elif hasattr(model, "encoder") and hasattr(model.encoder, "layer"):
            blocks = model.encoder.layer
        if blocks is not None:
            original_forwards = [b.forward for b in blocks]
            for i, block in enumerate(blocks):
                orig = original_forwards[i]
                def make_ckpt_forward(fn):
                    def ckpt_forward(*args, **kwargs):
                        def run(*a):
                            return fn(*a)
                        return ckpt_util.checkpoint(run, *args, use_reentrant=False)
                    return ckpt_forward
                block.forward = make_ckpt_forward(orig)
            print(f"  Gradient checkpointing enabled (manual patch, {len(blocks)} blocks) ✓")
            return
        print("  Warning: could not enable gradient checkpointing — proceeding without it")

    def _get_embed_dim(self, backbone) -> int:
        # DINOv2-Giant: 1536. ViT-L: 1024. ViT-B: 768
        if hasattr(backbone, "embed_dim"):
            return backbone.embed_dim
        if hasattr(backbone, "num_features"):
            return backbone.num_features
        # fallback: run a dummy forward
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 504, 504)
            out = backbone(dummy)
            return out.shape[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns raw logit (not sigmoid) — use BCEWithLogitsLoss."""
        features = self.backbone(x)
        # DINOv2 hub returns a dict or tensor depending on variant
        if isinstance(features, dict):
            features = features["x_norm_clstoken"]
        return self.head(features).squeeze(1)


# ─────────────────────────────────────────────
# Mixup augmentation
# ─────────────────────────────────────────────

def mixup_batch(
    x: torch.Tensor, y: torch.Tensor, alpha: float = 0.2
) -> tuple[torch.Tensor, torch.Tensor]:
    if alpha <= 0:
        return x, y
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(x.size(0), device=x.device)
    mixed_x = lam * x + (1 - lam) * x[idx]
    mixed_y = lam * y + (1 - lam) * y[idx]
    return mixed_x, mixed_y


# ─────────────────────────────────────────────
# Cosine LR schedule with warm-up
# ─────────────────────────────────────────────

def cosine_schedule_with_warmup(
    optimizer, warmup_steps: int, total_steps: int
):
    def lr_lambda(step: int):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def optimizer_step_due(batch_idx: int, num_batches: int, accumulate_grad: int) -> bool:
    """Return whether this 0-based batch closes an optimizer update.

    Full windows step on batches divisible by ``accumulate_grad``. A trailing
    window that does not fill the window still steps on the final batch so its
    gradients are not discarded.
    """
    if accumulate_grad < 1:
        raise ValueError(f"accumulate_grad must be >= 1, got {accumulate_grad}")
    if num_batches < 1:
        raise ValueError(f"num_batches must be >= 1, got {num_batches}")
    if not 0 <= batch_idx < num_batches:
        raise ValueError(f"batch_idx {batch_idx} outside 0..{num_batches - 1}")
    seen = batch_idx + 1
    on_full_window = seen % accumulate_grad == 0
    tail_pending = seen == num_batches and num_batches % accumulate_grad != 0
    return on_full_window or tail_pending


def optimizer_updates_per_epoch(num_batches: int, accumulate_grad: int) -> int:
    """Count optimizer updates in one epoch, including a partial tail window."""
    if accumulate_grad < 1:
        raise ValueError(f"accumulate_grad must be >= 1, got {accumulate_grad}")
    if num_batches < 0:
        raise ValueError(f"num_batches must be >= 0, got {num_batches}")
    if num_batches == 0:
        return 0
    return math.ceil(num_batches / accumulate_grad)


def microbatch_loss_divisor(batch_idx: int, num_batches: int, accumulate_grad: int) -> int:
    """How many microbatches share the optimizer update that includes this batch.

    A full window divides by ``accumulate_grad``. A trailing window divides by
    the number of batches actually in that window, so a single leftover batch
    is not scaled by the full accumulation factor.
    """
    if accumulate_grad < 1:
        raise ValueError(f"accumulate_grad must be >= 1, got {accumulate_grad}")
    if num_batches < 1:
        raise ValueError(f"num_batches must be >= 1, got {num_batches}")
    if not 0 <= batch_idx < num_batches:
        raise ValueError(f"batch_idx {batch_idx} outside 0..{num_batches - 1}")
    window_start = (batch_idx // accumulate_grad) * accumulate_grad
    window_end = min(window_start + accumulate_grad, num_batches)
    return window_end - window_start


def rescale_accumulated_gradients(model, planned_divisor: int, actual_count: int) -> None:
    """Correct a window that was divided by a size it never reached.

    Each microbatch already divided the loss by ``planned_divisor``. If the
    loader stopped after ``actual_count`` of those batches, multiply the
    summed gradient by ``planned_divisor / actual_count`` once. When the two
    counts match, the gradient is left alone.
    """
    if actual_count < 1:
        raise ValueError(f"actual_count must be >= 1, got {actual_count}")
    if planned_divisor < 1:
        raise ValueError(f"planned_divisor must be >= 1, got {planned_divisor}")
    if planned_divisor == actual_count:
        return
    factor = planned_divisor / actual_count
    for parameter in model.parameters():
        if parameter.grad is not None:
            parameter.grad.mul_(factor)


def autocast_for(device: torch.device, amp_dtype: str):
    if device.type == "cuda" and amp_dtype == "bf16":
        return autocast("cuda", dtype=torch.bfloat16)
    if device.type == "cuda" and amp_dtype == "fp16":
        return autocast("cuda", dtype=torch.float16)
    return nullcontext()


def apply_accumulated_step(model, optimizer, scaler, scheduler, grad_clip: float) -> None:
    """Clip, step, zero, and advance the LR schedule once per optimizer update."""
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)
    scheduler.step()


# ─────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────

def compute_metrics(logits: torch.Tensor, labels: torch.Tensor):
    preds = (torch.sigmoid(logits) >= 0.5).long()
    labels_int = labels.long()
    acc = (preds == labels_int).float().mean().item()

    tp = ((preds == 1) & (labels_int == 1)).sum().item()
    fp = ((preds == 1) & (labels_int == 0)).sum().item()
    fn = ((preds == 0) & (labels_int == 1)).sum().item()
    precision = tp / max(tp + fp, 1)
    recall    = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    return {"acc": acc, "precision": precision, "recall": recall, "f1": f1}


# ─────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────

def train_epoch(
    model, loader, optimizer, criterion, scaler, scheduler,
    device, cfg, step_offset: int, epoch: int
) -> tuple[float, dict, int]:
    model.train()
    total_loss = 0.0
    all_logits, all_labels = [], []
    step = step_offset

    amp_dtype = cfg["amp_dtype"]
    lr_now = scheduler.get_last_lr()[-1]
    num_batches = len(loader)
    accumulate_grad = int(cfg["accumulate_grad"])
    label_smoothing = float(cfg.get("label_smoothing", 0.0))
    pending_count = 0
    pending_divisor = 1

    pbar = tqdm(
        enumerate(loader),
        total=num_batches,
        desc=f"Epoch {epoch} [Train]",
        ncols=110,
        leave=True,
    )
    for batch_idx, (images, labels) in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if cfg["mixup_alpha"] > 0:
            images, labels = mixup_batch(images, labels, cfg["mixup_alpha"])
        targets = smooth_binary_targets(labels, label_smoothing)

        divisor = microbatch_loss_divisor(batch_idx, num_batches, accumulate_grad)
        with autocast_for(device, amp_dtype):
            logits = model(images)
            loss = criterion(logits, targets)
            loss = loss / divisor

        scaler.scale(loss).backward()
        pending_count += 1
        pending_divisor = divisor

        if optimizer_step_due(batch_idx, num_batches, accumulate_grad):
            apply_accumulated_step(model, optimizer, scaler, scheduler, cfg["grad_clip"])
            pending_count = 0
            lr_now = scheduler.get_last_lr()[-1]
            step += 1

        batch_loss = loss.item() * divisor
        total_loss += batch_loss
        all_logits.append(logits.detach().float())
        all_labels.append(labels.detach().float())

        pbar.set_postfix(loss=f"{batch_loss:.4f}", head_lr=f"{lr_now:.2e}", refresh=False)

    # Loader length can exceed batches actually yielded. Flush that tail,
    # rescaling if those batches were divided by a window that never filled.
    if pending_count:
        rescale_accumulated_gradients(model, pending_divisor, pending_count)
        apply_accumulated_step(model, optimizer, scaler, scheduler, cfg["grad_clip"])
        step += 1

    metrics = compute_metrics(
        torch.cat(all_logits),
        torch.cat(all_labels).round().long()
    )
    return total_loss / len(loader), metrics, step


@torch.no_grad()
def val_epoch(model, loader, criterion, device, cfg, epoch: int | None = None) -> tuple[float, dict]:
    model.eval()
    total_loss = 0.0
    all_logits, all_labels = [], []

    amp_dtype = cfg["amp_dtype"]
    desc = f"Epoch {epoch} [ Val ]" if epoch is not None else "Evaluating"

    pbar = tqdm(loader, total=len(loader), desc=desc, ncols=110, leave=True)
    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        # Validation stays on the original images and the original hard labels.
        with autocast_for(device, amp_dtype):
            logits = model(images)
            loss = criterion(logits, labels)
        total_loss += loss.item()
        all_logits.append(logits.float())
        all_labels.append(labels.float())
        pbar.set_postfix(loss=f"{loss.item():.4f}", refresh=False)

    all_logits_cat = torch.cat(all_logits)
    all_labels_cat = torch.cat(all_labels)
    metrics = compute_metrics(all_logits_cat, all_labels_cat)
    # attach raw tensors for plotting (caller can ignore)
    metrics["_logits"] = all_logits_cat
    metrics["_labels"] = all_labels_cat
    return total_loss / len(loader), metrics


# ─────────────────────────────────────────────
# Plots
# ─────────────────────────────────────────────

def plot_training_curves(
    history: list[dict],
    out_dir: Path,
    best_epoch: int,
    subtitle: str | None = None,
) -> None:
    if not HAS_PLOT or not history:
        return
    epochs = [r["epoch"] for r in history]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), facecolor="#111111")
    fig.patch.set_facecolor("#111111")
    best_val_acc = max(r["val_acc"] for r in history) * 100
    best_val_f1  = max(r["val_f1"]  for r in history)

    title = (
        f"HypeVault — DINOv2-Base · 504px (RTX 5080)\n"
        f"Epoch {best_epoch} | val_acc={best_val_acc:.2f}%  val_f1={best_val_f1:.4f}"
    )
    if subtitle:
        title += f"\n{subtitle}"
    fig.suptitle(title, color="white", fontsize=13, fontweight="bold", y=0.98)

    C = {"train": "#FF6B35", "val": "#00B4D8", "grid": "#333333", "text": "white",
         "warn": "#FFA500", "stop": "#FF4444", "target": "#888888"}

    def _style(ax, title):
        ax.set_facecolor("#1a1a1a"); ax.set_title(title, color=C["text"], fontsize=11)
        ax.tick_params(colors=C["text"]); ax.xaxis.label.set_color(C["text"])
        ax.yaxis.label.set_color(C["text"])
        for spine in ax.spines.values(): spine.set_edgecolor(C["grid"])
        ax.grid(True, color=C["grid"], linewidth=0.5)

    # Loss
    ax = axes[0, 0]
    ax.plot(epochs, [r["train_loss"] for r in history], "o-", color=C["train"], label="Train loss")
    ax.plot(epochs, [r["val_loss"]   for r in history], "s--", color=C["val"],   label="Val loss")
    ax.set_xlabel("Epoch"); ax.set_ylabel("BCE Loss"); _style(ax, "Loss")
    ax.legend(facecolor="#222", edgecolor=C["grid"], labelcolor=C["text"])

    # Accuracy
    ax = axes[0, 1]
    train_accs = [r["train_acc"] * 100 for r in history]
    val_accs   = [r["val_acc"]   * 100 for r in history]
    ax.plot(epochs, train_accs, "o-",  color=C["train"], label="Train acc")
    ax.plot(epochs, val_accs,   "s--", color=C["val"],   label="Val acc")
    ax.axhline(95, color=C["target"], linestyle=":", linewidth=1, label="95% target")
    if val_accs:
        best_idx = val_accs.index(max(val_accs))
        ax.annotate(f"{max(val_accs):.1f}%", xy=(epochs[best_idx], max(val_accs)),
                    color=C["val"], fontsize=9, ha="left")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Accuracy (%)"); _style(ax, "Accuracy")
    ax.legend(facecolor="#222", edgecolor=C["grid"], labelcolor=C["text"])

    # F1
    ax = axes[1, 0]
    ax.plot(epochs, [r["train_f1"] for r in history], "o-",  color=C["train"], label="Train F1")
    ax.plot(epochs, [r["val_f1"]   for r in history], "s--", color=C["val"],   label="Val F1")
    ax.axhline(0.95, color=C["target"], linestyle=":", linewidth=1, label="F1=0.95 target")
    if history:
        best_f1_idx = [r["val_f1"] for r in history].index(max(r["val_f1"] for r in history))
        ax.annotate(f"{history[best_f1_idx]['val_f1']:.4f}",
                    xy=(epochs[best_f1_idx], history[best_f1_idx]["val_f1"]),
                    color=C["val"], fontsize=9, ha="left")
    ax.set_xlabel("Epoch"); ax.set_ylabel("F1"); _style(ax, "F1 Score")
    ax.legend(facecolor="#222", edgecolor=C["grid"], labelcolor=C["text"])

    # Overfitting monitor
    ax = axes[1, 1]
    gaps = [(r["train_acc"] - r["val_acc"]) * 100 for r in history]
    bars = ax.bar(epochs, gaps, color="#FFD700", alpha=0.7)
    ax.axhline(10, color=C["warn"], linestyle="--", linewidth=1.2, label="Warn threshold (10%)")
    ax.axhline(20, color=C["stop"], linestyle="--", linewidth=1.2, label="Stop threshold (20%)")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Gap (%)"); _style(ax, "Overfitting Monitor\n(Train Acc — Val Acc)")
    ax.legend(facecolor="#222", edgecolor=C["grid"], labelcolor=C["text"], fontsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out_path = out_dir / "training_curves.png"
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved → {out_path}")


def plot_confusion_eval(
    logits: torch.Tensor,
    labels: torch.Tensor,
    out_dir: Path,
    num_val: int,
    best_val_acc: float,
    best_val_f1: float,
    threshold: float = 0.5,
) -> None:
    if not HAS_PLOT:
        return
    probs  = torch.sigmoid(logits).cpu().numpy().astype(float)
    y_true = labels.cpu().numpy().astype(int)
    y_pred = (probs >= threshold).astype(int)

    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    fpr, tpr, _ = roc_curve(y_true, probs)
    roc_auc = auc(fpr, tpr)
    prec_curve, rec_curve, _ = precision_recall_curve(y_true, probs)
    pr_auc = auc(rec_curve, prec_curve)

    fig = plt.figure(figsize=(18, 11), facecolor="#111111")
    fig.patch.set_facecolor("#111111")
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)
    axes = [fig.add_subplot(gs[r, c]) for r in range(2) for c in range(3)]

    title = (
        f"HypeVault — DINOv2-Base · 504px  |  Val set: {num_val:,} images\n"
        f"Accuracy={best_val_acc*100:.2f}%  F1={best_val_f1:.4f}  "
        f"ROC-AUC={roc_auc:.4f}  PR-AUC={pr_auc:.4f}\n"
        f"TN={tn:,}  FP={fp}  FN={fn}  TP={tp:,}"
    )
    fig.suptitle(title, color="white", fontsize=11, fontweight="bold", y=0.98)

    C = {"text": "white", "grid": "#333333"}
    def _style(ax, title):
        ax.set_facecolor("#1a1a1a"); ax.set_title(title, color=C["text"], fontsize=10)
        ax.tick_params(colors=C["text"]); ax.xaxis.label.set_color(C["text"])
        ax.yaxis.label.set_color(C["text"])
        for spine in ax.spines.values(): spine.set_edgecolor(C["grid"])

    # Raw CM
    ax = axes[0]
    im = ax.imshow(cm, cmap="RdYlGn", aspect="auto")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i,j]:,}", ha="center", va="center",
                    color="white", fontsize=14, fontweight="bold")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Pred Authentic", "Pred Deepfake"], color=C["text"])
    ax.set_yticklabels(["True Authentic", "True Deepfake"], color=C["text"])
    plt.colorbar(im, ax=ax)
    _style(ax, "Confusion Matrix (raw counts)")

    # Normalised CM
    ax = axes[1]
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
    im2 = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=100, aspect="auto")
    labels_txt = [["True Authentic", "True Deepfake"], ["True Authentic", "True Deepfake"]]
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm_norm[i,j]:.2f}%", ha="center", va="center",
                    color="white", fontsize=13, fontweight="bold")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Pred Authentic", "Pred Deepfake"], color=C["text"])
    ax.set_yticklabels(["True Authentic", "True Deepfake"], color=C["text"])
    plt.colorbar(im2, ax=ax)
    _style(ax, "Confusion Matrix (normalised %)")

    # ROC
    ax = axes[2]
    ax.set_facecolor("#1a1a1a")
    ax.plot(fpr, tpr, color="#00B4D8", lw=2, label=f"AUC = {roc_auc:.4f}")
    ax.plot([0, 1], [0, 1], "w--", lw=0.8)
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.legend(facecolor="#222", edgecolor=C["grid"], labelcolor=C["text"])
    _style(ax, "ROC Curve")

    # PR Curve
    ax = axes[3]
    ax.set_facecolor("#1a1a1a")
    ax.plot(rec_curve, prec_curve, color="#00E676", lw=2, label=f"PR-AUC = {pr_auc:.4f}")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.legend(facecolor="#222", edgecolor=C["grid"], labelcolor=C["text"])
    _style(ax, "Precision-Recall Curve")

    # Probability distribution
    ax = axes[4]
    ax.set_facecolor("#1a1a1a")
    auth_probs = probs[y_true == 0]
    fake_probs = probs[y_true == 1]
    ax.hist(auth_probs, bins=50, color="#00B4D8", alpha=0.7, density=True, label="Authentic (true)")
    ax.hist(fake_probs, bins=50, color="#FF6B35", alpha=0.7, density=True, label="Deepfake (true)")
    ax.axvline(0.5, color="#FFD700", linestyle="--", linewidth=1.5, label="Threshold = 0.5")
    ax.set_xlabel("P(Deepfake)"); ax.set_ylabel("Density")
    ax.legend(facecolor="#222", edgecolor=C["grid"], labelcolor=C["text"], fontsize=8)
    _style(ax, "Prediction Probability Distribution")

    # Per-class breakdown
    ax = axes[5]
    ax.set_facecolor("#1a1a1a")
    cats  = ["TN\nAuth→Auth", "FP\nAuth→Fake", "FN\nFake→Auth", "TP\nFake→Fake"]
    vals  = [tn, fp, fn, tp]
    total_auth = tn + fp
    total_fake = fn + tp
    pcts  = [tn/max(total_auth,1)*100, fp/max(total_auth,1)*100,
             fn/max(total_fake,1)*100, tp/max(total_fake,1)*100]
    colors = ["#00E676", "#FF4444", "#FF4444", "#00B4D8"]
    bars = ax.bar(cats, vals, color=colors, alpha=0.85)
    for bar, val, pct in zip(bars, vals, pcts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(vals)*0.01,
                f"{val:,}\n{pct:.2f}%", ha="center", va="bottom", color=C["text"], fontsize=8)
    ax.set_ylabel("Count"); _style(ax, "Per-Class Breakdown")

    for ax in axes:
        ax.tick_params(colors=C["text"])
        for spine in ax.spines.values(): spine.set_edgecolor(C["grid"])

    out_path = out_dir / "confusion_matrix_eval.png"
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved → {out_path}")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def build_optimizer(model: nn.Module, cfg: dict):
    """AdamW over real per-block learning rates, with decay split from no-decay."""
    groups = build_layerwise_param_groups(
        model,
        base_backbone_lr=float(cfg["backbone_lr"]),
        head_lr=float(cfg["head_lr"]),
        layer_decay=float(cfg["layer_decay"]),
        weight_decay=float(cfg["weight_decay"]),
        separate_no_decay=True,
    )
    table = format_param_group_table(groups)
    summary = format_decay_summary(groups)
    print(table)
    print(summary)
    optimizer = torch.optim.AdamW(groups, betas=(0.9, 0.999), eps=1e-8)
    return optimizer, groups, table


def build_train_loader(dataset, cfg: dict, labels: list[int]):
    decision = sampler_decision(labels, weighted_sampler=bool(int(cfg["weighted_sampler"])))
    counts = decision["counts"]
    print(
        f"Class counts — authentic: {counts[0]}  deepfake: {counts[1]}  "
        f"minority/majority: {decision['minority_ratio']:.4f}"
    )
    print(f"Sampler: {decision['sampler']}  shuffle: {decision['shuffle']}")
    if decision["materially_imbalanced"] and decision["sampler"] == "none":
        print(
            "Class counts differ by more than 10%. WeightedRandomSampler stays off "
            "unless --weighted_sampler 1."
        )
    workers = int(cfg["num_workers"])
    common = dict(
        batch_size=int(cfg["batch_size"]),
        num_workers=workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=workers > 0,
    )
    if decision["sampler"] == "weighted_random":
        loader = DataLoader(
            dataset,
            sampler=make_weighted_sampler(dataset),
            shuffle=False,
            **common,
        )
    else:
        loader = DataLoader(dataset, shuffle=True, **common)
    return loader, decision


def split_census(name: str, records: list) -> dict:
    authentic = sum(1 for record in records if record.label == 0)
    fake = sum(1 for record in records if record.label == 1)
    brands = Counter(record.brand for record in records)
    return {
        "split": name,
        "samples": len(records),
        "authentic": authentic,
        "fake": fake,
        "group_count": len({record.split_group for record in records}),
        "brand_counts": dict(sorted(brands.items())),
    }


def write_history(out_dir: Path, history: list[dict], test_summary: dict | None = None) -> None:
    payload: dict = {"epochs": history}
    if test_summary is not None:
        payload["test"] = test_summary
    (out_dir / "history.json").write_text(json.dumps(payload, indent=2))


def main():
    parser = argparse.ArgumentParser(description="HypeVault DINOv2 (RTX 5080 / 504px) fine-tuning")
    for k, v in DEFAULTS.items():
        if v is None:
            parser.add_argument(f"--{k}", type=str, default=None)
        else:
            parser.add_argument(f"--{k}", type=type(v), default=v)
    cfg = vars(parser.parse_args())
    cfg["family"] = str(cfg["family"])
    cfg["classifier_arch"] = validate_classifier_arch(str(cfg["classifier_arch"]))
    cfg["img_size"] = resolve_family_image_size(cfg["family"], int(cfg["img_size"]))
    cfg["model_name"] = FAMILY_TIMM_NAME[cfg["family"]]
    cfg["mixup_alpha"] = validate_mixup_alpha(cfg["mixup_alpha"])
    cfg["label_smoothing"] = validate_label_smoothing(cfg["label_smoothing"])
    cfg["max_false_authentic_rate"] = validate_max_false_authentic_rate(cfg["max_false_authentic_rate"])
    cfg["metric_tolerance"] = validate_metric_tolerance(cfg["metric_tolerance"])
    cfg["validation_threshold"] = validate_validation_threshold(cfg["validation_threshold"])
    cfg["threshold_policy"] = THRESHOLD_POLICY
    cfg.update(preprocessing_record(cfg["family"], str(cfg["augmentation_profile"]), cfg["classifier_arch"]))
    if int(cfg["input_size"]) != int(cfg["img_size"]):
        raise ValueError(
            f"{cfg['family']} preprocessing is {cfg['input_size']}px, but img_size is {cfg['img_size']}"
        )
    if int(cfg["batch_size"]) < 1 or int(cfg["accumulate_grad"]) < 1:
        raise ValueError("batch_size and accumulate_grad must be >= 1")

    # ── Paths ──────────────────────────────────
    out_dir = assert_experiment_output_dir(Path(cfg["output_dir"]))
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg["output_dir"] = str(out_dir)

    # ── Reproducibility ────────────────────────
    # One seed for every experiment. cudnn.benchmark stays on, so the run is
    # seeded but not bitwise deterministic.
    random.seed(int(cfg["seed"]))
    np.random.seed(int(cfg["seed"]))
    torch.manual_seed(int(cfg["seed"]))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(cfg["seed"]))
    cfg["python_seed"] = int(cfg["seed"])
    cfg["numpy_seed"] = int(cfg["seed"])
    cfg["torch_seed"] = int(cfg["seed"])
    cfg["cuda_seed"] = int(cfg["seed"]) if torch.cuda.is_available() else None
    cfg["deterministic_algorithms"] = False

    # ── Device ────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    cfg["cudnn_benchmark"] = True
    if device.type == "cuda":
        cfg["gpu_name"] = torch.cuda.get_device_name(0)
    else:
        cfg["gpu_name"] = None

    # ── Dataset ───────────────────────────────
    # Validation on the same scraped distribution can look strong when products
    # leak across splits; use --test_root for held-out products before production claims.
    data_root = Path(cfg["data_root"])
    print(f"\nLoading dataset from: {data_root}")
    records = load_records(data_root)
    catalog = [(record.image_path, record.label) for record in records]
    total = len(catalog)
    print(f"Total images: {total:,}")

    fixed_manifest = cfg.get("fixed_split_manifest")
    if fixed_manifest:
        if cfg.get("test_root"):
            raise RuntimeError("A fixed split manifest run must not load --test_root")
        splits, split_meta = records_from_fixed_manifest(
            records,
            Path(fixed_manifest),
            expected_hash=AUTHORITATIVE_MEMBERSHIP_HASH,
        )
        cfg["split_strategy"] = split_meta["strategy"]
        cfg["fixed_split_manifest"] = split_meta["fixed_split_manifest"]
        cfg["manifest_membership_hash"] = split_meta["manifest_membership_hash"]
        cfg["calibration_membership_source"] = split_meta["calibration_membership_source"]
        cfg["product_disjoint_guarantee"] = False
        write_fixed_membership_copy(out_dir, splits, split_meta)
        source_report = {
            "status": "not_used",
            "reason": "fixed manifest membership was used; source was not a split key",
        }
        product_report = {"product_id_status": "unavailable"}
        print(
            f"Fixed manifest {split_meta['fixed_split_manifest']} "
            f"hash {split_meta['manifest_membership_hash']}"
        )
        print("Calibration membership source: split_manifest_v2. No directory carve was applied.")
    else:
        splits, split_meta = split_records(
            records,
            cfg["split_strategy"],
            cfg["seed"],
            cfg["val_split"],
            cfg["test_fraction"],
        )
        if not 0.0 <= float(cfg["calibration_fraction"]) < 1.0:
            raise ValueError("calibration_fraction must be in [0, 1)")
        validation_membership = list(splits["validation"])
        test_membership = list(splits["test"])
        weight_train, calibration_records = carve_calibration_records(
            splits["train"], float(cfg["calibration_fraction"]), int(cfg["seed"])
        )
        splits = {
            "train": weight_train,
            "calibration": calibration_records,
            "validation": validation_membership,
            "test": test_membership,
        }
        split_meta = {**split_meta, "calibration_fraction": float(cfg["calibration_fraction"])}
        verify_no_image_overlap(splits)
        product_report = verify_no_product_overlap(splits)
        source_report = verify_no_source_leakage(splits)
        write_split_manifest(
            out_dir,
            records,
            splits,
            strategy=cfg["split_strategy"],
            seed=cfg["seed"],
            val_fraction=cfg["val_split"],
            test_fraction=cfg["test_fraction"],
            split_meta=split_meta,
            source_report=source_report,
            product_report=product_report,
        )
    train_samples = [(record.image_path, record.label) for record in splits["train"]]
    val_samples = [(record.image_path, record.label) for record in splits["validation"]]
    dataset_report = {
        "total_samples": len(records),
        "seed": int(cfg["seed"]),
        "split_strategy": cfg["split_strategy"],
        "calibration_fraction": None if fixed_manifest else float(cfg["calibration_fraction"]),
        "calibration_membership_source": cfg.get("calibration_membership_source"),
        "manifest_membership_hash": cfg.get("manifest_membership_hash"),
        "fixed_split_manifest": cfg.get("fixed_split_manifest"),
        "splits": {
            name: split_census(name, splits.get(name, []))
            for name in ("train", "calibration", "validation", "test")
        },
    }
    (out_dir / "dataset_report.json").write_text(json.dumps(dataset_report, indent=2) + "\n")
    print(
        f"Split {cfg['split_strategy']}: "
        f"train {len(train_samples):,} | calibration {len(splits['calibration']):,} | "
        f"val {len(val_samples):,} | holdout {len(splits['test']):,} "
        f"(saved split_manifest.json; calibration is not used for weight updates)"
    )
    for name in ("train", "calibration", "validation", "test"):
        census = dataset_report["splits"][name]
        print(
            f"  {name}: {census['samples']:,} images | "
            f"authentic {census['authentic']:,} | fake {census['fake']:,} | "
            f"groups {census['group_count']} | brands {census['brand_counts']}"
        )
    if len(train_samples) < 100:
        raise RuntimeError(
            f"Training split has only {len(train_samples)} images after the calibration carve. "
            "Refusing to start a full run. The split was not changed."
        )
    if source_report["status"] == "unavailable":
        print(f"Source check: {source_report['reason']}")
    if fixed_manifest:
        print(
            "Product ids are UNKNOWN. Membership comes from the fixed brand-stratified "
            "manifest, not from parent-directory disjointness."
        )
    elif product_report["product_id_status"] == "unavailable":
        print("Product ids are UNKNOWN; disjointness is enforced on parent directories.")

    train_tf = build_transforms(cfg["family"], train=True, profile=cfg["augmentation_profile"])
    val_tf = build_transforms(cfg["family"], train=False, profile=cfg["augmentation_profile"])

    train_ds = HypeVaultDataset(
        data_root,
        transform=train_tf,
        samples=train_samples,
        shuffle=False,
        seed=cfg["seed"],
    )
    val_ds = HypeVaultDataset(
        data_root,
        transform=val_tf,
        samples=val_samples,
        shuffle=False,
    )
    print(f"Train: {len(train_ds):,} | Val: {len(val_ds):,}")
    print(f"Class distribution — train: {np.bincount(train_ds.labels)}")
    effective_batch = int(cfg["batch_size"]) * int(cfg["accumulate_grad"])
    print(
        f"Effective batch: {effective_batch} "
        f"(batch_size={cfg['batch_size']} × accumulate_grad={cfg['accumulate_grad']})."
    )
    cfg["effective_batch"] = effective_batch
    if cfg.get("test_root"):
        test_root = Path(cfg["test_root"])
        print(f"Held-out test root: {test_root}")
        test_catalog = load_hypevault_samples(test_root)
        train_products = {product_group_key(path) for path, _ in train_samples}
        test_products = {product_group_key(path) for path, _ in test_catalog}
        test_overlap = train_products & test_products
        if test_overlap:
            examples = sorted(test_overlap)[:5]
            raise RuntimeError(
                "Held-out test root shares product folders with training data. "
                f"Examples: {examples}"
            )
    elif fixed_manifest:
        print(
            "Final test membership is recorded and is not evaluated. "
            "Architecture selection uses validation only."
        )
    else:
        print(
            "No --test_root provided: validation metrics are same-distribution only "
            "and may overstate production accuracy."
        )

    train_loader, sampler_info = build_train_loader(
        train_ds, cfg, [int(label) for label in train_ds.labels]
    )
    cfg["sampler"] = sampler_info["sampler"]
    cfg["shuffle"] = sampler_info["shuffle"]
    cfg["class_counts"] = {
        "authentic": int(sampler_info["counts"][0]),
        "deepfake": int(sampler_info["counts"][1]),
    }
    cfg["minority_ratio"] = sampler_info["minority_ratio"]
    cfg["preprocessing_version"] = cfg.get("evaluation_transform")
    cfg["dataset_counts"] = dataset_report["splits"]
    cfg["val_batch_size"] = int(cfg["batch_size"]) * 2
    val_loader = DataLoader(
        val_ds, batch_size=cfg["val_batch_size"], shuffle=False,
        num_workers=cfg["num_workers"], pin_memory=True,
        persistent_workers=cfg["num_workers"] > 0,
    )

    # ── AMP, recorded before the model build so config.json matches the run ──
    bf16_supported = bool(device.type == "cuda" and torch.cuda.is_bf16_supported())
    amp = resolve_amp(cfg["amp_dtype"], cuda=device.type == "cuda", bf16_supported=bf16_supported)
    cfg["amp_dtype_requested"] = cfg["amp_dtype"]
    cfg["amp_dtype"] = amp["amp_dtype"]
    cfg["amp_scaler"] = amp["amp_scaler"]
    cfg["amp_fallback"] = amp["amp_fallback"]
    write_experiment_config(out_dir / "config.json", cfg)
    if amp["amp_fallback"]:
        print(f"AMP requested {cfg['amp_dtype_requested']}; using {cfg['amp_dtype']}")

    # ── Model ─────────────────────────────────
    print(f"\nBuilding model: {cfg['family']} {cfg['classifier_arch']} ({cfg['model_name']})")
    model = build_authenticity_classifier(
        cfg["family"],
        cfg["classifier_arch"],
        image_size=int(cfg["img_size"]),
        pretrained=True,
        dropout=float(cfg["dropout"]),
    )
    model = model.to(device)

    param_count = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Total parameters: {param_count:.2f}M")

    optimizer, _param_groups, _table = build_optimizer(model, cfg)

    # ── LR Schedule ───────────────────────────
    # Count every optimizer update, including a partial accumulation window.
    steps_per_epoch = optimizer_updates_per_epoch(len(train_loader), cfg["accumulate_grad"])
    total_steps     = steps_per_epoch * cfg["num_epochs"]
    warmup_steps    = steps_per_epoch * cfg["warmup_epochs"]
    scheduler = cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    print(
        f"Schedule: {warmup_steps} warmup optimizer steps, "
        f"{total_steps} total optimizer steps "
        f"({steps_per_epoch} per epoch, including a partial accumulation window)."
    )
    print("Group table above is the base rate. Warmup and cosine scale every group together.")

    # ── Loss ──────────────────────────────────
    # BCE-with-logits on the raw logit. Label smoothing edits the targets.
    criterion = build_bce_loss()

    # ── AMP Scaler ────────────────────────────
    scaler = GradScaler("cuda", enabled=bool(cfg["amp_scaler"]))
    print(
        f"AMP mode: {cfg['amp_dtype']}"
        + (" (scaler on)" if cfg["amp_scaler"] else " (scaler off)")
    )

    # ── Training ──────────────────────────────
    # best_model.pt tracks the safety-constrained winner, not the highest F1.
    best_epoch = cfg["start_epoch"]
    best_val_logits = None
    best_val_labels = None
    history = []
    selection_state = new_selection_state(
        max_false_authentic_rate=cfg["max_false_authentic_rate"],
        metric_tolerance=cfg["metric_tolerance"],
        validation_threshold=cfg["validation_threshold"],
        model_family=cfg["family"],
        classifier_arch=cfg["classifier_arch"],
    )

    # ── Resume from checkpoint ─────────────────
    resumed_scheduler = False
    if cfg.get("resume_from"):
        resume_path = Path(cfg["resume_from"])
        if resume_path.exists():
            print(f"\nResuming from: {resume_path}")
            ckpt = load_training_checkpoint(
                resume_path, model, optimizer, scheduler, map_location=device
            )
            print(f"  Loaded weights from epoch {ckpt.get('epoch')}.")
            print("  Checkpoint selection uses the safety report, not validation F1.")
            resumed_scheduler = ckpt.get("scheduler_state") is not None
            selection_path = out_dir / "selection" / "checkpoint_selection.json"
            if selection_path.is_file():
                selection_state = load_selection_state(selection_path)
                print(
                    f"  Restored selection report — "
                    f"safety_validated={selection_state['safety_validated']} "
                    f"selected={selection_state['selected_checkpoint']}"
                )
            # Load history from disk so plots include prior epochs
            hist_path = out_dir / "history.json"
            if hist_path.exists():
                saved = json.loads(hist_path.read_text())
                history = saved.get("epochs", [])
                print(f"  Loaded {len(history)} epoch(s) of history from history.json")
        else:
            print(f"WARNING: resume_from path not found: {resume_path} — starting fresh")

    # Fast-forward LR scheduler past already-completed steps when the checkpoint
    # did not carry scheduler state (older checkpoints).
    steps_already_done = (cfg["start_epoch"] - 1) * steps_per_epoch
    global_step = steps_already_done
    if resumed_scheduler:
        global_step = int(ckpt.get("global_step", scheduler.last_epoch))
    else:
        for _ in range(steps_already_done):
            scheduler.step()

    print(f"\n{'='*60}")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    end_epoch = cfg["start_epoch"] + cfg["num_epochs"] - 1
    print(f"Starting training — epochs {cfg['start_epoch']}→{end_epoch}, {steps_per_epoch} steps/epoch")
    print(f"{'='*60}\n")
    for epoch in range(cfg["start_epoch"], end_epoch + 1):
        t0 = time.time()
        print(f"[Epoch {epoch}/{cfg['num_epochs']}] — TRAIN")

        train_loss, train_metrics, global_step = train_epoch(
            model, train_loader, optimizer, criterion,
            scaler, scheduler, device, cfg, global_step, epoch
        )

        val_loss, val_metrics = val_epoch(model, val_loader, criterion, device, cfg, epoch)
        # Fixed validation threshold. Not calibrated, and not chosen from the test set.
        scored = evaluate_logits(
            val_metrics["_logits"],
            val_metrics["_labels"],
            threshold=cfg["validation_threshold"],
        )
        far_value = scored["false_authentic_rate"]
        pr_value = scored["pr_auc"]
        far_text = "n/a" if far_value is None else f"{far_value:.4f}"
        pr_text = "n/a" if pr_value is None else f"{pr_value:.4f}"

        elapsed = time.time() - t0
        print(
            f"\n{'─'*55}\n"
            f"  Epoch {epoch:3d} | {elapsed:.0f}s\n"
            f"  Train | loss={train_loss:.4f} acc={train_metrics['acc']*100:.2f}% f1={train_metrics['f1']:.4f}\n"
            f"  Val   | loss={val_loss:.4f}  acc={val_metrics['acc']*100:.2f}%  "
            f"f1={val_metrics['f1']:.4f}  far={far_text}  pr_auc={pr_text}\n"
            f"{'─'*55}\n"
        )

        # strip internal tensors before logging
        train_metrics_log = {k: v for k, v in train_metrics.items() if not k.startswith("_")}
        val_metrics_log   = {k: v for k, v in val_metrics.items()   if not k.startswith("_")}
        row = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
               **{f"train_{k}": v for k, v in train_metrics_log.items()},
               **{f"val_{k}": v for k, v in val_metrics_log.items()},
               "val_false_authentic_rate": far_value,
               "val_pr_auc": pr_value,
               "val_threshold": scored["threshold"],
               "epoch_seconds": elapsed,
               "learning_rates": [float(value) for value in scheduler.get_last_lr()]}
        for key in (
            "roc_auc",
            "pr_auc",
            "accuracy",
            "precision",
            "recall",
            "f1",
            "false_authentic",
            "false_authentic_rate",
            "false_fake",
            "false_fake_rate",
            "threshold",
            "threshold_policy",
        ):
            row[f"val_{key}"] = scored[key]
        if device.type == "cuda":
            row["peak_allocated_bytes"] = int(torch.cuda.max_memory_allocated())
            row["peak_reserved_bytes"] = int(torch.cuda.max_memory_reserved())
            row["throughput_images_per_sec"] = (len(train_ds) + len(val_ds)) / max(elapsed, 1e-6)
        history.append(row)

        checkpoint_kwargs = dict(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            config=cfg,
            global_step=global_step,
            val_acc=val_metrics["acc"],
            val_f1=val_metrics["f1"],
        )
        # Only an epoch file that is actually written can be selected. save_every
        # greater than 1 leaves the skipped epochs in the metric history only.
        epoch_path = epoch_artifact_path(out_dir, epoch)
        if epoch % int(cfg["save_every"]) == 0:
            save_training_checkpoint(epoch_path, **checkpoint_kwargs)
            print(f"  Saved checkpoint: {epoch_path.name}")
        selection_state = update_selection(
            selection_state,
            candidate_from_validation(
                checkpoint_id=f"epoch_{epoch:03d}",
                epoch=epoch,
                global_step=global_step,
                validation_loss=val_loss,
                model_family=cfg["family"],
                classifier_arch=cfg["classifier_arch"],
                metrics=scored,
                checkpoint_path=str(epoch_path),
                checkpoint_available=epoch_path.is_file(),
            ),
        )
        row["selected_checkpoint"] = selection_state["selected_checkpoint"]
        row["safety_validated"] = bool(selection_state["safety_validated"])
        row["selection_action"] = selection_state["action"]
        write_history(out_dir, history)
        write_selection_report(out_dir, selection_state)

        save_training_checkpoint(out_dir / "latest.pt", **checkpoint_kwargs)

        chosen = selection_state["summary"]["selected_metrics"]
        if should_save_production_checkpoint(selection_state):
            best_epoch = epoch
            best_val_logits = val_metrics.get("_logits")
            best_val_labels = val_metrics.get("_labels")
            save_training_checkpoint(out_dir / "best_model.pt", **checkpoint_kwargs)
            print(
                "  Safety-validated checkpoint "
                f"{chosen['checkpoint_id']} "
                f"pr_auc={chosen['pr_auc']:.4f} f1={chosen['f1']:.4f} "
                f"false_authentic_rate={chosen['false_authentic_rate']:.4f} "
                "→ best_model.pt"
            )
        elif (
            selection_state["action"] == "increment"
            and selection_state["epochs_without_improvement"] >= cfg["early_stop_patience"]
        ):
            print(
                "\nEarly stopping after "
                f"{selection_state['epochs_without_improvement']} validation epochs "
                "without a safety-ranking improvement."
            )
            break
        elif selection_state["action"] == "hold":
            print(
                "  Not safety-eligible. best_model.pt was not updated. "
                "Early-stopping patience was not advanced."
            )
        else:
            print(
                "  No safety-ranking improvement. "
                f"Patience {selection_state['epochs_without_improvement']}"
                f"/{cfg['early_stop_patience']}."
            )

    # ── Final save ────────────────────────────
    # final_model.pt is the last epoch, kept for analysis. It is not the
    # safety selection. checkpoint_selection.json is the source of truth.
    final_ckpt = out_dir / "final_model.pt"
    save_training_checkpoint(
        final_ckpt,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=epoch,
        config=cfg,
        global_step=global_step,
        val_acc=val_metrics["acc"],
        val_f1=val_metrics["f1"],
    )
    write_selection_report(out_dir, selection_state)
    resource = {
        "gpu_name": cfg.get("gpu_name"),
        "amp_dtype": cfg["amp_dtype"],
        "batch_size": int(cfg["batch_size"]),
        "accumulate_grad": int(cfg["accumulate_grad"]),
        "effective_batch": int(cfg["batch_size"]) * int(cfg["accumulate_grad"]),
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()) if device.type == "cuda" else None,
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()) if device.type == "cuda" else None,
        "test_root": cfg.get("test_root"),
        "test_set_used_for_selection": False,
        "production_threshold": None,
    }
    (out_dir / "resource.json").write_text(json.dumps(resource, indent=2) + "\n")

    test_summary = None
    if cfg.get("test_root") and selection_state["safety_validated"]:
        test_root = Path(cfg["test_root"])
        print(f"\nEvaluating held-out test set from: {test_root}")
        print("Test metrics are reported only. They do not change checkpoint selection.")
        test_ds = HypeVaultDataset(
            test_root,
            transform=val_tf,
            samples=test_catalog,
            shuffle=False,
        )
        test_loader = DataLoader(
            test_ds,
            batch_size=cfg["batch_size"] * 2,
            shuffle=False,
            num_workers=cfg["num_workers"],
            pin_memory=True,
            persistent_workers=cfg["num_workers"] > 0,
        )
        best_ckpt = out_dir / "best_model.pt"
        state = torch.load(best_ckpt, map_location=device)
        model.load_state_dict(state["model_state"])
        test_loss, test_metrics = val_epoch(model, test_loader, criterion, device, cfg)
        test_summary = {
            "data_root": str(test_root),
            "num_images": len(test_ds),
            "loss": test_loss,
            "selection_used": False,
            **{key: value for key, value in test_metrics.items() if not key.startswith("_")},
        }
        write_history(out_dir, history, test_summary)
        print(
            f"Held-out test | loss={test_loss:.4f} "
            f"acc={test_metrics['acc']*100:.2f}% f1={test_metrics['f1']:.4f}"
        )
    else:
        write_history(out_dir, history)

    selected_metrics = selection_state["summary"]["selected_metrics"]
    print(f"\n{'='*60}")
    print("Training complete.")
    if selection_state["safety_validated"] and selected_metrics is not None:
        best_epoch = int(selected_metrics["epoch"])
        print(f"Safety-validated checkpoint: {selected_metrics['checkpoint_id']}")
        print(
            f"Validation PR-AUC={selected_metrics['pr_auc']:.4f}  "
            f"F1={selected_metrics['f1']:.4f}  "
            f"false_authentic_rate={selected_metrics['false_authentic_rate']:.4f}  "
            f"threshold={selected_metrics['threshold']:.2f}"
        )
    else:
        print(UNQUALIFIED_MESSAGE)
        print("final_model.pt is retained for analysis and is not a production checkpoint.")
    if test_summary:
        print(
            f"Held-out test accuracy: {test_summary['acc']*100:.2f}%   "
            f"test F1: {test_summary['f1']:.4f}"
        )
    print(f"Selection report: {out_dir / 'selection' / 'checkpoint_selection.json'}")
    print(f"Checkpoints in: {out_dir}")
    print(f"{'='*60}\n")

    # ── Model contract ─────────────────────────
    # Records the trained architecture. Does not read backend defaults and does
    # not overwrite checkpoint weights. Thresholds stay unset until calibration.
    try:
        best_ckpt = out_dir / "best_model.pt"
        manifest_checkpoint = (
            best_ckpt if selection_state["safety_validated"] and best_ckpt.is_file() else None
        )
        manifest = write_training_manifest(
            out_dir,
            backbone=cfg["model_name"],
            image_size=int(cfg["img_size"]),
            training_config=cfg,
            checkpoint_path=manifest_checkpoint,
            dataset_samples=catalog,
            repo_root=Path(__file__).resolve().parents[1],
        )
        print(f"Wrote model manifest {manifest['model_version']}")
    except Exception as e:
        print(f"Model manifest write failed (non-fatal): {e}")

    # ── Generate plots ─────────────────────────
    print("Generating training plots...")
    try:
        plot_training_curves(
            history,
            out_dir,
            best_epoch,
            subtitle="Checkpoint selection is safety-constrained PR-AUC, not max F1.",
        )
        if (
            selection_state["safety_validated"]
            and best_val_logits is not None
            and best_val_labels is not None
            and selected_metrics is not None
        ):
            plot_confusion_eval(
                best_val_logits, best_val_labels,
                out_dir, len(val_ds),
                selected_metrics["accuracy"], selected_metrics["f1"],
                threshold=float(selected_metrics["threshold"]),
            )
    except Exception as e:
        print(f"  Plot generation failed (non-fatal): {e}")

    # ── Export for Triton ─────────────────────
    print("Exporting ONNX for TensorRT conversion...")
    try:
        export_onnx(model, cfg, out_dir, device)
    except Exception as e:
        print(f"ONNX export failed (run export_tensorrt.py separately): {e}")


def export_onnx(model, cfg: dict, out_dir: Path, device):
    """Export to ONNX (used by export_tensorrt.py to build TRT FP16 engine)."""
    model.eval()
    dummy = torch.zeros(1, 3, cfg["img_size"], cfg["img_size"], device=device)
    onnx_path = out_dir / "dinov2_hypevault.onnx"

    torch.onnx.export(
        model,
        dummy,
        str(onnx_path),
        export_params=True,
        opset_version=18,
        do_constant_folding=True,
        input_names=["input__0"],
        output_names=["output__0"],
        dynamic_axes={
            "input__0": {0: "batch_size"},
            "output__0": {0: "batch_size"},
        },
    )
    print(f"ONNX exported → {onnx_path}")


if __name__ == "__main__":
    main()
