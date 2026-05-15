"""
HypeVault — DINOv2 local training (RTX 5080 profile)
=====================================================
Binary classification: Authentic (0) vs Deepfake (1)

Smaller backbone (DINOv2-Base via timm), **504×504** input (multiple of patch 14; 512² is invalid for ViT-P/14), batch settings sized
for ~16 GB VRAM. Original `ml/train.py` remains the DINOv2-Giant / 518px path.

Dataset layout expected:
  Label_0_Sneakers/<brand>/*.jpg   → label 0
  Label_0_Watches/<brand>/*.jpg    → label 0
  Label_1_Sneakers/<brand>/*.jpg   → label 1
  Label_1_Watches/<brand>/*.jpg    → label 1
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

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
    output_dir="/home/tensortorch26/Desktop/HypeVault/ml_rtx5080/checkpoints",
    # DINOv2-Base (timm id); ~86M backbone params — fits 16 GB at 512²
    model_name="vit_base_patch14_dinov2.lvd142m",
    # DINOv2 patch14 → H and W must be divisible by 14 (512 is invalid). 504 = 36×14, smaller than 518.
    img_size=504,
    batch_size=16,
    num_epochs=20,
    lr=2e-5,
    weight_decay=0.05,
    warmup_epochs=2,
    label_smoothing=0.05,
    dropout=0.2,
    val_split=0.1,
    num_workers=6,
    seed=42,
    grad_clip=1.0,
    accumulate_grad=2,                  # effective batch = 16 * 2 = 32
    amp_dtype="bf16",
    save_every=2,
    resume_from=None,                   # path to best_model.pt to resume from
    start_epoch=1,                      # epoch to start from when resuming
    early_stop_patience=5,
    mixup_alpha=0.2,
    test_root=None,
    # Confidence threshold for AUTHENTIC verdict (raise if too many false-negatives)
    # Default 0.5 = 50% sigmoid. Raise to 0.65–0.75 if real items get flagged as fake.
    confidence_threshold=0.5,
)

# ─────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────

LABEL_DIRS = {
    0: ["Label_0_Sneakers", "Label_0_Watches"],
    1: ["Label_1_Sneakers", "Label_1_Watches"],
}


def product_group_key(path: Path) -> str:
    """Stable product identity: label-folder brand/model directory (image parent)."""
    return str(path.parent.resolve())


def load_hypevault_samples(root: Path) -> list[tuple[Path, int]]:
    """Deterministic catalog order: label folder, brand, then filename."""
    samples: list[tuple[Path, int]] = []
    for label, dirs in LABEL_DIRS.items():
        for d in dirs:
            folder = root / d
            if not folder.exists():
                raise FileNotFoundError(f"Dataset folder not found: {folder}")
            for brand_dir in sorted(folder.iterdir()):
                if not brand_dir.is_dir():
                    continue
                for img_path in sorted(brand_dir.iterdir()):
                    if img_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}:
                        samples.append((img_path, label))
    return samples


def split_indices_by_product(
    samples: list[tuple[Path, int]],
    val_split: float,
    seed: int,
) -> tuple[list[int], list[int]]:
    """Assign whole products to train or val; stratify by class at the product level."""
    groups: dict[str, list[int]] = {}
    for idx, (path, _) in enumerate(samples):
        groups.setdefault(product_group_key(path), []).append(idx)

    if not groups:
        return [], []

    labels_by_key = {key: samples[idxs[0]][1] for key, idxs in groups.items()}
    keys_by_label: dict[int, list[str]] = {}
    for key in groups:
        keys_by_label.setdefault(labels_by_key[key], []).append(key)

    rng = np.random.default_rng(seed)
    val_key_set: set[str] = set()
    for label in sorted(keys_by_label):
        keys = list(keys_by_label[label])
        rng.shuffle(keys)
        if val_split <= 0 or len(keys) <= 1:
            n_val_groups = 0
        else:
            n_val_groups = int(round(len(keys) * val_split))
            n_val_groups = max(1, min(len(keys) - 1, n_val_groups))
        val_key_set.update(keys[:n_val_groups])

    train_idx: list[int] = []
    val_idx: list[int] = []
    for key, idxs in groups.items():
        if key in val_key_set:
            val_idx.extend(idxs)
        else:
            train_idx.extend(idxs)
    return train_idx, val_idx


def verify_product_split(
    train_samples: list[tuple[Path, int]],
    val_samples: list[tuple[Path, int]],
) -> dict[str, int]:
    """Fail fast if any product folder or image path appears in both splits."""
    train_products = {product_group_key(path) for path, _ in train_samples}
    val_products = {product_group_key(path) for path, _ in val_samples}
    product_overlap = train_products & val_products
    if product_overlap:
        examples = sorted(product_overlap)[:5]
        raise RuntimeError(
            "Product-level split leak: "
            f"{len(product_overlap)} folders appear in both train and val. "
            f"Examples: {examples}"
        )

    train_files = {str(path.resolve()) for path, _ in train_samples}
    val_files = {str(path.resolve()) for path, _ in val_samples}
    file_overlap = train_files & val_files
    if file_overlap:
        examples = sorted(file_overlap)[:5]
        raise RuntimeError(
            "Image-level split leak: "
            f"{len(file_overlap)} files appear in both train and val. "
            f"Examples: {examples}"
        )

    return {
        "train_products": len(train_products),
        "val_products": len(val_products),
        "train_images": len(train_samples),
        "val_images": len(val_samples),
        "product_overlap": 0,
        "file_overlap": 0,
    }


def write_split_manifest(
    out_dir: Path,
    train_samples: list[tuple[Path, int]],
    val_samples: list[tuple[Path, int]],
    seed: int,
    val_split: float,
) -> None:
    payload = {
        "seed": seed,
        "val_split": val_split,
        "train_product_folders": sorted({product_group_key(p) for p, _ in train_samples}),
        "val_product_folders": sorted({product_group_key(p) for p, _ in val_samples}),
        "train_images": len(train_samples),
        "val_images": len(val_samples),
    }
    (out_dir / "split_manifest.json").write_text(json.dumps(payload, indent=2))


class HypeVaultDataset(Dataset):
    """
    Loads images from the 4 label folders:
      Label_0_Sneakers, Label_0_Watches → 0 (Authentic)
      Label_1_Sneakers, Label_1_Watches → 1 (Deepfake)
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


def build_transforms(img_size: int, train: bool) -> transforms.Compose:
    """
    Augmentation strategy designed to close the distribution gap between
    clean training images (professional product photography) and real
    seller uploads (phone photos, varied lighting, compression artifacts).

    Key additions vs. vanilla DINOv2 fine-tune:
      - JPEG compression simulation  → handles listing photo artifacts
      - Random perspective            → handles phone-camera angles
      - Random rotation (±15°)        → handles non-studio shots
      - Heavy blur augmentation       → handles out-of-focus uploads
      - Aggressive crop (0.55 min)    → handles partial/cropped images
    """
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]

    class SimulateJPEG:
        """Simulate JPEG compression artifacts (common in real listings)."""
        def __init__(self, quality_range=(40, 95)):
            self.quality_range = quality_range
        def __call__(self, img: Image.Image) -> Image.Image:
            import io, random
            quality = random.randint(*self.quality_range)
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=quality)
            buf.seek(0)
            return Image.open(buf).copy()

    if train:
        return transforms.Compose([
            # Simulate phone crop / partial product shot
            transforms.RandomResizedCrop(
                img_size,
                scale=(0.55, 1.0),   # more aggressive than default 0.7
                ratio=(0.8, 1.25),   # allow portrait/landscape crops
                interpolation=InterpolationMode.BICUBIC,
            ),
            transforms.RandomHorizontalFlip(),
            # Phone camera rotation / tilt
            transforms.RandomRotation(degrees=15, interpolation=InterpolationMode.BICUBIC),
            # Perspective distortion (table/shelf photos)
            transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
            # Lighting variation
            transforms.ColorJitter(brightness=0.35, contrast=0.35, saturation=0.25, hue=0.08),
            transforms.RandomGrayscale(p=0.03),
            # Focus blur (phone photos)
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=7, sigma=(0.1, 3.0))], p=0.2),
            # JPEG compression artifacts
            transforms.RandomApply([SimulateJPEG(quality_range=(40, 90))], p=0.3),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
    else:
        return transforms.Compose([
            transforms.Resize(img_size + 16, interpolation=InterpolationMode.BICUBIC),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])


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

    amp_dtype = torch.bfloat16 if cfg["amp_dtype"] == "bf16" else torch.float16
    lr_now = scheduler.get_last_lr()[0]

    pbar = tqdm(
        enumerate(loader),
        total=len(loader),
        desc=f"Epoch {epoch} [Train]",
        ncols=110,
        leave=True,
    )
    for batch_idx, (images, labels) in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if cfg["mixup_alpha"] > 0:
            images, labels = mixup_batch(images, labels, cfg["mixup_alpha"])

        with autocast("cuda", dtype=amp_dtype):
            logits = model(images)
            loss = criterion(logits, labels)
            loss = loss / cfg["accumulate_grad"]

        scaler.scale(loss).backward()

        if (batch_idx + 1) % cfg["accumulate_grad"] == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            lr_now = scheduler.get_last_lr()[0]
            step += 1

        batch_loss = loss.item() * cfg["accumulate_grad"]
        total_loss += batch_loss
        all_logits.append(logits.detach().float())
        all_labels.append(labels.detach().float())

        pbar.set_postfix(loss=f"{batch_loss:.4f}", lr=f"{lr_now:.2e}", refresh=False)

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

    amp_dtype = torch.bfloat16 if cfg["amp_dtype"] == "bf16" else torch.float16
    desc = f"Epoch {epoch} [ Val ]" if epoch is not None else "Evaluating"

    pbar = tqdm(loader, total=len(loader), desc=desc, ncols=110, leave=True)
    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with autocast("cuda", dtype=amp_dtype):
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
) -> None:
    if not HAS_PLOT:
        return
    probs  = torch.sigmoid(logits).cpu().numpy().astype(float)
    y_true = labels.cpu().numpy().astype(int)
    y_pred = (probs >= 0.5).astype(int)

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

    # ── Paths ──────────────────────────────────
    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(cfg, indent=2))

    # ── Reproducibility ────────────────────────
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])

    # ── Device ────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    # ── Dataset ───────────────────────────────
    # Validation on the same scraped distribution can look strong when products
    # leak across splits; use --test_root for held-out products before production claims.
    data_root = Path(cfg["data_root"])
    print(f"\nLoading dataset from: {data_root}")
    catalog = load_hypevault_samples(data_root)
    total = len(catalog)
    print(f"Total images: {total:,}")

    train_idx, val_idx = split_indices_by_product(
        catalog,
        cfg["val_split"],
        cfg["seed"],
    )
    train_samples = [catalog[i] for i in train_idx]
    val_samples = [catalog[i] for i in val_idx]
    split_stats = verify_product_split(train_samples, val_samples)
    write_split_manifest(out_dir, train_samples, val_samples, cfg["seed"], cfg["val_split"])
    print(
        f"Product groups: train {split_stats['train_products']:,} | "
        f"val {split_stats['val_products']:,} "
        f"(saved split_manifest.json)"
    )

    train_tf = build_transforms(cfg["img_size"], train=True)
    val_tf   = build_transforms(cfg["img_size"], train=False)

    train_ds = HypeVaultDataset(
        data_root,
        transform=train_tf,
        samples=train_samples,
        shuffle=True,
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
    else:
        print(
            "No --test_root provided: validation metrics are same-distribution only "
            "and may overstate production accuracy."
        )

    sampler = make_weighted_sampler(train_ds)

    train_loader = DataLoader(
        train_ds, batch_size=cfg["batch_size"], sampler=sampler,
        num_workers=cfg["num_workers"], pin_memory=True, drop_last=True,
        persistent_workers=cfg["num_workers"] > 0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg["batch_size"] * 2, shuffle=False,
        num_workers=cfg["num_workers"], pin_memory=True,
        persistent_workers=cfg["num_workers"] > 0,
    )

    # ── Model ─────────────────────────────────
    print(f"\nBuilding model: {cfg['model_name']}")
    model = DINOv2Classifier(cfg["model_name"], dropout=cfg["dropout"])
    model = model.to(device)

    param_count = sum(p.numel() for p in model.parameters()) / 1e9
    print(f"Total parameters: {param_count:.2f}B")

    # ── Optimizer — layer-wise LR decay (optional but helps) ──
    # Give backbone a 10× lower LR than the head
    backbone_params = [p for n, p in model.named_parameters() if "head" not in n and p.requires_grad]
    head_params     = [p for n, p in model.named_parameters() if "head" in  n and p.requires_grad]
    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": cfg["lr"],      "weight_decay": cfg["weight_decay"]},
            {"params": head_params,     "lr": cfg["lr"] * 10, "weight_decay": 0.0},
        ],
        betas=(0.9, 0.999),
        eps=1e-8,
    )

    # ── LR Schedule ───────────────────────────
    steps_per_epoch = len(train_loader) // cfg["accumulate_grad"]
    total_steps     = steps_per_epoch * cfg["num_epochs"]
    warmup_steps    = steps_per_epoch * cfg["warmup_epochs"]
    scheduler = cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    # ── Loss ──────────────────────────────────
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=None  # classes balanced by sampler
    )

    # ── AMP Scaler ────────────────────────────
    use_bf16 = (cfg["amp_dtype"] == "bf16") and torch.cuda.is_bf16_supported()
    scaler = GradScaler("cuda", enabled=not use_bf16)  # bf16 doesn't need loss scaling
    print(f"AMP mode: {'bf16 (no scaler)' if use_bf16 else 'fp16 (with scaler)'}")

    # ── Training ──────────────────────────────
    best_val_acc = 0.0
    best_val_f1  = 0.0
    best_epoch   = cfg["start_epoch"]
    best_val_logits = None
    best_val_labels = None
    patience_count = 0
    history = []

    # ── Resume from checkpoint ─────────────────
    if cfg.get("resume_from"):
        resume_path = Path(cfg["resume_from"])
        if resume_path.exists():
            print(f"\nResuming from: {resume_path}")
            ckpt = torch.load(resume_path, map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model_state"])
            best_val_acc = ckpt.get("val_acc", 0.0)
            best_val_f1  = ckpt.get("val_f1",  0.0)
            print(f"  Loaded weights — previous best val_acc={best_val_acc*100:.2f}% val_f1={best_val_f1:.4f}")
            # Load history from disk so plots include prior epochs
            hist_path = out_dir / "history.json"
            if hist_path.exists():
                saved = json.loads(hist_path.read_text())
                history = saved.get("epochs", [])
                print(f"  Loaded {len(history)} epoch(s) of history from history.json")
        else:
            print(f"WARNING: resume_from path not found: {resume_path} — starting fresh")

    # Fast-forward LR scheduler past already-completed steps
    steps_already_done = (cfg["start_epoch"] - 1) * steps_per_epoch
    global_step = steps_already_done
    for _ in range(steps_already_done):
        scheduler.step()

    print(f"\n{'='*60}")
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

        elapsed = time.time() - t0
        print(
            f"\n{'─'*55}\n"
            f"  Epoch {epoch:3d} | {elapsed:.0f}s\n"
            f"  Train | loss={train_loss:.4f} acc={train_metrics['acc']*100:.2f}% f1={train_metrics['f1']:.4f}\n"
            f"  Val   | loss={val_loss:.4f}  acc={val_metrics['acc']*100:.2f}%  f1={val_metrics['f1']:.4f}\n"
            f"{'─'*55}\n"
        )

        # strip internal tensors before logging
        train_metrics_log = {k: v for k, v in train_metrics.items() if not k.startswith("_")}
        val_metrics_log   = {k: v for k, v in val_metrics.items()   if not k.startswith("_")}
        row = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
               **{f"train_{k}": v for k, v in train_metrics_log.items()},
               **{f"val_{k}": v for k, v in val_metrics_log.items()}}
        history.append(row)
        write_history(out_dir, history)

        # ── Save checkpoint ──────────────────
        if epoch % cfg["save_every"] == 0:
            ckpt = out_dir / f"epoch_{epoch:03d}.pt"
            torch.save({"epoch": epoch, "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(), "val_acc": val_metrics["acc"], "val_f1": val_metrics["f1"]}, ckpt)
            print(f"  Saved checkpoint: {ckpt.name}")

        # ── Save best ────────────────────────
        if val_metrics["f1"] > best_val_f1:
            best_val_f1  = val_metrics["f1"]
            best_val_acc = val_metrics["acc"]
            best_epoch   = epoch
            best_val_logits = val_metrics.get("_logits")
            best_val_labels = val_metrics.get("_labels")
            patience_count = 0
            best_ckpt = out_dir / "best_model.pt"
            torch.save({"epoch": epoch, "model_state": model.state_dict(), "val_acc": val_metrics["acc"], "val_f1": val_metrics["f1"]}, best_ckpt)
            print(f"  ★ New best! val_acc={best_val_acc*100:.2f}% val_f1={best_val_f1:.4f} → saved to best_model.pt")
        else:
            patience_count += 1
            if patience_count >= cfg["early_stop_patience"]:
                print(f"\nEarly stopping after {patience_count} epochs without improvement.")
                break

    # ── Final save ────────────────────────────
    final_ckpt = out_dir / "final_model.pt"
    torch.save({"epoch": epoch, "model_state": model.state_dict()}, final_ckpt)

    test_summary = None
    if cfg.get("test_root"):
        test_root = Path(cfg["test_root"])
        print(f"\nEvaluating held-out test set from: {test_root}")
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
        if best_ckpt.exists():
            state = torch.load(best_ckpt, map_location=device)
            model.load_state_dict(state["model_state"])
        test_loss, test_metrics = val_epoch(model, test_loader, criterion, device, cfg)
        test_summary = {
            "data_root": str(test_root),
            "num_images": len(test_ds),
            "loss": test_loss,
            **test_metrics,
        }
        write_history(out_dir, history, test_summary)
        print(
            f"Held-out test | loss={test_loss:.4f} "
            f"acc={test_metrics['acc']*100:.2f}% f1={test_metrics['f1']:.4f}"
        )
    else:
        write_history(out_dir, history)

    print(f"\n{'='*60}")
    print(f"Training complete.")
    print(f"Best val accuracy: {best_val_acc*100:.2f}%   Best val F1: {best_val_f1:.4f}")
    if test_summary:
        print(
            f"Held-out test accuracy: {test_summary['acc']*100:.2f}%   "
            f"test F1: {test_summary['f1']:.4f}"
        )
    print(f"Checkpoints in: {out_dir}")
    print(f"{'='*60}\n")

    # ── Generate plots ─────────────────────────
    print("Generating training plots...")
    try:
        plot_training_curves(history, out_dir, best_epoch)
        if best_val_logits is not None and best_val_labels is not None:
            plot_confusion_eval(
                best_val_logits, best_val_labels,
                out_dir, len(val_ds),
                best_val_acc, best_val_f1,
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
