"""
HypeVault — DINOv2-Giant Full Fine-Tuning
==========================================
Binary classification: Authentic (0) vs Deepfake (1)

Dataset layout expected:
  Label_0_Sneakers/<brand>/*.jpg   → label 0
  Label_0_Watches/<brand>/*.jpg    → label 0
  Label_1_Sneakers/<brand>/*.jpg   → label 1
  Label_1_Watches/<brand>/*.jpg    → label 1

Hardware: RTX 6000 Pro Blackwell (96 GB VRAM)
Resolution: 518 × 518 (DINOv2 native patch grid)
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
from torch.cuda.amp import GradScaler, autocast
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from PIL import Image, ImageFilter
import numpy as np

try:
    import timm
    HAS_TIMM = True
except ImportError:
    HAS_TIMM = False

# ─────────────────────────────────────────────
# Hyper-parameters (all overridable via CLI)
# ─────────────────────────────────────────────
DEFAULTS = dict(
    data_root="/home/tensortorch26/Desktop/scraper",
    output_dir="/home/tensortorch26/Desktop/HypeVault/ml/checkpoints",
    model_name="dinov2_vitg14_reg",    # DINOv2-Giant with registers (best variant)
    img_size=518,
    batch_size=32,                      # 96 GB lets us go up to 64; 32 is safe
    num_epochs=20,
    lr=1e-5,                            # low LR for full fine-tune of 1.1B params
    weight_decay=0.05,
    warmup_epochs=2,
    label_smoothing=0.05,
    dropout=0.2,
    val_split=0.1,
    num_workers=8,
    seed=42,
    grad_clip=1.0,
    accumulate_grad=1,                  # gradient accumulation steps
    amp_dtype="bf16",                   # bf16 for Blackwell; use fp16 if needed
    save_every=5,                       # save checkpoint every N epochs
    early_stop_patience=5,
    mixup_alpha=0.2,
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
      Label_0_Sneakers, Label_0_Watches → 0 (Authentic)
      Label_1_Sneakers, Label_1_Watches → 1 (Deepfake)
    """

    LABEL_DIRS = {
        0: ["Label_0_Sneakers", "Label_0_Watches"],
        1: ["Label_1_Sneakers", "Label_1_Watches"],
    }

    def __init__(
        self,
        root: Path,
        transform=None,
        indices: list[int] | None = None,
    ):
        self.root = root
        self.transform = transform
        self.samples: list[tuple[Path, int]] = []

        for label, dirs in self.LABEL_DIRS.items():
            for d in dirs:
                folder = root / d
                if not folder.exists():
                    raise FileNotFoundError(f"Dataset folder not found: {folder}")
                for brand_dir in sorted(folder.iterdir()):
                    if not brand_dir.is_dir():
                        continue
                    for img_path in sorted(brand_dir.iterdir()):
                        if img_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}:
                            self.samples.append((img_path, label))

        if indices is not None:
            self.samples = [self.samples[i] for i in indices]

        # pre-compute class counts for weighted sampler
        self.labels = [s[1] for s in self.samples]

    def __len__(self):
        return len(self.samples)

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
    DINOv2-Giant backbone + binary classification head.
    Full fine-tuning — all 1.1B parameters trained end-to-end.
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
            }
            timm_name = timm_names.get(model_name, model_name)
            print(f"Loading {timm_name} via timm...")
            model = timm.create_model(timm_name, pretrained=True, num_classes=0)
            print(f"  Loaded via timm ✓")
            return model

        raise RuntimeError(
            "Could not load DINOv2. Install timm or ensure torch.hub can reach github."
        )

    def _get_embed_dim(self, backbone) -> int:
        # DINOv2-Giant: 1536. ViT-Large: 1024
        if hasattr(backbone, "embed_dim"):
            return backbone.embed_dim
        if hasattr(backbone, "num_features"):
            return backbone.num_features
        # fallback: run a dummy forward
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 518, 518)
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
    device, cfg, step_offset: int
) -> tuple[float, dict, int]:
    model.train()
    total_loss = 0.0
    all_logits, all_labels = [], []
    step = step_offset

    amp_dtype = torch.bfloat16 if cfg["amp_dtype"] == "bf16" else torch.float16

    for batch_idx, (images, labels) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # Mixup
        if cfg["mixup_alpha"] > 0:
            images, labels = mixup_batch(images, labels, cfg["mixup_alpha"])

        with autocast(dtype=amp_dtype):
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
            step += 1

        total_loss += loss.item() * cfg["accumulate_grad"]
        all_logits.append(logits.detach().float())
        all_labels.append(labels.detach().float())

        if batch_idx % 50 == 0:
            lr_now = scheduler.get_last_lr()[0]
            print(f"  [batch {batch_idx:4d}/{len(loader)}] loss={loss.item() * cfg['accumulate_grad']:.4f} lr={lr_now:.2e}")

    metrics = compute_metrics(
        torch.cat(all_logits),
        torch.cat(all_labels).round().long()
    )
    return total_loss / len(loader), metrics, step


@torch.no_grad()
def val_epoch(model, loader, criterion, device, cfg) -> tuple[float, dict]:
    model.eval()
    total_loss = 0.0
    all_logits, all_labels = [], []

    amp_dtype = torch.bfloat16 if cfg["amp_dtype"] == "bf16" else torch.float16

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with autocast(dtype=amp_dtype):
            logits = model(images)
            loss = criterion(logits, labels)
        total_loss += loss.item()
        all_logits.append(logits.float())
        all_labels.append(labels.float())

    metrics = compute_metrics(torch.cat(all_logits), torch.cat(all_labels))
    return total_loss / len(loader), metrics


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="HypeVault DINOv2-Giant fine-tuning")
    for k, v in DEFAULTS.items():
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
    data_root = Path(cfg["data_root"])
    print(f"\nLoading dataset from: {data_root}")
    full_ds = HypeVaultDataset(data_root, transform=None)
    total = len(full_ds)
    print(f"Total images: {total:,}")

    val_size = int(total * cfg["val_split"])
    train_size = total - val_size
    rng = np.random.default_rng(cfg["seed"])
    perm = rng.permutation(total).tolist()
    train_idx, val_idx = perm[:train_size], perm[train_size:]

    train_tf = build_transforms(cfg["img_size"], train=True)
    val_tf   = build_transforms(cfg["img_size"], train=False)

    train_ds = HypeVaultDataset(data_root, transform=train_tf, indices=train_idx)
    val_ds   = HypeVaultDataset(data_root, transform=val_tf,   indices=val_idx)
    print(f"Train: {len(train_ds):,} | Val: {len(val_ds):,}")
    print(f"Class distribution — train: {np.bincount(train_ds.labels)}")

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
    scaler = GradScaler(enabled=not use_bf16)  # bf16 doesn't need loss scaling
    print(f"AMP mode: {'bf16 (no scaler)' if use_bf16 else 'fp16 (with scaler)'}")

    # ── Training ──────────────────────────────
    best_val_acc = 0.0
    best_val_f1  = 0.0
    patience_count = 0
    global_step = 0
    history = []

    print(f"\n{'='*60}")
    print(f"Starting training — {cfg['num_epochs']} epochs, {steps_per_epoch} steps/epoch")
    print(f"{'='*60}\n")

    for epoch in range(1, cfg["num_epochs"] + 1):
        t0 = time.time()
        print(f"[Epoch {epoch}/{cfg['num_epochs']}] — TRAIN")

        train_loss, train_metrics, global_step = train_epoch(
            model, train_loader, optimizer, criterion,
            scaler, scheduler, device, cfg, global_step
        )

        print(f"[Epoch {epoch}/{cfg['num_epochs']}] — VALIDATE")
        val_loss, val_metrics = val_epoch(model, val_loader, criterion, device, cfg)

        elapsed = time.time() - t0
        print(
            f"\n{'─'*55}\n"
            f"  Epoch {epoch:3d} | {elapsed:.0f}s\n"
            f"  Train | loss={train_loss:.4f} acc={train_metrics['acc']*100:.2f}% f1={train_metrics['f1']:.4f}\n"
            f"  Val   | loss={val_loss:.4f}  acc={val_metrics['acc']*100:.2f}%  f1={val_metrics['f1']:.4f}\n"
            f"{'─'*55}\n"
        )

        row = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, **{f"train_{k}": v for k, v in train_metrics.items()}, **{f"val_{k}": v for k, v in val_metrics.items()}}
        history.append(row)
        (out_dir / "history.json").write_text(json.dumps(history, indent=2))

        # ── Save checkpoint ──────────────────
        if epoch % cfg["save_every"] == 0:
            ckpt = out_dir / f"epoch_{epoch:03d}.pt"
            torch.save({"epoch": epoch, "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(), "val_acc": val_metrics["acc"], "val_f1": val_metrics["f1"]}, ckpt)
            print(f"  Saved checkpoint: {ckpt.name}")

        # ── Save best ────────────────────────
        if val_metrics["f1"] > best_val_f1:
            best_val_f1  = val_metrics["f1"]
            best_val_acc = val_metrics["acc"]
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

    print(f"\n{'='*60}")
    print(f"Training complete.")
    print(f"Best val accuracy: {best_val_acc*100:.2f}%   Best val F1: {best_val_f1:.4f}")
    print(f"Checkpoints in: {out_dir}")
    print(f"{'='*60}\n")

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
        input_names=["input"],
        output_names=["logit"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "logit": {0: "batch_size"},
        },
    )
    print(f"ONNX exported → {onnx_path}")


if __name__ == "__main__":
    main()
