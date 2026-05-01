"""
HypeVault — Real-World Validation & Threshold Calibration
===========================================================
Run this AFTER training on the best_model.pt checkpoint to:

  1. Measure performance per brand / per category
  2. Find the ideal confidence threshold (not just 0.5)
  3. Detect distribution shift (where the model struggles)
  4. Generate a full confusion matrix + classification report
  5. Test against deliberately degraded images (low-res, JPEG, rotation)
     — simulates real seller uploads

Usage:
    python validate_real_world.py \
        --checkpoint checkpoints/best_model.pt \
        --data_root ~/scraper \
        --threshold 0.5
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode


# ─── reuse model from train.py ───────────────────────────────────────────────
import sys
sys.path.insert(0, str(Path(__file__).parent))
from train import DINOv2Classifier, build_transforms


# ─── Dataset: per-brand breakout ─────────────────────────────────────────────

class DetailedDataset(Dataset):
    LABEL_DIRS = {0: ["Label_0_Sneakers", "Label_0_Watches"],
                  1: ["Label_1_Sneakers", "Label_1_Watches"]}

    def __init__(self, root: Path, transform=None, degradation=None):
        self.transform = transform
        self.degradation = degradation   # applied AFTER base transform
        self.samples: list[tuple[Path, int, str, str]] = []  # path, label, brand, category

        for label, dirs in self.LABEL_DIRS.items():
            for d in dirs:
                category = "sneaker" if "Sneakers" in d else "watch"
                folder = root / d
                if not folder.exists():
                    continue
                for brand_dir in sorted(folder.iterdir()):
                    if not brand_dir.is_dir():
                        continue
                    for img_path in sorted(brand_dir.iterdir()):
                        if img_path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                            self.samples.append((img_path, label, brand_dir.name, category))

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        path, label, brand, category = self.samples[idx]
        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            img = Image.new("RGB", (224, 224))
        if self.transform:
            img = self.transform(img)
        if self.degradation:
            img = self.degradation(img)
        return img, label, brand, category


def collate(batch):
    imgs, labels, brands, cats = zip(*batch)
    return torch.stack(imgs), list(labels), list(brands), list(cats)


# ─── Degradation transforms (simulate real uploads) ──────────────────────────

def make_degradation(mode: str, img_size: int):
    """Stress-test how the model handles non-studio images."""
    import io, random

    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]

    class JPEG:
        def __call__(self, t):
            # t is already a tensor — convert back, compress, re-normalise
            inv = transforms.Normalize([-m/s for m,s in zip(mean,std)], [1/s for s in std])
            pil = transforms.ToPILImage()(inv(t))
            buf = io.BytesIO()
            pil.save(buf, "JPEG", quality=random.randint(20, 50))
            buf.seek(0)
            return transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])(Image.open(buf).copy())

    if mode == "clean":
        return None
    elif mode == "lowres":
        # Simulate 240p phone photo resized up
        return transforms.Compose([
            transforms.Resize(int(img_size * 0.25), interpolation=InterpolationMode.NEAREST),
            transforms.Resize(img_size, interpolation=InterpolationMode.NEAREST),
        ])
    elif mode == "jpeg":
        return JPEG()
    elif mode == "rotate":
        return transforms.RandomRotation(degrees=30)
    elif mode == "dark":
        return transforms.ColorJitter(brightness=(0.2, 0.4))
    return None


# ─── Threshold calibration ────────────────────────────────────────────────────

def find_best_threshold(probs: np.ndarray, labels: np.ndarray) -> dict:
    """
    Sweep thresholds 0.3–0.8 and find the one that maximises F1.
    Critical: real-world use-case prefers HIGH RECALL for fakes
    (better to ask a human to review a borderline authentic item
    than to let a fake slip through).
    """
    best = {"threshold": 0.5, "f1": 0, "precision": 0, "recall": 0, "acc": 0}
    results = []

    for t in np.arange(0.20, 0.85, 0.025):
        preds = (probs >= t).astype(int)
        tp = ((preds == 1) & (labels == 1)).sum()
        fp = ((preds == 1) & (labels == 0)).sum()
        fn = ((preds == 0) & (labels == 1)).sum()
        tn = ((preds == 0) & (labels == 0)).sum()

        precision = tp / max(tp + fp, 1)
        recall    = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-8)
        acc = (tp + tn) / max(len(labels), 1)

        results.append({
            "threshold": round(float(t), 3),
            "f1": round(float(f1), 4),
            "precision": round(float(precision), 4),
            "recall": round(float(recall), 4),
            "acc": round(float(acc), 4),
            "fake_caught": int(tp),
            "real_wrongly_blocked": int(fp),
            "fakes_missed": int(fn),
        })

        if f1 > best["f1"]:
            best = results[-1]

    return best, results


# ─── Main ─────────────────────────────────────────────────────────────────────

@torch.no_grad()
def run_inference(model, loader, device, amp_dtype):
    model.eval()
    all_probs, all_labels, all_brands, all_cats = [], [], [], []

    for imgs, labels, brands, cats in loader:
        imgs = imgs.to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=amp_dtype):
            logits = model(imgs)
        probs = torch.sigmoid(logits).float().cpu().numpy()
        all_probs.extend(probs.tolist())
        all_labels.extend(labels)
        all_brands.extend(brands)
        all_cats.extend(cats)

    return (
        np.array(all_probs),
        np.array(all_labels),
        np.array(all_brands),
        np.array(all_cats),
    )


def per_group_metrics(probs, labels, groups, threshold):
    """Break down accuracy per brand or category."""
    out = {}
    for g in sorted(set(groups)):
        mask = groups == g
        p = probs[mask]
        l = labels[mask]
        preds = (p >= threshold).astype(int)
        tp = ((preds == 1) & (l == 1)).sum()
        fp = ((preds == 1) & (l == 0)).sum()
        fn = ((preds == 0) & (l == 1)).sum()
        tn = ((preds == 0) & (l == 0)).sum()
        acc = (tp + tn) / max(len(l), 1)
        recall = tp / max(tp + fn, 1)
        out[g] = {
            "n": int(len(l)),
            "acc": round(float(acc), 4),
            "fake_recall": round(float(recall), 4),
            "n_authentic": int((l == 0).sum()),
            "n_deepfake":  int((l == 1).sum()),
            "wrongly_blocked_authentic": int(fp),   # ← the deployment problem
            "fakes_missed": int(fn),                # ← the safety problem
        }
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint",  type=Path, default=Path("checkpoints/best_model.pt"))
    parser.add_argument("--data_root",   type=Path, default=Path("/home/tensortorch26/Desktop/scraper"))
    parser.add_argument("--model_name",  type=str,  default="dinov2_vitg14_reg")
    parser.add_argument("--img_size",    type=int,  default=518)
    parser.add_argument("--batch_size",  type=int,  default=64)
    parser.add_argument("--threshold",   type=float, default=0.5)
    parser.add_argument("--num_workers", type=int,  default=8)
    parser.add_argument("--out",         type=Path, default=Path("checkpoints/real_world_report.json"))
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    # Load model
    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device)
    model = DINOv2Classifier(args.model_name)
    model.load_state_dict(ckpt["model_state"])
    model = model.to(device).eval()
    print(f"  Loaded (val_acc={ckpt.get('val_acc', '?'):.4f}  val_f1={ckpt.get('val_f1', '?'):.4f})")

    val_tf = build_transforms(args.img_size, train=False)
    report = {}

    # ── 1. Clean validation ────────────────────────────────────────────────
    print("\n[1/5] Clean validation set (same distribution as training)...")
    clean_ds = DetailedDataset(args.data_root, transform=val_tf)
    clean_loader = DataLoader(clean_ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True,
                              collate_fn=collate)
    probs, labels, brands, cats = run_inference(model, clean_loader, device, amp_dtype)

    best_thresh, thresh_sweep = find_best_threshold(probs, labels)
    print(f"  Default threshold (0.5): acc={((probs>=0.5)==labels).mean():.4f}")
    print(f"  Best threshold:  {best_thresh['threshold']:.3f}  →  acc={best_thresh['acc']:.4f}  f1={best_thresh['f1']:.4f}")
    print(f"  At best threshold: {best_thresh['real_wrongly_blocked']} authentic items wrongly blocked, {best_thresh['fakes_missed']} fakes missed")

    report["clean"] = {
        "best_threshold": best_thresh,
        "by_brand":    per_group_metrics(probs, labels, brands, best_thresh["threshold"]),
        "by_category": per_group_metrics(probs, labels, cats,   best_thresh["threshold"]),
    }

    # ── 2–5. Degradation stress tests ─────────────────────────────────────
    degradation_modes = [
        ("lowres", "Low-res (240p phone upscale)"),
        ("jpeg",   "JPEG compressed (quality 20–50)"),
        ("rotate", "Rotated ±30° (tilted phone shot)"),
        ("dark",   "Underexposed (dim lighting)"),
    ]
    for mode, description in degradation_modes:
        print(f"\n[Stress] {description}...")
        deg = make_degradation(mode, args.img_size)
        ds = DetailedDataset(args.data_root, transform=val_tf, degradation=deg)
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True,
                            collate_fn=collate)
        p, l, _, _ = run_inference(model, loader, device, amp_dtype)
        acc = ((p >= best_thresh["threshold"]) == l).mean()
        fp  = ((p >= best_thresh["threshold"]) & (l == 0)).sum()
        print(f"  Accuracy: {acc:.4f}  |  Authentic items wrongly blocked: {fp}")
        report[mode] = {"acc": round(float(acc), 4), "wrongly_blocked_authentic": int(fp)}

    # Save report
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print(f"\nFull report saved → {args.out}")

    # ── Print final recommendation ─────────────────────────────────────────
    rec_thresh = best_thresh["threshold"]
    print(f"\n{'='*60}")
    print(f"RECOMMENDATION FOR PRODUCTION:")
    print(f"  Use threshold = {rec_thresh:.2f} in backend/inference/routes.py")
    print(f"  (instead of the default 0.5)")
    print(f"  If 'wrongly blocked authentic' is still high:")
    print(f"  → Re-train with --label_smoothing 0.1 --mixup_alpha 0.3")
    print(f"  → Check which brands are most affected in by_brand above")
    print(f"{'='*60}\n")

    return rec_thresh


if __name__ == "__main__":
    main()
