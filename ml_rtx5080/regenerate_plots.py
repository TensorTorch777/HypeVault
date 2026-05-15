#!/usr/bin/env python3
"""Regenerate training_curves.png and confusion_matrix_eval.png for ml_rtx5080 runs.

Uses this folder's train.py (DINOv2-Base, 504px defaults) so checkpoints load correctly.

  python ml_rtx5080/regenerate_plots.py --subtitle "Stage 2" --eval \\
    --data_root /home/tensortorch26/Desktop/HypeVault
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_PKG = Path(__file__).resolve().parent
if str(_PKG) not in sys.path:
    sys.path.insert(0, str(_PKG))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import train as tv


def _load_history(path: Path) -> list[dict]:
    data = json.loads(path.read_text())
    return data.get("epochs") or data.get("history") or []


def _best_epoch_from_history(history: list[dict]) -> int:
    if not history:
        return 1
    best_i = max(range(len(history)), key=lambda i: history[i].get("val_f1", 0.0))
    return int(history[best_i].get("epoch", best_i + 1))


def _merge_cfg(checkpoint_dir: Path) -> dict:
    cfg = dict(tv.DEFAULTS)
    cand = checkpoint_dir / "config.json"
    if cand.exists():
        cfg.update(json.loads(cand.read_text()))
    return cfg


@torch.no_grad()
def _val_epoch_cpu_safe(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    cfg: dict,
    epoch: int | None,
) -> tuple[float, dict]:
    if device.type == "cuda":
        return tv.val_epoch(model, loader, criterion, device, cfg, epoch)

    model.eval()
    total_loss = 0.0
    all_logits, all_labels = [], []
    desc = f"Epoch {epoch} [ Val ]" if epoch is not None else "Evaluating"
    for images, labels in tv.tqdm(loader, total=len(loader), desc=desc, ncols=110, leave=True):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(images)
        loss = criterion(logits, labels)
        total_loss += loss.item()
        all_logits.append(logits.float())
        all_labels.append(labels.float())
    all_logits_cat = torch.cat(all_logits)
    all_labels_cat = torch.cat(all_labels)
    metrics = tv.compute_metrics(all_logits_cat, all_labels_cat)
    metrics["_logits"] = all_logits_cat
    metrics["_labels"] = all_labels_cat
    return total_loss / len(loader), metrics


def run_eval_plots(
    data_root: Path,
    checkpoint: Path,
    out_dir: Path,
    cfg_override: dict | None,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = _merge_cfg(checkpoint.parent)
    if cfg_override:
        cfg.update(cfg_override)
    cfg["data_root"] = str(data_root)

    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])

    catalog = tv.load_hypevault_samples(data_root)
    train_idx, val_idx = tv.split_indices_by_product(
        catalog, cfg["val_split"], cfg["seed"]
    )
    train_samples = [catalog[i] for i in train_idx]
    val_samples = [catalog[i] for i in val_idx]
    tv.verify_product_split(train_samples, val_samples)

    val_tf = tv.build_transforms(cfg["img_size"], train=False)
    val_ds = tv.HypeVaultDataset(
        data_root,
        transform=val_tf,
        samples=val_samples,
        shuffle=False,
    )
    if device.type == "cpu":
        val_bs = max(1, min(int(cfg["batch_size"]), 4))
    else:
        gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        cap = 16 if gb >= 40 else 8 if gb >= 24 else 4
        val_bs = max(1, min(int(cfg["batch_size"]) * 2, cap))
    val_loader = DataLoader(
        val_ds,
        batch_size=val_bs,
        shuffle=False,
        num_workers=min(cfg["num_workers"], 2) if device.type == "cpu" else cfg["num_workers"],
        pin_memory=device.type == "cuda",
        persistent_workers=False,
    )
    print(f"Eval batch size: {val_bs} (device={device})")

    model = tv.DINOv2Classifier(cfg["model_name"], dropout=cfg["dropout"])
    model = model.to(device)
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])

    criterion = nn.BCEWithLogitsLoss()
    _, metrics = _val_epoch_cpu_safe(model, val_loader, criterion, device, cfg, epoch=None)

    val_acc = float(metrics["acc"])
    val_f1 = float(metrics["f1"])
    tv.plot_confusion_eval(
        metrics["_logits"],
        metrics["_labels"],
        out_dir,
        len(val_ds),
        val_acc,
        val_f1,
    )
    print(f"Eval metrics: acc={val_acc*100:.2f}%  f1={val_f1:.4f}  (n={len(val_ds):,})")


def main() -> None:
    p = argparse.ArgumentParser(description="Regenerate RTX5080 training / eval plots")
    here = Path(__file__).resolve().parent
    p.add_argument("--history", type=Path, default=here / "checkpoints" / "history.json")
    p.add_argument("--out_dir", type=Path, default=here / "checkpoints")
    p.add_argument("--subtitle", type=str, default=None)
    p.add_argument("--eval", action="store_true")
    p.add_argument("--data_root", type=Path, default=None)
    p.add_argument("--checkpoint", type=Path, default=None)
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    history = _load_history(args.history)
    if not history:
        raise SystemExit(f"No epochs in {args.history}")

    best_ep = _best_epoch_from_history(history)
    tv.plot_training_curves(history, args.out_dir, best_ep, subtitle=args.subtitle)
    print(f"Wrote training curves (best epoch by val_f1: {best_ep}) → {args.out_dir / 'training_curves.png'}")

    if args.eval:
        ckpt = args.checkpoint or (args.out_dir / "best_model.pt")
        if not ckpt.is_file():
            raise SystemExit(f"Checkpoint not found: {ckpt}")
        root = args.data_root
        if root is None:
            raise SystemExit("--eval requires --data_root")
        if not root.is_dir():
            raise SystemExit(f"data_root not found: {root}")
        run_eval_plots(root, ckpt, args.out_dir, cfg_override=None)


if __name__ == "__main__":
    main()
