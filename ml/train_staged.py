"""
HypeVault — Staged DINOv2-Giant Training  [$6 budget mode]
===========================================================
Two-stage progressive fine-tuning:

  Stage 1 — Linear Probe       (~7 min,  ~$0.22)
    Backbone: FULLY FROZEN. Only classification head trained.
    Baseline: ~80–88% accuracy.

  Stage 2 — Partial Fine-Tune  (~2 hrs, ~$3.90)
    Backbone: last 8 transformer blocks unfrozen.
    Expected accuracy: ~91–95%

Usage:
    python train_staged.py --stage both        # run stage 1 then 2 (default)
    python train_staged.py --stage 1           # linear probe only
    python train_staged.py --stage 2           # partial fine-tune only
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # non-interactive backend — works on headless GPU server
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent))
from train import (
    HypeVaultDataset,
    DINOv2Classifier,
    build_transforms,
    make_weighted_sampler,
    cosine_schedule_with_warmup,
    mixup_batch,
)

# ─────────────────────────────────────────────
# Stage configs
# ─────────────────────────────────────────────

STAGE1 = dict(
    epochs=3,
    lr_head=1e-3,
    weight_decay=0.01,
    warmup_epochs=1,
    batch_size=48,
    label_smoothing=0.05,
    mixup_alpha=0.0,
    freeze_blocks="all",
)

STAGE2 = dict(
    epochs=8,
    lr_backbone=5e-6,
    lr_head=5e-5,
    weight_decay=0.05,
    warmup_epochs=1,
    batch_size=32,
    label_smoothing=0.05,
    mixup_alpha=0.2,
    freeze_blocks=32,
)

DEFAULTS = dict(
    data_root="/root/scraper",
    output_dir="/root/checkpoints",
    model_name="dinov2_vitg14_reg",
    img_size=518,
    num_workers=8,
    seed=42,
    grad_clip=1.0,
    amp_dtype="bf16",
    stage="both",
    resume=None,
    overfit_warn_gap=0.10,   # warn if train_acc - val_acc > this
    overfit_stop_gap=0.20,   # early stop if gap exceeds this for 3 epochs
)


# ─────────────────────────────────────────────
# Freeze helpers
# ─────────────────────────────────────────────

def _get_backbone_blocks(backbone):
    if hasattr(backbone, "blocks"):
        return backbone.blocks
    if hasattr(backbone, "layers"):
        return backbone.layers
    for _, mod in backbone.named_children():
        if isinstance(mod, nn.ModuleList) and len(mod) > 4:
            return mod
    return None


def freeze_backbone(model: DINOv2Classifier, freeze_blocks):
    for param in model.backbone.parameters():
        param.requires_grad = False

    if freeze_blocks == "all":
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
        print(f"  Backbone: FULLY FROZEN | Trainable: {trainable:.1f}M params", flush=True)
        return

    blocks = _get_backbone_blocks(model.backbone)
    if blocks is None:
        for p in model.backbone.parameters():
            p.requires_grad = True
        print("  WARNING: could not find blocks — unfreezing all backbone", flush=True)
        return

    total = len(blocks)
    for i, blk in enumerate(blocks):
        if i >= freeze_blocks:
            for p in blk.parameters():
                p.requires_grad = True

    for attr in ["norm", "ln_post", "norm_post", "fc_norm"]:
        if hasattr(model.backbone, attr):
            for p in getattr(model.backbone, attr).parameters():
                p.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e9
    unfrozen = total - freeze_blocks
    print(
        f"  Backbone: blocks 0–{freeze_blocks-1} FROZEN | "
        f"blocks {freeze_blocks}–{total-1} ACTIVE ({unfrozen} blocks) | "
        f"Trainable: {trainable:.2f}B params",
        flush=True,
    )


# ─────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────

def compute_metrics(logits: torch.Tensor, labels: torch.Tensor) -> dict:
    probs = torch.sigmoid(logits.float())
    preds = (probs >= 0.5).long()
    labels_i = labels.long()
    acc = (preds == labels_i).float().mean().item()
    tp = ((preds == 1) & (labels_i == 1)).sum().item()
    fp = ((preds == 1) & (labels_i == 0)).sum().item()
    fn = ((preds == 0) & (labels_i == 1)).sum().item()
    prec = tp / max(tp + fp, 1)
    rec  = tp / max(tp + fn, 1)
    f1   = 2 * prec * rec / max(prec + rec, 1e-8)
    return {"acc": acc, "precision": prec, "recall": rec, "f1": f1}


# ─────────────────────────────────────────────
# Overfitting monitor
# ─────────────────────────────────────────────

class OverfitMonitor:
    def __init__(self, warn_gap: float = 0.10, stop_gap: float = 0.20, patience: int = 3):
        self.warn_gap = warn_gap
        self.stop_gap = stop_gap
        self.patience = patience
        self._overfit_count = 0
        self.history: list[dict] = []

    def check(self, epoch: int, train_acc: float, val_acc: float, train_loss: float, val_loss: float) -> str:
        gap_acc  = train_acc  - val_acc
        gap_loss = val_loss   - train_loss  # positive = overfitting
        self.history.append({
            "epoch": epoch, "train_acc": train_acc, "val_acc": val_acc,
            "gap_acc": gap_acc, "train_loss": train_loss, "val_loss": val_loss,
        })
        status = "OK"
        if gap_acc > self.stop_gap:
            self._overfit_count += 1
            status = f"SEVERE ({self._overfit_count}/{self.patience})"
        elif gap_acc > self.warn_gap:
            self._overfit_count = 0
            status = "WARNING"
        else:
            self._overfit_count = 0
            status = "OK"
        return status, gap_acc, gap_loss

    @property
    def should_stop(self) -> bool:
        return self._overfit_count >= self.patience


# ─────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────

def save_plots(history: list[dict], out_dir: Path):
    """
    Save a 2×2 figure (loss, accuracy, F1, overfit gap) to PNG.
    Called after every epoch so you can scp/download anytime.
    """
    if not history:
        return

    epochs    = [r["epoch"]      for r in history]
    tr_loss   = [r["train_loss"] for r in history]
    vl_loss   = [r["val_loss"]   for r in history]
    tr_acc    = [r["train_acc"]  * 100 for r in history]
    vl_acc    = [r["val_acc"]    * 100 for r in history]
    tr_f1     = [r.get("train_f1", 0) for r in history]
    vl_f1     = [r["val_f1"]           for r in history]
    gap       = [r.get("overfit_gap_acc", 0) * 100 for r in history]
    stages    = [r.get("stage", 1) for r in history]

    plt.style.use("dark_background")
    fig = plt.figure(figsize=(14, 9))
    fig.patch.set_facecolor("#0d0d0d")
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.32)

    ax_loss = fig.add_subplot(gs[0, 0])
    ax_acc  = fig.add_subplot(gs[0, 1])
    ax_f1   = fig.add_subplot(gs[1, 0])
    ax_gap  = fig.add_subplot(gs[1, 1])

    palette = {"train": "#FF3B00", "val": "#00C8FF", "f1": "#00E676", "gap": "#FFD740"}
    bg_col  = "#111111"
    grid_col = "rgba(255,255,255,0.07)" if False else "#1f1f1f"  # matplotlib doesn't do rgba text

    def _style(ax, title, ylabel, xlabel="Epoch"):
        ax.set_facecolor(bg_col)
        ax.set_title(title, color="white", fontsize=11, fontweight="bold", pad=8)
        ax.set_xlabel(xlabel, color="#888888", fontsize=9)
        ax.set_ylabel(ylabel, color="#888888", fontsize=9)
        ax.tick_params(colors="#666666", labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor("#333333")
        ax.grid(True, color=grid_col, linestyle="--", linewidth=0.5, alpha=0.6)
        ax.legend(fontsize=9, facecolor="#1a1a1a", edgecolor="#333333", labelcolor="white")

    # Mark stage boundary
    s1_end = max((i + 1 for i, s in enumerate(stages) if s == 1), default=None)

    def _mark_stages(ax):
        if s1_end and s1_end < max(epochs):
            ax.axvline(x=s1_end + 0.5, color="#444444", linestyle=":", linewidth=1.2)
            ax.text(s1_end + 0.7, ax.get_ylim()[1] * 0.97, "S2→", color="#555555", fontsize=7)

    # ── Loss ──────────────────────────────────
    ax_loss.plot(epochs, tr_loss, color=palette["train"], marker="o", markersize=4, linewidth=1.8, label="Train loss")
    ax_loss.plot(epochs, vl_loss, color=palette["val"],   marker="s", markersize=4, linewidth=1.8, label="Val loss",   linestyle="--")
    _style(ax_loss, "Loss", "BCE Loss")
    _mark_stages(ax_loss)
    if len(epochs) >= 2:
        ax_loss.annotate(f"{vl_loss[-1]:.4f}", (epochs[-1], vl_loss[-1]),
                         textcoords="offset points", xytext=(5, 4),
                         color=palette["val"], fontsize=8)

    # ── Accuracy ──────────────────────────────
    ax_acc.plot(epochs, tr_acc, color=palette["train"], marker="o", markersize=4, linewidth=1.8, label="Train acc")
    ax_acc.plot(epochs, vl_acc, color=palette["val"],   marker="s", markersize=4, linewidth=1.8, label="Val acc",   linestyle="--")
    ax_acc.set_ylim(max(0, min(tr_acc + vl_acc) - 5), 101)
    ax_acc.axhline(y=95, color="#FFD740", linestyle=":", linewidth=0.8, alpha=0.7, label="95% target")
    _style(ax_acc, "Accuracy", "Accuracy (%)")
    _mark_stages(ax_acc)
    if len(epochs) >= 1:
        ax_acc.annotate(f"{vl_acc[-1]:.1f}%", (epochs[-1], vl_acc[-1]),
                        textcoords="offset points", xytext=(5, 4),
                        color=palette["val"], fontsize=8)

    # ── F1 ────────────────────────────────────
    ax_f1.plot(epochs, tr_f1, color=palette["train"], marker="o", markersize=4, linewidth=1.8, label="Train F1")
    ax_f1.plot(epochs, vl_f1, color=palette["f1"],    marker="s", markersize=4, linewidth=1.8, label="Val F1",    linestyle="--")
    ax_f1.set_ylim(max(0, min(tr_f1 + vl_f1) - 0.05), 1.01)
    ax_f1.axhline(y=0.95, color="#FFD740", linestyle=":", linewidth=0.8, alpha=0.7, label="F1=0.95 target")
    _style(ax_f1, "F1 Score", "F1")
    _mark_stages(ax_f1)
    if len(epochs) >= 1:
        ax_f1.annotate(f"{vl_f1[-1]:.4f}", (epochs[-1], vl_f1[-1]),
                       textcoords="offset points", xytext=(5, 4),
                       color=palette["f1"], fontsize=8)

    # ── Overfitting gap ────────────────────────
    colors_gap = [palette["gap"] if abs(g) < 10 else "#FF4444" for g in gap]
    ax_gap.bar(epochs, gap, color=colors_gap, width=0.6, alpha=0.85, label="Train−Val acc gap (%)")
    ax_gap.axhline(y=10, color="#FF8800", linestyle="--", linewidth=0.9, alpha=0.8, label="Warn threshold (10%)")
    ax_gap.axhline(y=20, color="#FF3B00", linestyle="--", linewidth=0.9, alpha=0.8, label="Stop threshold (20%)")
    ax_gap.axhline(y=0,  color="#444444", linewidth=0.6)
    _style(ax_gap, "Overfitting Monitor\n(Train Acc − Val Acc)", "Gap (%)")
    _mark_stages(ax_gap)

    # ── Title ─────────────────────────────────
    latest = history[-1]
    subtitle = (
        f"Epoch {latest['epoch']} | "
        f"val_acc={latest['val_acc']*100:.2f}%  val_f1={latest['val_f1']:.4f}  "
        f"| Stage {latest.get('stage',1)}"
    )
    fig.suptitle(
        f"HypeVault — DINOv2-Giant Training\n{subtitle}",
        color="white", fontsize=12, fontweight="bold", y=0.98
    )

    plot_path = out_dir / "training_curves.png"
    plt.savefig(plot_path, dpi=130, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return plot_path


# ─────────────────────────────────────────────
# Training / validation loops
# ─────────────────────────────────────────────

LOG_EVERY = 50   # print a progress line every N batches


def _bar(current: int, total: int, width: int = 30) -> str:
    filled = int(width * current / max(total, 1))
    return f"[{'█' * filled}{'░' * (width - filled)}]"


def train_epoch(
    model, loader, optimizer, criterion, scaler, scheduler,
    device, cfg, stage_num, epoch, total_epochs
):
    model.train()
    amp_dtype = torch.bfloat16 if cfg["amp_dtype"] == "bf16" else torch.float16

    all_logits, all_labels = [], []
    running_loss = 0.0
    t0 = time.time()

    total_batches = len(loader)
    print(f"\n  ┌─ S{stage_num} Epoch {epoch}/{total_epochs} — TRAIN  ({total_batches} batches)", flush=True)

    for i, (imgs, labels) in enumerate(loader):
        imgs   = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if cfg.get("mixup_alpha", 0) > 0:
            imgs, labels = mixup_batch(imgs, labels, cfg["mixup_alpha"])

        with torch.amp.autocast("cuda", dtype=amp_dtype):
            logits = model(imgs)
            loss   = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.get("grad_clip", 1.0))
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        scheduler.step()

        running_loss += loss.item()
        all_logits.append(logits.detach().float())
        all_labels.append(labels.detach().float())

        if (i + 1) % LOG_EVERY == 0 or (i + 1) == total_batches:
            elapsed = time.time() - t0
            pct = (i + 1) / total_batches
            eta  = (elapsed / pct) * (1 - pct) if pct > 0 else 0
            avg_loss = running_loss / (i + 1)
            lr_now = scheduler.get_last_lr()[0]
            bar = _bar(i + 1, total_batches)
            print(
                f"  │  {bar} {(i+1):>5}/{total_batches}  "
                f"loss={avg_loss:.4f}  lr={lr_now:.2e}  "
                f"elapsed={elapsed:.0f}s  eta={eta:.0f}s",
                flush=True,
            )

    metrics = compute_metrics(torch.cat(all_logits), torch.cat(all_labels).round().long())
    print(f"  └─ Train done  acc={metrics['acc']*100:.2f}%  f1={metrics['f1']:.4f}", flush=True)
    return running_loss / len(loader), metrics


@torch.no_grad()
def val_epoch(model, loader, criterion, device, cfg, stage_num, epoch, total_epochs):
    model.eval()
    amp_dtype = torch.bfloat16 if cfg["amp_dtype"] == "bf16" else torch.float16
    total_loss = 0.0
    all_logits, all_labels = [], []
    total_batches = len(loader)

    print(f"\n  ┌─ S{stage_num} Epoch {epoch}/{total_epochs} — VAL    ({total_batches} batches)", flush=True)
    t0 = time.time()

    for i, (imgs, labels) in enumerate(loader):
        imgs   = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", dtype=amp_dtype):
            logits = model(imgs)
            loss   = criterion(logits, labels)
        total_loss += loss.item()
        all_logits.append(logits.float())
        all_labels.append(labels.float())

        if (i + 1) % LOG_EVERY == 0 or (i + 1) == total_batches:
            elapsed = time.time() - t0
            pct  = (i + 1) / total_batches
            eta  = (elapsed / pct) * (1 - pct) if pct > 0 else 0
            bar  = _bar(i + 1, total_batches)
            print(
                f"  │  {bar} {(i+1):>5}/{total_batches}  "
                f"loss={total_loss/(i+1):.4f}  "
                f"elapsed={elapsed:.0f}s  eta={eta:.0f}s",
                flush=True,
            )

    metrics = compute_metrics(torch.cat(all_logits), torch.cat(all_labels))
    print(f"  └─ Val done  acc={metrics['acc']*100:.2f}%  f1={metrics['f1']:.4f}", flush=True)
    return total_loss / len(loader), metrics


# ─────────────────────────────────────────────
# Run one stage
# ─────────────────────────────────────────────

def run_stage(
    stage_num: int,
    stage_cfg: dict,
    model: DINOv2Classifier,
    train_ds, val_ds,
    device,
    global_cfg: dict,
    out_dir: Path,
    history: list,
):
    overfit = OverfitMonitor(
        warn_gap=global_cfg["overfit_warn_gap"],
        stop_gap=global_cfg["overfit_stop_gap"],
        patience=3,
    )

    stage_label = (
        "Linear Probe — head only (frozen backbone)" if stage_num == 1
        else "Partial Fine-Tune — last 8 blocks + head"
    )
    print(f"\n{'='*65}")
    print(f"  STAGE {stage_num} — {stage_label}")
    print(f"{'='*65}\n")

    freeze_backbone(model, stage_cfg["freeze_blocks"])

    sampler = make_weighted_sampler(train_ds)
    train_loader = DataLoader(
        train_ds, batch_size=stage_cfg["batch_size"], sampler=sampler,
        num_workers=global_cfg["num_workers"], pin_memory=True, drop_last=True,
        persistent_workers=global_cfg["num_workers"] > 0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=stage_cfg["batch_size"] * 2, shuffle=False,
        num_workers=global_cfg["num_workers"], pin_memory=True,
        persistent_workers=global_cfg["num_workers"] > 0,
    )

    if stage_num == 1:
        param_groups = [{"params": list(model.head.parameters()),
                         "lr": stage_cfg["lr_head"], "weight_decay": stage_cfg["weight_decay"]}]
    else:
        backbone_params = [p for p in model.backbone.parameters() if p.requires_grad]
        head_params     = list(model.head.parameters())
        param_groups = [
            {"params": backbone_params, "lr": stage_cfg["lr_backbone"], "weight_decay": stage_cfg["weight_decay"]},
            {"params": head_params,     "lr": stage_cfg["lr_head"],     "weight_decay": 0.0},
        ]

    optimizer = torch.optim.AdamW(param_groups, betas=(0.9, 0.999), eps=1e-8)
    steps_per_epoch = len(train_loader)
    total_steps  = steps_per_epoch * stage_cfg["epochs"]
    warmup_steps = steps_per_epoch * stage_cfg["warmup_epochs"]
    scheduler = cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    criterion = nn.BCEWithLogitsLoss()

    use_bf16 = (global_cfg["amp_dtype"] == "bf16") and torch.cuda.is_bf16_supported()
    scaler   = torch.amp.GradScaler("cuda", enabled=not use_bf16)

    best_f1, best_acc = 0.0, 0.0
    merged_cfg = {**global_cfg, **stage_cfg}

    header = f"\n{'Epoch':>5}  {'TrLoss':>7}  {'TrAcc':>6}  {'ValLoss':>7}  {'ValAcc':>6}  {'F1':>6}  {'Gap':>7}  Overfit"
    sep = "─" * 70
    print(header, flush=True)
    print(sep, flush=True)

    for epoch in range(1, stage_cfg["epochs"] + 1):
        t0 = time.time()

        tr_loss, tr_m = train_epoch(
            model, train_loader, optimizer, criterion, scaler, scheduler,
            device, merged_cfg, stage_num, epoch, stage_cfg["epochs"]
        )
        vl_loss, vl_m = val_epoch(
            model, val_loader, criterion, device, merged_cfg,
            stage_num, epoch, stage_cfg["epochs"]
        )

        status, gap_acc, gap_loss = overfit.check(
            epoch, tr_m["acc"], vl_m["acc"], tr_loss, vl_loss
        )
        elapsed = time.time() - t0

        ok_sym = "OK" if status == "OK" else ("WARN" if "WARNING" in status else "OVERFIT!")
        print(
            f"{epoch:>5}  {tr_loss:>7.4f}  {tr_m['acc']*100:>5.2f}%  "
            f"{vl_loss:>7.4f}  {vl_m['acc']*100:>5.2f}%  "
            f"{vl_m['f1']:>6.4f}  {gap_acc*100:>+6.2f}%  {ok_sym}  ({elapsed:.0f}s)",
            flush=True,
        )

        if status.startswith("SEVERE") or "WARNING" in status:
            print(
                f"  ⚠  val_loss−train_loss={gap_loss:+.4f}  "
                f"train_acc−val_acc={gap_acc*100:+.2f}%",
                flush=True,
            )
            if overfit.should_stop:
                print(
                    f"\n  ⛔ Early stopping: overfit gap >{global_cfg['overfit_stop_gap']*100:.0f}% "
                    f"for 3 epochs. Tip: raise --label_smoothing or reduce LR.",
                    flush=True,
                )
                break

        row = {
            "stage": stage_num, "epoch": epoch,
            "train_loss": tr_loss, "val_loss": vl_loss,
            **{f"train_{k}": v for k, v in tr_m.items()},
            **{f"val_{k}": v for k, v in vl_m.items()},
            "overfit_gap_acc": gap_acc,
            "overfit_status": status,
        }
        history.append(row)
        (out_dir / "history.json").write_text(json.dumps(history, indent=2))

        # ── Metrics table ────────────────────
        print(
            f"\n  ┌─ Full metrics ─────────────────────────────┐\n"
            f"  │  {'Metric':<16} {'Train':>10} {'Val':>10}  │\n"
            f"  │  {'─'*38}  │\n"
            f"  │  {'Loss':<16} {tr_loss:>10.4f} {vl_loss:>10.4f}  │\n"
            f"  │  {'Accuracy':<16} {tr_m['acc']*100:>9.2f}% {vl_m['acc']*100:>9.2f}%  │\n"
            f"  │  {'F1':<16} {tr_m['f1']:>10.4f} {vl_m['f1']:>10.4f}  │\n"
            f"  │  {'Precision':<16} {tr_m['precision']:>10.4f} {vl_m['precision']:>10.4f}  │\n"
            f"  │  {'Recall':<16} {tr_m['recall']:>10.4f} {vl_m['recall']:>10.4f}  │\n"
            f"  └────────────────────────────────────────────┘",
            flush=True,
        )

        # ── Save plot ────────────────────────
        try:
            plot_path = save_plots(history, out_dir)
            print(f"  📊 Plot updated → {plot_path}", flush=True)
        except Exception as e:
            print(f"  Plot error: {e}", flush=True)

        if vl_m["f1"] > best_f1:
            best_f1  = vl_m["f1"]
            best_acc = vl_m["acc"]
            ckpt = out_dir / f"stage{stage_num}_best.pt"
            torch.save({
                "stage": stage_num, "epoch": epoch,
                "model_state": model.state_dict(),
                "val_acc": vl_m["acc"], "val_f1": vl_m["f1"],
            }, ckpt)
            print(f"  ★  New best! val_acc={best_acc*100:.2f}%  val_f1={best_f1:.4f}  → {ckpt.name}", flush=True)

        print(flush=True)

    print(f"\n  Stage {stage_num} complete — best val_acc={best_acc*100:.2f}%  val_f1={best_f1:.4f}", flush=True)
    return best_acc, best_f1


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    for k, v in DEFAULTS.items():
        if v is None:
            parser.add_argument(f"--{k}", type=str, default=None)
        else:
            parser.add_argument(f"--{k}", type=type(v), default=v)
    args = parser.parse_args()
    cfg  = vars(args)

    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(cfg, indent=2))

    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice : {device}", flush=True)
    if device.type == "cuda":
        print(f"GPU    : {torch.cuda.get_device_name(0)}", flush=True)
        print(f"VRAM   : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB", flush=True)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    # ── Dataset ─────────────────────────────────────────────────────
    data_root = Path(cfg["data_root"])
    full_ds = HypeVaultDataset(data_root, transform=None)
    total = len(full_ds)
    tqdm.write(f"\nDataset: {total:,} images loaded from {data_root}")

    val_size = int(total * 0.10)
    rng  = np.random.default_rng(cfg["seed"])
    perm = rng.permutation(total).tolist()
    train_idx, val_idx = perm[: total - val_size], perm[total - val_size :]

    train_tf = build_transforms(cfg["img_size"], train=True)
    val_tf   = build_transforms(cfg["img_size"], train=False)
    train_ds = HypeVaultDataset(data_root, transform=train_tf, indices=train_idx)
    val_ds   = HypeVaultDataset(data_root, transform=val_tf,   indices=val_idx)

    labels_arr = np.array(train_ds.labels)
    print(
        f"Train : {len(train_ds):,}  (authentic={int((labels_arr==0).sum())}, deepfake={int((labels_arr==1).sum())})",
        flush=True,
    )
    print(f"Val   : {len(val_ds):,}", flush=True)

    # ── Model ────────────────────────────────────────────────────────
    print(f"\nLoading {cfg['model_name']}...", flush=True)
    model = DINOv2Classifier(cfg["model_name"]).to(device)

    if cfg["resume"]:
        ckpt = torch.load(cfg["resume"], map_location=device)
        model.load_state_dict(ckpt["model_state"])
        print(f"Resumed from {cfg['resume']}", flush=True)

    history: list[dict] = []
    t_global = time.time()
    cost_hr  = 1.89

    run_s1 = cfg["stage"] in ("1", "both")
    run_s2 = cfg["stage"] in ("2", "both")

    if run_s1:
        acc1, f1_1 = run_stage(1, STAGE1, model, train_ds, val_ds, device, cfg, out_dir, history)
        hrs = (time.time() - t_global) / 3600
        print(f"\n  Elapsed: {hrs:.2f}h  |  Est. cost: ${hrs*cost_hr:.2f}", flush=True)

    if run_s2:
        if run_s1:
            s1_best = out_dir / "stage1_best.pt"
            if s1_best.exists():
                ckpt = torch.load(s1_best, map_location=device)
                model.load_state_dict(ckpt["model_state"])
                print("\nLoaded stage1_best.pt → starting stage 2", flush=True)

        acc2, f1_2 = run_stage(2, STAGE2, model, train_ds, val_ds, device, cfg, out_dir, history)
        hrs = (time.time() - t_global) / 3600
        print(f"\n  Elapsed: {hrs:.2f}h  |  Est. cost: ${hrs*cost_hr:.2f}", flush=True)

    # ── Final save ──────────────────────────────────────────────────
    final = out_dir / "final_staged_model.pt"
    torch.save({"model_state": model.state_dict()}, final)
    print(f"\nFinal model saved: {final}", flush=True)

    print("\nExporting ONNX...", flush=True)
    try:
        from train import export_onnx
        export_onnx(model, cfg, out_dir, device)
    except Exception as e:
        print(f"ONNX export error (run export_tensorrt.py later): {e}", flush=True)

    total_h = (time.time() - t_global) / 3600
    print(
        f"\n{'='*55}\n"
        f"  DONE  |  {total_h:.2f}h  |  ~${total_h*cost_hr:.2f}\n"
        f"  Run validate_real_world.py to calibrate threshold.\n"
        f"{'='*55}\n",
        flush=True,
    )


if __name__ == "__main__":
    main()
