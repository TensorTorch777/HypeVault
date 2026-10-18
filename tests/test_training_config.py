"""Sampler, loss, MixUp, BF16 fallback, and a tiny training smoke test."""

from __future__ import annotations

import inspect
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn as nn
from torch.amp import GradScaler
from torch.utils.data import DataLoader, Dataset, RandomSampler, WeightedRandomSampler

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ml_rtx5080"))

import classifier as clf
import dataset as ds
import train as tv
import training_config as tc


class _LabelSet(Dataset):
    def __init__(self, labels: list[int]) -> None:
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int):
        label = float(self.labels[index])
        return torch.zeros(1), torch.tensor(label)


class _SmokeImages(Dataset):
    def __init__(self, count: int, image_size: int) -> None:
        generator = torch.Generator().manual_seed(4)
        self.images = torch.randn(count, 3, image_size, image_size, generator=generator)
        self.labels = torch.tensor([0.0, 1.0, 0.0, 1.0, 0.0][:count])

    def __len__(self) -> int:
        return self.images.shape[0]

    def __getitem__(self, index: int):
        return self.images[index], self.labels[index]


class TrainingConfigTests(unittest.TestCase):
    def test_balanced_loader_shuffles_without_a_weighted_sampler(self) -> None:
        labels = [0, 1, 0, 1, 0, 1]
        decision = tc.sampler_decision(labels, weighted_sampler=False)
        self.assertEqual(decision["sampler"], "none")
        self.assertTrue(decision["shuffle"])
        self.assertFalse(decision["materially_imbalanced"])
        loader, info = tv.build_train_loader(
            _LabelSet(labels),
            {"batch_size": 2, "num_workers": 0, "weighted_sampler": 0},
            labels,
        )
        self.assertEqual(info["sampler"], "none")
        self.assertIsInstance(loader.sampler, RandomSampler)
        self.assertNotIsInstance(loader.sampler, WeightedRandomSampler)

    def test_imbalance_stays_opt_in(self) -> None:
        labels = [0, 0, 0, 0, 1]
        decision = tc.sampler_decision(labels, weighted_sampler=False)
        self.assertTrue(decision["materially_imbalanced"])
        self.assertEqual(decision["sampler"], "none")
        self.assertTrue(decision["shuffle"])
        weighted = tc.sampler_decision(labels, weighted_sampler=True)
        self.assertEqual(weighted["sampler"], "weighted_random")
        self.assertFalse(weighted["shuffle"])

    def test_watch_catalog_is_balanced_so_the_default_is_shuffle(self) -> None:
        root = Path(__file__).resolve().parents[1]
        if not (root / "Label_0_Watches").is_dir():
            self.skipTest("watch catalog is not in the workspace")
        records = ds.load_records(root)
        labels = [record.label for record in records]
        balance = tc.class_balance(labels)
        decision = tc.sampler_decision(labels, weighted_sampler=False)
        self.assertEqual(balance["counts"][0], 15000)
        self.assertEqual(balance["counts"][1], 15000)
        self.assertFalse(balance["materially_imbalanced"])
        self.assertEqual(decision["sampler"], "none")
        self.assertTrue(decision["shuffle"])
        self.assertEqual(tv.DEFAULTS["weighted_sampler"], 0)
        self.assertEqual(tv.DEFAULTS["mixup_alpha"], 0.0)

    def test_loss_is_bce_with_logits_and_smoothing_changes_targets_only(self) -> None:
        criterion = tc.build_bce_loss()
        self.assertIsInstance(criterion, nn.BCEWithLogitsLoss)
        targets = torch.tensor([0.0, 1.0])
        smoothed = tc.smooth_binary_targets(targets, 0.02)
        self.assertAlmostEqual(smoothed[0].item(), 0.01)
        self.assertAlmostEqual(smoothed[1].item(), 0.99)
        self.assertTrue(torch.equal(tc.smooth_binary_targets(targets, 0.0), targets))
        logits = torch.tensor([2.5, -3.0])
        loss = criterion(logits, smoothed)
        self.assertTrue(math.isfinite(loss.item()))
        self.assertGreater(float(logits[0]), 1.0)
        with self.assertRaises(ValueError):
            tc.validate_label_smoothing(0.05)
        self.assertNotIn("sigmoid", inspect.getsource(clf.AuthenticityClassifier.forward).lower())
        self.assertNotIn("sigmoid", inspect.getsource(clf.ClassifierTrunk.forward).lower())

    def test_mixup_is_recorded_and_stays_out_of_image_transforms(self) -> None:
        self.assertNotIn("mixup", inspect.getsource(tv.build_transforms))
        self.assertNotIn("mixup", inspect.getsource(tv.val_epoch))
        root = Path(__file__).resolve().parents[1] / "ml_rtx5080" / "experiments"
        root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=root) as tmp:
            path = Path(tmp) / "config.json"
            tc.write_experiment_config(
                path,
                {"mixup_alpha": 0.05, "label_smoothing": 0.02, "sampler": "none"},
            )
            payload = json.loads(path.read_text())
        self.assertEqual(payload["mixup_alpha"], 0.05)
        self.assertEqual(payload["label_smoothing"], 0.02)
        self.assertEqual(tc.validate_mixup_alpha(0.0), 0.0)
        with self.assertRaises(ValueError):
            tc.validate_mixup_alpha(0.2)

    def test_bf16_fallback(self) -> None:
        enabled = tc.resolve_amp("bf16", cuda=True, bf16_supported=True)
        self.assertEqual(enabled["amp_dtype"], "bf16")
        self.assertFalse(enabled["amp_scaler"])
        self.assertIsNone(enabled["amp_fallback"])
        cuda_fallback = tc.resolve_amp("bf16", cuda=True, bf16_supported=False)
        self.assertEqual(cuda_fallback["amp_dtype"], "fp16")
        self.assertTrue(cuda_fallback["amp_scaler"])
        self.assertEqual(cuda_fallback["amp_fallback"], "fp16")
        cpu_fallback = tc.resolve_amp("bf16", cuda=False, bf16_supported=False)
        self.assertEqual(cpu_fallback["amp_dtype"], "fp32")
        self.assertFalse(cpu_fallback["amp_scaler"])
        self.assertEqual(cpu_fallback["amp_fallback"], "fp32")

    def test_checkpoint_directories_are_refused(self) -> None:
        with self.assertRaises(ValueError):
            tc.assert_experiment_output_dir(
                Path("/home/tensortorch26/Desktop/HypeVault/ml_rtx5080/checkpoints")
            )
        with self.assertRaises(ValueError):
            tc.assert_experiment_output_dir(
                Path("/home/tensortorch26/Desktop/HypeVault/ml_rtx5080/checkpoints_watches/run")
            )

    def test_tiny_training_smoke(self) -> None:
        torch.manual_seed(0)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        bf16_supported = bool(device.type == "cuda" and torch.cuda.is_bf16_supported())
        amp = tc.resolve_amp("bf16", cuda=device.type == "cuda", bf16_supported=bf16_supported)
        image_size = 504
        model = clf.build_authenticity_classifier(
            "dinov2", clf.CLS_ONLY, image_size=image_size, pretrained=False
        ).to(device)
        cfg = {
            "backbone_lr": 1e-4,
            "head_lr": 1e-3,
            "layer_decay": 0.75,
            "weight_decay": 0.05,
            "accumulate_grad": 2,
            "mixup_alpha": 0.0,
            "label_smoothing": 0.02,
            "grad_clip": 1.0,
            "amp_dtype": amp["amp_dtype"],
            "family": "dinov2",
            "classifier_arch": "cls_only",
        }
        optimizer, groups, _table = tv.build_optimizer(model, cfg)
        self.assertIn("block_11", [group["name"] for group in groups])
        self.assertGreater(len({group["lr"] for group in groups}), 2)
        scheduler = tv.cosine_schedule_with_warmup(optimizer, warmup_steps=0, total_steps=3)
        scaler = GradScaler("cuda", enabled=bool(amp["amp_scaler"]))
        loader = DataLoader(_SmokeImages(5, image_size), batch_size=1, shuffle=False)
        before = model.trunk.fc2.weight.detach().clone()
        start_epoch = scheduler.last_epoch
        loss, metrics, step = tv.train_epoch(
            model, loader, optimizer, tc.build_bce_loss(), scaler, scheduler, device, cfg, 0, 1
        )
        self.assertEqual(step, 3)
        self.assertEqual(tv.optimizer_updates_per_epoch(len(loader), 2), 3)
        self.assertEqual(scheduler.last_epoch - start_epoch, 3)
        self.assertTrue(math.isfinite(loss))
        self.assertTrue(math.isfinite(metrics["f1"]))
        self.assertFalse(torch.equal(before, model.trunk.fc2.weight.detach()))
        for parameter in model.parameters():
            self.assertTrue(torch.isfinite(parameter).all().item())
        val_loader = DataLoader(_SmokeImages(2, image_size), batch_size=1, shuffle=False)
        val_loss, val_metrics = tv.val_epoch(
            model, val_loader, tc.build_bce_loss(), device, cfg, epoch=1
        )
        self.assertTrue(math.isfinite(val_loss))
        self.assertTrue(math.isfinite(val_metrics["acc"]))

        root = Path(__file__).resolve().parents[1] / "ml_rtx5080" / "experiments"
        root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=root) as tmp:
            destination = Path(tmp)
            tc.write_experiment_config(destination / "config.json", cfg)
            recorded = json.loads((destination / "config.json").read_text())
            self.assertEqual(recorded["mixup_alpha"], 0.0)
            self.assertEqual(recorded["label_smoothing"], 0.02)
            weights = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            tv_path = destination / "latest.pt"
            tc.save_training_checkpoint(
                tv_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=1,
                config=recorded,
                global_step=step,
                val_acc=val_metrics["acc"],
                val_f1=val_metrics["f1"],
            )
            with torch.no_grad():
                model.trunk.fc2.weight.add_(1.0)
            resumed = tc.load_training_checkpoint(
                tv_path, model, optimizer, scheduler, map_location=device
            )
            self.assertIn("optimizer_state", resumed)
            self.assertIn("scheduler_state", resumed)
            self.assertEqual(resumed["config"]["mixup_alpha"], 0.0)
            self.assertEqual(resumed["global_step"], 3)
            for key, value in model.state_dict().items():
                self.assertTrue(torch.equal(value.detach().cpu(), weights[key]))


if __name__ == "__main__":
    unittest.main()
