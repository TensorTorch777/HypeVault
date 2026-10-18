"""Metric definitions, fixed threshold, and read-only evaluation."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ml_rtx5080"))

import evaluation as ev
import train as tv
from classifier import CLS_ONLY, CLS_PATCH_ATTENTION, build_authenticity_classifier
from evaluate import main as evaluate_main


class _TensorSet(Dataset):
    def __init__(self, images: torch.Tensor, labels: torch.Tensor) -> None:
        self.images = images
        self.labels = labels

    def __len__(self) -> int:
        return int(self.images.shape[0])

    def __getitem__(self, index: int):
        return self.images[index], int(self.labels[index]), f"sample-{index}"


class _Probe(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.bn = nn.BatchNorm2d(3)
        self.score = nn.Parameter(torch.tensor(0.25))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        pooled = self.bn(images).mean(dim=(1, 2, 3))
        return pooled + self.score


class MetricTests(unittest.TestCase):
    def test_known_logits_match_the_false_authentic_definition(self) -> None:
        logits = torch.tensor([-2.0, 2.0, -2.0, 2.0, 3.0])
        labels = torch.tensor([0, 0, 1, 1, 1])
        metrics = ev.evaluate_logits(logits, labels, threshold=0.50)
        self.assertEqual(metrics["positive_class"], "fake")
        self.assertEqual(metrics["threshold_policy"], "fixed_not_calibrated")
        self.assertNotIn("best_threshold", metrics)
        self.assertEqual(metrics["authentic_count"], 2)
        self.assertEqual(metrics["fake_count"], 3)
        self.assertEqual(metrics["sample_count"], 5)
        self.assertEqual(metrics["confusion_matrix"]["false_authentic"], 1)
        self.assertEqual(metrics["confusion_matrix"]["false_fake"], 1)
        self.assertEqual(metrics["confusion_matrix"]["true_positive"], 2)
        self.assertEqual(metrics["confusion_matrix"]["true_negative"], 1)
        self.assertAlmostEqual(metrics["false_authentic_rate"], 1 / 3)
        self.assertAlmostEqual(metrics["false_fake_rate"], 1 / 2)
        self.assertAlmostEqual(metrics["accuracy"], 3 / 5)
        self.assertAlmostEqual(metrics["precision"], 2 / 3)
        self.assertAlmostEqual(metrics["recall"], 2 / 3)
        self.assertAlmostEqual(metrics["f1"], 2 / 3)
        self.assertAlmostEqual(metrics["roc_auc"], 2 / 3)
        self.assertAlmostEqual(metrics["pr_auc"], (1 / 3) + (2 / 9) + (1 / 5))
        self.assertIsNone(ev.evaluate_logits(torch.zeros(3), torch.zeros(3))["false_authentic_rate"])
        self.assertIsNone(ev.evaluate_logits(torch.ones(3), torch.ones(3))["false_fake_rate"])

    def test_threshold_changes_predictions_and_not_logits(self) -> None:
        logits = torch.tensor([0.4054651081, -1.0])
        original = logits.clone()
        labels = torch.tensor([1, 0])
        low = ev.evaluate_logits(logits, labels, threshold=0.50)
        high = ev.evaluate_logits(logits, labels, threshold=0.70)
        self.assertEqual(low["confusion_matrix"]["true_positive"], 1)
        self.assertEqual(high["confusion_matrix"]["false_authentic"], 1)
        self.assertTrue(torch.equal(logits, original))
        probabilities = ev.probabilities_from_logits(logits)
        self.assertAlmostEqual(float(probabilities[0]), 0.6, places=6)
        self.assertFalse(torch.allclose(probabilities, logits))
        sweep = ev.threshold_sweep(logits, labels, thresholds=(0.50, 0.70))
        self.assertEqual(len(sweep), 2)
        self.assertTrue(all("best_threshold" not in row for row in sweep))
        with self.assertRaises(ValueError):
            ev.threshold_sweep(logits, labels, thresholds=())

    def test_evaluation_does_not_change_weights_optimizer_or_scheduler(self) -> None:
        torch.manual_seed(0)
        model = _Probe()
        model.train()
        images = torch.randn(4, 3, 8, 8)
        labels = torch.tensor([0, 1, 0, 1])
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        scheduler = tv.cosine_schedule_with_warmup(optimizer, warmup_steps=0, total_steps=4)
        loss = model(images).sum()
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        before_weights = {key: value.detach().clone() for key, value in model.state_dict().items()}
        before_optimizer = deepcopy(optimizer.state_dict())
        before_scheduler = deepcopy(scheduler.state_dict())
        loader = DataLoader(_TensorSet(images, labels), batch_size=2, shuffle=False)
        logits, out_labels, sample_ids = ev.collect_logits(model, loader, torch.device("cpu"))
        self.assertEqual(sample_ids, ["sample-0", "sample-1", "sample-2", "sample-3"])
        self.assertEqual(tuple(logits.shape), (4,))
        self.assertTrue(torch.equal(out_labels, labels))
        for key, value in model.state_dict().items():
            self.assertTrue(torch.equal(value, before_weights[key]))
        self.assertEqual(optimizer.state_dict()["param_groups"], before_optimizer["param_groups"])
        for key, value in optimizer.state_dict()["state"].items():
            for state_key, state_value in value.items():
                previous = before_optimizer["state"][key][state_key]
                if torch.is_tensor(state_value):
                    self.assertTrue(torch.equal(state_value, previous))
                else:
                    self.assertEqual(state_value, previous)
        self.assertEqual(scheduler.state_dict(), before_scheduler)
        self.assertTrue(model.training)

    def test_all_four_models_eval_without_changing_weights_or_accepting_504_for_dinov3(self) -> None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        specs = (
            ("dinov2", CLS_ONLY, 504),
            ("dinov2", CLS_PATCH_ATTENTION, 504),
            ("dinov3", CLS_ONLY, 512),
            ("dinov3", CLS_PATCH_ATTENTION, 512),
        )
        for family, arch, side in specs:
            with self.subTest(family=family, arch=arch):
                model = build_authenticity_classifier(
                    family, arch, image_size=side, pretrained=False
                ).to(device)
                self.assertNotIn("Sigmoid", [type(module).__name__ for module in model.modules()])
                if family == "dinov3":
                    with self.assertRaises(ValueError):
                        model(torch.zeros(1, 3, 504, 504, device=device))
                image = torch.zeros(1, 3, side, side, device=device)
                before = {key: value.detach().clone() for key, value in model.state_dict().items()}
                loader = DataLoader(
                    _TensorSet(image.cpu(), torch.tensor([1])),
                    batch_size=1,
                    shuffle=False,
                )
                logits, _, _ = ev.collect_logits(model, loader, device)
                self.assertEqual(tuple(logits.shape), (1,))
                self.assertTrue(torch.isfinite(logits).all())
                metrics = ev.evaluate_logits(logits, torch.tensor([1]), threshold=0.50)
                self.assertEqual(metrics["fake_count"], 1)
                self.assertEqual(metrics["threshold"], 0.50)
                for key, value in model.state_dict().items():
                    self.assertTrue(torch.equal(value, before[key]))
                del model
                if device.type == "cuda":
                    torch.cuda.empty_cache()

    def test_checkpoint_evaluation_writes_only_under_experiments(self) -> None:
        root = Path(__file__).resolve().parents[1] / "ml_rtx5080" / "experiments"
        root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=root) as tmp:
            experiment = Path(tmp)
            image_root = experiment / "images"
            authentic = image_root / "Label_0_Watches" / "brand"
            fake = image_root / "Label_1_Watches" / "brand"
            authentic.mkdir(parents=True)
            fake.mkdir(parents=True)
            Image.new("RGB", (96, 48), (180, 40, 40)).save(authentic / "wide.png")
            Image.new("RGB", (40, 90), (40, 40, 180)).save(fake / "tall.png")
            membership = {
                "train": [],
                "validation": [str((authentic / "wide.png").resolve()), str((fake / "tall.png").resolve())],
                "test": [],
            }
            manifest = experiment / "split_manifest.json"
            manifest.write_text(json.dumps({"membership": membership}))
            manifest_bytes = manifest.read_bytes()
            model = build_authenticity_classifier("dinov2", CLS_ONLY, pretrained=False)
            checkpoint = experiment / "probe.pt"
            torch.save({"model_state": model.state_dict(), "epoch": 1}, checkpoint)
            checkpoint_bytes = checkpoint.read_bytes()
            forbidden = Path(__file__).resolve().parents[1] / "ml_rtx5080" / "checkpoints"
            with self.assertRaises(ValueError):
                ev.write_evaluation_artifacts(forbidden, {"confusion_matrix": {}}, [])
            evaluate_main(
                [
                    "--checkpoint",
                    str(checkpoint),
                    "--split_manifest",
                    str(manifest),
                    "--output_dir",
                    str(experiment),
                    "--model",
                    "dinov2",
                    "--head",
                    "cls_only",
                    "--split",
                    "validation",
                    "--threshold",
                    "0.50",
                    "--batch_size",
                    "1",
                    "--device",
                    "cpu",
                ]
            )
            self.assertEqual(manifest.read_bytes(), manifest_bytes)
            self.assertEqual(checkpoint.read_bytes(), checkpoint_bytes)
            metrics = json.loads((experiment / "evaluation" / "metrics.json").read_text())
            self.assertEqual(metrics["sample_count"], 2)
            self.assertEqual(metrics["input_size"], 504)
            self.assertEqual(metrics["threshold"], 0.50)
            self.assertEqual(metrics["threshold_policy"], "fixed_not_calibrated")
            self.assertNotIn("best_threshold", metrics)
            self.assertIn("false_authentic_rate", metrics)
            matrix = json.loads((experiment / "evaluation" / "confusion_matrix.json").read_text())
            self.assertEqual(matrix["labels"], ["authentic", "fake"])
            predictions = (experiment / "evaluation" / "predictions.csv").read_text().strip().splitlines()
            self.assertEqual(
                predictions[0],
                "sample_id,brand,label,logit,probability,prediction,threshold",
            )
            self.assertEqual(len(predictions), 3)
            self.assertFalse((experiment / "evaluation" / "brand_metrics.json").exists())
            self.assertNotIn("pixel", predictions[1])


if __name__ == "__main__":
    unittest.main()
