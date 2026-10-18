"""Brand-balanced evaluation. Diagnostic only; no training and no selection changes."""

from __future__ import annotations

import json
import random
import sys
import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ml_rtx5080"))

import brand_evaluation as brand_eval
import checkpoint_selection as selection
import dataset as ds
import evaluation as ev
from classifier import CLS_ONLY, build_authenticity_classifier
from evaluate import main as evaluate_main


def _row(brand: str, label: int, logit: float, index: int) -> dict:
    folder = "Label_0_Watches" if label == 0 else "Label_1_Watches"
    if brand == ds.UNKNOWN:
        sample_id = str(Path("/catalog") / folder / f"{index}.jpg")
    else:
        sample_id = str(Path("/catalog") / folder / brand / f"{index}.jpg")
    probability = float(torch.sigmoid(torch.tensor(logit)))
    return {
        "sample_id": sample_id,
        "brand": brand,
        "label": label,
        "logit": logit,
        "probability": probability,
        "prediction": 1 if probability >= 0.5 else 0,
        "threshold": 0.5,
    }


def _brand_rows(
    brand: str,
    authentic: int,
    fake: int,
    false_authentic: int = 0,
    false_fake: int = 0,
) -> list[dict]:
    rows = []
    for index in range(authentic):
        logit = 4.0 if index < false_fake else -4.0
        rows.append(_row(brand, 0, logit, index))
    for index in range(fake):
        logit = -4.0 if index < false_authentic else 4.0
        rows.append(_row(brand, 1, logit, authentic + index))
    return rows


class BrandEvaluationTests(unittest.TestCase):
    def test_brand_identity_matches_the_dataset_loader(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for folder in ("Label_0_Watches", "Label_1_Watches"):
                for brand in ("Omega", "Rolex"):
                    directory = root / folder / brand
                    directory.mkdir(parents=True)
                    (directory / "0.jpg").write_bytes(b"img")
            records = ds.load_records(root)
            self.assertGreater(len(records), 0)
            for record in records:
                self.assertEqual(ds.brand_from_image_path(record.image_path), record.brand)
        self.assertEqual(
            ds.brand_from_image_path(Path("/catalog/Label_1_Watches/only.jpg")),
            ds.UNKNOWN,
        )

    def test_prediction_rows_group_known_brands_and_keep_unknown(self) -> None:
        sample_ids = [
            "/catalog/Label_0_Watches/Omega/a.jpg",
            "/catalog/Label_1_Watches/Rolex/b.jpg",
            "/catalog/Label_0_Watches/loose.jpg",
        ]
        rows = ev.prediction_rows(
            sample_ids,
            torch.tensor([0, 1, 0]),
            torch.tensor([-2.0, 2.0, -2.0]),
            threshold=0.5,
        )
        self.assertEqual([row["brand"] for row in rows], ["Omega", "Rolex", ds.UNKNOWN])
        report = brand_eval.build_brand_report(rows, minimum_brand_samples=1)
        names = [brand["brand"] for brand in report["brands"]]
        self.assertEqual(names, ["Omega", "Rolex", ds.UNKNOWN])
        unknown = report["brands"][-1]
        self.assertEqual(unknown["brand_status"], "UNKNOWN_BRAND")
        self.assertEqual(unknown["sample_count"], 1)
        self.assertEqual(report["diagnostics"]["unknown_brand_groups"], 1)

    def test_minimum_sample_rule_excludes_low_sample_from_macro(self) -> None:
        small = _brand_rows("Small", authentic=10, fake=9, false_authentic=1)
        large = _brand_rows("Large", authentic=20, fake=20, false_authentic=0)
        report = brand_eval.build_brand_report(small + large, minimum_brand_samples=20)
        by_name = {brand["brand"]: brand for brand in report["brands"]}
        self.assertEqual(by_name["Small"]["sample_status"], "low_sample")
        self.assertFalse(by_name["Small"]["metrics_reliable"])
        self.assertEqual(by_name["Small"]["sample_count"], 19)
        self.assertEqual(by_name["Large"]["sample_status"], "eligible")
        self.assertNotIn("Small", report["macro"]["false_authentic_rate_macro_brands"])
        self.assertEqual(report["macro"]["false_authentic_rate_macro_brands"], ["Large"])
        self.assertEqual(report["macro"]["false_authentic_rate_macro_n"], 1)
        at_minimum = brand_eval.build_brand_report(
            _brand_rows("Edge", authentic=10, fake=10, false_authentic=0),
            minimum_brand_samples=20,
        )
        self.assertEqual(at_minimum["brands"][0]["sample_status"], "eligible")

    def test_synthetic_brands_separate_macro_risk_and_class_limits(self) -> None:
        # FAR 0.01 needs 1 false authentic in 100 fakes. Fifty fakes cannot
        # produce that rate with an integer count, so brand A uses 100 fakes.
        # Brand C has 20 images. The default minimum is 20, which includes 20.
        # This case sets the minimum to 21 so those 20 images are low-sample.
        rows = []
        rows += _brand_rows("A", authentic=50, fake=100, false_authentic=1)
        rows += _brand_rows("B", authentic=50, fake=50, false_authentic=2)
        rows += _brand_rows("C", authentic=10, fake=10, false_authentic=1)
        rows += _brand_rows("D", authentic=50, fake=0)
        report = brand_eval.build_brand_report(
            rows,
            minimum_brand_samples=21,
            max_false_authentic_rate=0.02,
        )
        by_name = {brand["brand"]: brand for brand in report["brands"]}
        self.assertEqual(brand_eval.DEFAULT_MINIMUM_BRAND_SAMPLES, 20)
        self.assertAlmostEqual(by_name["A"]["false_authentic_rate"], 0.01)
        self.assertAlmostEqual(by_name["B"]["false_authentic_rate"], 0.04)
        self.assertEqual(by_name["A"]["false_authentic"], 1)
        self.assertEqual(by_name["B"]["false_authentic"], 2)
        self.assertTrue(by_name["A"]["brand_far_eligible"])
        self.assertFalse(by_name["B"]["brand_far_eligible"])
        self.assertGreater(by_name["A"]["far_gap_to_limit"], 0)
        self.assertLess(by_name["B"]["far_gap_to_limit"], 0)
        self.assertEqual(by_name["C"]["sample_status"], "low_sample")
        self.assertIsNone(by_name["C"]["brand_far_eligible"])
        self.assertEqual(by_name["C"]["false_authentic"], 1)
        self.assertTrue(by_name["D"]["zero_fake"])
        self.assertFalse(by_name["D"]["zero_authentic"])
        self.assertIsNone(by_name["D"]["false_authentic_rate"])
        self.assertIsNone(by_name["D"]["roc_auc"])
        self.assertIsNone(by_name["D"]["pr_auc"])
        self.assertEqual(by_name["D"]["roc_auc_status"], "unavailable")
        self.assertEqual(by_name["D"]["false_fake_rate"], 0.0)
        self.assertEqual(report["macro"]["false_authentic_rate_macro_brands"], ["A", "B"])
        self.assertAlmostEqual(report["macro"]["false_authentic_rate"], (0.01 + 0.04) / 2)
        self.assertEqual(report["macro"]["roc_auc_macro_n"], 2)
        self.assertNotIn("D", report["macro"]["roc_auc_macro_brands"])
        self.assertNotIn("C", report["macro"]["false_authentic_rate_macro_brands"])
        self.assertEqual(report["diagnostics"]["eligible_brands"], 3)
        self.assertEqual(report["diagnostics"]["low_sample_brands"], 1)
        self.assertEqual(report["diagnostics"]["brands_above_far_limit"], 1)
        self.assertEqual(report["diagnostics"]["brands_at_or_below_far_limit"], 1)
        self.assertAlmostEqual(report["diagnostics"]["max_eligible_brand_far"], 0.04)
        self.assertAlmostEqual(report["diagnostics"]["min_eligible_brand_far"], 0.01)
        self.assertAlmostEqual(report["diagnostics"]["far_spread"], 0.03)
        self.assertAlmostEqual(report["diagnostics"]["brand_far_std"], 0.015)
        self.assertEqual(
            [item["brand"] for item in report["rankings"]["highest_false_authentic_rate"]],
            ["B", "A"],
        )
        self.assertEqual(
            [item["brand"] for item in report["low_sample_rankings"]["highest_false_authentic_rate"]],
            ["C"],
        )
        self.assertFalse(report["checkpoint_selection_affected"])
        self.assertFalse(report["threshold_affected"])
        self.assertEqual(report["threshold_policy"], "fixed_not_calibrated")

    def test_zero_authentic_brand_does_not_invent_rates(self) -> None:
        rows = _brand_rows("FakesOnly", authentic=0, fake=5, false_authentic=1)
        report = brand_eval.build_brand_report(rows, minimum_brand_samples=1)
        brand = report["brands"][0]
        self.assertTrue(brand["zero_authentic"])
        self.assertIsNone(brand["false_fake_rate"])
        self.assertIsNone(brand["roc_auc"])
        self.assertIsNone(brand["pr_auc"])
        self.assertAlmostEqual(brand["false_authentic_rate"], 0.2)

    def test_macro_weights_brands_equally_and_global_stays_micro(self) -> None:
        rows = _brand_rows("A", authentic=50, fake=200, false_authentic=2)
        rows += _brand_rows("B", authentic=50, fake=50, false_authentic=2)
        report = brand_eval.build_brand_report(rows, minimum_brand_samples=20)
        direct = ev.evaluate_logits(
            torch.tensor([row["logit"] for row in rows], dtype=torch.float32),
            torch.tensor([row["label"] for row in rows], dtype=torch.long),
            threshold=0.5,
        )
        for key in (
            "sample_count",
            "authentic_count",
            "fake_count",
            "roc_auc",
            "pr_auc",
            "accuracy",
            "precision",
            "recall",
            "f1",
            "false_authentic_rate",
            "false_fake_rate",
        ):
            self.assertEqual(report["global"][key], direct[key])
        self.assertAlmostEqual(report["macro"]["false_authentic_rate"], 0.025)
        self.assertNotAlmostEqual(
            report["global"]["false_authentic_rate"],
            report["macro"]["false_authentic_rate"],
        )
        self.assertEqual(report["global_scope"], "micro")
        self.assertEqual(report["macro"]["scope"], "macro-brand")

    def test_rankings_do_not_depend_on_row_order(self) -> None:
        rows = []
        rows += _brand_rows("A", authentic=50, fake=100, false_authentic=1)
        rows += _brand_rows("B", authentic=50, fake=50, false_authentic=2)
        rows += _brand_rows("C", authentic=10, fake=10, false_authentic=1)
        first = brand_eval.build_brand_report(rows, minimum_brand_samples=21)
        rng = random.Random(7)
        shuffled = list(rows)
        rng.shuffle(shuffled)
        second = brand_eval.build_brand_report(shuffled, minimum_brand_samples=21)
        self.assertEqual(first["rankings"], second["rankings"])
        self.assertEqual(first["low_sample_rankings"], second["low_sample_rankings"])
        self.assertEqual(
            [brand["brand"] for brand in first["brands"]],
            [brand["brand"] for brand in second["brands"]],
        )

    def test_brand_report_does_not_change_weights_or_checkpoint_selection(self) -> None:
        layer = nn.Linear(2, 1)
        before = {key: value.detach().clone() for key, value in layer.state_dict().items()}
        rows = _brand_rows("A", authentic=30, fake=30, false_authentic=0)
        rows += _brand_rows("B", authentic=30, fake=30, false_authentic=3)
        brand_eval.build_brand_report(rows, minimum_brand_samples=20)
        for key, value in layer.state_dict().items():
            self.assertTrue(torch.equal(value, before[key]))
        source = Path(brand_eval.__file__).read_text()
        self.assertNotIn("select_checkpoints", source)
        self.assertNotIn("update_selection", source)
        candidates = [
            {
                "checkpoint_id": "unsafe",
                "epoch": 1,
                "global_step": 1,
                "validation_loss": 0.2,
                "model_family": "dinov2",
                "classifier_arch": "cls_only",
                "roc_auc": 0.99,
                "pr_auc": 0.99,
                "accuracy": 0.99,
                "precision": 0.99,
                "recall": 0.99,
                "f1": 0.99,
                "false_authentic_rate": 0.08,
                "false_fake_rate": 0.01,
                "threshold": 0.5,
                "eligible": True,
            },
            {
                "checkpoint_id": "safe",
                "epoch": 2,
                "global_step": 2,
                "validation_loss": 0.3,
                "model_family": "dinov2",
                "classifier_arch": "cls_only",
                "roc_auc": 0.9,
                "pr_auc": 0.9,
                "accuracy": 0.9,
                "precision": 0.9,
                "recall": 0.9,
                "f1": 0.9,
                "false_authentic_rate": 0.01,
                "false_fake_rate": 0.02,
                "threshold": 0.5,
                "eligible": True,
            },
        ]
        kwargs = {
            "max_false_authentic_rate": 0.02,
            "metric_tolerance": 1e-6,
            "expected_family": "dinov2",
            "expected_classifier_arch": "cls_only",
        }
        chosen = selection.select_checkpoints(candidates, **kwargs)["selected_checkpoint"]
        brand_eval.build_brand_report(rows)
        again = selection.select_checkpoints(candidates, **kwargs)["selected_checkpoint"]
        self.assertEqual(chosen, "safe")
        self.assertEqual(again, "safe")

    def test_brand_cli_writes_under_evaluation_and_leaves_files_unchanged(self) -> None:
        root = Path(__file__).resolve().parents[1] / "ml_rtx5080" / "experiments"
        root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=root) as tmp:
            experiment = Path(tmp)
            authentic = experiment / "images" / "Label_0_Watches" / "Omega"
            fake = experiment / "images" / "Label_1_Watches" / "Rolex"
            authentic.mkdir(parents=True)
            fake.mkdir(parents=True)
            Image.new("RGB", (32, 32), (20, 20, 20)).save(authentic / "a.png")
            Image.new("RGB", (32, 48), (20, 20, 20)).save(fake / "b.png")
            manifest = experiment / "split_manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "membership": {
                            "validation": [
                                str((authentic / "a.png").resolve()),
                                str((fake / "b.png").resolve()),
                            ]
                        }
                    }
                )
            )
            manifest_bytes = manifest.read_bytes()
            model = build_authenticity_classifier("dinov2", CLS_ONLY, pretrained=False)
            checkpoint = experiment / "probe.pt"
            torch.save({"model_state": model.state_dict()}, checkpoint)
            checkpoint_bytes = checkpoint.read_bytes()
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
                    "--threshold",
                    "0.50",
                    "--batch_size",
                    "1",
                    "--device",
                    "cpu",
                    "--brand_eval",
                    "1",
                    "--minimum_brand_samples",
                    "20",
                ]
            )
            self.assertEqual(manifest.read_bytes(), manifest_bytes)
            self.assertEqual(checkpoint.read_bytes(), checkpoint_bytes)
            evaluation_dir = experiment / "evaluation"
            metrics = json.loads((evaluation_dir / "metrics.json").read_text())
            brand_metrics = json.loads((evaluation_dir / "brand_metrics.json").read_text())
            summary = json.loads((evaluation_dir / "brand_summary.json").read_text())
            self.assertEqual(metrics["sample_count"], brand_metrics["global"]["sample_count"])
            self.assertEqual(metrics["false_authentic_rate"], brand_metrics["global"]["false_authentic_rate"])
            self.assertEqual(metrics["threshold"], 0.5)
            self.assertFalse(summary["checkpoint_selection_affected"])
            self.assertEqual({brand["brand"] for brand in brand_metrics["brands"]}, {"Omega", "Rolex"})
            self.assertTrue((evaluation_dir / "brand_metrics.csv").is_file())
            self.assertTrue((evaluation_dir / "brand_risk.csv").is_file())
            self.assertTrue((evaluation_dir / "brand_far.png").is_file())
            refused = Path(__file__).resolve().parents[1] / "ml_rtx5080" / "checkpoints"
            with self.assertRaises(ValueError):
                brand_eval.write_brand_evaluation(refused, _brand_rows("A", 5, 5))


if __name__ == "__main__":
    unittest.main()
