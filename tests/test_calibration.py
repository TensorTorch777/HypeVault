"""Temperature scaling, calibration split, and threshold candidates."""

from __future__ import annotations

import csv
import json
import math
import sys
import tempfile
import unittest
import unittest.mock
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ml_rtx5080"))

import calibration as cal
import calibration_phase20 as phase20
import dataset as ds
import evaluation as ev
from calibrate import main as calibrate_main
from classifier import CLS_ONLY, build_authenticity_classifier


def _records(root: Path, brands: list[str], per_brand: int = 2) -> list[ds.SampleRecord]:
    records = []
    for label, folder in ((0, "Label_0_Watches"), (1, "Label_1_Watches")):
        for brand in brands:
            directory = root / folder / brand
            directory.mkdir(parents=True, exist_ok=True)
            for index in range(per_brand):
                path = directory / f"{index}.jpg"
                path.write_bytes(b"img")
                records.append(
                    ds.SampleRecord(
                        image_path=path,
                        label=label,
                        brand=brand,
                        product_id=ds.UNKNOWN,
                        marketplace=ds.UNKNOWN,
                        source=ds.UNKNOWN,
                        split_group=ds.product_group_key(path),
                    )
                )
    return records


class CalibrationMathTests(unittest.TestCase):
    def test_temperature_one_matches_baseline_sigmoid(self) -> None:
        logits = torch.tensor([-2.0, -0.5, 0.0, 0.5, 3.0])
        scaled = cal.probabilities_from_temperature(logits, 1.0)
        self.assertTrue(torch.allclose(scaled, torch.sigmoid(logits)))

    def test_temperature_preserves_probability_order(self) -> None:
        logits = torch.tensor([-4.0, -1.0, 0.2, 1.5, 6.0])
        baseline_order = torch.argsort(torch.sigmoid(logits))
        for temperature in (0.5, 2.0):
            scaled = cal.probabilities_from_temperature(logits, temperature)
            self.assertTrue(torch.equal(torch.argsort(scaled), baseline_order))
            self.assertTrue(torch.all((scaled > 0) & (scaled < 1)))
            self.assertGreater(float(scaled[-1]), float(scaled[0]))

    def test_fitting_does_not_change_model_parameters_or_logits(self) -> None:
        layer = nn.Linear(2, 1)
        before = {name: value.detach().clone() for name, value in layer.state_dict().items()}
        inputs = torch.tensor([[1.0, -1.0], [0.5, 0.25], [-2.0, 0.5], [3.0, -3.0]])
        logits = layer(inputs).reshape(-1)
        original = logits.detach().clone()
        labels = torch.tensor([0, 0, 1, 1])
        fitted = cal.fit_temperature(logits, labels)
        for name, value in layer.state_dict().items():
            self.assertTrue(torch.equal(value, before[name]))
        self.assertTrue(torch.equal(logits.detach(), original))
        self.assertGreater(fitted["temperature"], 0.0)
        self.assertTrue(torch.isfinite(torch.tensor(fitted["temperature"])))

    def test_overconfident_logits_improve_nll_without_changing_rank(self) -> None:
        labels = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
        logits = torch.tensor([-3.0, -3.0, -3.0, 3.0, 3.0, 3.0, 3.0, -3.0])
        fitted = cal.fit_temperature(logits, labels)
        self.assertGreater(fitted["temperature"], 1.0)
        self.assertLessEqual(fitted["fitted_nll"], fitted["baseline_nll"])
        baseline = cal.score_temperature(logits, labels, 1.0, n_bins=10)
        calibrated = cal.score_temperature(logits, labels, fitted["temperature"], n_bins=10)
        comparison = cal.compare_calibration(baseline, calibrated)
        self.assertGreater(comparison["nll_absolute_improvement"], 0.0)
        self.assertTrue(comparison["roc_auc_unchanged"])
        self.assertTrue(comparison["pr_auc_unchanged"])
        self.assertFalse(comparison["classification_change_claimed"])
        self.assertLessEqual(calibrated["brier"], baseline["brier"])
        self.assertLessEqual(calibrated["ece"], baseline["ece"])
        order = torch.argsort(cal.probabilities_from_temperature(logits, 1.0))
        scaled_order = torch.argsort(
            cal.probabilities_from_temperature(logits, fitted["temperature"])
        )
        self.assertTrue(torch.equal(order, scaled_order))

    def test_brier_score_matches_hand_values(self) -> None:
        self.assertAlmostEqual(
            cal.brier_score(torch.tensor([0.0, 1.0]), torch.tensor([0, 1])), 0.0
        )
        self.assertAlmostEqual(
            cal.brier_score(torch.tensor([1.0, 0.0]), torch.tensor([0, 1])), 1.0
        )
        self.assertAlmostEqual(
            cal.brier_score(torch.tensor([0.5, 0.5]), torch.tensor([0, 1])), 0.25
        )

    def test_ece_matches_hand_example(self) -> None:
        # All four predicted-class confidences are 0.9 and land in the top bin.
        # Two of the four predictions are wrong, so the gap is 0.4 and ECE is 0.4.
        probabilities = torch.tensor([0.9, 0.9, 0.1, 0.1])
        labels = torch.tensor([1, 0, 0, 1])
        ece, bins = cal.expected_calibration_error(probabilities, labels, n_bins=10)
        self.assertAlmostEqual(ece, 0.4)
        occupied = [row for row in bins if row["sample_count"] > 0]
        self.assertEqual(len(occupied), 1)
        self.assertEqual(occupied[0]["bin"], 9)
        self.assertEqual(occupied[0]["sample_count"], 4)
        self.assertAlmostEqual(occupied[0]["confidence_mean"], 0.9)
        self.assertAlmostEqual(occupied[0]["observed_accuracy"], 0.5)
        self.assertAlmostEqual(occupied[0]["absolute_gap"], 0.4)
        empty = [row for row in bins if row["sample_count"] == 0]
        self.assertEqual(len(empty), 9)
        self.assertIsNone(empty[0]["confidence_mean"])

    def test_threshold_sweep_far_matches_evaluator(self) -> None:
        logits = torch.tensor([-4.0, -4.0, 4.0, 4.0, -1.0, -1.0])
        labels = torch.tensor([0, 0, 1, 1, 1, 1])
        rows = cal.threshold_sweep(logits, labels, temperature=1.0)
        self.assertEqual([row["threshold"] for row in rows], [step / 100 for step in range(5, 100, 5)])
        weak = float(torch.sigmoid(torch.tensor(-1.0)))
        for row in rows:
            metrics = ev.evaluate_logits(logits, labels, threshold=row["threshold"])
            self.assertEqual(row["false_authentic"], metrics["false_authentic"])
            self.assertAlmostEqual(row["false_authentic_rate"], metrics["false_authentic_rate"])
            expected_far = 0.0 if weak >= row["threshold"] else 0.5
            self.assertAlmostEqual(row["false_authentic_rate"], expected_far)
        candidates = cal.calibration_threshold_candidates(rows, max_false_authentic_rate=0.02)
        self.assertIsNone(candidates["production_threshold"])
        self.assertEqual(candidates["role"], "calibration_candidates")
        for threshold in candidates["safe_thresholds"]:
            matched = next(row for row in rows if row["threshold"] == threshold)
            self.assertLessEqual(matched["false_authentic_rate"], 0.02)
        self.assertNotIn(0.50, candidates["safe_thresholds"])
        self.assertEqual(candidates["highest_f1"]["threshold"], 0.25)
        self.assertIsNotNone(candidates["highest_recall"])
        self.assertIsNotNone(candidates["lowest_far"])
        self.assertEqual(candidates["closest_to_0_50"]["threshold"], 0.25)
        neighbors = cal.threshold_sensitivity(logits, labels, 1.0, candidates["closest_to_0_50"])
        self.assertEqual([row["threshold"] for row in neighbors], [0.23, 0.25, 0.27])

    def test_normal_resolution_refuses_validation_and_test(self) -> None:
        held_out = "/tmp/Label_1_Watches/TestBrand/0.jpg"
        validation = "/tmp/Label_0_Watches/ValBrand/0.jpg"
        with self.assertRaises(ValueError):
            cal.resolve_calibration_membership(
                {"membership": {"validation": [validation], "test": [held_out]}},
                seed=1,
            )
        with self.assertRaises(ValueError):
            cal.resolve_calibration_membership(
                {
                    "membership": {
                        "train": ["/tmp/Label_0_Watches/TrainBrand/0.jpg"],
                        "calibration": [held_out],
                        "test": [held_out],
                    }
                },
                seed=1,
            )
        resolved = cal.resolve_calibration_membership(
            {"membership": {"validation": [validation], "test": [held_out]}},
            seed=1,
            debug_nonproduction_split="test",
        )
        self.assertTrue(resolved["non_production"])
        self.assertEqual(resolved["source"], "test")

    def test_resolution_runs_without_a_test_split(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            records = _records(root, ["A", "B", "C", "D"])
            manifest = {
                "membership": {
                    "train": [str(record.image_path.resolve()) for record in records],
                    "validation": [],
                }
            }
            resolved = cal.resolve_calibration_membership(manifest, fraction=0.10, seed=7)
            self.assertNotIn("test", manifest["membership"])
            self.assertEqual(resolved["test_path_count"], 0)
            self.assertFalse(resolved["non_production"])
            self.assertTrue(resolved["paths"])
            self.assertFalse(set(resolved["paths"]) & set(resolved["weight_train_paths"]))

    def test_brand_summaries_reuse_the_one_global_temperature(self) -> None:
        rows = []
        for brand in ("Alpha", "Beta"):
            for _ in range(20):
                rows.append({"brand": brand, "label": 0, "logit": -2.0})
                rows.append({"brand": brand, "label": 1, "logit": 2.0})
        summaries = cal.brand_calibration_summaries(rows, temperature=1.7, minimum_brand_samples=20)
        self.assertEqual([row["temperature"] for row in summaries], [1.7, 1.7])
        self.assertTrue(all(row["sample_status"] == "eligible" for row in summaries))
        short = cal.brand_calibration_summaries(rows[:10], temperature=1.7, minimum_brand_samples=20)
        self.assertTrue(all(row["nll"] is None for row in short))
        self.assertTrue(all(row["temperature"] == 1.7 for row in short))

    def test_calibration_module_does_not_rerun_checkpoint_selection(self) -> None:
        source = Path(cal.__file__).read_text()
        self.assertNotIn("select_checkpoints", source)
        self.assertNotIn("update_selection", source)
        self.assertNotIn("import checkpoint_selection", source)


class CalibrationSplitTests(unittest.TestCase):
    def test_carve_is_deterministic_and_directory_disjoint(self) -> None:
        self.assertEqual(ds.DEFAULT_CALIBRATION_FRACTION, 0.10)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            records = _records(root, ["A", "B", "C", "D"])
            splits, meta = ds.split_records(records, "final_holdout", 11, 0.25, 0.25)
            validation_paths = [record.image_path.resolve() for record in splits["validation"]]
            test_paths = [record.image_path.resolve() for record in splits["test"]]
            first_train, first_cal = ds.carve_calibration_records(splits["train"], 0.10, 11)
            second_train, second_cal = ds.carve_calibration_records(splits["train"], 0.10, 11)
            self.assertEqual(
                [record.image_path for record in first_cal],
                [record.image_path for record in second_cal],
            )
            self.assertEqual(
                [record.image_path for record in first_train],
                [record.image_path for record in second_train],
            )
            carved = {
                "train": first_train,
                "calibration": first_cal,
                "validation": list(splits["validation"]),
                "test": list(splits["test"]),
            }
            self.assertEqual(
                [record.image_path.resolve() for record in carved["validation"]],
                validation_paths,
            )
            self.assertEqual(
                [record.image_path.resolve() for record in carved["test"]],
                test_paths,
            )
            ds.verify_no_image_overlap(carved)
            ds.verify_no_product_overlap(carved)
            train_groups = {record.split_group for record in first_train}
            calibration_groups = {record.split_group for record in first_cal}
            validation_groups = {record.split_group for record in splits["validation"]}
            test_groups = {record.split_group for record in splits["test"]}
            self.assertTrue(calibration_groups)
            self.assertFalse(calibration_groups & train_groups)
            self.assertFalse(calibration_groups & validation_groups)
            self.assertFalse(calibration_groups & test_groups)
            for group in calibration_groups:
                images = [record for record in records if record.split_group == group]
                self.assertGreater(len(images), 1)
                self.assertTrue(all(record in first_cal for record in images))
            payload = ds.write_split_manifest(
                root / "out",
                records,
                carved,
                strategy="final_holdout",
                seed=11,
                val_fraction=0.25,
                test_fraction=0.25,
                split_meta={**meta, "calibration_fraction": 0.10},
                source_report={"status": "unavailable"},
                product_report={"product_id_status": "unavailable"},
            )
            record = payload["calibration_split"]
            self.assertFalse(record["weight_updates_allowed"])
            self.assertFalse(record["used_for_checkpoint_selection"])
            self.assertFalse(record["used_for_final_test"])
            self.assertEqual(record["parent_split"], "train")
            self.assertEqual(record["leakage_unit"], "parent_directory")
            self.assertEqual(record["product_id_status"], "unavailable")
            self.assertEqual(record["source_status"], "unavailable")
            self.assertEqual(record["marketplace_status"], "unavailable")
            self.assertIn("not marketplace-disjoint", record["leakage_note"])
            self.assertEqual(record["random_seed"], 11)
            self.assertEqual(record["sample_count"], len(first_cal))
            self.assertEqual(
                payload["membership"]["validation"],
                [str(path) for path in validation_paths],
            )
            self.assertEqual(
                payload["membership"]["test"],
                [str(path) for path in test_paths],
            )
            trainer = Path(__file__).resolve().parents[1].joinpath("ml_rtx5080", "train.py").read_text()
            carve_call = trainer.index("weight_train, calibration_records = carve_calibration_records")
            loader_line = trainer.index("train_samples = ")
            self.assertLess(carve_call, loader_line)


class CalibrationCliTests(unittest.TestCase):
    def test_cli_refuses_final_test_and_writes_no_production_threshold(self) -> None:
        root = Path(__file__).resolve().parents[1] / "ml_rtx5080" / "experiments"
        root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=root) as tmp:
            experiment = Path(tmp)
            image_root = experiment / "images"
            authentic = image_root / "Label_0_Watches" / "CalBrand"
            fake = image_root / "Label_1_Watches" / "CalBrand"
            validation = image_root / "Label_0_Watches" / "ValBrand"
            held_out = image_root / "Label_1_Watches" / "TestBrand"
            for directory in (authentic, fake, validation, held_out):
                directory.mkdir(parents=True)
            Image.new("RGB", (32, 32), (20, 20, 20)).save(authentic / "a.png")
            Image.new("RGB", (32, 48), (200, 20, 20)).save(fake / "b.png")
            Image.new("RGB", (32, 32), (20, 200, 20)).save(validation / "v.png")
            Image.new("RGB", (32, 32), (20, 20, 200)).save(held_out / "t.png")
            calibration_paths = [
                str((authentic / "a.png").resolve()),
                str((fake / "b.png").resolve()),
            ]
            validation_path = str((validation / "v.png").resolve())
            test_path = str((held_out / "t.png").resolve())
            manifest = experiment / "split_manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "membership": {
                            "validation": [validation_path],
                            "test": [test_path],
                        }
                    }
                )
            )
            dummy = experiment / "not-a-model.pt"
            dummy.write_bytes(b"checkpoint")
            with self.assertRaises(ValueError):
                calibrate_main(
                    [
                        "--checkpoint",
                        str(dummy),
                        "--split_manifest",
                        str(manifest),
                        "--experiment_dir",
                        str(experiment),
                        "--model",
                        "dinov2",
                        "--head",
                        "cls_only",
                    ]
                )
            refused = json.loads(manifest.read_text())
            refused["membership"]["calibration"] = [test_path]
            refused["membership"]["test"] = [test_path]
            manifest.write_text(json.dumps(refused))
            with self.assertRaises(ValueError):
                calibrate_main(
                    [
                        "--checkpoint",
                        str(dummy),
                        "--split_manifest",
                        str(manifest),
                        "--experiment_dir",
                        str(experiment),
                        "--model",
                        "dinov2",
                        "--head",
                        "cls_only",
                    ]
                )
            manifest.write_text(
                json.dumps(
                    {
                        "dataset_hash": "smoke",
                        "membership": {
                            "calibration": calibration_paths,
                            "train": [],
                            "validation": [str((validation / "v.png").resolve())],
                        },
                    }
                )
            )
            manifest_bytes = manifest.read_bytes()
            model = build_authenticity_classifier("dinov2", CLS_ONLY, pretrained=False)
            checkpoint = experiment / "probe.pt"
            torch.save({"model_state": model.state_dict()}, checkpoint)
            checkpoint_bytes = checkpoint.read_bytes()
            calibrate_main(
                [
                    "--checkpoint",
                    str(checkpoint),
                    "--split_manifest",
                    str(manifest),
                    "--experiment_dir",
                    str(experiment),
                    "--model",
                    "dinov2",
                    "--head",
                    "cls_only",
                    "--bins",
                    "10",
                    "--batch_size",
                    "1",
                    "--device",
                    "cpu",
                ]
            )
            self.assertEqual(manifest.read_bytes(), manifest_bytes)
            self.assertEqual(checkpoint.read_bytes(), checkpoint_bytes)
            destination = experiment / "calibration"
            for name in (
                "calibration_predictions.csv",
                "temperature.json",
                "calibration_metrics.json",
                "threshold_candidates.csv",
                "reliability_bins.csv",
                "calibration_summary.json",
            ):
                self.assertTrue((destination / name).is_file(), name)
            summary = json.loads((destination / "calibration_summary.json").read_text())
            temperature = json.loads((destination / "temperature.json").read_text())
            metrics = json.loads((destination / "calibration_metrics.json").read_text())
            self.assertTrue(summary["calibration_succeeded"])
            self.assertTrue(summary["valid_temperature"])
            self.assertIsNone(summary["production_threshold"])
            self.assertFalse(summary["production_threshold_selected"])
            self.assertTrue(summary["checkpoint_selection_completed"])
            self.assertTrue(summary["calibration_fitted_after_selection"])
            self.assertFalse(summary["checkpoint_selection_rerun"])
            self.assertFalse(summary["non_production"])
            self.assertGreater(temperature["temperature"], 0.0)
            self.assertIsNone(temperature["production_threshold"])
            self.assertNotIn("model_state", temperature)
            self.assertFalse(metrics["classification_change_claimed"])
            self.assertIn("nll", metrics["baseline"])
            self.assertIn("brier", metrics["baseline"])
            self.assertIn("ece", metrics["baseline"])
            self.assertIn("roc_auc", metrics["baseline"])
            self.assertIn("pr_auc", metrics["baseline"])
            predictions = (destination / "calibration_predictions.csv").read_text()
            self.assertIn(calibration_paths[0], predictions)
            self.assertNotIn(test_path, predictions)
            self.assertNotIn(str((validation / "v.png").resolve()), predictions)
            forbidden = Path(__file__).resolve().parents[1] / "ml_rtx5080" / "checkpoints"
            self.assertFalse(forbidden.joinpath("calibration").exists())


def _manifest_path() -> Path:
    return (
        Path(__file__).resolve().parents[1]
        / "ml_rtx5080"
        / "experiments"
        / "dataset_audit"
        / "split_manifest_v2.json"
    )


def _synthetic_records() -> tuple[list[dict], torch.Tensor]:
    records = []
    logits = []
    pattern = [-3.0, -3.0, -3.0, 3.0, 3.0, 3.0, 3.0, -3.0]
    labels = [0, 0, 0, 0, 1, 1, 1, 1]
    for brand_index, brand in enumerate(("Alpha", "Beta")):
        for index, (logit, label) in enumerate(zip(pattern, labels)):
            records.append(
                {
                    "sample_id": f"{brand}-{index}",
                    "path": f"/tmp/{brand}/{index}.jpg",
                    "label": label,
                    "brand": brand,
                    "duplicate_group_id": f"{brand}-{index}",
                }
            )
            logits.append(logit + brand_index * 0.0)
    return records, torch.tensor(logits, dtype=torch.float32)


class CalibrationPhase20Tests(unittest.TestCase):
    def test_real_manifest_membership_counts_brands_and_no_image_opens(self) -> None:
        manifest = json.loads(_manifest_path().read_text())
        opened = []

        def reject_open(*args, **kwargs):
            opened.append(args[0] if args else kwargs.get("fp"))
            raise AssertionError("calibration validation opened an image")

        with unittest.mock.patch("PIL.Image.open", reject_open):
            records = phase20.validate_phase16_calibration_manifest(manifest)
        self.assertEqual(opened, [])
        self.assertEqual(len(records), 2998)
        self.assertEqual(sum(row["label"] == 0 for row in records), 1498)
        self.assertEqual(sum(row["label"] == 1 for row in records), 1500)
        brands = {row["brand"] for row in records}
        self.assertEqual(brands, set(phase20.CATALOG_BRANDS))
        for brand in phase20.CATALOG_BRANDS:
            self.assertTrue(any(row["brand"] == brand and row["label"] == 0 for row in records))
            self.assertTrue(any(row["brand"] == brand and row["label"] == 1 for row in records))
        test_paths = {str(Path(path).resolve()) for path in manifest["membership"]["test"]}
        train_paths = {str(Path(path).resolve()) for path in manifest["membership"]["train"]}
        validation_paths = {str(Path(path).resolve()) for path in manifest["membership"]["validation"]}
        calibration_paths = {row["path"] for row in records}
        self.assertFalse(calibration_paths & test_paths)
        self.assertFalse(calibration_paths & train_paths)
        self.assertFalse(calibration_paths & validation_paths)
        self.assertEqual(manifest["membership_hash"], phase20.AUTHORITATIVE_MEMBERSHIP_HASH)

    def test_overlap_and_hash_mismatch_stop_validation(self) -> None:
        manifest = json.loads(_manifest_path().read_text())
        manifest["membership"]["calibration"][0] = manifest["membership"]["train"][0]
        with self.assertRaises(ValueError):
            phase20.validate_phase16_calibration_manifest(manifest)

    def test_one_positive_temperature_preserves_logits_and_repeated_fits(self) -> None:
        records, logits = _synthetic_records()
        original = logits.clone()
        first = phase20.choose_temperature(logits, torch.tensor([row["label"] for row in records]))
        second = phase20.choose_temperature(logits, torch.tensor([row["label"] for row in records]))
        self.assertTrue(torch.equal(logits, original))
        self.assertGreater(first["temperature"], 0.0)
        self.assertTrue(math.isfinite(first["temperature"]))
        self.assertLessEqual(
            abs(first["temperature"] - second["temperature"]),
            phase20.TEMPERATURE_REPEAT_TOLERANCE,
        )
        self.assertIsInstance(first["temperature"], float)
        result = phase20.calibration_result(
            records,
            logits,
            spec={"family": "dinov2", "head": "cls_only", "input_size": 504},
            checkpoint_path=Path("/tmp/epoch_008.pt"),
        )
        temperatures = {row["temperature"] for row in result["brand_calibrated"]}
        self.assertEqual(temperatures, {result["temperature_fit"]["temperature"]})
        self.assertFalse(result["temperature_improved_nll"] if False else result["architecture_ranking_affected"])
        self.assertLess(result["temperature_fit"]["calibrated_nll"], result["temperature_fit"]["baseline_nll"])
        self.assertLessEqual(result["calibrated"]["brier"], result["baseline"]["brier"])
        self.assertLessEqual(result["calibrated"]["ece"], result["baseline"]["ece"])
        self.assertTrue(result["comparison"]["roc_auc_unchanged"])
        self.assertTrue(result["comparison"]["pr_auc_unchanged"])
        self.assertEqual(result["baseline"]["roc_auc"], result["calibrated"]["roc_auc"])
        self.assertEqual(result["baseline"]["pr_auc"], result["calibrated"]["pr_auc"])
        self.assertTrue(torch.equal(logits, original))

    def test_checkpoint_bytes_and_production_threshold_stay_untouched(self) -> None:
        records, logits = _synthetic_records()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = root / "epoch_008.pt"
            checkpoint.write_bytes(b"not-a-weight-update")
            digest = phase20.file_sha256(checkpoint)
            result = phase20.calibration_result(
                records,
                logits,
                spec={"family": "dinov2", "head": "cls_only", "input_size": 504},
                checkpoint_path=checkpoint,
            )
            phase20.write_model_artifacts(root / "experiment", result)
            self.assertEqual(checkpoint.read_bytes(), b"not-a-weight-update")
            self.assertEqual(phase20.file_sha256(checkpoint), digest)
            temperature = json.loads((root / "experiment" / "calibration" / "temperature.json").read_text())
            summary = json.loads((root / "experiment" / "calibration" / "calibration_summary.json").read_text())
            self.assertIsNone(temperature["production_threshold"])
            self.assertIsNone(summary["production_threshold"])
            self.assertFalse(summary["architecture_ranking_affected"])
            self.assertFalse(summary["test_images_loaded"])
            self.assertFalse(summary["ood_used"])
            self.assertTrue(summary["checkpoint_selection_completed"])
            self.assertTrue(summary["calibration_fitted_after_selection"])

    def test_false_authentic_rate_uses_the_fake_denominator(self) -> None:
        labels = torch.tensor([0, 0, 0, 0, 1, 1], dtype=torch.float32)
        logits = torch.tensor([-10.0, -10.0, -10.0, -10.0, -10.0, 10.0])
        point = phase20._operating_point(logits, labels, 0.5, 1.0)
        self.assertEqual(point["fake_count"], 2)
        self.assertEqual(point["false_authentic_count"], 1)
        self.assertAlmostEqual(point["false_authentic_rate"], 0.5)
        self.assertNotAlmostEqual(point["false_authentic_rate"], 1 / 6)

    def test_artifact_schema_and_writer_guards(self) -> None:
        records, logits = _synthetic_records()
        original = [float(value) for value in logits]
        result = phase20.calibration_result(
            records,
            logits,
            spec={"family": "dinov2", "head": "cls_only", "input_size": 504},
            checkpoint_path=Path("/tmp/epoch_008.pt"),
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            destination = phase20.write_model_artifacts(root / "experiment", result)
            comparison = phase20.write_comparison(root / "calibration_comparison_v1", [result])
            expected = {
                "calibration_predictions.csv",
                "temperature.json",
                "calibration_metrics.json",
                "threshold_candidates.csv",
                "reliability_bins_baseline.csv",
                "reliability_bins_calibrated.csv",
                "calibration_summary.json",
            }
            self.assertEqual(expected, {path.name for path in destination.iterdir()})
            self.assertTrue((comparison / "calibration_comparison.json").is_file())
            self.assertTrue((comparison / "calibration_comparison.csv").is_file())
            temperature = json.loads((destination / "temperature.json").read_text())
            for key in (
                "checkpoint",
                "model_family",
                "model_architecture",
                "head",
                "input_size",
                "manifest_hash",
                "calibration_sample_count",
                "authentic_count",
                "fake_count",
                "temperature",
                "initial_temperature",
                "baseline_nll",
                "calibrated_nll",
                "temperature_improved_nll",
                "fitting_method",
                "fitting_status",
            ):
                self.assertIn(key, temperature)
            self.assertEqual(temperature["initial_temperature"], 1.0)
            self.assertGreater(temperature["temperature"], 0.0)
            metrics = json.loads((destination / "calibration_metrics.json").read_text())
            for side in ("baseline", "calibrated"):
                for key in ("nll", "brier", "ece", "roc_auc", "pr_auc"):
                    self.assertIn(key, metrics[side])
            bins = (destination / "reliability_bins_baseline.csv").read_text().splitlines()
            self.assertEqual(
                bins[0],
                "bin_lower,bin_upper,sample_count,mean_confidence,observed_accuracy,absolute_gap",
            )
            self.assertEqual(len(bins), 11)
            with (destination / "calibration_predictions.csv").open(newline="") as handle:
                stored = list(csv.DictReader(handle))
            self.assertEqual([float(row["logit"]) for row in stored], original)
            self.assertIsNone(json.loads((comparison / "calibration_comparison.json").read_text())["production_threshold"])
            self.assertFalse(
                json.loads((comparison / "calibration_comparison.json").read_text())["architecture_ranking_affected"]
            )
            repo = Path(__file__).resolve().parents[1] / "ml_rtx5080"
            with self.assertRaises(RuntimeError):
                phase20.write_model_artifacts(repo / "experiments" / "dinov2_cls_only", result)
            with self.assertRaises(RuntimeError):
                phase20.write_comparison(repo / "checkpoints" / "calibration_comparison_v1", [result])

    def test_runner_does_not_select_checkpoints_carve_or_read_the_test_split(self) -> None:
        source = Path(phase20.__file__).read_text()
        self.assertNotIn("select_checkpoints", source)
        self.assertNotIn("carve_calibration_records", source)
        self.assertNotIn("resolve_calibration_membership", source)
        self.assertNotIn("load_split_membership", source)
        self.assertNotIn("membership.test", source.replace('manifest["membership"]["test"]', ""))
        self.assertIn('manifest["membership"]["test"]', source)


if __name__ == "__main__":
    unittest.main()
