"""Safety-constrained checkpoint selection. No GPU and no catalog training."""

from __future__ import annotations

import csv
import json
import random
import sys
import tempfile
import unittest
import unittest.mock
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ml_rtx5080"))

import checkpoint_policy_v2 as policy
import checkpoint_policy_v3 as policy_v3
import checkpoint_selection as selection
import train as tv


def _candidate(
    name: str,
    pr_auc: float,
    f1: float,
    far: float,
    *,
    epoch: int = 1,
    step: int = 1,
    family: str = "dinov2",
    arch: str = "cls_only",
    threshold: float = 0.5,
    **extra,
) -> dict:
    row = {
        "checkpoint_id": name,
        "epoch": epoch,
        "global_step": step,
        "validation_loss": 0.2,
        "model_family": family,
        "classifier_arch": arch,
        "roc_auc": pr_auc,
        "pr_auc": pr_auc,
        "accuracy": f1,
        "precision": f1,
        "recall": f1,
        "f1": f1,
        "false_authentic_rate": far,
        "false_fake_rate": 0.05,
        "threshold": threshold,
        "eligible": True,
    }
    row.update(extra)
    return row


_DINOV2_WEIGHTS = Path(__file__).resolve().parents[1] / "ml_rtx5080" / "experiments" / "v2_dinov2_cls_only" / "best_model.pt"


class CheckpointSelectionTests(unittest.TestCase):
    def _select(self, candidates, **kwargs):
        settings = {
            "max_false_authentic_rate": 0.02,
            "metric_tolerance": 1e-6,
            "expected_family": "dinov2",
            "expected_classifier_arch": "cls_only",
        }
        settings.update(kwargs)
        return selection.select_checkpoints(candidates, **settings)

    def test_defaults_and_trainer_no_longer_selects_max_f1(self) -> None:
        self.assertEqual(selection.DEFAULT_MAX_FALSE_AUTHENTIC_RATE, 0.02)
        self.assertEqual(tv.DEFAULTS["max_false_authentic_rate"], 0.02)
        self.assertEqual(tv.DEFAULTS["validation_threshold"], 0.5)
        self.assertEqual(tv.DEFAULTS["batch_size"], 16)
        self.assertEqual(tv.DEFAULTS["accumulate_grad"], 2)
        source = Path(tv.__file__).read_text()
        self.assertNotIn('if val_metrics["f1"] > best_val_f1', source)
        self.assertIn("update_selection", source)

    def test_unsafe_high_f1_loses_to_a_lower_f1_safe_checkpoint(self) -> None:
        summary = self._select(
            [
                _candidate("high_f1", pr_auc=0.99, f1=0.99, far=0.08),
                _candidate("safe", pr_auc=0.90, f1=0.80, far=0.01, epoch=2),
            ]
        )
        self.assertEqual(summary["selected_checkpoint"], "safe")
        self.assertTrue(summary["safety_validated"])
        by_id = {row["checkpoint_id"]: row for row in summary["candidates"]}
        self.assertFalse(by_id["high_f1"]["far_eligible"])
        self.assertFalse(by_id["high_f1"]["selected"])
        self.assertTrue(by_id["safe"]["selected"])

    def test_synthetic_far_constraint_prefers_higher_pr_auc(self) -> None:
        summary = self._select(
            [
                _candidate("A", pr_auc=0.995, f1=0.990, far=0.080, epoch=1),
                _candidate("B", pr_auc=0.970, f1=0.960, far=0.015, epoch=2),
                _candidate("C", pr_auc=0.980, f1=0.965, far=0.018, epoch=3),
            ]
        )
        self.assertEqual(summary["far_eligible_checkpoints"], 2)
        self.assertEqual(summary["selected_checkpoint"], "C")
        self.assertEqual(summary["selected_metrics"]["checkpoint_id"], "C")
        self.assertTrue(summary["safety_validated"])
        self.assertIsNone(summary["status_message"])
        on_the_limit = self._select([_candidate("limit", pr_auc=0.5, f1=0.5, far=0.02)])
        self.assertTrue(on_the_limit["candidates"][0]["far_eligible"])
        over_the_limit = self._select([_candidate("over", pr_auc=0.5, f1=0.5, far=0.0200001)])
        self.assertFalse(over_the_limit["candidates"][0]["far_eligible"])

    def test_pr_auc_tie_uses_f1_then_lower_far_then_earlier_checkpoint(self) -> None:
        f1_winner = self._select(
            [
                _candidate("low_f1", pr_auc=0.97, f1=0.90, far=0.01, epoch=5),
                _candidate("high_f1", pr_auc=0.97, f1=0.94, far=0.019, epoch=1),
            ]
        )
        self.assertEqual(f1_winner["selected_checkpoint"], "high_f1")

        far_winner = self._select(
            [
                _candidate("higher_far", pr_auc=0.97, f1=0.94, far=0.018, epoch=4),
                _candidate("lower_far", pr_auc=0.97, f1=0.94, far=0.012, epoch=1),
            ]
        )
        self.assertEqual(far_winner["selected_checkpoint"], "lower_far")

        earlier = self._select(
            [
                _candidate("early", pr_auc=0.97, f1=0.94, far=0.012, epoch=1, step=3),
                _candidate("later", pr_auc=0.97, f1=0.94, far=0.012, epoch=2, step=1),
            ]
        )
        self.assertEqual(earlier["selected_checkpoint"], "early")

        lower_step = self._select(
            [
                _candidate("step_low", pr_auc=0.97, f1=0.94, far=0.012, epoch=2, step=4),
                _candidate("step_high", pr_auc=0.97, f1=0.94, far=0.012, epoch=2, step=9),
            ]
        )
        self.assertEqual(lower_step["selected_checkpoint"], "step_low")

        named = self._select(
            [
                _candidate("epoch_a", pr_auc=0.97, f1=0.94, far=0.012, epoch=2, step=9),
                _candidate("epoch_b", pr_auc=0.97, f1=0.94, far=0.012, epoch=2, step=9),
            ]
        )
        self.assertEqual(named["selected_checkpoint"], "epoch_a")

    def test_no_qualified_checkpoint_is_explicit(self) -> None:
        summary = self._select(
            [
                _candidate("only", pr_auc=0.999, f1=0.999, far=0.04),
            ]
        )
        self.assertFalse(summary["safety_validated"])
        self.assertIsNone(summary["selected_checkpoint"])
        self.assertEqual(summary["safety_status"], selection.UNQUALIFIED_STATUS)
        self.assertEqual(summary["status_message"], "NO SAFETY-VALIDATED CHECKPOINT")
        self.assertEqual(summary["far_eligible_checkpoints"], 0)
        self.assertFalse(summary["candidates"][0]["selected"])
        state = selection.new_selection_state(
            max_false_authentic_rate=0.02,
            metric_tolerance=1e-6,
            validation_threshold=0.5,
            model_family="dinov2",
            classifier_arch="cls_only",
        )
        state = selection.update_selection(state, _candidate("only", 0.999, 0.999, 0.04))
        self.assertEqual(state["action"], "hold")
        self.assertEqual(state["epochs_without_improvement"], 0)
        self.assertFalse(selection.should_save_production_checkpoint(state))

    def test_test_metrics_cannot_change_the_winner(self) -> None:
        unsafe = _candidate("unsafe", pr_auc=0.99, f1=0.99, far=0.08)
        safe = _candidate("safe", pr_auc=0.91, f1=0.90, far=0.01, epoch=2)
        unsafe["test_pr_auc"] = 0.10
        unsafe["test_false_authentic_rate"] = 0.0
        safe["test_pr_auc"] = 0.01
        safe["test_false_authentic_rate"] = 0.90
        without_test = self._select(
            [
                {key: value for key, value in unsafe.items() if not key.startswith("test_")},
                {key: value for key, value in safe.items() if not key.startswith("test_")},
            ]
        )
        with_test = self._select([unsafe, safe])
        self.assertEqual(without_test["selected_checkpoint"], "safe")
        self.assertEqual(with_test["selected_checkpoint"], "safe")
        for row in with_test["candidates"]:
            self.assertNotIn("test_pr_auc", row)
            self.assertNotIn("test_false_authentic_rate", row)
        built = selection.candidate_from_validation(
            checkpoint_id="safe",
            epoch=2,
            global_step=1,
            validation_loss=0.2,
            model_family="dinov2",
            classifier_arch="cls_only",
            metrics={
                "pr_auc": 0.91,
                "f1": 0.90,
                "false_authentic_rate": 0.01,
                "threshold": 0.5,
                "test_pr_auc": 0.01,
            },
        )
        self.assertNotIn("test_pr_auc", built)

    def test_threshold_is_recorded_and_not_described_as_calibrated(self) -> None:
        summary = self._select([_candidate("one", 0.95, 0.93, 0.01, threshold=0.5)])
        chosen = summary["selected_metrics"]
        self.assertEqual(chosen["threshold"], 0.5)
        self.assertEqual(chosen["threshold_policy"], "fixed_not_calibrated")
        self.assertEqual(summary["threshold_policy"], "fixed_not_calibrated")
        self.assertNotIn("temperature", summary["selection_rule"])

    def test_selection_is_independent_of_candidate_order(self) -> None:
        base = [
            _candidate("A", pr_auc=0.995, f1=0.990, far=0.080, epoch=1),
            _candidate("B", pr_auc=0.970, f1=0.960, far=0.015, epoch=2),
            _candidate("C", pr_auc=0.980, f1=0.965, far=0.018, epoch=3),
            _candidate("D", pr_auc=0.975, f1=0.970, far=0.010, epoch=4),
        ]
        expected = self._select(base)["selected_checkpoint"]
        rng = random.Random(0)
        for _ in range(12):
            shuffled = list(base)
            rng.shuffle(shuffled)
            summary = self._select(shuffled)
            self.assertEqual(summary["selected_checkpoint"], expected)
            self.assertEqual(
                [row["checkpoint_id"] for row in summary["candidates"]],
                ["A", "B", "C", "D"],
            )

    def test_tie_tolerance_is_a_rounding_band(self) -> None:
        tied = self._select(
            [
                _candidate("narrow_pr", pr_auc=0.98000, f1=0.99, far=0.01, epoch=1),
                _candidate("within_band", pr_auc=0.98004, f1=0.90, far=0.01, epoch=2),
            ],
            metric_tolerance=1e-4,
        )
        self.assertEqual(tied["selected_checkpoint"], "narrow_pr")
        separated = self._select(
            [
                _candidate("lower_pr", pr_auc=0.98000, f1=0.99, far=0.01, epoch=3),
                _candidate("higher_pr", pr_auc=0.98006, f1=0.80, far=0.019, epoch=1),
            ],
            metric_tolerance=1e-4,
        )
        self.assertEqual(separated["selected_checkpoint"], "higher_pr")
        self.assertEqual(selection.metric_band(0.98004, 1e-4), selection.metric_band(0.98000, 1e-4))
        self.assertNotEqual(selection.metric_band(0.98006, 1e-4), selection.metric_band(0.98000, 1e-4))

    def test_ineligible_and_corrupt_checkpoints_are_not_selected(self) -> None:
        excellent = _candidate(
            "wrong_family",
            pr_auc=0.999,
            f1=0.999,
            far=0.0,
            family="dinov3",
        )
        marked = _candidate("corrupt", pr_auc=0.999, f1=0.999, far=0.0, epoch=4)
        marked["eligible"] = False
        marked["rejection_reason"] = "unreadable checkpoint"
        held_out = _candidate("from_test", pr_auc=0.999, f1=0.999, far=0.0, epoch=5, split="test")
        usable = _candidate("usable", pr_auc=0.90, f1=0.88, far=0.01, epoch=2)
        summary = self._select([excellent, marked, held_out, usable])
        self.assertEqual(summary["selected_checkpoint"], "usable")
        rejected = {row["checkpoint_id"]: row for row in summary["candidates"]}
        self.assertFalse(rejected["wrong_family"]["eligible"])
        self.assertFalse(rejected["corrupt"]["far_eligible"])
        self.assertFalse(rejected["from_test"]["eligible"])

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            garbage = root / "broken.pt"
            garbage.write_bytes(b"this is not a checkpoint")
            broken = selection.inspect_checkpoint_file(
                garbage, expected_family="dinov2", expected_classifier_arch="cls_only"
            )
            self.assertFalse(broken["eligible"])
            self.assertFalse(broken["selected"])
            optimizer_only = root / "optimizer.pt"
            torch.save({"optimizer_state": {"step": 1}}, optimizer_only)
            missing_weights = selection.inspect_checkpoint_file(
                optimizer_only, expected_family="dinov2", expected_classifier_arch="cls_only"
            )
            self.assertIn("missing model weights", missing_weights["rejection_reason"])
            other_family = root / "dinov3.pt"
            torch.save(
                {
                    "model_state": {"weight": torch.zeros(1)},
                    "config": {"family": "dinov3", "classifier_arch": "cls_only"},
                    "epoch": 1,
                    "split": "validation",
                },
                other_family,
            )
            mismatch = selection.inspect_checkpoint_file(
                other_family, expected_family="dinov2", expected_classifier_arch="cls_only"
            )
            self.assertFalse(mismatch["eligible"])
            self.assertIn("model family", mismatch["rejection_reason"])

    def test_online_updates_follow_the_safety_ranking(self) -> None:
        state = selection.new_selection_state(
            max_false_authentic_rate=0.02,
            metric_tolerance=1e-6,
            validation_threshold=0.5,
            model_family="dinov2",
            classifier_arch="cls_only",
        )
        for epoch in (1, 2, 3):
            state = selection.update_selection(
                state, _candidate(f"unsafe_{epoch}", pr_auc=0.99, f1=0.99, far=0.08, epoch=epoch)
            )
            self.assertEqual(state["action"], "hold")
            self.assertEqual(state["epochs_without_improvement"], 0)
            self.assertFalse(selection.should_save_production_checkpoint(state))
        state = selection.update_selection(
            state, _candidate("B", pr_auc=0.970, f1=0.960, far=0.015, epoch=4)
        )
        self.assertEqual(state["action"], "reset")
        self.assertTrue(selection.should_save_production_checkpoint(state))
        self.assertEqual(state["selected_checkpoint"], "B")
        state = selection.update_selection(
            state, _candidate("C", pr_auc=0.980, f1=0.965, far=0.018, epoch=5)
        )
        self.assertEqual(state["selected_checkpoint"], "C")
        self.assertEqual(state["action"], "reset")
        self.assertEqual(state["epochs_without_improvement"], 0)
        state = selection.update_selection(
            state, _candidate("A", pr_auc=0.995, f1=0.990, far=0.080, epoch=6)
        )
        self.assertEqual(state["selected_checkpoint"], "C")
        self.assertEqual(state["action"], "increment")
        self.assertEqual(state["epochs_without_improvement"], 1)
        self.assertFalse(selection.should_save_production_checkpoint(state))

    def test_report_roundtrip_stays_inside_the_experiment(self) -> None:
        state = selection.new_selection_state(
            max_false_authentic_rate=0.02,
            metric_tolerance=1e-6,
            validation_threshold=0.5,
            model_family="dinov2",
            classifier_arch="cls_only",
        )
        state = selection.update_selection(
            state, _candidate("A", pr_auc=0.995, f1=0.990, far=0.080, epoch=1)
        )
        state = selection.update_selection(
            state, _candidate("C", pr_auc=0.980, f1=0.965, far=0.018, epoch=2, threshold=0.5)
        )
        root = Path(__file__).resolve().parents[1] / "ml_rtx5080" / "experiments"
        root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=root) as tmp:
            destination = selection.write_selection_report(Path(tmp), state)
            self.assertEqual(destination.name, "selection")
            payload = json.loads((destination / "checkpoint_selection.json").read_text())
            self.assertEqual(payload["selected_checkpoint"], "C")
            self.assertEqual(payload["max_false_authentic_rate"], 0.02)
            self.assertEqual(payload["validation_threshold"], 0.5)
            self.assertTrue(payload["safety_validated"])
            self.assertEqual(payload["checkpoints_evaluated"], 2)
            self.assertEqual(payload["far_eligible_checkpoints"], 1)
            restored = selection.load_selection_state(destination / "checkpoint_selection.json")
            self.assertEqual(restored["selected_checkpoint"], "C")
            with (destination / "checkpoint_metrics.csv").open() as handle:
                table = list(csv.DictReader(handle))
            self.assertEqual([row["checkpoint_id"] for row in table], ["A", "C"])
            self.assertEqual(table[1]["selected"], "True")
            self.assertEqual(table[1]["far_eligible"], "True")
            self.assertEqual(table[0]["far_eligible"], "False")
            self.assertEqual(table[1]["threshold"], "0.5")
        refused = Path(__file__).resolve().parents[1] / "ml_rtx5080" / "checkpoints"
        with self.assertRaises(ValueError):
            selection.write_selection_report(refused, state)

    def test_ceiling_tie_keeps_the_earliest_epoch_and_does_not_reset_patience(self) -> None:
        rows = [
            _candidate("epoch_005", 1.0, 1.0, 0.0, epoch=5, step=100, validation_loss=0.0100),
            _candidate("epoch_010", 1.0, 1.0, 0.0, epoch=10, step=200, validation_loss=0.0100),
            _candidate("epoch_015", 1.0, 1.0, 0.0, epoch=15, step=300, validation_loss=0.0100),
        ]
        summary = self._select(rows)
        self.assertEqual(summary["selected_checkpoint"], "epoch_005")
        self.assertEqual(summary["selected_metrics"]["epoch"], 5)

        state = selection.new_selection_state(
            max_false_authentic_rate=0.02,
            metric_tolerance=1e-6,
            validation_threshold=0.5,
            model_family="dinov2",
            classifier_arch="cls_only",
        )
        for row in rows:
            state = selection.update_selection(state, row)
        self.assertEqual(state["selected_checkpoint"], "epoch_005")
        self.assertEqual(state["epochs_without_improvement"], 2)
        self.assertEqual(state["action"], "increment")
        self.assertFalse(selection.should_save_production_checkpoint(state))

    def test_validation_loss_within_tolerance_does_not_replace_the_earlier_epoch(self) -> None:
        """Loss is a ranking band, not a reason for a later epoch to win by itself.

        With tolerance 1e-4, 0.01000 and 0.00996 fall in the same band
        (round(value / 1e-4) * 1e-4). That later checkpoint stays tied and loses.
        0.00900 is a lower band, so it wins even though its epoch is later.
        """
        tolerance = 1e-4
        self.assertEqual(selection.metric_band(0.01000, tolerance), selection.metric_band(0.00996, tolerance))
        self.assertNotEqual(selection.metric_band(0.01000, tolerance), selection.metric_band(0.00900, tolerance))
        tied_loss = self._select(
            [
                _candidate("epoch_005", 1.0, 1.0, 0.0, epoch=5, step=100, validation_loss=0.01000),
                _candidate("epoch_010", 1.0, 1.0, 0.0, epoch=10, step=200, validation_loss=0.00996),
            ],
            metric_tolerance=tolerance,
        )
        self.assertEqual(tied_loss["selected_checkpoint"], "epoch_005")
        improved_loss = self._select(
            [
                _candidate("epoch_005", 1.0, 1.0, 0.0, epoch=5, step=100, validation_loss=0.01000),
                _candidate("epoch_010", 1.0, 1.0, 0.0, epoch=10, step=200, validation_loss=0.00900),
            ],
            metric_tolerance=tolerance,
        )
        self.assertEqual(improved_loss["selected_checkpoint"], "epoch_010")
        self.assertEqual(improved_loss["selected_metrics"]["validation_loss"], 0.00900)

    def test_far_constraint_still_beats_loss_and_an_earlier_epoch(self) -> None:
        summary = self._select(
            [
                _candidate("early_unsafe", 1.0, 1.0, 0.05, epoch=1, step=1, validation_loss=0.001),
                _candidate("later_safe", 0.90, 0.80, 0.02, epoch=9, step=90, validation_loss=0.2),
            ]
        )
        self.assertEqual(summary["selected_checkpoint"], "later_safe")
        self.assertFalse(
            next(row for row in summary["candidates"] if row["checkpoint_id"] == "early_unsafe")["far_eligible"]
        )


class CheckpointPolicyV2Tests(unittest.TestCase):
    @unittest.skipUnless(_DINOV2_WEIGHTS.is_file(), "local checkpoint, not in git")
    def test_phase17_histories_rerank_deterministically_without_touching_checkpoints(self) -> None:
        root = Path(__file__).resolve().parents[1] / "ml_rtx5080" / "experiments"
        watched = [root / "dataset_audit" / "split_manifest_v2.json"]
        for spec in policy.PHASE17_EXPERIMENTS:
            directory = root / spec["directory"]
            watched.append(directory / "selection" / "checkpoint_selection.json")
            watched.append(directory / "history.json")
            watched.append(directory / "best_model.pt")
            watched.append(directory / "epoch_030.pt")
        before = {path: (path.stat().st_mtime_ns, path.stat().st_size) for path in watched}
        first = policy.build_policy_report(root)
        second = policy.build_policy_report(root)
        self.assertEqual(first, second)
        self.assertIsNone(first["architecture_winner"])
        self.assertFalse(first["training_performed"])
        self.assertFalse(first["test_images_loaded"])
        self.assertEqual(first["temperature"], "not_fitted")
        self.assertIsNone(first["production_threshold"])
        self.assertEqual(len(first["experiments"]), 4)
        for row in first["experiments"]:
            self.assertEqual(row["old_selected_epoch"], 30)
            self.assertFalse(row["best_model_pt_replaced"])
            self.assertIn("new_selected_epoch", row)
        after = {path: (path.stat().st_mtime_ns, path.stat().st_size) for path in watched}
        self.assertEqual(before, after)

    def test_policy_writer_refuses_phase17_and_checkpoint_directories(self) -> None:
        report = {"experiments": []}
        root = Path(__file__).resolve().parents[1] / "ml_rtx5080"
        with self.assertRaises(RuntimeError):
            policy.write_policy_report(report, root / "experiments" / "v2_dinov2_cls_only")
        with self.assertRaises(RuntimeError):
            policy.write_policy_report(report, root / "checkpoints" / "checkpoint_policy_v2")

    def test_calibration_readiness_accepts_the_phase16_manifest_and_rejects_overlap(self) -> None:
        manifest_path = (
            Path(__file__).resolve().parents[1]
            / "ml_rtx5080"
            / "experiments"
            / "dataset_audit"
            / "split_manifest_v2.json"
        )
        manifest = json.loads(manifest_path.read_text())
        ready = policy.assess_calibration_readiness(manifest)
        self.assertTrue(ready["calibration_ready"], ready["failures"])
        self.assertEqual(ready["samples"], 2998)
        self.assertEqual(ready["authentic"], 1498)
        self.assertEqual(ready["fake"], 1500)
        self.assertTrue(ready["all_five_brands_in_both_classes"])
        self.assertEqual(ready["overlaps"], {"train": 0, "validation": 0, "test": 0})
        self.assertEqual(ready["duplicate_groups_cross_splits"], 0)
        self.assertEqual(ready["statistical_sufficiency"], "not_declared")
        self.assertFalse(ready["test_images_loaded"])
        self.assertFalse(ready["test_metrics_computed"])

        overlapped = json.loads(json.dumps(manifest))
        calibration_row = next(row for row in overlapped["samples"] if row["split"] == "calibration")
        validation_row = next(row for row in overlapped["samples"] if row["split"] == "validation")
        validation_row["path"] = calibration_row["path"]
        validation_row["duplicate_group_id"] = calibration_row["duplicate_group_id"]
        rejected = policy.assess_calibration_readiness(overlapped)
        self.assertFalse(rejected["calibration_ready"])
        self.assertGreater(rejected["overlaps"]["validation"], 0)
        self.assertGreater(rejected["duplicate_groups_cross_splits"], 0)

    def test_final_test_images_and_checkpoint_bytes_are_not_opened(self) -> None:
        root = Path(__file__).resolve().parents[1] / "ml_rtx5080" / "experiments"
        source = Path(policy.__file__).read_text()
        self.assertNotIn("run_calibration", source)
        self.assertNotIn("fit_temperature", source)
        self.assertNotIn("Image.open", source)
        real_open = open

        def guarded_open(file, *args, **kwargs):
            name = str(file)
            if name.endswith((".jpg", ".jpeg", ".png", ".pt", ".onnx")):
                raise AssertionError(f"opened {name}")
            return real_open(file, *args, **kwargs)

        with unittest.mock.patch("builtins.open", guarded_open):
            report = policy.build_policy_report(root)
        self.assertFalse(report["test_images_loaded"])
        self.assertEqual(report["calibration"]["membership_hash"], policy.AUTHORITATIVE_MEMBERSHIP_HASH)


_MEMBERSHIP_HASH = "78a38ebac1102977ec5598577126ade77fed8edd6563e13dbdee73dba389f678"
_PREPROCESSING = "resize_pad_square_eval_v1"


def _artifact_weights(arch: str) -> dict:
    weights = {
        "backbone.patch.weight": torch.zeros(1),
        "trunk.norm.weight": torch.zeros(1),
        "trunk.fc1.weight": torch.zeros(1),
        "trunk.fc2.weight": torch.zeros(1),
    }
    if arch == "cls_patch_attention":
        weights["fusion.weight"] = torch.zeros(1)
        weights["attention_pool.query.weight"] = torch.zeros(1)
    return weights


def _write_checkpoint(
    directory: Path,
    epoch: int,
    *,
    family: str = "dinov2",
    arch: str = "cls_only",
    membership_hash: str = _MEMBERSHIP_HASH,
    input_size: int = 504,
    preprocessing: str = _PREPROCESSING,
    output_dir: Path | None = None,
) -> Path:
    path = directory / f"epoch_{epoch:03d}.pt"
    torch.save(
        {
            "epoch": epoch,
            "global_step": epoch,
            "model_state": _artifact_weights(arch),
            "config": {
                "family": family,
                "classifier_arch": arch,
                "img_size": input_size,
                "input_size": input_size,
                "manifest_membership_hash": membership_hash,
                "preprocessing_version": preprocessing,
                "output_dir": str(output_dir or directory),
            },
        },
        path,
    )
    return path


def _expectations(directory: Path, *, family: str = "dinov2", arch: str = "cls_only", input_size: int = 504) -> dict:
    return {
        "expected_input_size": input_size,
        "expected_membership_hash": _MEMBERSHIP_HASH,
        "expected_preprocessing_version": _PREPROCESSING,
        "expected_experiment_dir": directory,
    }


class CheckpointPolicyV3Tests(unittest.TestCase):
    def _select(self, candidates, directory: Path, **kwargs):
        settings = {
            "max_false_authentic_rate": 0.02,
            "metric_tolerance": 1e-6,
            "expected_family": "dinov2",
            "expected_classifier_arch": "cls_only",
            "artifact_expectations": _expectations(directory),
        }
        settings.update(kwargs)
        return selection.select_checkpoints(candidates, **settings)

    def test_metric_record_with_missing_checkpoint_is_excluded(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            missing = _candidate(
                "epoch_019",
                1.0,
                1.0,
                0.0,
                epoch=19,
                step=19,
                validation_loss=0.009,
                checkpoint_path=str(directory / "epoch_019.pt"),
                checkpoint_available=False,
            )
            summary = self._select([missing], directory)
        self.assertIsNone(summary["selected_checkpoint"])
        self.assertEqual(summary["candidates"][0]["rejection_reason"], "no checkpoint artifact")
        self.assertFalse(summary["candidates"][0]["far_eligible"])

    def test_existing_checkpoint_wins_when_better_metric_row_has_no_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            _write_checkpoint(directory, 8)
            missing = _candidate(
                "epoch_019",
                1.0,
                1.0,
                0.0,
                epoch=19,
                step=19,
                validation_loss=0.001,
                checkpoint_path=str(directory / "epoch_019.pt"),
                checkpoint_available=False,
            )
            present = _candidate(
                "epoch_008",
                0.90,
                0.80,
                0.01,
                epoch=8,
                step=8,
                validation_loss=0.2,
                checkpoint_path=str(directory / "epoch_008.pt"),
                checkpoint_available=True,
            )
            summary = self._select([missing, present], directory)
        self.assertEqual(summary["selected_checkpoint"], "epoch_008")
        rejected = next(row for row in summary["candidates"] if row["checkpoint_id"] == "epoch_019")
        self.assertIn("no checkpoint artifact", rejected["rejection_reason"])

    def test_corrupt_checkpoint_is_excluded(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            corrupt = directory / "epoch_004.pt"
            corrupt.write_bytes(b"not a checkpoint")
            _write_checkpoint(directory, 2)
            better = _candidate(
                "epoch_004",
                1.0,
                1.0,
                0.0,
                epoch=4,
                step=4,
                validation_loss=0.001,
                checkpoint_path=str(corrupt),
                checkpoint_available=True,
            )
            worse = _candidate(
                "epoch_002",
                0.5,
                0.5,
                0.01,
                epoch=2,
                step=2,
                checkpoint_path=str(directory / "epoch_002.pt"),
                checkpoint_available=True,
            )
            summary = self._select([better, worse], directory)
        self.assertEqual(summary["selected_checkpoint"], "epoch_002")
        rejected = next(row for row in summary["candidates"] if row["checkpoint_id"] == "epoch_004")
        self.assertIn("unreadable checkpoint", rejected["rejection_reason"])

    def test_wrong_model_checkpoint_is_excluded(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            _write_checkpoint(directory, 2, family="dinov3", input_size=512)
            _write_checkpoint(directory, 4)
            wrong = _candidate(
                "epoch_002", 1.0, 1.0, 0.0, epoch=2, step=2, validation_loss=0.001,
                checkpoint_path=str(directory / "epoch_002.pt"), checkpoint_available=True,
            )
            right = _candidate(
                "epoch_004", 0.5, 0.5, 0.01, epoch=4, step=4,
                checkpoint_path=str(directory / "epoch_004.pt"), checkpoint_available=True,
            )
            summary = self._select([wrong, right], directory)
        self.assertEqual(summary["selected_checkpoint"], "epoch_004")
        rejected = next(row for row in summary["candidates"] if row["checkpoint_id"] == "epoch_002")
        self.assertIn("model family", rejected["rejection_reason"])

    def test_wrong_head_checkpoint_is_excluded(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            _write_checkpoint(directory, 2, arch="cls_patch_attention")
            _write_checkpoint(directory, 4)
            wrong = _candidate(
                "epoch_002", 1.0, 1.0, 0.0, epoch=2, step=2, validation_loss=0.001,
                checkpoint_path=str(directory / "epoch_002.pt"), checkpoint_available=True,
            )
            right = _candidate(
                "epoch_004", 0.5, 0.5, 0.01, epoch=4, step=4,
                checkpoint_path=str(directory / "epoch_004.pt"), checkpoint_available=True,
            )
            summary = self._select([wrong, right], directory)
        self.assertEqual(summary["selected_checkpoint"], "epoch_004")
        rejected = next(row for row in summary["candidates"] if row["checkpoint_id"] == "epoch_002")
        self.assertIn("classifier head", rejected["rejection_reason"])

    def test_manifest_mismatch_is_excluded(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            _write_checkpoint(directory, 2, membership_hash="0" * 64)
            _write_checkpoint(directory, 4)
            wrong = _candidate(
                "epoch_002", 1.0, 1.0, 0.0, epoch=2, step=2, validation_loss=0.001,
                checkpoint_path=str(directory / "epoch_002.pt"), checkpoint_available=True,
            )
            right = _candidate(
                "epoch_004", 0.5, 0.5, 0.01, epoch=4, step=4,
                checkpoint_path=str(directory / "epoch_004.pt"), checkpoint_available=True,
            )
            summary = self._select([wrong, right], directory)
        self.assertEqual(summary["selected_checkpoint"], "epoch_004")
        rejected = next(row for row in summary["candidates"] if row["checkpoint_id"] == "epoch_002")
        self.assertIn("manifest membership hash does not match", rejected["rejection_reason"])

    def test_valid_checkpoint_is_selectable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            path = _write_checkpoint(directory, 6)
            before = path.read_bytes()
            candidate = _candidate(
                "epoch_006", 0.91, 0.88, 0.0, epoch=6, step=12, validation_loss=0.01,
                checkpoint_path=str(path), checkpoint_available=True,
            )
            summary = self._select([candidate], directory)
            self.assertEqual(path.read_bytes(), before)
        chosen = summary["selected_metrics"]
        self.assertEqual(summary["selected_checkpoint"], "epoch_006")
        self.assertTrue(chosen["checkpoint_validation"]["checkpoint_load_valid"])
        self.assertEqual(chosen["checkpoint_validation"]["manifest_membership_hash"], _MEMBERSHIP_HASH)

    def test_ranking_among_actual_artifacts_is_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            _write_checkpoint(directory, 4)
            _write_checkpoint(directory, 10)
            rows = [
                _candidate(
                    "epoch_010", 1.0, 1.0, 0.0, epoch=10, step=20, validation_loss=0.01,
                    checkpoint_path=str(directory / "epoch_010.pt"), checkpoint_available=True,
                ),
                _candidate(
                    "epoch_004", 1.0, 1.0, 0.0, epoch=4, step=8, validation_loss=0.01,
                    checkpoint_path=str(directory / "epoch_004.pt"), checkpoint_available=True,
                ),
            ]
            first = self._select(rows, directory)["selected_checkpoint"]
            random.shuffle(rows)
            second = self._select(rows, directory)["selected_checkpoint"]
        self.assertEqual(first, "epoch_004")
        self.assertEqual(second, "epoch_004")

    @unittest.skipUnless(_DINOV2_WEIGHTS.is_file(), "local checkpoint, not in git")
    def test_selection_does_not_mutate_checkpoint_bytes(self) -> None:
        root = Path(__file__).resolve().parents[1] / "ml_rtx5080" / "experiments"
        watched = [root / "dataset_audit" / "split_manifest_v2.json"]
        for spec in policy.PHASE17_EXPERIMENTS:
            directory = root / spec["directory"]
            watched.append(directory / "selection" / "checkpoint_selection.json")
            watched.append(directory / "history.json")
            watched.append(directory / "best_model.pt")
            watched.append(directory / "epoch_008.pt")
            watched.append(directory / "epoch_030.pt")
        before = {path: (path.stat().st_mtime_ns, path.stat().st_size) for path in watched}
        report, _audit = policy_v3.build_policy_report(root, validate_selected=False)
        after = {path: (path.stat().st_mtime_ns, path.stat().st_size) for path in watched}
        self.assertEqual(before, after)
        self.assertFalse(report["training_performed"])
        self.assertFalse(report["test_images_loaded"])
        self.assertEqual(report["temperature"], "not_fitted")
        self.assertIsNone(report["production_threshold"])
        self.assertFalse(report["phase17_checkpoint_bytes_modified"])
        by_dir = {row["directory"]: row for row in report["experiments"]}
        cls = by_dir["v2_dinov2_cls_only"]
        self.assertEqual(cls["history_selected_epoch"], 8)
        self.assertFalse(cls["history_candidate_excluded"])
        self.assertEqual(cls["artifact_selected_epoch"], 8)
        self.assertTrue(cls["selected_checkpoint_exists"])
        self.assertTrue(str(cls["selected_checkpoint_path"]).endswith("epoch_008.pt"))
        for name, history_epoch in (
            ("v2_dinov2_cls_patch_attention", 19),
            ("v2_dinov3_cls_only", 13),
            ("v2_dinov3_cls_patch_attention", 15),
        ):
            row = by_dir[name]
            self.assertEqual(row["history_selected_epoch"], history_epoch)
            self.assertTrue(row["history_candidate_excluded"])
            self.assertEqual(row["excluded_candidate_reason"], "no checkpoint artifact")
            self.assertEqual(row["artifact_selected_epoch"] % 2, 0)
            self.assertNotEqual(row["artifact_selected_epoch"], history_epoch)
            self.assertTrue(row["selected_checkpoint_exists"])
            self.assertNotIn("best_model.pt", row["selected_checkpoint_path"])

    def test_phase18_selection_rule_remains_intact(self) -> None:
        tied = [
            _candidate("epoch_b", 1.0, 1.0, 0.0, epoch=10, step=20, validation_loss=0.01),
            _candidate("epoch_a", 1.0, 1.0, 0.0, epoch=4, step=8, validation_loss=0.01),
        ]
        metric_only = selection.select_checkpoints(
            tied,
            max_false_authentic_rate=0.02,
            metric_tolerance=1e-6,
            expected_family="dinov2",
            expected_classifier_arch="cls_only",
        )
        self.assertEqual(metric_only["selected_checkpoint"], "epoch_a")
        self.assertEqual(metric_only["selection_version"], "phase18-far-v2")
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            _write_checkpoint(directory, 4)
            _write_checkpoint(directory, 10)
            later_better = [
                _candidate(
                    "epoch_004", 0.80, 0.80, 0.0, epoch=4, step=8, validation_loss=0.01,
                    checkpoint_path=str(directory / "epoch_004.pt"), checkpoint_available=True,
                ),
                _candidate(
                    "epoch_010", 0.95, 0.90, 0.0, epoch=10, step=20, validation_loss=0.02,
                    checkpoint_path=str(directory / "epoch_010.pt"), checkpoint_available=True,
                ),
            ]
            ranked = self._select(later_better, directory)
        self.assertEqual(ranked["selected_checkpoint"], "epoch_010")

    def test_save_every_greater_than_one_cannot_select_an_unsaved_epoch(self) -> None:
        self.assertFalse(selection.epoch_artifact_is_saved(19, 2))
        self.assertTrue(selection.epoch_artifact_is_saved(8, 2))
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            _write_checkpoint(directory, 2)
            unsaved = _candidate(
                "epoch_001", 1.0, 1.0, 0.0, epoch=1, step=1, validation_loss=0.001,
                checkpoint_path=str(selection.epoch_artifact_path(directory, 1)),
                checkpoint_available=selection.epoch_artifact_is_saved(1, 2),
            )
            saved = _candidate(
                "epoch_002", 0.70, 0.70, 0.01, epoch=2, step=2, validation_loss=0.2,
                checkpoint_path=str(selection.epoch_artifact_path(directory, 2)),
                checkpoint_available=selection.epoch_artifact_is_saved(2, 2),
            )
            summary = self._select([unsaved, saved], directory)
        self.assertEqual(summary["selected_checkpoint"], "epoch_002")
        source = Path(tv.__file__).read_text()
        self.assertLess(
            source.index("save_training_checkpoint(epoch_path"),
            source.index("selection_state = update_selection"),
        )

    def test_policy_writer_refuses_phase17_and_checkpoint_directories(self) -> None:
        report = {"experiments": []}
        audit = {"experiments": []}
        root = Path(__file__).resolve().parents[1] / "ml_rtx5080"
        with self.assertRaises(RuntimeError):
            policy_v3.write_policy_report(report, audit, root / "experiments" / "v2_dinov2_cls_only")
        with self.assertRaises(RuntimeError):
            policy_v3.write_policy_report(report, audit, root / "checkpoints" / "checkpoint_policy_v3")


if __name__ == "__main__":
    unittest.main()
