"""Frozen-candidate selection and one-time final-test scoring."""

from __future__ import annotations

import inspect
import json
import sys
import tempfile
import unittest
import unittest.mock
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ml_rtx5080"))

import final_holdout as holdout


def _candidate(
    name: str,
    nll: float,
    *,
    ece: float = 0.0,
    brier: float = 0.1,
    head: str = "cls_only",
    family: str = "dinov2",
    far: float = 0.0,
    peak: int = 100,
    throughput: float = 10.0,
    test_roc_auc: float = 0.0,
) -> dict:
    return {
        "name": name,
        "family": family,
        "head": head,
        "architecture_name": f"{family}:{head}",
        "checkpoint": f"/tmp/{name}.pt",
        "temperature": 0.25,
        "validation_false_authentic_rate": far,
        "validation_pr_auc": 1.0,
        "validation_f1": 1.0,
        "validation_roc_auc": 1.0,
        "calibrated_nll": nll,
        "calibrated_ece": ece,
        "calibrated_brier": brier,
        "peak_allocated_bytes": peak,
        "throughput_images_per_sec": throughput,
        "test_roc_auc": test_roc_auc,
    }


def _pair(left: np.ndarray, right: np.ndarray, name_a: str, name_b: str) -> dict:
    comparison = holdout.paired_bootstrap_nll_difference(left, right, resamples=200, seed=42)
    comparison["model_a"] = name_a
    comparison["model_b"] = name_b
    return comparison


class FinalHoldoutSelectionTests(unittest.TestCase):
    def test_paired_bootstrap_is_deterministic_and_reports_the_nll_difference(self) -> None:
        left = np.array([1.0, 3.0, 5.0, 7.0])
        right = np.array([2.0, 4.0, 6.0, 8.0])
        first = holdout.paired_bootstrap_nll_difference(left, right, resamples=300, seed=42)
        second = holdout.paired_bootstrap_nll_difference(left, right, resamples=300, seed=42)
        self.assertEqual(first, second)
        self.assertAlmostEqual(first["observed_delta_nll"], -1.0)
        self.assertAlmostEqual(first["mean_nll_a"], 4.0)
        self.assertAlmostEqual(first["mean_nll_b"], 5.0)
        self.assertLessEqual(first["ci_low"], first["ci_high"])

    def test_constant_difference_interval_excludes_zero(self) -> None:
        comparison = holdout.paired_bootstrap_nll_difference(
            np.zeros(16),
            np.ones(16),
            resamples=100,
            seed=7,
        )
        self.assertEqual(comparison["observed_delta_nll"], -1.0)
        self.assertEqual(comparison["ci_low"], -1.0)
        self.assertEqual(comparison["ci_high"], -1.0)
        self.assertTrue(comparison["excludes_zero"])
        self.assertTrue(holdout.materially_better(comparison))

    def test_material_rule_rejects_a_difference_whose_interval_includes_zero(self) -> None:
        shared = np.array([0.2, 0.4, 0.1, 0.5, 0.3, 0.6])
        comparison = _pair(shared, shared, "alpha", "beta")
        self.assertEqual(comparison["observed_delta_nll"], 0.0)
        self.assertFalse(comparison["excludes_zero"])
        self.assertFalse(holdout.materially_better(comparison))
        worse_brier = _candidate("alpha", 0.35, brier=0.2)
        better_brier = _candidate("beta", 0.35, brier=0.01, head="cls_patch_attention")
        decision = holdout.select_frozen_candidate([worse_brier, better_brier], [comparison])
        self.assertEqual(decision["selected_name"], "beta")
        self.assertTrue(decision["tie_break_applied"])
        self.assertIsNone(decision["production_threshold"])
        self.assertFalse(decision["test_used_for_selection"])

    def test_material_nll_improvement_beats_a_lower_brier_and_ignores_test_scores(self) -> None:
        comparison = _pair(np.zeros(24), np.ones(24), "good", "bad")
        good = _candidate("good", 0.0, brier=0.9, test_roc_auc=0.1)
        bad = _candidate("bad", 1.0, brier=0.01, test_roc_auc=0.99)
        decision = holdout.select_frozen_candidate([good, bad], [comparison])
        self.assertEqual(decision["selected_name"], "good")
        self.assertEqual(decision["calibration_survivors"], ["good"])
        self.assertFalse(decision["tie_break_applied"])
        self.assertTrue(decision["validation_metrics_tied"])

    def test_cls_only_wins_a_remaining_calibration_tie(self) -> None:
        shared = np.linspace(0.1, 0.4, 8)
        comparison = _pair(shared, shared, "patch", "plain")
        decision = holdout.select_frozen_candidate(
            [
                _candidate("patch", 0.2, head="cls_patch_attention"),
                _candidate("plain", 0.2, head="cls_only"),
            ],
            [comparison],
        )
        self.assertEqual(decision["selected_name"], "plain")

    def test_unsafe_validation_candidate_cannot_win(self) -> None:
        decision = holdout.select_frozen_candidate(
            [
                _candidate("unsafe", 0.0, far=0.5),
                _candidate("safe", 0.2, far=0.0),
            ],
            [],
        )
        self.assertEqual(decision["selected_name"], "safe")

    def test_freeze_artifact_states_the_rule_and_leaves_the_threshold_null(self) -> None:
        decision = {
            "selection_rule": holdout.SELECTION_RULE,
            "validation_metrics_tied": True,
            "selected_model": "plain",
            "selected_checkpoint": "/tmp/epoch_008.pt",
            "selected_temperature": 0.25,
            "calibration_survivors": ["plain"],
            "tie_break_applied": False,
            "production_threshold": None,
            "final_test_opened_at_selection": False,
            "test_used_for_selection": False,
            "candidates": [_candidate("plain", 0.1)],
            "comparisons": [
                {
                    "model_a": "plain",
                    "model_b": "other",
                    "mean_nll_a": 0.1,
                    "mean_nll_b": 0.2,
                    "observed_delta_nll": -0.1,
                    "ci_low": -0.2,
                    "ci_high": -0.05,
                    "excludes_zero": True,
                    "materially_better_a": True,
                    "materially_better_b": False,
                    "resamples": 200,
                    "seed": 42,
                }
            ],
        }
        with tempfile.TemporaryDirectory() as tmp:
            destination = holdout.emit_freeze_files(Path(tmp) / "final_model_freeze", decision)
            text = (destination / "model_selection_freeze.md").read_text()
            payload = json.loads((destination / "model_selection_freeze.json").read_text())
        self.assertIn("Validation metrics were tied.", text)
        self.assertIn("Calibration was used only as the predeclared secondary criterion.", text)
        self.assertIn("Paired calibration bootstrap was used.", text)
        self.assertIn("The final test had not yet been opened at selection time.", text)
        self.assertIn("Production threshold: null", text)
        self.assertIsNone(payload["production_threshold"])
        self.assertFalse(payload["test_used_for_selection"])
        self.assertEqual(payload["selected_temperature"], 0.25)
        freeze_source = "\n".join(
            inspect.getsource(function)
            for function in (
                holdout.write_model_freeze,
                holdout.build_freeze_decision,
                holdout.emit_freeze_files,
            )
        )
        self.assertNotIn("Image.open", freeze_source)
        self.assertNotIn("fit_temperature", inspect.getsource(holdout.score_holdout))

    def test_final_test_scoring_keeps_the_frozen_temperature_and_selection(self) -> None:
        records = [
            {"sample_id": "authentic", "brand": "Richard Mille", "label": 0},
            {"sample_id": "fake-clear", "brand": "Richard Mille", "label": 1},
            {"sample_id": "fake-low", "brand": "Richard Mille", "label": 1},
        ]
        logits = torch.tensor([-3.0, 3.0, -2.0])
        baseline = holdout.score_holdout(records, logits, 1.0)
        sharpened = holdout.score_holdout(records, logits, 0.5)
        self.assertNotEqual(
            baseline["predictions"][0]["calibrated_probability"],
            sharpened["predictions"][0]["calibrated_probability"],
        )
        self.assertEqual(sharpened["temperature"], 0.5)
        self.assertFalse(sharpened["temperature_fitted_on_test"])
        self.assertFalse(sharpened["threshold_tuned_on_test"])
        self.assertIsNone(sharpened["production_threshold"])
        self.assertEqual(sharpened["severe_false_authentic"]["actual_fake_and_calibrated_p_fake_below_0_50"], 1)
        self.assertEqual(sharpened["error_audit"][0]["error_type"], "false_authentic")
        self.assertEqual(sharpened["error_audit"][0]["sample_id"], "fake-low")
        freeze = {
            "selected_model": "plain",
            "selected_checkpoint": "/tmp/epoch_008.pt",
            "selected_temperature": 0.5,
            "production_threshold": None,
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            freeze_dir = root / "final_model_freeze"
            holdout.emit_freeze_files(
                freeze_dir,
                {
                    "selection_rule": holdout.SELECTION_RULE,
                    "validation_metrics_tied": True,
                    "selected_model": "plain",
                    "selected_checkpoint": freeze["selected_checkpoint"],
                    "selected_temperature": 0.5,
                    "calibration_survivors": ["plain"],
                    "tie_break_applied": False,
                    "candidates": [_candidate("plain", 0.1)],
                    "comparisons": [],
                },
            )
            freeze_bytes = (freeze_dir / "model_selection_freeze.json").read_bytes()
            destination = holdout.write_final_test_artifacts(
                root / "final_test_v1",
                score=sharpened,
                freeze=freeze,
                freeze_sha256="abc",
                test_membership_sha256="def",
                checkpoint_sha256="ghi",
            )
            self.assertEqual((freeze_dir / "model_selection_freeze.json").read_bytes(), freeze_bytes)
            header = (destination / "final_test_predictions.csv").read_text().splitlines()[0]
            self.assertEqual(header, ",".join(holdout.PREDICTION_FIELDS))
            summary = json.loads((destination / "final_test_summary.json").read_text())
            self.assertTrue(summary["model_selection_completed_before_test"])
            self.assertFalse(summary["test_used_for_selection"])
            self.assertFalse(summary["threshold_tuned_on_test"])
            self.assertFalse(summary["temperature_fitted_on_test"])
            self.assertFalse(summary["checkpoint_changed_after_test"])
            self.assertIsNone(summary["production_threshold"])
            with self.assertRaises(RuntimeError):
                holdout.write_final_test_artifacts(
                    destination,
                    score=sharpened,
                    freeze=freeze,
                    freeze_sha256="abc",
                    test_membership_sha256="def",
                    checkpoint_sha256="ghi",
                )

    def test_real_test_membership_is_locked_and_does_not_open_images(self) -> None:
        manifest_path = (
            Path(__file__).resolve().parents[1]
            / "ml_rtx5080"
            / "experiments"
            / "dataset_audit"
            / "split_manifest_v2.json"
        )
        manifest = json.loads(manifest_path.read_text())

        def reject_open(*args, **kwargs):
            raise AssertionError("final-test validation opened an image")

        with unittest.mock.patch("PIL.Image.open", reject_open):
            records = holdout.validate_final_test_membership(manifest)
        self.assertEqual(len(records), 3006)
        self.assertEqual(sum(row["label"] == 0 for row in records), 1506)
        self.assertEqual(sum(row["label"] == 1 for row in records), 1500)
        brands = {row["brand"] for row in records}
        self.assertEqual(brands, set(holdout.CATALOG_BRANDS))
        for brand in holdout.CATALOG_BRANDS:
            self.assertTrue(any(row["brand"] == brand and row["label"] == 0 for row in records))
            self.assertTrue(any(row["brand"] == brand and row["label"] == 1 for row in records))
        test_paths = {row["path"] for row in records}
        for split in ("train", "calibration", "validation"):
            other = {str(Path(path).resolve()) for path in manifest["membership"][split]}
            self.assertFalse(test_paths & other)
        self.assertEqual(manifest["membership_hash"], holdout.AUTHORITATIVE_MEMBERSHIP_HASH)
        overlapped = {
            "membership_hash": manifest["membership_hash"],
            "samples": manifest["samples"],
            "membership": {name: list(paths) for name, paths in manifest["membership"].items()},
        }
        overlapped["membership"]["test"][0] = overlapped["membership"]["train"][0]
        with self.assertRaises(ValueError):
            holdout.validate_final_test_membership(overlapped)


if __name__ == "__main__":
    unittest.main()
