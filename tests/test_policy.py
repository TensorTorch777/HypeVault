"""Shadow policy precedence, OOD availability, and final-test isolation."""

from __future__ import annotations

import inspect
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ml_rtx5080"))

import policy
import policy_eval


class PolicyTests(unittest.TestCase):
    def test_low_probability_without_quality_flag_is_authentic(self) -> None:
        result = policy.evaluate_shadow_policy(probability=0.029, quality_flags=())
        self.assertEqual(result["decision"], "AUTHENTIC")
        self.assertEqual(result["reason_codes"], ["AUTHENTICITY_BELOW_THRESHOLD", "QUALITY_OK"])
        self.assertFalse(result["quality_review"])

    def test_high_probability_without_quality_flag_is_fake(self) -> None:
        result = policy.evaluate_shadow_policy(probability=0.694, quality_flags=("EXTREME_ASPECT_RATIO",))
        self.assertEqual(result["decision"], "FAKE")
        self.assertEqual(result["reason_codes"], ["AUTHENTICITY_THRESHOLD"])
        self.assertFalse(result["quality_review"])

    def test_quality_flag_overrides_authenticity(self) -> None:
        low = policy.evaluate_shadow_policy(probability=0.01, quality_flags=("LOW_RESOLUTION",))
        high = policy.evaluate_shadow_policy(probability=0.99, quality_flags=("LOW_CONTRAST",))
        self.assertEqual(low["decision"], "REVIEW")
        self.assertEqual(high["decision"], "REVIEW")
        self.assertEqual(low["reason_codes"], ["QUALITY_LOW_RESOLUTION"])
        self.assertEqual(high["reason_codes"], ["QUALITY_LOW_CONTRAST"])

    def test_multiple_quality_flags_review(self) -> None:
        result = policy.evaluate_shadow_policy(
            probability=0.2,
            quality_flags=("LOW_RESOLUTION", "LOW_CONTRAST", "BLURRY"),
        )
        self.assertEqual(result["decision"], "REVIEW")
        self.assertEqual(
            result["reason_codes"],
            ["QUALITY_MULTIPLE_FLAGS", "QUALITY_LOW_RESOLUTION", "QUALITY_LOW_CONTRAST"],
        )

    def test_ood_unavailable_stays_null(self) -> None:
        result = policy.evaluate_shadow_policy(probability=0.1, quality_flags=())
        self.assertEqual(result["ood_status"], "unavailable")
        self.assertIsNone(result["ood_score"])
        self.assertIsNone(result["ood_flag"])
        self.assertFalse(result["ood_checked"])
        self.assertFalse(result["in_distribution_assumed"])

    def test_ood_unavailable_is_not_false(self) -> None:
        with self.assertRaises(ValueError):
            policy.evaluate_shadow_policy(probability=0.1, quality_flags=(), ood_flag=False)
        with self.assertRaises(ValueError):
            policy.evaluate_shadow_policy(probability=0.1, quality_flags=(), ood_score=0.0)

    def test_false_authentic_is_fake_routed_to_authentic(self) -> None:
        rows = [
            {"label": 1, "decision": "AUTHENTIC"},
            {"label": 1, "decision": "REVIEW"},
            {"label": 1, "decision": "FAKE"},
        ]
        summary = policy.summarize_policy(rows)
        self.assertEqual(summary["false_authentic_count"], 1)
        self.assertAlmostEqual(summary["false_authentic_rate"], 1 / 3)
        self.assertEqual(summary["confusion"]["actual_fake"]["REVIEW"], 1)

    def test_false_fake_is_authentic_routed_to_fake(self) -> None:
        rows = [
            {"label": 0, "decision": "FAKE"},
            {"label": 0, "decision": "REVIEW"},
            {"label": 0, "decision": "AUTHENTIC"},
        ]
        summary = policy.summarize_policy(rows)
        self.assertEqual(summary["false_fake_count"], 1)
        self.assertEqual(summary["authentic_to_review_count"], 1)
        self.assertAlmostEqual(summary["false_fake_rate"], 1 / 3)

    def test_review_is_excluded_from_binary_accuracy(self) -> None:
        rows = [
            {"label": 1, "decision": "FAKE"},
            {"label": 0, "decision": "AUTHENTIC"},
            {"label": 1, "decision": "REVIEW"},
        ]
        summary = policy.summarize_policy(rows)
        self.assertEqual(summary["binary_decision_count"], 2)
        self.assertAlmostEqual(summary["binary_accuracy_review_excluded"], 1.0)
        self.assertTrue(summary["confusion"]["review_excluded_from_binary_metrics"])

    def test_reason_codes_are_deterministic(self) -> None:
        first = policy.reason_codes("AUTHENTIC", ())
        second = policy.reason_codes("AUTHENTIC", ())
        self.assertEqual(first, second)
        self.assertEqual(first, ("AUTHENTICITY_BELOW_THRESHOLD", "QUALITY_OK"))
        self.assertEqual(policy.reason_codes("FAKE", ()), ("AUTHENTICITY_THRESHOLD",))

    def test_final_test_samples_are_rejected(self) -> None:
        with self.assertRaises(RuntimeError):
            policy.screen_splits([{"sample_id": "hidden", "split": "test", "label": 1}], set())
        with self.assertRaises(RuntimeError):
            policy.screen_splits([{"sample_id": "hidden", "split": "train", "label": 1}], {"hidden"})
        with self.assertRaises(RuntimeError):
            policy_eval.run_phase26(
                Path("/tmp"),
                Path("/tmp/missing.json"),
                input_splits=("test",),
                enforce_expected_sha=False,
            )

    def test_checkpoint_bytes_stay_unchanged(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = root / "epoch_018.pt"
            checkpoint.write_bytes(b"frozen-shadow-policy")
            _write_fixture(root)
            policy_eval.run_phase26(
                root,
                root / "manifest.json",
                checkpoint=checkpoint,
                output_dir=root / "policy_shadow_v1",
                quality_features=root / "features.csv",
                stress_linkage=root / "stress.csv",
                phase22_summary=root / "phase22.json",
                calibration_predictions=root / "empty.csv",
                validation_predictions=root / "empty.csv",
                validation_checkpoint=root / "missing.pt",
                low_margin=root / "empty.csv",
                high_confidence=root / "probs.csv",
                quality_thresholds=root / "missing-thresholds.json",
                enforce_expected_sha=False,
            )
            self.assertEqual(checkpoint.read_bytes(), b"frozen-shadow-policy")
        source = inspect.getsource(policy) + inspect.getsource(policy_eval)
        self.assertNotIn("torch.save", source)
        self.assertNotIn("fit_temperature", source)
        self.assertNotIn("backward", source)

    def test_temperature_is_unchanged(self) -> None:
        self.assertEqual(policy.FROZEN_TEMPERATURE, 0.24038200410185356)
        probability = policy.calibrated_probability_from_logit(0.0)
        self.assertAlmostEqual(probability, 0.5)
        self.assertNotIn("fit_temperature", inspect.getsource(policy_eval.run_phase26))

    def test_policy_output_is_deterministic(self) -> None:
        sample = {
            "sample_id": "a.jpg",
            "source_split": "train",
            "brand": "Rolex",
            "label": 0,
            "quality_flags": ("LOW_CONTRAST",),
            "probability_source": "cached",
        }
        first = policy.build_policy_trace(sample, probability=0.2)
        second = policy.build_policy_trace(sample, probability=0.2)
        self.assertEqual(first, second)
        self.assertEqual(first["decision"], "REVIEW")
        self.assertEqual(first["ood_status"], "unavailable")
        self.assertEqual(first["production_policy_status"], "not_promoted")

    def test_quality_rule_thresholds_match_phase23(self) -> None:
        self.assertEqual(policy.LOW_RESOLUTION_CUTOFF, 512.0)
        self.assertEqual(policy.LOW_CONTRAST_CUTOFF, 0.1933616647502347)
        self.assertEqual(policy.QUALITY_RULE, "low_resolution_or_low_contrast")
        path = Path(__file__).resolve().parents[1] / "ml_rtx5080" / "experiments" / "image_quality_v1" / "quality_thresholds.json"
        cutoffs = json.loads(path.read_text())["cutoffs"]
        self.assertEqual(cutoffs["min_dimension_p5"], policy.LOW_RESOLUTION_CUTOFF)
        self.assertEqual(cutoffs["luminance_std_p5"], policy.LOW_CONTRAST_CUTOFF)
        self.assertEqual(policy.THRESHOLD_STATUS, "research_policy_candidate")
        self.assertNotEqual(policy.THRESHOLD_STATUS, "production_threshold")


def _write_fixture(root: Path) -> None:
    manifest = {
        "membership_hash": policy_eval.MEMBERSHIP_HASH,
        "membership": {
            "train": [str(root / "catalog-train.jpg")],
            "calibration": [str(root / "catalog-calibration.jpg")],
            "validation": [str(root / "catalog-validation.jpg")],
            "test": [str(root / "catalog-final.jpg")],
        },
        "samples": [
            {"sample_id": "kept.jpg", "path": str(root / "catalog-train.jpg"), "sha256": "abc"},
            {"sample_id": "final.jpg", "path": str(root / "catalog-final.jpg"), "sha256": "def"},
        ],
    }
    (root / "manifest.json").write_text(json.dumps(manifest))
    (root / "features.csv").write_text(
        "sample_id,source_split,brand,label,quality_flags,min_dimension,luminance_std\n"
        "kept.jpg,train,Audemars Piguet,0,LOW_RESOLUTION,400,0.25\n"
    )
    (root / "stress.csv").write_text(
        "sample_id,source_split,brand,label,condition,prediction_flip,false_authentic,false_fake,"
        "original_calibrated_probability,transformed_calibrated_probability,probability_change,"
        "flags_before,flags_after,before_min_dimension,after_min_dimension,before_sharpness,after_sharpness,"
        "before_mean_luminance,after_mean_luminance,delta_min_dimension,delta_sharpness,delta_mean_luminance,"
        "delta_luminance_std,delta_pixel_count,delta_foreground_occupancy\n"
        "kept.jpg,train,Audemars Piguet,0,lower_resolution,1,false,true,0.01,0.99,0.98,"
        ",LOW_RESOLUTION,800,200,0.1,0.1,0.5,0.5,-600,0,0,0,0,0\n"
    )
    (root / "phase22.json").write_text(json.dumps({"test_samples_used": 1}))
    (root / "probs.csv").write_text("sample_id,source_split,brand,label,calibrated_probability\nkept.jpg,train,Audemars Piguet,0,0.01\n")
    (root / "empty.csv").write_text("sample_id,logit\n")


if __name__ == "__main__":
    unittest.main()
