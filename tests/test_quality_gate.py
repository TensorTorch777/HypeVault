"""Research quality-gate accounting stays off the final test and the checkpoint."""

from __future__ import annotations

import inspect
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ml_rtx5080"))

import quality_gate as gate


def _record(sample_id: str, label: int, flags: tuple[str, ...], min_dimension: float = 800.0, split: str = "train") -> dict:
    return {
        "sample_id": sample_id,
        "source_split": split,
        "brand": "Richard Mille",
        "label": label,
        "quality_flags": flags,
        "min_dimension": min_dimension,
        "authenticity_prediction": "fake" if label == 1 else "authentic",
        "aspect_ratio": 1.0,
        "mean_luminance": 0.5,
        "luminance_std": 0.25,
        "foreground_occupancy": 0.7,
        "bytes_per_pixel": 0.3,
        "file_bytes": 1000,
        "sharpness_laplacian_variance": 0.02,
        "calibrated_probability": 0.99 if label == 1 else 0.01,
    }


def _failure(
    sample_id: str,
    condition: str,
    flags_after: tuple[str, ...],
    *,
    flags_before: tuple[str, ...] = (),
    after_dimension: float = 200.0,
    before_dimension: float = 800.0,
) -> dict:
    return {
        "event_id": f"{sample_id}|{condition}",
        "sample_id": sample_id,
        "source_split": "validation",
        "condition": condition,
        "false_authentic": False,
        "false_fake": True,
        "original_calibrated_probability": 0.02,
        "transformed_calibrated_probability": 0.97,
        "flags_before": flags_before,
        "flags_after": flags_after,
        "features_before": {"min_dimension": before_dimension},
        "features_after": {"min_dimension": after_dimension},
        "before_min_dimension": before_dimension,
        "after_min_dimension": after_dimension,
        "before_mean_luminance": 0.58,
        "after_mean_luminance": 0.49,
        "delta_foreground_occupancy": 0.05,
    }


class QualityGateTests(unittest.TestCase):
    def test_quality_level_assignment_is_deterministic(self) -> None:
        plain = {"min_dimension": 800.0}
        low = {"min_dimension": 400.0}
        self.assertEqual(gate.assign_quality_level((), plain), 0)
        self.assertEqual(gate.assign_quality_level(("BLURRY",), plain), 1)
        self.assertEqual(gate.assign_quality_level(("LOW_RESOLUTION",), low), 2)
        self.assertEqual(gate.assign_quality_level(("BLURRY", "TOO_BRIGHT", "EXCESSIVE_CLIPPING"), plain), 2)
        self.assertEqual(gate.assign_quality_level(("LOW_CONTRAST", "TOO_DARK"), plain), 2)
        self.assertEqual(gate.assign_quality_level(("BLURRY",), plain), gate.assign_quality_level(("BLURRY",), plain))

    def test_individual_flag_rule(self) -> None:
        self.assertTrue(gate.rule_matches("low_resolution", ("LOW_RESOLUTION",), {"min_dimension": 400.0}))
        self.assertFalse(gate.rule_matches("low_resolution", ("BLURRY",), {"min_dimension": 800.0}))
        self.assertTrue(gate.rule_matches("any_flag", ("BLURRY",), {"min_dimension": 800.0}))
        self.assertFalse(gate.rule_matches("any_flag", (), {"min_dimension": 800.0}))

    def test_combination_rules(self) -> None:
        features = {"min_dimension": 800.0}
        self.assertTrue(gate.rule_matches("low_resolution_or_low_contrast", ("LOW_CONTRAST",), features))
        self.assertTrue(gate.rule_matches("low_resolution_or_suspicious_framing", ("SUSPICIOUS_FRAMING",), features))
        self.assertFalse(gate.rule_matches("low_resolution_or_too_dark", ("LOW_CONTRAST",), features))
        self.assertTrue(gate.rule_matches("two_or_more_flags", ("BLURRY", "TOO_BRIGHT"), features))
        self.assertFalse(gate.rule_matches("three_or_more_flags", ("BLURRY", "TOO_BRIGHT"), features))
        self.assertTrue(
            gate.rule_matches(
                "critical_resolution_or_dark_contrast_or_three_flags",
                ("LOW_CONTRAST", "TOO_DARK"),
                features,
            )
        )

    def test_review_rate(self) -> None:
        records = [
            _record("a", 0, ()),
            _record("b", 1, ("LOW_RESOLUTION",), 400.0),
            _record("c", 0, ()),
            _record("d", 1, ()),
        ]
        result = gate.evaluate_rule("low_resolution", records, [])
        self.assertEqual(result["affected_images"], 1)
        self.assertEqual(result["review_rate"], 0.25)
        self.assertEqual(result["clean_pass_rate"], 0.75)
        self.assertIsNone(gate.review_rate(1, 0))

    def test_rule_ranking_is_exact(self) -> None:
        wide = _rule("wide", rate=0.28, caught=9)
        mid = _rule("mid", rate=0.09, caught=8)
        narrow = _rule("narrow", rate=0.02, caught=7)
        ranked = gate.rank_rules([narrow, wide, mid])
        self.assertEqual([row["rule_id"] for row in ranked], ["wide", "mid", "narrow"])
        self.assertEqual(ranked[0]["research_rank"], 1)
        budget = gate.match_review_budget(ranked, 0.04)
        self.assertEqual(budget["rule_id"], "narrow")
        self.assertFalse(budget["production_optimal"])
        self.assertIsNone(budget["production_quality_gate"])
        recommended = gate.recommend_research_candidate(ranked)
        self.assertEqual(recommended["rule_id"], "mid")
        self.assertEqual(recommended["label"], "research_quality_gate_candidate")

    def test_known_failure_coverage(self) -> None:
        records = [_record("kept", 0, ()), _record("small", 1, ("LOW_RESOLUTION",), 300.0)]
        failures = [
            _failure("crop", "mild_crop", ("LOW_RESOLUTION",), after_dimension=495.0),
            _failure("dark", "darker", ("LOW_CONTRAST", "EXTREME_ASPECT_RATIO"), after_dimension=1126.0, before_dimension=1126.0),
        ]
        low_resolution = gate.evaluate_rule("low_resolution", records, failures)
        contrast = gate.evaluate_rule("low_resolution_or_low_contrast", records, failures)
        self.assertEqual(low_resolution["stress_failures_caught"], 1)
        self.assertEqual(low_resolution["failure_linked_catches"], 1)
        self.assertEqual(low_resolution["missed_ids"], ["dark|darker"])
        self.assertEqual(contrast["stress_failures_caught"], 2)
        self.assertEqual(contrast["failure_linked_catches"], 2)
        self.assertEqual(contrast["stress_failures_missed"], 0)
        preexisting = _failure(
            "wide",
            "resize_recompression",
            ("EXTREME_ASPECT_RATIO",),
            flags_before=("EXTREME_ASPECT_RATIO",),
            after_dimension=957.0,
            before_dimension=1126.0,
        )
        aspect = gate.evaluate_rule("low_resolution_or_extreme_aspect", records, [preexisting])
        self.assertEqual(aspect["stress_failures_caught"], 1)
        self.assertEqual(aspect["failure_linked_catches"], 0)

    def test_false_authentic_is_not_confused_with_false_fake(self) -> None:
        rows = [
            _record("auth", 0, ()),
            {
                **_record("hidden-fake", 1, ()),
                "authenticity_prediction": "authentic",
            },
        ]
        metrics = gate.subset_classification(rows)
        self.assertEqual(metrics["false_authentic_count"], 1)
        self.assertEqual(metrics["false_fake_count"], 0)
        self.assertEqual(metrics["false_authentic_rate"], 1.0)
        self.assertEqual(metrics["false_fake_rate"], 0.0)
        authentic_only = [_record("only", 0, ())]
        only_metrics = gate.subset_classification(authentic_only)
        self.assertIsNone(only_metrics["false_authentic_rate"])
        self.assertEqual(only_metrics["false_fake_rate"], 0.0)

    def test_final_test_samples_are_rejected(self) -> None:
        leaked = _record("hidden", 1, (), split="test")
        with self.assertRaises(RuntimeError):
            gate.screen_records([leaked], set())
        with self.assertRaises(RuntimeError):
            gate.screen_records([_record("hidden", 1, ())], {"hidden"})
        with self.assertRaises(RuntimeError):
            gate.screen_failures([_failure("hidden", "mild_crop", ("LOW_RESOLUTION",)) | {"source_split": "test"}])

    def test_hash_fields_keep_membership_and_file_sha_distinct(self) -> None:
        self.assertEqual(gate.MEMBERSHIP_HASH, "78a38ebac1102977ec5598577126ade77fed8edd6563e13dbdee73dba389f678")
        source = inspect.getsource(gate.run_phase24)
        self.assertIn('"membership_hash"', source)
        self.assertIn('"manifest_file_sha256"', source)
        self.assertNotIn('"manifest_sha256"', source)
        self.assertNotIn('"manifest_hash"', source)
        result = gate.analyze([_record("a", 0, ())], [])
        self.assertNotIn("manifest_sha256", result)
        self.assertNotIn("manifest_hash", result)
        self.assertEqual(result["final_test_samples_used"], 0)
        self.assertIsNone(result["production_quality_gate"])

    def test_repeated_analysis_matches(self) -> None:
        records = [
            _record("a", 0, ()),
            _record("b", 1, ("LOW_RESOLUTION",), 300.0),
            _record("c", 0, ("BLURRY", "TOO_BRIGHT", "EXCESSIVE_CLIPPING")),
        ]
        failures = [_failure("crop", "mild_crop", ("LOW_RESOLUTION",))]
        first = json.dumps(gate.analyze(records, failures), sort_keys=True)
        second = json.dumps(gate.analyze(records, failures), sort_keys=True)
        self.assertEqual(first, second)

    def test_checkpoint_bytes_stay_unchanged(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "epoch_018.pt"
            checkpoint.write_bytes(b"frozen-quality-gate")
            digest = gate.file_sha256(checkpoint)
            gate.analyze([_record("a", 0, ())], [])
            self.assertEqual(checkpoint.read_bytes(), b"frozen-quality-gate")
            self.assertEqual(gate.file_sha256(checkpoint), digest)
        source = inspect.getsource(gate)
        self.assertNotIn("torch.save", source)
        self.assertNotIn("fit_temperature", source)
        self.assertNotIn("backward", source)

    def test_temperature_is_unchanged(self) -> None:
        self.assertEqual(gate.FROZEN_TEMPERATURE, 0.24038200410185356)
        self.assertEqual(gate.EXPECTED_CHECKPOINT_SHA256, "5a38c93fd442b03653c65d2a5ecc9c2687ef152f7c5c020763e4ce1fd9c7d28f")
        self.assertNotIn("fit_temperature", inspect.getsource(gate.run_phase24))


def _rule(rule_id: str, *, rate: float, caught: int) -> dict:
    return {
        "rule_id": rule_id,
        "definition": rule_id,
        "review_rate": rate,
        "stress_failures_caught": caught,
        "stress_failures_missed": 9 - caught,
        "false_authentic_count": 0,
        "false_fake_count": 0,
        "hidden_false_authentic_count": 0,
        "missed_ids": [],
        "label": "research_quality_gate_candidate",
        "production_quality_gate": None,
    }


if __name__ == "__main__":
    unittest.main()
