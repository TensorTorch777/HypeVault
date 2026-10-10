"""Unit tests for the frozen DINOv2 versus DINOv3 comparison rules."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PKG = REPO / "ml_rtx5080"
CORE = PKG / "experiments" / "dinov2_vs_dinov3_full_scale"
sys.path.insert(0, str(PKG))
sys.path.insert(0, str(CORE))

import compare_core as core


def _row(sample_id, model_id, label, decision, logit=0.0, error=None, brand="Brand"):
    return {
        "sample_id": sample_id,
        "model_id": model_id,
        "label": label,
        "brand": brand,
        "decision": decision,
        "raw_logit": logit,
        "error": error,
    }


class SplitMembershipTests(unittest.TestCase):
    def test_common_validation_excludes_train_and_calibration(self) -> None:
        blocked = {"/data/test.jpg"}
        self.assertEqual(
            core.assign_role(
                parent_folder="/watches/lange",
                resolved_path="/watches/lange/a.jpg",
                dinov2_train_folders={"/watches/rm"},
                dinov2_val_folders={"/watches/lange"},
                dinov3_split="validation",
                blocked=blocked,
            ),
            "primary_validation",
        )
        self.assertEqual(
            core.assign_role(
                parent_folder="/watches/rm",
                resolved_path="/watches/rm/a.jpg",
                dinov2_train_folders={"/watches/rm"},
                dinov2_val_folders={"/watches/lange"},
                dinov3_split="validation",
                blocked=blocked,
            ),
            "diagnostic",
        )
        self.assertEqual(
            core.assign_role(
                parent_folder="/watches/lange",
                resolved_path="/watches/lange/cal.jpg",
                dinov2_train_folders={"/watches/rm"},
                dinov2_val_folders={"/watches/lange"},
                dinov3_split="calibration",
                blocked=blocked,
            ),
            "diagnostic",
        )
        self.assertEqual(
            core.assign_role(
                parent_folder="/watches/lange",
                resolved_path="/data/test.jpg",
                dinov2_train_folders=set(),
                dinov2_val_folders={"/watches/lange"},
                dinov3_split="validation",
                blocked=blocked,
            ),
            "excluded",
        )
        self.assertEqual(
            core.assign_role(
                parent_folder="/watches/lange",
                resolved_path="/watches/lange/heldout.jpg",
                dinov2_train_folders=set(),
                dinov2_val_folders={"/watches/lange"},
                dinov3_split="test",
                blocked=set(),
            ),
            "excluded",
        )

    def test_selection_is_reproducible_and_rejects_duplicates(self) -> None:
        records = [
            {"sample_id": "b", "resolved_path": "/b", "label": 0, "brand": "B", "role": "diagnostic"},
            {"sample_id": "a", "resolved_path": "/a", "label": 1, "brand": "A", "role": "primary_validation"},
        ]
        first = core.select_catalog(records, set())
        second = core.select_catalog(list(reversed(records)), set())
        self.assertEqual([row["sample_id"] for row in first["diagnostic"]], ["b"])
        self.assertEqual(first, second)
        records.append(dict(records[0]))
        with self.assertRaises(core.ComparisonGuardError):
            core.select_catalog(records, set())

    def test_blocked_path_cannot_enter_the_sweep(self) -> None:
        records = [
            {
                "sample_id": "t",
                "resolved_path": "/locked/t.jpg",
                "label": 1,
                "brand": "X",
                "role": "diagnostic",
            }
        ]
        with self.assertRaises(core.ComparisonGuardError):
            core.select_catalog(records, {"/locked/t.jpg"})


class IdentityTests(unittest.TestCase):
    def test_protocol_pins_checkpoint_and_preprocess_identity(self) -> None:
        protocol = json.loads((CORE / "protocol_v1.json").read_text())
        dinov2 = protocol["models"]["dinov2_legacy"]
        dinov3 = protocol["models"]["dinov3_experimental"]
        self.assertEqual(dinov2["checkpoint_sha256"], core.DINOV2_SHA256)
        self.assertEqual(dinov3["checkpoint_sha256"], core.DINOV3_SHA256)
        self.assertEqual(dinov2["preprocessing_id"], core.DINOV2_PREPROCESS)
        self.assertEqual(dinov3["preprocessing_id"], core.DINOV3_PREPROCESS)
        self.assertEqual(dinov2["input_hw"], [504, 504])
        self.assertEqual(dinov3["input_hw"], [512, 512])
        self.assertEqual(dinov2["min_authentic_confidence"], core.DINOV2_MIN_AUTHENTIC)
        self.assertEqual(dinov3["temperature"], core.DINOV3_TEMPERATURE)
        self.assertEqual(dinov3["decision_threshold"], core.DINOV3_THRESHOLD)
        self.assertFalse(protocol["winner_may_be_declared"])
        self.assertEqual(protocol["primary_validation"]["data_readiness"], "NO_GO_FOR_AUTHENTICITY_CLAIMS")

    def test_hash_mismatch_fails_closed(self) -> None:
        path = CORE / "protocol_v1.json"
        with self.assertRaises(core.ComparisonGuardError):
            core.require_sha256(path, "0" * 64, "protocol")


class DecisionAndMetricTests(unittest.TestCase):
    def test_label_direction_and_denominators(self) -> None:
        rows = [
            _row("a", core.DINOV2_ID, 1, "AUTHENTIC", logit=0.2),
            _row("b", core.DINOV2_ID, 1, "FAKE", logit=2),
            _row("c", core.DINOV2_ID, 0, "FAKE", logit=-0.2),
            _row("d", core.DINOV2_ID, 0, "AUTHENTIC", logit=-2),
        ]
        metrics = core.score_rows(rows)
        self.assertEqual(metrics["false_authentic_count"], 1)
        self.assertEqual(metrics["false_authentic_rate"], 0.5)
        self.assertEqual(metrics["false_fake_count"], 1)
        self.assertEqual(metrics["false_fake_rate"], 0.5)
        self.assertEqual(metrics["confusion"]["fake_labeled"]["AUTHENTIC"], 1)
        self.assertEqual(metrics["accuracy"], 0.5)
        self.assertIsNotNone(metrics["roc_auc_raw_logit"])
        self.assertGreater(metrics["roc_auc_raw_logit"], 0.5)

    def test_review_is_not_authentic_or_fake(self) -> None:
        rows = [
            _row("a", core.DINOV3_ID, 1, "REVIEW", logit=3),
            _row("b", core.DINOV3_ID, 1, "FAKE", logit=3),
            _row("c", core.DINOV3_ID, 0, "AUTHENTIC", logit=-3),
        ]
        metrics = core.score_rows(rows)
        self.assertEqual(metrics["review_count"], 1)
        self.assertEqual(metrics["binary_denominator"], 2)
        self.assertEqual(metrics["false_authentic_count"], 0)
        self.assertEqual(metrics["false_authentic_rate"], 0.0)
        self.assertEqual(metrics["accuracy"], 1.0)
        self.assertEqual(metrics["selective_accuracy"], 1.0)
        self.assertAlmostEqual(metrics["decision_coverage"], 2 / 3)
        with self.assertRaises(core.ComparisonGuardError):
            core.score_rows([_row("z", core.DINOV2_ID, 0, "REVIEW")])

    def test_one_class_ranking_is_undefined(self) -> None:
        rows = [_row("a", core.DINOV3_ID, 0, "AUTHENTIC", logit=-1)]
        metrics = core.score_rows(rows)
        self.assertIsNone(metrics["roc_auc_raw_logit"])
        self.assertIsNone(metrics["pr_auc_raw_logit"])
        self.assertIsNone(metrics["false_authentic_rate"])
        self.assertEqual(metrics["false_fake_rate"], 0.0)

    def test_failed_missing_and_duplicate_rows(self) -> None:
        failed = _row("a", core.DINOV2_ID, 1, "FAKE", error="corrupt")
        failed["decision"] = None
        ok = _row("b", core.DINOV2_ID, 1, "FAKE", logit=1)
        metrics = core.score_rows([failed, ok])
        self.assertEqual(metrics["failed"], 1)
        self.assertEqual(metrics["successful"], 1)
        self.assertEqual(metrics["fake_labeled"], 1)
        ids = ["a", "b"]
        rows = [
            _row("a", core.DINOV2_ID, 0, "AUTHENTIC"),
            _row("a", core.DINOV3_ID, 0, "AUTHENTIC"),
            _row("b", core.DINOV2_ID, 1, "FAKE"),
            _row("b", core.DINOV3_ID, 1, "FAKE"),
        ]
        core.assert_complete_pairs(rows, ids)
        with self.assertRaises(core.ComparisonGuardError):
            core.assert_complete_pairs(rows[:-1], ids)
        with self.assertRaises(core.ComparisonGuardError):
            core.assert_complete_pairs(rows + [_row("a", core.DINOV2_ID, 0, "FAKE")], ids)

    def test_native_decisions_match_the_frozen_policy(self) -> None:
        self.assertEqual(core.dinov2_decision(0.0), ("FAKE", 0.5))
        authentic = core.dinov2_decision(-3.0)
        self.assertEqual(authentic[0], "AUTHENTIC")
        borderline = core.sigmoid(-1.0)
        self.assertLess(1.0 - borderline, 0.88)
        self.assertEqual(core.dinov2_decision(-1.0)[0], "FAKE")
        self.assertEqual(core.dinov3_decision(0.0, ())[0], "FAKE")
        self.assertEqual(core.dinov3_decision(-1.0, ())[0], "AUTHENTIC")
        self.assertEqual(core.dinov3_decision(-1.0, ("LOW_RESOLUTION",))[0], "REVIEW")
        self.assertNotEqual(core.sigmoid(1.0), core.dinov3_fake_score(1.0))

    def test_primary_cohort_guard(self) -> None:
        rows = [
            {"brand": core.PRIMARY_BRAND, "label": 0, "sample_id": f"{index:03d}"}
            for index in range(core.PRIMARY_COUNT)
        ]
        core.assert_primary_cohort(rows)
        rows[0]["label"] = 1
        with self.assertRaises(core.ComparisonGuardError):
            core.assert_primary_cohort(rows)

    def test_agreement_counts_disagreement(self) -> None:
        left = [
            _row("a", core.DINOV2_ID, 1, "FAKE"),
            _row("b", core.DINOV2_ID, 1, "AUTHENTIC"),
        ]
        right = [
            _row("a", core.DINOV3_ID, 1, "FAKE"),
            _row("b", core.DINOV3_ID, 1, "REVIEW"),
        ]
        report = core.paired_agreement(left, right)
        self.assertEqual(report["agreement"], 1)
        self.assertEqual(report["disagreement"], 1)
        self.assertEqual(report["both_binary_and_both_wrong"], 0)


class LockedPathTests(unittest.TestCase):
    def test_opener_is_not_called_for_a_locked_path(self) -> None:
        calls = []

        def opener(path):
            calls.append(path)
            return "opened"

        with self.assertRaises(core.ComparisonGuardError):
            core.open_if_allowed("/tmp/locked.jpg", {str(Path("/tmp/locked.jpg").resolve())}, opener)
        self.assertEqual(calls, [])
        self.assertEqual(core.open_if_allowed("/tmp/allowed.jpg", set(), opener), "opened")
        self.assertEqual(calls, ["/tmp/allowed.jpg"])


if __name__ == "__main__":
    unittest.main()
