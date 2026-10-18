"""Phase 35 OOD rules. Image opens are guarded, and the final test is not a fixture."""

from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "ml_rtx5080"))


def _row(**overrides) -> dict:
    base = {
        "image_path": "/tmp/ood.jpg",
        "ood_category": "food",
        "taxonomy": "A_unrelated",
        "source": "wikimedia_commons",
        "source_url": "https://commons.wikimedia.org/wiki/File:Example.jpg",
        "source_type": "wikimedia_commons_category_member",
        "brand": "UNKNOWN",
        "is_watch": False,
        "is_trained_brand": False,
        "license_or_usage_note": "CC BY 4.0",
        "sha256": "a" * 64,
        "width": 64,
        "height": 64,
        "aspect_ratio": 1.0,
        "duplicate_group": "a" * 64,
        "split": "development",
        "collection_method": "wikimedia_commons_category_api",
    }
    base.update(overrides)
    return base


def _pass_facts() -> dict:
    groups = {name: 40 for name in (
        "A_unrelated",
        "B_product_non_watch",
        "C_watch_outside_scope",
        "D_unseen_luxury_brand",
        "E_structurally_different",
    )}
    escape = {name: 0.0 for name in groups}
    return {
        "real_corpus": True,
        "scored_count": 200,
        "group_counts": groups,
        "holdout_counts": {name: 10 for name in groups},
        "provenance_failures": 0,
        "no_gate_authentic_rate": escape,
        "gates": [
            {"name": "moderate", "calibration_review_increase": 0.02, "authentic_escape": escape},
        ],
        "holdout_authentic_escape": {"moderate": escape},
        "scored_license_unknown_count": 0,
        "independent_photo_count_known": True,
        "unseen_luxury_mahalanobis_auroc": 0.95,
    }


class OODValidationTests(unittest.TestCase):
    def test_manifest_row_requires_provenance_and_rejects_trained_brand(self) -> None:
        from ood_validation import OODArtifactError, validate_row

        validate_row(_row())
        with self.assertRaises(OODArtifactError):
            validate_row(_row(source="UNKNOWN"))
        with self.assertRaises(OODArtifactError):
            validate_row(_row(sha256="abc"))
        with self.assertRaises(OODArtifactError):
            validate_row(_row(is_trained_brand=True))

    def test_split_is_deterministic_and_has_a_holdout_bucket(self) -> None:
        from ood_validation import assign_split

        digest = "12345678" + "0" * 56
        self.assertEqual(assign_split(digest), assign_split(digest))
        splits = {assign_split(f"{index:08x}" + "0" * 56) for index in range(40)}
        self.assertIn("holdout", splits)
        self.assertIn("development", splits)

    def test_duplicate_screen_rejects_catalog_sha_and_does_not_open_blocked_paths(self) -> None:
        from ood_validation import screen_against_catalog

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image_path = root / "ood.png"
            Image.new("RGB", (32, 24), (10, 20, 30)).save(image_path)
            blocked = root / "blocked.png"
            Image.new("RGB", (8, 8), (1, 2, 3)).save(blocked)
            digest = hashlib.sha256(image_path.read_bytes()).hexdigest()
            opened = {"blocked": False}
            original = Image.open

            def guard(path, *args, **kwargs):
                if Path(path).resolve() == blocked.resolve():
                    opened["blocked"] = True
                    raise AssertionError("blocked image was opened")
                return original(path, *args, **kwargs)

            Image.open = guard
            try:
                leaked = screen_against_catalog(
                    [_row(image_path=str(image_path), sha256=digest)],
                    catalog_sha={digest},
                    catalog_hash=np.array([1], dtype=np.uint64),
                    blocked={str(blocked.resolve())},
                )
                self.assertEqual(leaked["admitted"], [])
                self.assertEqual(leaked["rejected"][0]["rejection"], "exact_sha256_match_to_catalog")
                blocked_row = screen_against_catalog(
                    [_row(image_path=str(blocked), sha256="b" * 64)],
                    catalog_sha=set(),
                    catalog_hash=np.array([1], dtype=np.uint64),
                    blocked={str(blocked.resolve())},
                )
            finally:
                Image.open = original
        self.assertFalse(opened["blocked"])
        self.assertEqual(blocked_row["rejected"][0]["rejection"], "final_test_path")

    def test_popcount_and_ood_scores_are_deterministic(self) -> None:
        import torch
        import ood
        from ood_validation import _popcount, fpr_at_tpr

        values = np.array([0, 1, 7, 255], dtype=np.uint64)
        self.assertEqual(_popcount(values).tolist(), [0, 1, 3, 8])
        generator = torch.Generator().manual_seed(4)
        features = torch.randn(12, 6, generator=generator)
        reference = ood.fit_reference(features, torch.tensor([0, 1] * 6))
        first = ood.score_features(features, reference, "mahalanobis")
        second = ood.score_features(features, reference, "mahalanobis")
        self.assertTrue(torch.equal(first, second))
        rates = fpr_at_tpr(np.linspace(0, 1, 40), np.linspace(2, 3, 40))
        self.assertLess(rates["fpr_at_95_tpr"], 0.05)

    def test_threshold_evaluation_and_fail_closed_artifact(self) -> None:
        from ood_validation import OODArtifactError, decide_case, threshold_from_percentile

        self.assertEqual(decide_case(0.1, ("LOW_RESOLUTION",), 10.0, None), "REVIEW")
        self.assertEqual(decide_case(0.2, (), 10.0, 3.0), "REVIEW")
        self.assertEqual(decide_case(0.8, (), 0.1, 3.0), "FAKE")
        self.assertEqual(decide_case(0.2, (), 0.1, 3.0), "AUTHENTIC")
        scores = np.linspace(0, 1, 100)
        self.assertAlmostEqual(threshold_from_percentile(scores, 0.95), 0.95, places=2)
        with self.assertRaises(OODArtifactError):
            threshold_from_percentile(np.ones(5), 0.95)

    def test_status_rules(self) -> None:
        from ood_validation import OOD_BLOCKED, OOD_FAIL, OOD_PARTIAL, OOD_PASS, classify_ood_status

        self.assertEqual(classify_ood_status({"real_corpus": False, "scored_count": 0})["status"], OOD_BLOCKED)
        ready = _pass_facts()
        self.assertEqual(classify_ood_status(ready)["status"], OOD_PASS)
        pending = _pass_facts()
        pending["group_counts"]["E_structurally_different"] = 10
        pending["scored_count"] = 170
        self.assertEqual(classify_ood_status(pending)["status"], OOD_PARTIAL)
        unsafe = _pass_facts()
        unsafe["no_gate_authentic_rate"] = {name: 0.4 for name in unsafe["group_counts"]}
        unsafe["gates"] = [
            {
                "name": "moderate",
                "calibration_review_increase": 0.25,
                "authentic_escape": {name: 0.2 for name in unsafe["group_counts"]},
            }
        ]
        self.assertEqual(classify_ood_status(unsafe)["status"], OOD_FAIL)

    def test_recorded_phase_does_not_promote_or_touch_final_test(self) -> None:
        status_path = REPO / "ml_rtx5080" / "experiments" / "ood_validation_v1" / "ood_status.json"
        if not status_path.is_file():
            self.skipTest("phase 35 artifacts are written by the evaluation run")
        status = json.loads(status_path.read_text())
        manifest_path = status_path.parent / "ood_manifest.json"
        digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
        self.assertEqual((status_path.parent / "ood_manifest.sha256").read_text().strip(), digest)
        self.assertIsNone(status["production_ood_threshold"])
        self.assertFalse(status["promoted"])
        self.assertEqual(status["final_test_samples_used"], 0)
        self.assertEqual(status["frozen_checkpoint_sha256"], "5a38c93fd442b03653c65d2a5ecc9c2687ef152f7c5c020763e4ce1fd9c7d28f")
        self.assertEqual(status["frozen_temperature"], 0.24038200410185356)
        self.assertEqual(status["policy_version"], "shadow_v1")
        summary = REPO / "ml_rtx5080" / "experiments" / "final_test_v1" / "final_test_summary.json"
        self.assertTrue(summary.is_file())


if __name__ == "__main__":
    unittest.main()
