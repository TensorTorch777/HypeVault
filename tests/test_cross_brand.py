"""Phase 45 leave-one-brand-out protocol. No images and no checkpoints are opened."""

from __future__ import annotations

import hashlib
import sys
import unittest
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "ml_rtx5080"))

BRANDS = (
    "A. Lange & Söhne",
    "Audemars Piguet",
    "Patek Philippe",
    "Richard Mille",
    "Vacheron Constantin",
)


def _catalog() -> list[dict]:
    rows = []
    for brand_index, brand in enumerate(BRANDS):
        for label in (0, 1):
            for copy in range(6):
                group = f"{brand}-{label}-{copy // 2}"
                rows.append(
                    {
                        "path": f"/data/{brand}/{label}-{copy}.jpg",
                        "label": label,
                        "brand": brand,
                        "duplicate_group_id": group,
                        "sha256": f"hash-{brand}-{label}-{copy}",
                        "split": "train",
                        "sample_id": f"{brand_index}-{label}-{copy}",
                    }
                )
    locked = {
        "path": "/data/locked/final-test.jpg",
        "label": 1,
        "brand": "Patek Philippe",
        "duplicate_group_id": "locked-group",
        "sha256": "locked-hash",
        "split": "test",
        "sample_id": "locked",
    }
    copy = {
        "path": "/data/Patek Philippe/copy-of-final.jpg",
        "label": 1,
        "brand": "Patek Philippe",
        "duplicate_group_id": "locked-group",
        "sha256": "copy-hash",
        "split": "train",
        "sample_id": "copy",
    }
    shared = {
        "path": "/data/Audemars Piguet/near-lange.jpg",
        "label": 0,
        "brand": "Audemars Piguet",
        "duplicate_group_id": "A. Lange & Söhne-0-0",
        "sha256": "shared-hash",
        "split": "train",
        "sample_id": "shared",
    }
    return rows + [locked, copy, shared]


class CrossBrandProtocolTests(unittest.TestCase):
    def test_held_out_brand_is_absent_from_train_calibration_and_validation(self) -> None:
        from cross_brand import FOLDS, build_fold

        for _directory, brand in FOLDS:
            fold = build_fold(_catalog(), brand, seed=45)
            for role in ("train", "calibration", "validation"):
                brands = {row["brand"] for row in fold["rows"] if row["role"] == role}
                self.assertNotIn(brand, brands)
            test_brands = {row["brand"] for row in fold["rows"] if row["role"] == "cross_brand_test"}
            self.assertEqual(test_brands, {brand})

    def test_duplicate_groups_and_final_test_do_not_leak(self) -> None:
        from cross_brand import build_fold, locked_test_paths

        samples = _catalog()
        fold = build_fold(samples, "A. Lange & Söhne", seed=45)
        test_paths = locked_test_paths(samples)
        self.assertTrue(test_paths.isdisjoint({row["path"] for row in fold["rows"]}))
        self.assertNotIn("/data/Patek Philippe/copy-of-final.jpg", {row["path"] for row in fold["rows"]})
        quarantined = {row["path"] for row in fold["rows"] if row["role"] == "leakage_quarantine"}
        self.assertIn("/data/Audemars Piguet/near-lange.jpg", quarantined)
        self.assertNotIn("/data/Audemars Piguet/near-lange.jpg", {row["path"] for row in fold["rows"] if row["role"] == "train"})

    def test_fold_generation_is_deterministic(self) -> None:
        from cross_brand import build_fold

        first = build_fold(_catalog(), "Richard Mille", seed=45)
        second = build_fold(_catalog(), "Richard Mille", seed=45)
        self.assertEqual(first["rows"], second["rows"])

    def test_calibration_and_checkpoint_selection_reject_the_held_out_brand(self) -> None:
        from cross_brand import assert_calibration_isolation, assert_selection_isolation, build_fold

        fold = build_fold(_catalog(), "Vacheron Constantin", seed=45)
        calibration = assert_calibration_isolation(fold["rows"], "Vacheron Constantin")
        validation = assert_selection_isolation(fold["rows"], "Vacheron Constantin")
        self.assertTrue(calibration)
        self.assertTrue(validation)
        contaminated = fold["rows"] + [
            {
                "path": "/data/Vacheron Constantin/0-0.jpg",
                "label": 0,
                "brand": "Vacheron Constantin",
                "role": "validation",
                "duplicate_group_id": "x",
                "sha256": "y",
            }
        ]
        with self.assertRaises(RuntimeError):
            assert_selection_isolation(contaminated, "Vacheron Constantin")

    def test_gap_and_temperature_use_the_declared_inputs(self) -> None:
        from calibration import fit_temperature, scaled_logits
        from cross_brand import classify_generalization, generalization_gap
        from evaluation import evaluate_logits

        in_domain = {"roc_auc": 0.99, "pr_auc": 0.98, "f1": 0.97, "false_authentic_rate": 0.01, "false_fake_rate": 0.02, "nll": 0.1, "brier": 0.02, "ece": 0.01}
        unseen = {"roc_auc": 0.70, "pr_auc": 0.62, "f1": 0.60, "false_authentic_rate": 0.20, "false_fake_rate": 0.18, "nll": 0.5, "brier": 0.2, "ece": 0.1}
        gap = generalization_gap(in_domain, unseen)
        self.assertAlmostEqual(gap["pr_auc"], 0.36)
        self.assertAlmostEqual(gap["false_authentic_rate"], -0.19)
        rows = [
            {"unseen": unseen, "gap": gap}
            for _ in range(5)
        ]
        self.assertEqual(classify_generalization(rows), "WEAK")
        calibration_logits = torch.tensor([2.0, -2.0, 1.5, -1.5])
        calibration_labels = torch.tensor([1, 0, 1, 0])
        held_logits = torch.tensor([4.0, -4.0])
        temperature = float(fit_temperature(calibration_logits, calibration_labels)["temperature"])
        frozen = scaled_logits(held_logits, temperature)
        self.assertTrue(torch.allclose(frozen, held_logits / temperature))
        scored = evaluate_logits(frozen, torch.tensor([1, 0]), threshold=0.5)
        self.assertEqual(scored["false_authentic"], 0)
        self.assertGreater(scored["pr_auc"], 0.9)

    def test_recorded_manifest_matches_the_isolation_rules(self) -> None:
        manifest_path = REPO / "ml_rtx5080" / "experiments" / "v3_cross_brand" / "fold_manifest.json"
        source = REPO / "ml_rtx5080" / "experiments" / "dataset_audit" / "split_manifest_v2.json"
        if not manifest_path.is_file():
            self.skipTest("fold manifest is written by the protocol builder")
        before = hashlib.sha256(source.read_bytes()).hexdigest()
        import json
        from cross_brand import assert_final_test_excluded, assert_fold_isolation, locked_test_paths

        payload = json.loads(manifest_path.read_text())
        source_payload = json.loads(source.read_text())
        test_paths = locked_test_paths(source_payload["samples"])
        self.assertEqual(len(payload["folds"]), 5)
        for fold in payload["folds"].values():
            assert_fold_isolation(fold)
            assert_final_test_excluded(fold, test_paths)
        assert_final_test_excluded(payload["random_split_control"], test_paths)
        self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(), before)


if __name__ == "__main__":
    unittest.main()
