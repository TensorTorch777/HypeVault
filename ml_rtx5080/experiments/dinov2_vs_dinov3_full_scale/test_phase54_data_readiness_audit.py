"""Synthetic fixtures for the Phase 54 metadata audit."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import phase54_data_readiness_audit as audit


def _sample(label_dir: str, brand: str, name: str, split: str, **extra):
    return {
        "path": f"/data/{label_dir}/{brand}/{name}.jpg",
        "split": split,
        "brand": brand,
        "sha256": extra.get("sha256", f"hash-{name}"),
        "duplicate_group_id": extra.get("duplicate_group_id", f"group-{name}"),
        "excluded_ood": extra.get("excluded_ood", False),
        "manual_review": False,
    }


def _folders():
    return {
        "seed": 42,
        "val_split": 0.1,
        "train_product_folders": [
            "/data/Label_0_Watches/Patek Philippe",
            "/data/Label_1_Watches/Richard Mille",
        ],
        "val_product_folders": ["/data/Label_0_Watches/A. Lange & Söhne"],
        "calibration_product_folders": [],
    }


class MembershipTests(unittest.TestCase):
    def test_primary_is_validation_intersection_without_train_or_calibration(self) -> None:
        samples = [
            _sample("Label_0_Watches", "A. Lange & Söhne", "held", "validation"),
            _sample("Label_0_Watches", "A. Lange & Söhne", "fit", "train"),
            _sample("Label_0_Watches", "A. Lange & Söhne", "cal", "calibration"),
            _sample("Label_1_Watches", "Richard Mille", "other", "validation"),
            _sample("Label_0_Watches", "A. Lange & Söhne", "locked", "test"),
        ]
        result = audit.build_audit(samples, _folders())
        self.assertEqual([row["canonical_id"].split("/")[-1] for row in result["primary"]], ["held.jpg"])
        self.assertEqual(result["intersection"]["train"]["validation"], 1)
        self.assertEqual(result["intersection"]["validation"]["calibration"], 1)

    def test_duplicate_split_membership_fails_closed(self) -> None:
        samples = [
            _sample("Label_0_Watches", "A. Lange & Söhne", "held", "validation"),
            _sample("Label_0_Watches", "A. Lange & Söhne", "held", "train"),
        ]
        with self.assertRaises(audit.AuditStop):
            audit.build_audit(samples, _folders())

    def test_missing_manifest_fails_closed(self) -> None:
        with self.assertRaises(audit.AuditStop):
            audit.require_file(Path("/tmp/phase54-missing-manifest.json"), None, "manifest")

    def test_brand_and_class_confounding_is_explicit(self) -> None:
        samples = [_sample("Label_0_Watches", "A. Lange & Söhne", "held", "validation")]
        result = audit.build_audit(samples, _folders())
        self.assertTrue(result["primary_summary"]["single_brand"])
        self.assertTrue(result["primary_summary"]["single_historical_label"])
        self.assertEqual(result["primary_summary"]["historical_labels"], {"authentic_labeled": 1})

    def test_unknown_provenance_cannot_emit_go(self) -> None:
        samples = [
            _sample("Label_0_Watches", "A. Lange & Söhne", "real", "validation"),
            _sample("Label_1_Watches", "A. Lange & Söhne", "fake", "validation"),
        ]
        folders = _folders()
        folders["val_product_folders"].append("/data/Label_1_Watches/A. Lange & Söhne")
        result = audit.build_audit(samples, folders)
        fields = audit.provenance_status(result["primary"][0])
        self.assertEqual(fields["product_id"], "UNKNOWN")
        self.assertEqual(fields["independent_verification_status"], "NOT_RECORDED")
        self.assertEqual(fields["verified_label"], "NOT_RECORDED")
        self.assertEqual(result["decision"]["data_readiness"], audit.DATA_READINESS_NO_GO)
        self.assertIn("no_independently_verified_labels", result["decision"]["reasons"])

    def test_go_requires_verified_labels_and_known_products(self) -> None:
        primary = [
            {"historical_label": "authentic_labeled"},
            {"historical_label": "fake_labeled"},
        ]
        refused = audit.decide_readiness(
            primary=primary,
            independently_verified=0,
            product_ids_known=True,
            source_disjoint=True,
            product_count=300,
            required_products=300,
            membership_established=True,
        )
        self.assertEqual(refused["data_readiness"], audit.DATA_READINESS_NO_GO)
        accepted = audit.decide_readiness(
            primary=primary,
            independently_verified=2,
            product_ids_known=True,
            source_disjoint=True,
            product_count=300,
            required_products=300,
            membership_established=True,
        )
        self.assertEqual(accepted["data_readiness"], "GO")

    def test_locked_and_ood_paths_are_not_opened(self) -> None:
        calls = []

        def opener(path):
            calls.append(path)
            return "opened"

        blocked = {"Label_0_Watches/A. Lange & Söhne/locked.jpg"}
        with self.assertRaises(audit.AuditStop):
            audit.assert_not_opened("/data/Label_0_Watches/A. Lange & Söhne/locked.jpg", blocked, opener)
        self.assertEqual(calls, [])
        samples = [
            _sample("Label_0_Watches", "A. Lange & Söhne", "locked", "test"),
            _sample("Label_0_Watches", "A. Lange & Söhne", "ood", "validation", excluded_ood=True),
            _sample("Label_0_Watches", "A. Lange & Söhne", "held", "validation"),
        ]
        result = audit.build_audit(samples, _folders())
        self.assertEqual(len(result["primary"]), 1)
        self.assertTrue(all(row["image_sha256"] is None for row in result["rows"] if row["dinov3_split"] == "test" or row["excluded_ood"]))

    def test_outputs_are_deterministic(self) -> None:
        samples = [
            _sample("Label_0_Watches", "A. Lange & Söhne", "b", "validation"),
            _sample("Label_0_Watches", "A. Lange & Söhne", "a", "train"),
        ]
        first = audit.build_audit(samples, _folders())
        second = audit.build_audit(list(reversed(samples)), _folders())
        self.assertEqual(
            json.dumps(first["intersection"], sort_keys=True),
            json.dumps(second["intersection"], sort_keys=True),
        )
        self.assertEqual(
            [row["canonical_id"] for row in first["rows"]],
            [row["canonical_id"] for row in second["rows"]],
        )


class RealManifestGuardTests(unittest.TestCase):
    def test_hash_mismatch_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "manifest.json"
            path.write_text("{}")
            with self.assertRaises(audit.AuditStop):
                audit.require_file(path, "0" * 64, "manifest")


if __name__ == "__main__":
    unittest.main()
