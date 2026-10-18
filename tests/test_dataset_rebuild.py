"""Phase 48 dataset-rebuild protocol. No images and no checkpoints are opened."""

from __future__ import annotations

import hashlib
import json
import random
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO / "ml_rtx5080" / "experiments" / "dataset_rebuild_v1"
MANIFEST = REPO / "ml_rtx5080" / "experiments" / "dataset_audit" / "split_manifest_v2.json"
FROZEN_SHA = "5a38c93fd442b03653c65d2a5ecc9c2687ef152f7c5c020763e4ce1fd9c7d28f"


def _load(name: str) -> dict:
    return json.loads((ROOT / name).read_text())


class DatasetRebuildTests(unittest.TestCase):
    def test_protocol_hash_is_fixed_before_later_artifacts(self):
        raw = (ROOT / "dataset_rebuild_protocol_v1.json").read_bytes()
        recorded = (ROOT / "dataset_rebuild_protocol_v1.sha256").read_text().split()[0]
        self.assertEqual(hashlib.sha256(raw).hexdigest(), recorded)
        protocol = json.loads(raw)
        self.assertEqual(protocol["frozen"]["checkpoint_sha256"], FROZEN_SHA)
        self.assertEqual(protocol["frozen"]["temperature"], 0.24038200410185356)
        self.assertEqual(protocol["frozen"]["decision_threshold"], 0.5)
        self.assertTrue(protocol["frozen"]["no_training"])
        self.assertTrue(protocol["future_geometry_policy"]["splits"].startswith("No train"))
        for name in (
            "source_label_audit.json",
            "geometry_shortcut_audit.json",
            "duplicate_quarantine.json",
            "image_label_evidence_schema.json",
            "rebuild_candidate_manifest.json",
        ):
            self.assertEqual(_load(name)["protocol_sha256"], recorded)

    def test_visual_sample_matches_the_hashed_algorithm(self):
        protocol = _load("dataset_rebuild_protocol_v1.json")
        catalog = json.loads(MANIFEST.read_text())
        eligible = [row for row in catalog["samples"] if row["split"] != "test"]
        rng = random.Random(48)
        drawn = []
        for brand in protocol["visual_sample"]["brands_in_order"]:
            for label in protocol["visual_sample"]["labels_in_order"]:
                pool = [row for row in eligible if row["brand"] == brand and row["label"] == label]
                pool.sort(key=lambda row: row["sha256"])
                drawn.extend(rng.sample(pool, 3))
        sample = _load("geometry_shortcut_audit.json")["sample"]["records"]
        self.assertEqual([row["image_id"] for row in sample], [Path(row["path"]).name for row in drawn])
        self.assertTrue(all(row["split"] != "test" for row in sample))

    def test_shortcut_is_a_measurement_and_not_a_trained_classifier(self):
        geometry = _load("geometry_shortcut_audit.json")
        rule = geometry["pre_specified_512_rule"]
        self.assertFalse(geometry["classifier_trained"])
        self.assertEqual(geometry["final_test_files_opened"], 0)
        self.assertEqual(rule["counterfeit_and_512"], 13500)
        self.assertEqual(rule["authentic_and_512"], 1)
        self.assertEqual(rule["counterfeit_other_size"], 0)
        self.assertEqual(geometry["quantization"]["counterfeit_unique_tables"], 1)
        self.assertEqual(geometry["quantization"]["same_counterfeit_table_on_authentic_files"], 127)
        self.assertFalse(geometry["quantization"]["authentic_512_uses_counterfeit_table"])

    def test_no_image_is_training_ready(self):
        audit = _load("source_label_audit.json")
        candidate = _load("rebuild_candidate_manifest.json")
        self.assertEqual(audit["independently_verified"], 0)
        self.assertEqual(audit["catalog_rows"], 30000)
        self.assertFalse(any(row["training_eligible"] for row in audit["records"]))
        self.assertFalse(any(row["admitted_label"] is not None or row["assigned_split"] is not None for row in audit["records"]))
        self.assertFalse(candidate["dataset_ready_for_training"])
        self.assertFalse(candidate["splits_created"])
        self.assertEqual(candidate["training_eligible"], 0)
        self.assertEqual(candidate["sealed_final_test_files_opened"], 0)
        self.assertEqual(audit["disposition_counts"]["SEALED_FINAL_TEST"], 3006)
        self.assertEqual(audit["disposition_counts"]["QUARANTINE_PLACEHOLDER"], 9)

        quarantine = _load("duplicate_quarantine.json")
        self.assertFalse(quarantine["source_files_deleted"])
        self.assertEqual(quarantine["placeholder_copies"], 9)
        ids = {row["image_id"] for row in quarantine["records"]}
        for image_id in _load("dataset_rebuild_protocol_v1.json")["previously_identified_quarantine"]["expert_review_image_ids"]:
            self.assertIn(image_id, ids)

        report = (ROOT / "phase48_report.md").read_text()
        self.assertIn("DATASET_READY_FOR_TRAINING = false", report)
        self.assertIn("MODEL_CHANGED = false", report)
        self.assertIn("FINAL_TEST_TOUCHED = false", report)
        self.assertIn("PRODUCTION_PROMOTION_ALLOWED = false", report)
        schema = _load("image_label_evidence_schema.json")
        self.assertIn("directory name", schema["not_sufficient"])
        self.assertIn("model prediction", schema["not_sufficient"])
