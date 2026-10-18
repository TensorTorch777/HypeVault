"""Phase 47 dataset-integrity protocol. No images and no checkpoints are opened."""

from __future__ import annotations

import hashlib
import json
import random
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO / "ml_rtx5080" / "experiments" / "dataset_integrity_v1"
MANIFEST = REPO / "ml_rtx5080" / "experiments" / "dataset_audit" / "split_manifest_v2.json"
FROZEN_SHA = "5a38c93fd442b03653c65d2a5ecc9c2687ef152f7c5c020763e4ce1fd9c7d28f"
PREIDENTIFIED = (
    "3b5748de7b1b.jpg",
    "e12883ada111.jpg",
    "4daef03931ff.jpg",
    "d07c3aa1998f.jpg",
    "d5266fb5ab9a.jpg",
    "18730ea81b18.jpg",
    "ff49bf4ac8ab.jpg",
    "d6ca042cb997.jpg",
    "a6d20a112797.jpg",
    "debeefd1cdee.jpg",
    "7be465bc1cf6.jpg",
    "896a73ac8355.jpg",
    "5d747fb6eba2.jpg",
    "d14367e7ccfe.jpg",
    "4e3b5554ef77.jpg",
)


def _load(name: str) -> dict:
    return json.loads((ROOT / name).read_text())


class DatasetIntegrityTests(unittest.TestCase):
    def test_protocol_hash_matches_the_file(self):
        raw = (ROOT / "audit_protocol_v1.json").read_bytes()
        recorded = (ROOT / "audit_protocol_v1.sha256").read_text().split()[0]
        self.assertEqual(hashlib.sha256(raw).hexdigest(), recorded)
        protocol = json.loads(raw)
        self.assertEqual(protocol["sampling"]["seed"], 47)
        self.assertEqual(protocol["sampling"]["per_cell"], 4)
        self.assertEqual(protocol["frozen"]["checkpoint_sha256"], FROZEN_SHA)
        self.assertEqual(protocol["frozen"]["temperature"], 0.24038200410185356)
        self.assertEqual(protocol["frozen"]["decision_threshold"], 0.5)
        self.assertTrue(protocol["frozen"]["final_test_images_not_opened"])
        self.assertEqual(protocol["identity_recovery"]["preidentified_image_ids"], list(PREIDENTIFIED))

    def test_sample_manifest_hash_and_deterministic_redraw(self):
        raw = (ROOT / "audit_sample_manifest.json").read_bytes()
        recorded = (ROOT / "audit_sample_manifest.sha256").read_text().split()[0]
        manifest = json.loads(raw)
        self.assertEqual(hashlib.sha256(raw).hexdigest(), recorded)
        self.assertEqual(manifest["protocol_sha256"], (ROOT / "audit_protocol_v1.sha256").read_text().split()[0])
        self.assertEqual(manifest["unique_images"], 55)
        self.assertEqual(manifest["final_test_images_included"], 0)
        self.assertTrue(manifest["drawn_after_protocol_hash"])

        catalog = json.loads(MANIFEST.read_text())
        eligible = [row for row in catalog["samples"] if row["split"] != "test"]
        protocol = _load("audit_protocol_v1.json")
        rng = random.Random(47)
        drawn = []
        for brand in protocol["sampling"]["brands_in_order"]:
            for label in protocol["sampling"]["labels_in_order"]:
                pool = [row for row in eligible if row["brand"] == brand and row["label"] == label]
                pool.sort(key=lambda row: row["sha256"])
                drawn.extend(rng.sample(pool, 4))
        stratified = [row["image_id"] for row in manifest["images"] if row["stratum"] == "stratified_brand_label"]
        self.assertEqual(stratified, [Path(row["path"]).name for row in drawn])
        self.assertEqual(len(stratified), 40)

    def test_sample_excludes_the_locked_test_split(self):
        catalog = json.loads(MANIFEST.read_text())
        by_sha = {row["sha256"]: row for row in catalog["samples"]}
        for row in _load("audit_sample_manifest.json")["images"]:
            source = by_sha[row["sha256"]]
            self.assertNotEqual(source["split"], "test")
            self.assertNotEqual(row["split"], "test")
            self.assertEqual(Path(source["path"]).name, row["image_id"])

    def test_status_flags_and_richard_mille_recovery_limit(self):
        status = _load("status.json")
        self.assertEqual(status["DATASET_LABEL_INTEGRITY"], "FAIL")
        self.assertEqual(status["IMAGES_AUDITED"], 55)
        self.assertEqual(status["LABEL_PROVENANCE_COVERAGE"], "0/55")
        self.assertEqual(status["SUSPECT_IMAGES"], 31)
        self.assertEqual(status["RM_ERRORS_REVIEWED"], 15)
        self.assertEqual(status["RM_ERROR_IDENTITIES_UNAVAILABLE"], 5)
        self.assertEqual(status["PHASE45_GENERALIZATION_SCOPE"], "FIVE_BRAND_CATALOG_ONLY")
        self.assertEqual(status["EXTERNAL_SIXTH_BRAND_VALIDATION"], "BLOCKED")
        self.assertFalse(status["MODEL_CHANGED"])
        self.assertFalse(status["CALIBRATION_CHANGED"])
        self.assertFalse(status["THRESHOLD_CHANGED"])
        self.assertFalse(status["FINAL_TEST_TOUCHED"])
        self.assertFalse(status["PRODUCTION_PROMOTION_ALLOWED"])

        analysis = _load("rm_error_analysis.json")
        self.assertEqual(analysis["aggregate_false_authentic_count"], 20)
        self.assertEqual(len(analysis["cases"]), 15)
        self.assertEqual(analysis["identities_unavailable"], 5)
        self.assertFalse(analysis["threshold_changed"])
        self.assertFalse(analysis["checkpoint_changed"])
        ids = [row["image_id"] for row in analysis["cases"]]
        self.assertEqual(ids, list(PREIDENTIFIED))
        for row in analysis["cases"]:
            self.assertLess(row["p_fake"], 0.5)
            self.assertFalse(row["model_rerun"])

        visual = _load("visual_integrity_audit.json")
        self.assertEqual(visual["images_audited"], 55)
        self.assertEqual(visual["sample_counts"]["expert_review"], 31)
        self.assertEqual(visual["sample_counts"]["independent_provenance_known"], 0)
        provenance = _load("label_provenance_audit.json")
        self.assertTrue(provenance["labels_inherited_from_directory_names"])
        self.assertFalse(provenance["labels_changed"])
        self.assertEqual(provenance["coverage_in_audited_sample"]["with_independent_provenance"], 0)
