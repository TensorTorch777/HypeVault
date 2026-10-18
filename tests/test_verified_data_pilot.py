"""Phase 49 verified-data pilot. No images and no checkpoints are opened."""

from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO / "ml_rtx5080" / "experiments" / "verified_data_pilot_v1"
FROZEN_SHA = "5a38c93fd442b03653c65d2a5ecc9c2687ef152f7c5c020763e4ce1fd9c7d28f"


def _load(name: str) -> dict:
    return json.loads((ROOT / name).read_text())


class VerifiedDataPilotTests(unittest.TestCase):
    def test_protocol_and_manifest_hashes(self):
        protocol_raw = (ROOT / "pilot_protocol_v1.json").read_bytes()
        protocol_sha = (ROOT / "pilot_protocol_v1.sha256").read_text().split()[0]
        self.assertEqual(hashlib.sha256(protocol_raw).hexdigest(), protocol_sha)
        protocol = json.loads(protocol_raw)
        self.assertEqual(protocol["frozen"]["checkpoint_sha256"], FROZEN_SHA)
        self.assertTrue(protocol["frozen"]["no_training"])
        self.assertTrue(protocol["frozen"]["no_inference"])
        self.assertTrue(protocol["written_before_admission"])

        manifest_raw = (ROOT / "candidate_manifest.json").read_bytes()
        manifest_sha = (ROOT / "candidate_manifest.sha256").read_text().split()[0]
        self.assertEqual(hashlib.sha256(manifest_raw).hexdigest(), manifest_sha)
        manifest = json.loads(manifest_raw)
        self.assertEqual(manifest["protocol_sha256"], protocol_sha)
        self.assertEqual(manifest["images_acquired"], 0)
        self.assertEqual(manifest["admitted"], 0)
        self.assertEqual(manifest["final_test_files_opened"], 0)
        self.assertGreater(len(manifest["records"]), 0)
        for row in manifest["records"]:
            self.assertEqual(row["admission_status"], "REJECTED")
            self.assertEqual(row["sha256"], "UNKNOWN")
            self.assertNotEqual(row["label"], "authentic")
            self.assertNotEqual(row["label"], "counterfeit")

    def test_no_go_does_not_invent_a_training_set(self):
        decision = _load("go_no_go.json")
        self.assertEqual(decision["decision"], "NO_GO")
        self.assertEqual(decision["AUTHENTIC_VERIFIED_COUNT"], 0)
        self.assertEqual(decision["COUNTERFEIT_VERIFIED_COUNT"], 0)
        self.assertEqual(decision["INDEPENDENT_SOURCE_GROUPS"], 0)
        self.assertFalse(decision["LICENSE_AND_PERMISSION_CONFIRMED"])
        self.assertFalse(decision["LABEL_PROVENANCE_COMPLETE"])
        self.assertFalse(decision["DATASET_READY_FOR_TRAINING"])
        self.assertFalse(decision["MODEL_CHANGED"])
        self.assertFalse(decision["CALIBRATION_CHANGED"])
        self.assertFalse(decision["FINAL_TEST_TOUCHED"])
        self.assertFalse(decision["PRODUCTION_PROMOTION_ALLOWED"])
        self.assertFalse(decision["model_development_should_resume"])

        labels = _load("label_evidence_audit.json")
        rights = _load("license_and_permission_audit.json")
        duplicates = _load("duplicate_audit.json")
        self.assertEqual(labels["rows_admitted"], 0)
        self.assertFalse(labels["label_provenance_complete"])
        self.assertFalse(rights["confirmed_for_an_admitted_image"])
        self.assertEqual(duplicates["candidate_files_acquired"], 0)
        self.assertEqual(duplicates["final_test_files_opened"], 0)
        self.assertEqual(_load("source_feasibility.json")["images_downloaded"], 0)

        report = (ROOT / "pilot_report.md").read_text()
        self.assertIn("DATA_PILOT_STATUS = NO_GO", report)
        self.assertIn("DATASET_READY_FOR_TRAINING = false", report)
