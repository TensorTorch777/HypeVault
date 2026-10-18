"""The Phase 17 trainer must consume the frozen Phase 16 membership."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ml_rtx5080"))

import dataset as ds
import split_v2


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "ml_rtx5080" / "experiments" / "dataset_audit" / "split_manifest_v2.json"


_DATASET = Path(__file__).resolve().parents[1] / "Label_0_Watches"


class FixedSplitManifestTests(unittest.TestCase):
    @unittest.skipUnless(_DATASET.is_dir(), "local dataset, not in git")
    def test_authoritative_manifest_matches_phase16(self) -> None:
        records = ds.load_records(ROOT)
        splits, meta = split_v2.records_from_fixed_manifest(
            records,
            MANIFEST,
            expected_hash=split_v2.AUTHORITATIVE_MEMBERSHIP_HASH,
        )
        self.assertEqual(meta["calibration_membership_source"], "split_manifest_v2")
        self.assertEqual(meta["manifest_membership_hash"], split_v2.AUTHORITATIVE_MEMBERSHIP_HASH)
        self.assertFalse(meta["product_disjoint_guarantee"])
        for name, expected in split_v2.AUTHORITATIVE_COUNTS.items():
            rows = splits[name]
            self.assertEqual(len(rows), expected["samples"])
            self.assertEqual(sum(row.label == 0 for row in rows), expected["authentic"])
            self.assertEqual(sum(row.label == 1 for row in rows), expected["fake"])

    @unittest.skipUnless(_DATASET.is_dir(), "local dataset, not in git")
    def test_wrong_hash_stops_before_a_split_is_accepted(self) -> None:
        records = ds.load_records(ROOT)
        with self.assertRaises(RuntimeError) as raised:
            split_v2.records_from_fixed_manifest(
                records,
                MANIFEST,
                expected_hash="0" * 64,
            )
        self.assertIn("Refusing to train", str(raised.exception))

    def test_copy_refuses_phase15_directories(self) -> None:
        with self.assertRaises(RuntimeError):
            split_v2.write_fixed_membership_copy(
                Path("/tmp/dinov2_cls_only"),
                {},
                {"manifest_membership_hash": "abc"},
            )


if __name__ == "__main__":
    unittest.main()
