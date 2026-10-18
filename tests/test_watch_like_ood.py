"""Phase 36 rules. The holdout is not used to invent a threshold."""

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
        "image_path": "/tmp/watch.jpg",
        "sha256": "a" * 64,
        "source": "wikimedia_commons",
        "source_url": "https://commons.wikimedia.org/wiki/File:Watch.jpg",
        "brand": "Omega",
        "watch_category": "unseen_luxury",
        "collection_method": "wikimedia_commons_original_file",
        "is_original_or_derivative": "commons_original_upload",
        "license_or_usage_note": "CC BY 4.0",
        "width": 800,
        "height": 600,
        "aspect_ratio": 800 / 600,
        "is_trained_brand": False,
    }
    base.update(overrides)
    return base


def _facts(unseen=0.9, ordinary=0.8, smart=0.7, minimum=True) -> dict:
    return {
        "minimum_met": minimum,
        "minimum_failures": [] if minimum else ["holdout too small"],
        "holdout_escape": {
            "unseen_luxury": unseen,
            "ordinary_watch": ordinary,
            "smart_digital": smart,
            "wrist_store": 0.5,
        },
    }


class WatchLikeOODTests(unittest.TestCase):
    def test_protocol_is_stable_and_forbids_threshold_search(self) -> None:
        from watch_like_ood import protocol_document

        first = protocol_document()
        second = protocol_document()
        self.assertEqual(first, second)
        self.assertEqual(first["threshold_search"], "forbidden")
        self.assertIsNone(first["production_ood_threshold"])
        self.assertEqual(first["temperature"], 0.24038200410185356)
        self.assertEqual(first["checkpoint_sha256"], "5a38c93fd442b03653c65d2a5ecc9c2687ef152f7c5c020763e4ce1fd9c7d28f")
        self.assertFalse(first["final_test_used"])

    def test_provenance_and_category_rules(self) -> None:
        from watch_like_ood import OODArtifactError, validate_watch_row

        validate_watch_row(_row())
        with self.assertRaises(OODArtifactError):
            validate_watch_row(_row(source="UNKNOWN"))
        with self.assertRaises(OODArtifactError):
            validate_watch_row(_row(watch_category="handbag"))
        with self.assertRaises(OODArtifactError):
            validate_watch_row(_row(brand="Patek Philippe"))
        with self.assertRaises(OODArtifactError):
            validate_watch_row(_row(source="roboflow"))

    def test_split_and_escape_are_deterministic(self) -> None:
        from watch_like_ood import assign_split, escape_rate

        digest = "00000000" + "ab" * 28
        self.assertEqual(assign_split(digest), assign_split(digest))
        rows = [{"decision_current": "AUTHENTIC"}, {"decision_current": "REVIEW"}, {"decision_current": "AUTHENTIC"}]
        self.assertAlmostEqual(escape_rate(rows), 2 / 3)
        self.assertAlmostEqual(escape_rate(rows), escape_rate(list(reversed(rows))))

    def test_duplicate_screen_rejects_phase35_and_blocked_paths(self) -> None:
        from watch_like_ood import screen_rows

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image = root / "watch.jpg"
            Image.new("RGB", (40, 30), (20, 30, 40)).save(image)
            blocked = root / "final.jpg"
            Image.new("RGB", (8, 8), (1, 1, 1)).save(blocked)
            digest = hashlib.sha256(image.read_bytes()).hexdigest()
            row = _row(image_path=str(image), sha256=digest)
            opened = {"blocked": False}
            original = Image.open

            def guard(path, *args, **kwargs):
                if Path(path).resolve() == blocked.resolve():
                    opened["blocked"] = True
                    raise AssertionError("blocked image opened")
                return original(path, *args, **kwargs)

            Image.open = guard
            try:
                overlap = screen_rows([row], set(), np.array([], dtype=np.uint64), {digest}, np.array([], dtype=np.uint64), {str(blocked.resolve())})
                blocked_result = screen_rows(
                    [_row(image_path=str(blocked), sha256="b" * 64)],
                    set(),
                    np.array([], dtype=np.uint64),
                    set(),
                    np.array([], dtype=np.uint64),
                    {str(blocked.resolve())},
                )
            finally:
                Image.open = original
        self.assertFalse(opened["blocked"])
        self.assertEqual(overlap["rejected"][0]["rejection"], "exact_sha256_overlap")
        self.assertEqual(blocked_result["rejected"][0]["rejection"], "final_test_path")

    def test_status_rules_and_checkpoint_constants(self) -> None:
        from watch_like_ood import (
            CHECKPOINT_SHA256,
            CONFIRMED_FAIL,
            FROZEN_TEMPERATURE,
            IMPROVED,
            PASSED,
            UNRESOLVED,
            classify_watch_ood,
        )

        self.assertEqual(CHECKPOINT_SHA256, "5a38c93fd442b03653c65d2a5ecc9c2687ef152f7c5c020763e4ce1fd9c7d28f")
        self.assertEqual(FROZEN_TEMPERATURE, 0.24038200410185356)
        self.assertEqual(classify_watch_ood(_facts(minimum=False))["status"], UNRESOLVED)
        self.assertEqual(classify_watch_ood(_facts(0.88, 0.7, 0.6))["status"], CONFIRMED_FAIL)
        self.assertEqual(classify_watch_ood(_facts(0.40, 0.40, 0.40))["status"], IMPROVED)
        self.assertEqual(classify_watch_ood(_facts(0.01, 0.01, 0.01))["status"], PASSED)

    def test_recorded_run_keeps_the_frozen_contract(self) -> None:
        status_path = REPO / "ml_rtx5080" / "experiments" / "watch_like_ood_v1" / "watch_like_ood_status.json"
        if not status_path.is_file():
            self.skipTest("phase 36 artifacts are written by the evaluation run")
        status = json.loads(status_path.read_text())
        protocol = (status_path.parent / "watch_like_ood_protocol_v1.json").read_bytes()
        self.assertEqual((status_path.parent / "watch_like_ood_protocol_v1.sha256").read_text().strip(), hashlib.sha256(protocol).hexdigest())
        manifest = (status_path.parent / "watch_like_ood_manifest.json").read_bytes()
        self.assertEqual((status_path.parent / "watch_like_ood_manifest.sha256").read_text().strip(), hashlib.sha256(manifest).hexdigest())
        self.assertIsNone(status["production_ood_threshold"])
        self.assertFalse(status["threshold_changed"])
        self.assertFalse(status["promoted"])
        self.assertEqual(status["final_test_samples_used"], 0)
        self.assertTrue(status["final_test_untouched"])
        self.assertEqual(status["frozen_temperature"], 0.24038200410185356)


if __name__ == "__main__":
    unittest.main()
