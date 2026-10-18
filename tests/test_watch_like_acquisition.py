"""Phase 37 acquisition rules. No authenticity scores are consulted."""

from __future__ import annotations

import hashlib
import json
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "ml_rtx5080"))


def _row(**overrides) -> dict:
    base = {
        "image_path": "/tmp/phase37/watch.jpg",
        "sha256": "a" * 64,
        "source": "example_repository",
        "source_url": "https://example.test/watch.jpg",
        "source_id": "example:1",
        "brand": "Omega",
        "category": "unseen_luxury",
        "subcategory": "wristwatch",
        "license": "CC BY 4.0",
        "license_url": "https://creativecommons.org/licenses/by/4.0/",
        "collection_timestamp": "2026-10-07T00:00:00+00:00",
        "image_width": 800,
        "image_height": 600,
        "file_size": 1000,
        "format": "JPEG",
        "is_original_or_derivative": "original_photo",
        "independent_source_group": "example-session-1",
        "phase": 37,
        "admission_status": "admitted",
        "rejection_reason": "",
        "brand_verification": "source_verified",
    }
    base.update(overrides)
    return base


class WatchLikeAcquisitionTests(unittest.TestCase):
    def test_manifest_schema_rejects_model_scores(self) -> None:
        from watch_like_acquisition import AcquisitionError, validate_manifest_row

        validate_manifest_row(_row())
        with self.assertRaises(AcquisitionError):
            validate_manifest_row({key: value for key, value in _row().items() if key != "license"})
        with self.assertRaises(AcquisitionError):
            validate_manifest_row(_row(p_fake=0.1))

    def test_provenance_requires_source_license_and_original_bytes_class(self) -> None:
        from watch_like_acquisition import provenance_reason

        self.assertEqual(provenance_reason(_row()), "")
        self.assertEqual(provenance_reason(_row(source="UNKNOWN")), "missing_source")
        self.assertEqual(provenance_reason(_row(source_url="UNKNOWN")), "missing_source_url")
        self.assertEqual(provenance_reason(_row(license="UNKNOWN")), "missing_license")
        self.assertEqual(provenance_reason(_row(is_original_or_derivative="derivative")), "originality_not_admitted")
        self.assertEqual(provenance_reason(_row(is_original_or_derivative="unknown")), "originality_not_admitted")

    def test_duplicate_detection_uses_hash_url_and_perceptual_distance(self) -> None:
        from watch_like_acquisition import IdentityIndex, duplicate_reason

        index = IdentityIndex()
        index.add_record(
            {
                "sha256": "b" * 64,
                "source_url": "https://example.test/old.jpg",
                "source_id": "old:1",
                "image_path": "/tmp/old.jpg",
                "perceptual_hash": "0000000000000000",
            }
        )
        self.assertEqual(duplicate_reason(_row(sha256="b" * 64), index, None), "exact_sha256")
        self.assertEqual(
            duplicate_reason(_row(source_url="https://example.test/old.jpg"), index, None),
            "source_url",
        )
        self.assertEqual(duplicate_reason(_row(image_path="/tmp/old.jpg"), index, None), "filename")
        self.assertEqual(duplicate_reason(_row(), index, 1), "near_duplicate")
        self.assertEqual(duplicate_reason(_row(), index, int("ffffffffffffffff", 16)), "")

    def test_derivative_and_category_and_brand_rules(self) -> None:
        from watch_like_acquisition import category_reason, derivative_reason, rejection_reason

        self.assertEqual(derivative_reason(_row(image_path="/tmp/watch.rf.jpg")), "derivative_marker")
        self.assertEqual(
            derivative_reason(_row(source_url="https://live.staticflickr.com/1/2_ab_b.jpg")),
            "flickr_resized_rendition",
        )
        self.assertEqual(derivative_reason(_row()), "")
        self.assertEqual(category_reason(_row(category="UNKNOWN")), "ambiguous_category")
        self.assertEqual(category_reason(_row(brand="UNKNOWN", brand_verification="unknown")), "brand_not_source_verified")
        self.assertEqual(rejection_reason(_row(brand="Patek Philippe")), "excluded_trained_brand")
        self.assertEqual(rejection_reason(_row(brand="A. Lange & Söhne")), "excluded_trained_brand")
        self.assertEqual(rejection_reason(_row(brand="Rolex")), "")

    def test_source_groups_stay_together_and_assignment_is_deterministic(self) -> None:
        from watch_like_acquisition import assign_holdout, assign_source_group_split

        first = assign_source_group_split("session-9")
        self.assertEqual(first, assign_source_group_split("session-9"))
        self.assertIn(first, {"WATCH_OOD_DEV", "WATCH_OOD_HOLDOUT"})
        rows = assign_holdout(
            [
                _row(sha256="1" * 64, independent_source_group="same-family"),
                _row(sha256="2" * 64, independent_source_group="same-family", source_id="example:2"),
                _row(sha256="3" * 64, independent_source_group="other-family", source_id="example:3"),
            ]
        )
        family = [row["split"] for row in rows if row["independent_source_group"] == "same-family"]
        self.assertEqual(family[0], family[1])
        self.assertEqual(rows, assign_holdout(rows))

    def test_manifest_hash_is_stable(self) -> None:
        from watch_like_acquisition import manifest_sha256

        payload = {"samples": [_row()]}
        self.assertEqual(manifest_sha256(payload), manifest_sha256(json.loads(json.dumps(payload))))
        changed = {"samples": [_row(brand="Cartier")]}
        self.assertNotEqual(manifest_sha256(payload), manifest_sha256(changed))
        self.assertEqual(len(manifest_sha256(payload)), 64)

    def test_corpus_ready_is_computed_from_counts(self) -> None:
        from watch_like_acquisition import assign_source_group_split, corpus_ready, load_protocol

        protocol = load_protocol()
        self.assertFalse(corpus_ready([_row(admission_status="rejected", rejection_reason="ambiguous_category", category="UNKNOWN")], protocol))
        rows: list[dict] = []
        quotas = {
            "unseen_luxury": (300, 60),
            "ordinary_watch": (200, 40),
            "smart_digital": (100, 25),
            "wrist_store": (50, 25),
        }
        brands = {
            "unseen_luxury": ["Rolex", "Omega", "Cartier", "Tudor"],
            "ordinary_watch": ["Casio", "Seiko", "Citizen", "Timex"],
            "smart_digital": ["Apple", "Samsung", "Garmin", "Fitbit"],
            "wrist_store": ["UNKNOWN"],
        }
        serial = 0
        for category, (need_admitted, need_holdout) in quotas.items():
            brand_cycle = brands[category]
            holdout_for_brand = {brand: 0 for brand in brand_cycle}
            admitted = 0
            holdout = 0
            while admitted < need_admitted or holdout < need_holdout or any(
                count < 8 for count in holdout_for_brand.values()
            ):
                serial += 1
                brand = brand_cycle[serial % len(brand_cycle)]
                group = f"{category}-{serial}"
                split = assign_source_group_split(group)
                if split == "WATCH_OOD_HOLDOUT" and holdout >= need_holdout and admitted >= need_admitted:
                    if category != "unseen_luxury" or all(count >= 8 for count in holdout_for_brand.values()):
                        break
                rows.append(
                    _row(
                        sha256=f"{serial:064x}",
                        source_id=group,
                        source_url=f"https://example.test/{group}.jpg",
                        image_path=f"/tmp/{group}.jpg",
                        independent_source_group=group,
                        category=category,
                        brand=brand,
                        brand_verification="source_verified" if brand != "UNKNOWN" else "unknown",
                    )
                )
                admitted += 1
                if split == "WATCH_OOD_HOLDOUT":
                    holdout += 1
                    holdout_for_brand[brand] += 1
                if serial > 5000:
                    self.fail("could not build a deterministic fixture")
        self.assertTrue(corpus_ready(rows, protocol))

    def test_phase36_protocol_and_holdout_files_stay_unchanged(self) -> None:
        from watch_like_acquisition import PROTOCOL_SHA256, protocol_sha256

        self.assertEqual(protocol_sha256(), PROTOCOL_SHA256)
        manifest = REPO / "ml_rtx5080" / "experiments" / "watch_like_ood_v1" / "watch_like_ood_manifest.json"
        digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
        recorded = (manifest.parent / "watch_like_ood_manifest.sha256").read_text().strip()
        self.assertEqual(digest, recorded)
        self.assertNotIn("torch", (REPO / "ml_rtx5080" / "watch_like_acquisition.py").read_text())

    def test_recorded_phase37_corpus_is_not_ready(self) -> None:
        status_path = REPO / "ml_rtx5080" / "experiments" / "watch_like_ood_v2" / "acquisition_status.json"
        if not status_path.exists():
            self.skipTest("phase 37 artifacts have not been written")
        status = json.loads(status_path.read_text())
        self.assertFalse(status["WATCH_OOD_V2_CORPUS_READY"])
        self.assertTrue(status["PROTOCOL_HASH_UNCHANGED"])
        self.assertFalse(status["PHASE36_HOLDOUT_MODIFIED"])
        self.assertFalse(status["FINAL_TEST_TOUCHED"])
        self.assertFalse(status["MODEL_TOUCHED"])
        self.assertFalse(status["evaluation_corpus_locked"])
        manifest_path = status_path.parent / "acquisition_manifest.json"
        digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
        self.assertEqual(digest, (status_path.parent / "acquisition_manifest.sha256").read_text().strip())
        self.assertFalse((status_path.parent / "watch_like_ood_v2_manifest.sha256").exists())
        samples = json.loads(manifest_path.read_text())["samples"]
        self.assertTrue(samples)
        self.assertTrue(all(sample["admission_status"] == "rejected" for sample in samples))
        self.assertTrue(all("p_fake" not in sample for sample in samples))


if __name__ == "__main__":
    unittest.main()
