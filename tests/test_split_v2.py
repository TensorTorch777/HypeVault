"""Corrected brand-stratified split: duplicates stay together and brands do not confound labels."""

from __future__ import annotations

import json
import sys
import unittest
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ml_rtx5080"))

from PIL import Image

import dataset as ds
import split_v2


BRANDS = ["A. Lange & Söhne", "Audemars Piguet", "Patek Philippe", "Richard Mille", "Vacheron Constantin"]


def _jpeg(brand_index: int, label: int, index: int, quality: int = 95, code: int | None = None) -> bytes:
    # One bit per row, so distinct ids differ by at least 8 dHash bits.
    # A recompressed copy stays well inside the Hamming threshold of 4.
    if code is None:
        code = brand_index * 40 + label * 20 + index
    small = Image.new("L", (9, 8))
    for y in range(8):
        bit = (code >> y) & 1
        for x in range(9):
            small.putpixel((x, y), x * 28 if bit else (8 - x) * 28)
    image = small.resize((72, 64), Image.Resampling.NEAREST).convert("RGB")
    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=quality)
    return buffer.getvalue()


def _write_catalog(root: Path) -> None:
    for label, folder in ((0, "Label_0_Watches"), (1, "Label_1_Watches")):
        for brand_index, brand in enumerate(BRANDS):
            directory = root / folder / brand
            directory.mkdir(parents=True)
            for index in range(20):
                (directory / f"{index:02d}.jpg").write_bytes(_jpeg(brand_index, label, index))
            # Same bytes, second filename: an exact duplicate inside this brand and label.
            (directory / "exact-copy.jpg").write_bytes((directory / "00.jpg").read_bytes())
            # Recompressed copy of 01.jpg: a near duplicate, not a second product identity.
            source = Image.open(directory / "01.jpg")
            buffer = BytesIO()
            source.save(buffer, format="JPEG", quality=30)
            (directory / "near-copy.jpg").write_bytes(buffer.getvalue())


def _build(root: Path, seed: int = 42):
    records = ds.load_records(root)
    return split_v2.build_split_from_records(records, root, seed)


class SplitV2Tests(unittest.TestCase):
    def test_same_seed_reproduces_membership_hash(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_catalog(root)
            _, _, first = _build(root, 42)
            _, _, second = _build(root, 42)
            self.assertEqual(first["membership_hash"], second["membership_hash"])
            self.assertEqual(first["membership"], second["membership"])
            self.assertEqual(json.dumps(first, sort_keys=True), json.dumps(second, sort_keys=True))
            self.assertEqual(first["membership_hash"], split_v2.membership_hash(first["membership"]))

    def test_different_seed_changes_membership(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_catalog(root)
            _, _, first = _build(root, 1)
            _, _, second = _build(root, 2)
            self.assertNotEqual(first["membership_hash"], second["membership_hash"])

    def test_catalog_split_is_duplicate_safe_and_brand_stratified(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_catalog(root)
            rows, summary, manifest = _build(root, 42)
            self.assertGreater(summary["exact_duplicate_groups"], 0)
            self.assertGreater(summary["near_duplicate_groups"], 0)
            self.assertEqual(manifest["strategy"], "brand_stratified_duplicate_safe")
            self.assertFalse(manifest["product_disjoint_guarantee"])
            self.assertEqual(manifest["product_id_status"], "unavailable")
            self.assertFalse(manifest["calibration_policy"]["weight_updates_allowed"])
            report = split_v2.validate_split_manifest(manifest)
            self.assertTrue(report["ok"])
            for name in split_v2.SPLIT_NAMES:
                self.assertGreater(manifest["counts"][name], 0)
            for brand in BRANDS:
                for label in (0, 1):
                    for name in split_v2.SPLIT_NAMES:
                        present = [
                            row
                            for row in manifest["samples"]
                            if row["split"] == name and row["brand"] == brand and row["label"] == label
                        ]
                        self.assertTrue(present, f"{name} missing {brand} label {label}")
            owners: dict[str, str] = {}
            for row in rows:
                if row["manual_review"]:
                    self.assertEqual(row["split"], "manual_review")
                    continue
                previous = owners.get(row["duplicate_group_id"])
                self.assertTrue(previous is None or previous == row["split"])
                owners[row["duplicate_group_id"]] = row["split"]

    def test_mixed_label_duplicate_is_held_for_review(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_catalog(root)
            shared = _jpeg(0, 0, 0, code=250)
            for folder in ("Label_0_Watches", "Label_1_Watches"):
                (root / folder / BRANDS[0] / "conflict.jpg").write_bytes(shared)
            _, summary, manifest = _build(root, 42)
            self.assertGreater(summary["manual_review_images"], 0)
            conflicts = [row for row in manifest["samples"] if row["sample_id"].endswith("conflict.jpg")]
            self.assertEqual({row["split"] for row in conflicts}, {"manual_review"})
            self.assertEqual({row["label"] for row in conflicts}, {0, 1})
            split_v2.validate_split_manifest(manifest)

    def test_product_disjoint_claim_stays_false(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_catalog(root)
            records = ds.load_records(root)
            self.assertTrue(all(record.product_id == ds.UNKNOWN for record in records))
            audit_rows, _, manifest = _build(root, 42)
            audit = split_v2.structure_audit(records, audit_rows, {"exact_duplicate_groups": 0, "largest_exact_duplicate_group": 0, "exact_image_counts_by_label_brand": [], "near_duplicate_groups": 0, "largest_near_duplicate_group": 0, "near_groups": [], "exact_groups": [], "hamming_threshold": 4, "manual_review_images": 0})
            self.assertFalse(audit["product_disjoint_possible"])
            self.assertIn("brand", audit["current_group_meaning"])
            self.assertFalse(manifest["product_disjoint_guarantee"])

    def test_writer_does_not_overwrite_phase15_manifest(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "data"
            output = Path(tmp) / "audit"
            root.mkdir()
            _write_catalog(root)
            phase15 = output / "split_manifest.json"
            output.mkdir()
            phase15.write_text('{"phase": 15}\n')
            before = phase15.read_bytes()
            records = ds.load_records(root)
            split_v2.write_audit_directory(output, records, root, 42)
            self.assertEqual(phase15.read_bytes(), before)
            self.assertTrue((output / "split_manifest_v2.json").exists())
            with self.assertRaises(ValueError):
                split_v2._write_json(output / "split_manifest.json", {"no": True})
            with self.assertRaises(ValueError):
                split_v2._write_json(Path(tmp) / "checkpoints" / "split_manifest_v2.json", {"no": True})

    def test_leakage_checker_catches_overlap_duplicate_split_and_brand_confound(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_catalog(root)
            _, _, manifest = _build(root, 42)
            overlapped = json.loads(json.dumps(manifest))
            overlapped["samples"][0]["split"] = "train"
            overlapped["samples"][1]["path"] = overlapped["samples"][0]["path"]
            overlapped["samples"][1]["split"] = "validation"
            with self.assertRaises(split_v2.SplitLeakageError):
                split_v2.validate_split_manifest(overlapped)

            crossed = json.loads(json.dumps(manifest))
            group = crossed["samples"][0]["duplicate_group_id"]
            mates = [row for row in crossed["samples"] if row["duplicate_group_id"] == group]
            if len(mates) < 2:
                crossed["samples"][1]["duplicate_group_id"] = group
                mates = crossed["samples"][:2]
            mates[0]["split"] = "train"
            mates[1]["split"] = "validation"
            with self.assertRaises(split_v2.SplitLeakageError) as crossed_error:
                split_v2.validate_split_manifest(crossed)
            self.assertTrue(any("duplicate group" in message for message in crossed_error.exception.failures))

            confounded = json.loads(json.dumps(manifest))
            for row in confounded["samples"]:
                if row["split"] != "validation":
                    continue
                if row["label"] == 0:
                    row["brand"] = BRANDS[0]
                else:
                    row["brand"] = BRANDS[1]
            with self.assertRaises(split_v2.SplitLeakageError) as confound_error:
                split_v2.validate_split_manifest(confounded)
            text = " ".join(confound_error.exception.failures)
            self.assertIn("validation", text)
            self.assertIn("brand", text)

            empty_test = json.loads(json.dumps(manifest))
            for row in empty_test["samples"]:
                if row["split"] == "test":
                    row["split"] = "train"
            with self.assertRaises(split_v2.SplitLeakageError):
                split_v2.validate_split_manifest(empty_test)


if __name__ == "__main__":
    unittest.main()
