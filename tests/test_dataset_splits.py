"""Split membership must follow parent directories and must not invent metadata."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ml_rtx5080"))

import dataset as ds


def _write_tree(root: Path, brands: list[str], per_brand: int = 2) -> None:
    for label, folder in ((0, "Label_0_Watches"), (1, "Label_1_Watches")):
        for brand in brands:
            directory = root / folder / brand
            directory.mkdir(parents=True)
            for index in range(per_brand):
                (directory / f"{index}.jpg").write_bytes(b"img")


class DatasetSplitTests(unittest.TestCase):
    def test_loader_keeps_unknown_fields_and_reads_brand(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_tree(root, ["Rolex", "Omega"])
            records = ds.load_records(root)
            self.assertEqual(len(records), 8)
            self.assertTrue(all(record.product_id == ds.UNKNOWN for record in records))
            self.assertTrue(all(record.marketplace == ds.UNKNOWN for record in records))
            self.assertTrue(all(record.source == ds.UNKNOWN for record in records))
            self.assertEqual(sorted({record.brand for record in records}), ["Omega", "Rolex"])

    def test_product_disjoint_matches_index_split_and_blocks_overlap(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_tree(root, ["A", "B", "C", "D"])
            records = ds.load_records(root)
            samples = [(record.image_path, record.label) for record in records]
            train_idx, val_idx = ds.split_indices_by_product(samples, 0.25, 42)
            splits, meta = ds.split_records(records, "product_disjoint", 42, 0.25, 0.1)
            self.assertEqual(meta["source_disjoint_status"], "not_requested")
            self.assertEqual(
                [record.image_path for record in splits["train"]],
                [samples[i][0] for i in train_idx],
            )
            self.assertEqual(
                [record.image_path for record in splits["validation"]],
                [samples[i][0] for i in val_idx],
            )
            self.assertEqual(splits["test"], [])
            ds.verify_no_image_overlap(splits)
            report = ds.verify_no_product_overlap(splits)
            self.assertEqual(report["product_id_status"], "unavailable")
            source_report = ds.verify_no_source_leakage(splits)
            self.assertEqual(source_report["status"], "unavailable")

    def test_unknown_source_does_not_claim_source_disjointness(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_tree(root, ["A", "B", "C", "D"])
            records = ds.load_records(root)
            plain, _ = ds.split_records(records, "product_disjoint", 7, 0.25)
            sourced, meta = ds.split_records(records, "product_and_source_disjoint", 7, 0.25)
            self.assertEqual(meta["source_disjoint_status"], "unavailable")
            self.assertIn("UNKNOWN", meta["source_disjoint_reason"])
            self.assertEqual(
                [record.image_path for record in plain["train"]],
                [record.image_path for record in sourced["train"]],
            )

    def test_known_source_cannot_cross_splits(self) -> None:
        records = []
        for label, brand, source in (
            (0, "A", "ebay"),
            (0, "B", "ebay"),
            (0, "C", "chrono24"),
            (0, "D", "chrono24"),
            (1, "A", "ebay"),
            (1, "B", "stockx"),
            (1, "C", "chrono24"),
            (1, "D", "stockx"),
        ):
            path = Path(f"/data/Label_{label}/{brand}/0.jpg")
            records.append(
                ds.SampleRecord(path, label, brand, ds.UNKNOWN, ds.UNKNOWN, source, f"/data/Label_{label}/{brand}")
            )
        splits, meta = ds.split_records(records, "product_and_source_disjoint", 3, 0.5)
        self.assertEqual(meta["source_disjoint_status"], "enforced")
        ds.verify_no_source_leakage(splits)
        ds.verify_no_product_overlap(splits)

    def test_overlap_checks_raise(self) -> None:
        shared = ds.SampleRecord(
            Path("/tmp/same.jpg"), 0, "A", ds.UNKNOWN, ds.UNKNOWN, "ebay", "/tmp/A"
        )
        other = ds.SampleRecord(
            Path("/tmp/other.jpg"), 1, "B", "sku-1", ds.UNKNOWN, "ebay", "/tmp/B"
        )
        leaked = ds.SampleRecord(
            Path("/tmp/third.jpg"), 0, "C", "sku-1", ds.UNKNOWN, "ebay", "/tmp/A"
        )
        splits = {"train": [shared], "validation": [shared], "test": []}
        with self.assertRaises(RuntimeError):
            ds.verify_no_image_overlap(splits)
        splits = {"train": [other], "validation": [leaked], "test": []}
        with self.assertRaises(RuntimeError):
            ds.verify_no_product_overlap(splits)
        with self.assertRaises(RuntimeError):
            ds.verify_no_source_leakage(splits)

    def test_final_holdout_is_group_disjoint_and_not_trained(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_tree(root, ["A", "B", "C", "D"])
            records = ds.load_records(root)
            splits, _ = ds.split_records(records, "final_holdout", 42, 0.25, 0.25)
            self.assertGreater(len(splits["test"]), 0)
            self.assertGreater(len(splits["validation"]), 0)
            self.assertGreater(len(splits["train"]), 0)
            ds.verify_no_image_overlap(splits)
            ds.verify_no_product_overlap(splits)
            out = root / "out"
            payload = ds.write_split_manifest(
                out,
                records,
                splits,
                strategy="final_holdout",
                seed=42,
                val_fraction=0.25,
                test_fraction=0.25,
                split_meta={"source_disjoint_status": "not_requested", "source_disjoint_reason": ""},
                source_report={"status": "unavailable"},
                product_report={"product_id_status": "unavailable"},
            )
            saved = json.loads((out / "split_manifest.json").read_text())
            self.assertEqual(saved["products"], [ds.UNKNOWN])
            self.assertEqual(saved["sources"], [ds.UNKNOWN])
            self.assertEqual(saved["counts"]["test"], len(splits["test"]))
            self.assertEqual(payload["dataset_hash"], saved["dataset_hash"])
            train_groups = {record.split_group for record in splits["train"]}
            test_groups = {record.split_group for record in splits["test"]}
            self.assertFalse(train_groups & test_groups)


if __name__ == "__main__":
    unittest.main()
