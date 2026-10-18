"""Hard-negative mining stays off the final test and the frozen checkpoint."""

from __future__ import annotations

import inspect
import json
import sys
import tempfile
import unittest
import unittest.mock
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ml_rtx5080"))

import hard_negative as mining
from evaluation import evaluate_logits


def _sample(split: str, name: str, label: int, brand: str = "Richard Mille") -> dict:
    path = f"/tmp/hypevault-challenge/{split}/Label_{label}_Watches/{brand}/{name}.jpg"
    return {
        "path": path,
        "sample_id": f"{split}/{name}.jpg",
        "split": split,
        "label": label,
        "brand": brand,
    }


def _manifest(samples: list[dict] | None = None) -> dict:
    rows = samples or [
        _sample("train", "a", 0),
        _sample("calibration", "b", 1),
        _sample("validation", "c", 0),
        _sample("test", "d", 1),
    ]
    membership = {
        split: [row["path"] for row in rows if row["split"] == split]
        for split in ("train", "calibration", "validation", "test")
    }
    return {"membership": membership, "samples": rows}


def _signals() -> list[dict]:
    records = [
        {"sample_id": "near", "source_split": "train", "brand": "Richard Mille", "label": 1, "path": "/a"},
        {"sample_id": "sure-fake", "source_split": "calibration", "brand": "Patek Philippe", "label": 1, "path": "/b"},
        {"sample_id": "sure-auth", "source_split": "validation", "brand": "Richard Mille", "label": 0, "path": "/c"},
        {"sample_id": "wrong-auth", "source_split": "train", "brand": "Audemars Piguet", "label": 0, "path": "/d"},
    ]
    temperature = mining.FROZEN_TEMPERATURE
    logits = [
        temperature * float(np.log(0.51 / 0.49)),
        8.0,
        -8.0,
        8.0,
    ]
    return mining.build_signal_rows(records, logits)


class HardNegativeTests(unittest.TestCase):
    def test_final_test_samples_are_rejected(self) -> None:
        manifest = _manifest()
        manifest["membership"]["train"].append(manifest["membership"]["test"][0])
        with self.assertRaises(RuntimeError):
            mining.prepare_source_records(manifest)
        leaked = {
            "sample_id": "hidden",
            "source_split": "test",
            "brand": "Richard Mille",
            "label": 1,
            "path": "/test",
        }
        with self.assertRaises(RuntimeError):
            mining.build_signal_rows([leaked], [0.0])

    def test_low_margin_ranking_is_deterministic(self) -> None:
        rows = _signals()
        first = mining.rank_low_margin(rows, limit=3)
        second = mining.rank_low_margin(rows, limit=3)
        self.assertEqual([row["sample_id"] for row in first], [row["sample_id"] for row in second])
        self.assertEqual(first[0]["sample_id"], "near")
        self.assertLess(first[0]["margin"], first[1]["margin"])
        self.assertFalse(first[0]["misclassified"])
        self.assertNotIn("error", first[0])

    def test_high_confidence_filtering(self) -> None:
        _pool, selected = mining.select_high_confidence(_signals(), limit=10)
        chosen = {row["sample_id"] for row in selected}
        self.assertIn("sure-fake", chosen)
        self.assertIn("sure-auth", chosen)
        self.assertNotIn("near", chosen)
        self.assertNotIn("wrong-auth", chosen)
        self.assertTrue(all(not row["misclassified"] for row in selected))

    def test_same_brand_opposite_class_pairs(self) -> None:
        rows, embeddings = _pair_rows()
        pairs = mining.rank_opposite_pairs(rows, embeddings, same_brand=True, limit=10)
        self.assertTrue(pairs)
        self.assertTrue(all(row["brand_a"] == row["brand_b"] for row in pairs))
        self.assertTrue(all(row["label_a"] == 0 and row["label_b"] == 1 for row in pairs))
        self.assertNotIn("cross-fake", [row["sample_id_b"] for row in pairs])

    def test_cross_brand_pairs(self) -> None:
        rows, embeddings = _pair_rows()
        pairs = mining.rank_opposite_pairs(rows, embeddings, same_brand=False, limit=10)
        self.assertEqual([row["sample_id_b"] for row in pairs], ["cross-fake"])
        self.assertTrue(all(row["brand_a"] != row["brand_b"] for row in pairs))
        self.assertTrue(all(row["label_a"] != row["label_b"] for row in pairs))

    def test_cosine_similarity_ordering(self) -> None:
        rows, embeddings = _pair_rows()
        pairs = mining.rank_opposite_pairs(rows, embeddings, same_brand=True, limit=10)
        similarities = [row["cosine_similarity"] for row in pairs]
        self.assertEqual(similarities, sorted(similarities, reverse=True))
        self.assertEqual([row["sample_id_b"] for row in pairs], ["near-fake", "mid-fake", "far-fake"])
        self.assertGreater(similarities[0], similarities[1])
        self.assertGreater(similarities[1], similarities[2])

    def test_transformed_lineage_is_preserved(self) -> None:
        original = _signals()[0]
        measured = mining.transform_measurement(original, original["logit"] - 0.2, "mild_crop")
        self.assertEqual(measured["source_sample_id"], original["sample_id"])
        self.assertEqual(measured["source_split"], original["source_split"])
        self.assertEqual(measured["brand"], original["brand"])
        self.assertEqual(measured["label"], original["label"])
        self.assertEqual(measured["condition"], "mild_crop")

    def test_transformation_metrics_are_deterministic(self) -> None:
        image = Image.new("RGB", (32, 48), (20, 20, 20))
        image.paste(Image.new("RGB", (10, 12), (180, 30, 30)), (4, 6))
        before = list(image.getdata())
        for name in [spec["name"] for spec in mining.TRANSFORM_SPECS + mining.QUALITY_SPECS]:
            first = list(mining.apply_challenge_transform(image, name).getdata())
            second = list(mining.apply_challenge_transform(image, name).getdata())
            self.assertEqual(first, second)
            self.assertEqual(list(image.getdata()), before)
        original = _signals()[0]
        first_row = mining.transform_measurement(original, 1.25, "gaussian_blur")
        second_row = mining.transform_measurement(original, 1.25, "gaussian_blur")
        self.assertEqual(first_row, second_row)

    def test_probability_change(self) -> None:
        original = _signals()[2]
        transformed_logit = -1.0
        measured = mining.transform_measurement(original, transformed_logit, "brightness_shift")
        expected = float(torch.sigmoid(torch.tensor(transformed_logit / mining.FROZEN_TEMPERATURE)))
        self.assertAlmostEqual(
            measured["probability_change"],
            expected - original["calibrated_probability"],
        )
        self.assertAlmostEqual(
            measured["logit_change"],
            transformed_logit - original["logit"],
        )

    def test_far_uses_fake_denominator(self) -> None:
        labels = torch.tensor([0, 0, 1, 1, 1, 1])
        logits = torch.tensor([10.0, -10.0, -10.0, 10.0, 10.0, 10.0])
        metrics = evaluate_logits(logits, labels)
        self.assertEqual(metrics["false_authentic"], 1)
        self.assertEqual(metrics["fake_count"], 4)
        self.assertEqual(metrics["false_authentic_rate"], mining.false_authentic_rate(1, 4))
        self.assertEqual(mining.false_authentic_rate(1, 4), 0.25)
        self.assertIsNone(mining.false_authentic_rate(0, 0))

    def test_checkpoint_bytes_stay_unchanged(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = root / "epoch_018.pt"
            checkpoint.write_bytes(b"frozen-checkpoint-bytes")
            digest = mining.file_sha256(checkpoint)
            rows = _signals()
            mining.rank_low_margin(rows, limit=2)
            mining.select_high_confidence(rows, limit=2)
            self.assertEqual(checkpoint.read_bytes(), b"frozen-checkpoint-bytes")
            self.assertEqual(mining.file_sha256(checkpoint), digest)
            with self.assertRaises(Exception):
                mining._assert_challenge_destination(root / "checkpoints" / "hard_negative_v1")
        source = inspect.getsource(mining.collect_challenge_inference) + inspect.getsource(mining.run_phase22)
        self.assertNotIn("fit_temperature", source)
        self.assertNotIn("backward", source)
        self.assertNotIn("torch.save", source)

    def test_temperature_is_not_mutated(self) -> None:
        self.assertEqual(mining.FROZEN_TEMPERATURE, 0.24038200410185356)
        with self.assertRaises(ValueError):
            mining.build_signal_rows(_signals_records(), [0.0, 1.0, -1.0, 2.0], temperature=1.0)
        self.assertNotIn("fit_temperature", inspect.getsource(mining))

    def test_challenge_manifest_hash_is_reproducible(self) -> None:
        assembled = _assembled()
        first = mining.manifest_sha256(assembled["manifest"])
        second = mining.manifest_sha256(_assembled()["manifest"])
        self.assertEqual(first, second)
        with tempfile.TemporaryDirectory() as tmp:
            destination = mining.write_challenge_artifacts(
                Path(tmp) / "hard_negative_v1",
                assembled,
                metadata={"checkpoint_sha256": "abc", "split_manifest_sha256": "def"},
            )
            summary = json.loads((destination / "challenge_summary.json").read_text())
        self.assertEqual(summary["challenge_manifest_sha256"], first)
        self.assertIsNone(summary["production_threshold"])
        self.assertFalse(summary["temperature_changed"])
        self.assertEqual(summary["test_samples_used"], 0)

    def test_source_split_is_recorded(self) -> None:
        records, rejected = mining.prepare_source_records(_manifest())
        self.assertFalse(rejected)
        self.assertEqual({row["source_split"] for row in records}, {"train", "calibration", "validation"})
        manifest_path = (
            Path(__file__).resolve().parents[1]
            / "ml_rtx5080"
            / "experiments"
            / "dataset_audit"
            / "split_manifest_v2.json"
        )
        payload = json.loads(manifest_path.read_text())

        def reject_open(*_args, **_kwargs):
            raise AssertionError("challenge preparation opened an image")

        with unittest.mock.patch("PIL.Image.open", reject_open):
            real_records, real_rejected = mining.prepare_source_records(
                payload,
                enforce_authoritative_hash=True,
                require_files=False,
            )
        self.assertFalse(real_rejected)
        self.assertEqual(len(real_records), 20999 + 2998 + 2997)
        self.assertTrue(all(row["source_split"] in mining.SOURCE_SPLITS for row in real_records))
        test_paths = {str(Path(path).resolve()) for path in payload["membership"]["test"]}
        self.assertFalse(any(row["path"] in test_paths for row in real_records))
        for item in _assembled()["manifest"]["items"]:
            self.assertIn(item["source_split"], mining.SOURCE_SPLITS)

    def test_missing_or_invalid_samples_are_rejected_safely(self) -> None:
        rows = [
            _sample("train", "a", 0),
            _sample("calibration", "b", 1),
            _sample("validation", "c", 0),
            _sample("test", "d", 1),
        ]
        manifest = _manifest(rows)
        manifest["membership"]["train"].append("/tmp/hypevault-challenge/nowhere/Label_0_Watches/Richard Mille/missing.jpg")
        broken = _sample("validation", "bad", 0)
        broken["label"] = 5
        manifest["samples"].append(broken)
        manifest["membership"]["validation"].append(broken["path"])
        records, rejected = mining.prepare_source_records(manifest, require_files=True)
        reasons = {row["reason"] for row in rejected}
        self.assertIn("missing sample", reasons)
        self.assertIn("invalid label", reasons)
        self.assertIn("missing image file", reasons)
        self.assertFalse(any(row["sample_id"].endswith("bad.jpg") for row in records))
        self.assertTrue(all(row["source_split"] != "test" for row in records))
        with self.assertRaises(ValueError):
            mining.prepare_source_records({"samples": []})


def _signals_records() -> list[dict]:
    return [
        {"sample_id": "near", "source_split": "train", "brand": "Richard Mille", "label": 1, "path": "/a"},
        {"sample_id": "sure-fake", "source_split": "calibration", "brand": "Patek Philippe", "label": 1, "path": "/b"},
        {"sample_id": "sure-auth", "source_split": "validation", "brand": "Richard Mille", "label": 0, "path": "/c"},
        {"sample_id": "wrong-auth", "source_split": "train", "brand": "Audemars Piguet", "label": 0, "path": "/d"},
    ]


def _pair_rows() -> tuple[list[dict], np.ndarray]:
    rows = [
        _pair_row("auth", "Richard Mille", 0, 0.01),
        _pair_row("near-fake", "Richard Mille", 1, 0.99),
        _pair_row("mid-fake", "Richard Mille", 1, 0.80),
        _pair_row("far-fake", "Richard Mille", 1, 0.60),
        _pair_row("cross-fake", "Audemars Piguet", 1, 0.99),
    ]
    embeddings = np.asarray(
        [
            [1.0, 0.0],
            [1.0, 0.0],
            [0.6, 0.8],
            [0.0, 1.0],
            [0.99, 0.01],
        ],
        dtype=np.float64,
    )
    return rows, embeddings


def _pair_row(sample_id: str, brand: str, label: int, probability: float) -> dict:
    return {
        "sample_id": sample_id,
        "source_split": "train",
        "brand": brand,
        "label": label,
        "calibrated_probability": probability,
    }


def _assembly_inputs():
    records = _signals_records()
    temperature = mining.FROZEN_TEMPERATURE
    logits = np.asarray(
        [temperature * float(np.log(0.51 / 0.49)), 8.0, -8.0, 8.0],
        dtype=np.float32,
    )
    embeddings = np.eye(4, dtype=np.float32)
    return records, logits, embeddings


def _assembled() -> dict:
    records, logits, embeddings = _assembly_inputs()
    return mining.assemble_challenge(
        records,
        logits,
        embeddings,
        checkpoint="/tmp/epoch_018.pt",
        limits={
            "low_margin": 3,
            "high_confidence": 4,
            "same_brand_pairs": 2,
            "cross_brand_pairs": 2,
            "transform_originals": 1,
            "quality_originals": 1,
        },
    )


if __name__ == "__main__":
    unittest.main()
