"""Real-corpus OOD validation stays off the final test and a production threshold."""

from __future__ import annotations

import inspect
import json
import sys
import tempfile
import unittest
from pathlib import Path

import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ml_rtx5080"))

import ood
import ood_real


def _image(path: Path, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8), color).save(path)


def _manifest(root: Path, sha: str = "abc") -> Path:
    payload = {
        "membership_hash": ood_real.MEMBERSHIP_HASH,
        "membership": {
            "train": [str(root / "catalog-train.jpg")],
            "calibration": [str(root / "catalog-calibration.jpg")],
            "validation": [str(root / "catalog-validation.jpg")],
            "test": [str(root / "catalog-final.jpg")],
        },
        "samples": [
            {"sample_id": "catalog-train.jpg", "path": str(root / "catalog-train.jpg"), "sha256": sha},
        ],
    }
    path = root / "split_manifest.json"
    path.write_text(json.dumps(payload))
    return path


class OodRealTests(unittest.TestCase):
    def test_exact_duplicate_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            kept = root / "electronics" / "radio.png"
            copy = root / "electronics" / "radio-copy.png"
            blocked = root / "shoes" / "runner.png"
            _image(kept, (20, 20, 20))
            _image(copy, (20, 20, 20))
            _image(blocked, (30, 40, 50))
            digest = ood_real.file_sha256(kept)
            accepted, rejected = ood_real.collect_ood_images(
                root,
                catalog_sha256={digest},
                blocked_paths={str(blocked.resolve())},
            )
            self.assertEqual(accepted, [])
            reasons = {row["sample_id"]: row["rejection_reason"] for row in rejected}
            self.assertEqual(reasons["electronics/radio.png"], "hypevault_sha256_collision")
            self.assertEqual(reasons["electronics/radio-copy.png"], "hypevault_sha256_collision")
            self.assertEqual(reasons["shoes/runner.png"], "split_membership_overlap")

    def test_ood_manifest_is_deterministic(self) -> None:
        samples = [
            {"sample_id": "b.png", "path": "/ood/b.png", "category": "food", "dataset_type": "OOD", "sha256": "2", "ood_split": "OOD_FINAL"},
            {"sample_id": "a.png", "path": "/ood/a.png", "category": "food", "dataset_type": "OOD", "sha256": "1", "ood_split": "OOD_DEVELOPMENT"},
        ]
        first = ood_real.build_manifest(samples, [], available=True)
        second = ood_real.build_manifest(list(reversed(samples)), [], available=True)
        self.assertEqual(json.dumps(first, sort_keys=True), json.dumps(second, sort_keys=True))
        self.assertEqual([row["sample_id"] for row in first["samples"]], ["a.png", "b.png"])
        self.assertTrue(all(row["dataset_type"] == "OOD" for row in first["samples"]))

    def test_id_is_negative_and_ood_is_positive(self) -> None:
        self.assertEqual(ood.POSITIVE_CLASS, "ood")
        self.assertEqual(ood.NEGATIVE_CLASS, "in_distribution")
        separated = ood.ranking_metrics(torch.tensor([0.1, 0.2]), torch.tensor([0.8, 0.9]))
        reversed_scores = ood.ranking_metrics(torch.tensor([0.8, 0.9]), torch.tensor([0.1, 0.2]))
        self.assertEqual(separated["positive_class"], "ood")
        self.assertAlmostEqual(separated["roc_auc"], 1.0)
        self.assertAlmostEqual(reversed_scores["roc_auc"], 0.0)

    def test_larger_score_means_more_ood(self) -> None:
        ood_real.assert_score_direction()
        self.assertEqual(ood.SCORE_DIRECTION, "larger_is_more_ood")

    def test_threshold_candidates_are_deterministic(self) -> None:
        generator = torch.Generator().manual_seed(5)
        id_scores = torch.rand(80, generator=generator)
        ood_scores = torch.rand(30, generator=generator) + 1.0
        first = ood_real.development_operating_points(id_scores, ood_scores)
        second = ood_real.development_operating_points(id_scores, ood_scores)
        self.assertEqual(
            [row["strategy"] for row in first],
            ["id_flag_rate_0_01", "id_flag_rate_0_02", "id_flag_rate_0_05", "id_percentile_95", "id_percentile_99"],
        )
        self.assertEqual(first, second)
        for row in first:
            self.assertIsNone(row["production_ood_threshold"])
            self.assertEqual(row["role"], "research_ood_threshold_candidate")
        chosen = ood_real.select_research_threshold(first)
        again = ood_real.select_research_threshold(list(reversed(first)))
        self.assertEqual(chosen["strategy"], again["strategy"])
        self.assertEqual(chosen["threshold"], again["threshold"])
        self.assertIsNone(chosen["production_ood_threshold"])

    def test_id_flag_rate(self) -> None:
        scores = torch.tensor([0.1, 0.6, 0.7, 0.2])
        self.assertAlmostEqual(ood_real.id_ood_flag_rate(scores, 0.5), 0.5)
        point = ood.operating_point(scores, torch.tensor([0.8]), 0.5)
        self.assertAlmostEqual(point["id_ood_flag_rate"], 0.5)

    def test_ood_false_accept_rate(self) -> None:
        scores = torch.tensor([0.2, 0.8, 0.9])
        self.assertAlmostEqual(ood_real.ood_false_accept_rate(scores, 0.5), 1 / 3)
        point = ood.operating_point(torch.tensor([0.1, 0.2]), scores, 0.5)
        self.assertAlmostEqual(point["ood_false_accept_rate"], 1 / 3)
        self.assertAlmostEqual(point["ood_detection_rate"] + point["ood_false_accept_rate"], 1.0)

    def test_development_and_final_ood_splits_stay_isolated(self) -> None:
        samples = [{"sample_id": f"{index:02d}.png", "path": f"/ood/{index:02d}.png", "category": "vehicles"} for index in range(20)]
        development, final, status = ood_real.split_development_final(list(reversed(samples)))
        again_dev, again_final, again_status = ood_real.split_development_final(samples)
        self.assertEqual(status, "split")
        self.assertEqual(again_status, "split")
        self.assertEqual([row["sample_id"] for row in development], [row["sample_id"] for row in again_dev])
        self.assertEqual([row["sample_id"] for row in final], [row["sample_id"] for row in again_final])
        self.assertEqual(len(development), 16)
        self.assertEqual(len(final), 4)
        self.assertFalse({row["sample_id"] for row in development} & {row["sample_id"] for row in final})
        small_dev, small_final, small_status = ood_real.split_development_final(samples[:19])
        self.assertEqual(small_status, "corpus_too_small")
        self.assertEqual(small_dev, [])
        self.assertEqual(small_final, [])
        scored = ood_real._score_explicit_corpus(
            load_batch=_separated_batch,
            train_paths=[f"/id/train/{index}.png" for index in range(8)],
            validation_paths=[f"/id/validation/{index}.png" for index in range(40)],
            development=development,
            final=final,
            thresholds=None,
        )
        self.assertEqual(scored["scoring_stages"], ["train", "validation", "ood_development", "ood_final"])
        self.assertEqual(scored["reference_fit_splits"], ["train"])
        self.assertFalse(scored["reference_refit_on_ood"])
        self.assertIsNone(scored["production_ood_threshold"])
        primary = scored["methods"]["mahalanobis"]
        self.assertFalse(primary["held_out"]["used_for_threshold_selection"])
        self.assertFalse(primary["held_out"]["threshold_changed_after_freeze"])
        self.assertEqual(primary["held_out"]["threshold"], scored["research_ood_threshold"]["mahalanobis"]["threshold"])

    def test_final_test_paths_are_rejected(self) -> None:
        with self.assertRaises(RuntimeError):
            ood_real.screen_ood_paths(
                ["/catalog/final.jpg"],
                {"train": set(), "calibration": set(), "validation": set(), "test": {"/catalog/final.jpg"}},
            )

    def test_checkpoint_bytes_stay_unchanged(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = root / "epoch_018.pt"
            checkpoint.write_bytes(b"frozen-ood-phase")
            manifest = _manifest(root)
            destination = root / "ood_v1"
            ood_real.run_phase25(
                root,
                manifest,
                ood_root=None,
                checkpoint=checkpoint,
                output_dir=destination,
                enforce_expected_sha=False,
            )
            self.assertEqual(checkpoint.read_bytes(), b"frozen-ood-phase")
        source = inspect.getsource(ood_real)
        self.assertNotIn("torch.save", source)
        self.assertNotIn("fit_temperature", source)
        self.assertNotIn("backward", source)

    def test_temperature_is_unchanged(self) -> None:
        self.assertEqual(ood_real.FROZEN_TEMPERATURE, 0.24038200410185356)
        self.assertNotIn("fit_temperature", inspect.getsource(ood_real.run_phase25))
        probability = float(ood_real.calibrated_p_fake(torch.tensor([0.0])))
        self.assertAlmostEqual(probability, 0.5)

    def test_production_ood_threshold_stays_null(self) -> None:
        candidates = ood_real.development_operating_points(torch.linspace(0.0, 1.0, 50), torch.linspace(0.5, 2.0, 20))
        chosen = ood_real.select_research_threshold(candidates)
        self.assertIsNone(chosen["production_ood_threshold"])
        self.assertEqual(ood_real.PRODUCTION_QUALITY_GATE, None)
        self.assertEqual(ood_real.RESEARCH_QUALITY_CANDIDATE, "low_resolution_or_low_contrast")

    def test_missing_corpus_is_reported_honestly(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = root / "epoch_018.pt"
            checkpoint.write_bytes(b"frozen-ood-phase")
            summary = ood_real.run_phase25(
                root,
                _manifest(root),
                ood_root=None,
                checkpoint=checkpoint,
                output_dir=root / "ood_v1",
                enforce_expected_sha=False,
            )
        self.assertFalse(summary["REAL_OOD_CORPUS_AVAILABLE"])
        self.assertFalse(summary["ood_performance_claimed"])
        self.assertIsNone(summary["production_ood_threshold"])
        self.assertIsNone(summary["research_ood_threshold"])
        self.assertEqual(summary["research_threshold_status"], "not_selected_corpus_unavailable")
        self.assertEqual(summary["final_test_samples_used"], 0)
        self.assertFalse(summary["training_performed"])
        self.assertFalse(summary["reference_fit_performed"])
        self.assertEqual(summary["reference_fit_splits"], [])
        self.assertEqual(summary["reference_excluded_splits"], ["validation", "calibration", "test"])
        self.assertIsNone(summary["methods"]["mahalanobis"]["roc_auc"])
        self.assertIsNone(summary["methods"]["cosine"]["roc_auc"])
        self.assertEqual(summary["membership_hash"], "78a38ebac1102977ec5598577126ade77fed8edd6563e13dbdee73dba389f678")
        self.assertIn("manifest_file_sha256", summary)
        self.assertIn("ood_manifest_sha256", summary)
        self.assertIn("checkpoint_sha256", summary)
        self.assertNotIn("manifest_sha256", summary)
        self.assertNotIn("manifest_hash", summary)
        self.assertEqual(summary["temperature"], 0.24038200410185356)


def _separated_batch(stage: str, paths: list[str], labeled: bool) -> dict:
    if stage in {"test", "calibration"}:
        raise RuntimeError(stage)
    if stage in {"train", "validation"}:
        base = torch.tensor([1.0, 0.0, 0.0, 0.0])
        scale = 0.05
    else:
        base = torch.tensor([0.0, 12.0, 0.0, 0.0])
        scale = 0.01
    features = torch.stack([base + torch.tensor([index * scale, 0.0, 0.0, 0.0]) for index, _path in enumerate(paths)])
    labels = [index % 2 for index in range(len(paths))] if labeled else [-1 for _path in paths]
    return {
        "features": features,
        "logits": torch.linspace(-2.0, 2.0, len(paths)),
        "sample_ids": list(paths),
        "labels": torch.tensor(labels, dtype=torch.long),
        "brands": ["Richard Mille" if labeled else "" for _path in paths],
    }


if __name__ == "__main__":
    unittest.main()
