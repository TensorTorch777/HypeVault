"""Feature-distance OOD scores, thresholds, and split isolation."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ml_rtx5080"))

import ood
from classifier import CLS_ONLY, build_authenticity_classifier
from ood_eval import main as ood_main


def _cluster() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(4)
    base = torch.tensor([1.0, 0.0, 0.0, 0.0])
    noise = torch.randn(6, 4, generator=generator) * 0.01
    near = base + noise
    far = torch.tensor(
        [
            [0.0, 8.0, 0.0, 0.0],
            [0.0, 0.0, 9.0, 0.0],
        ]
    )
    labels = torch.tensor([0, 0, 0, 1, 1, 1])
    return near, far, labels


class OodScoreTests(unittest.TestCase):
    def test_distant_features_score_higher_for_both_detectors(self) -> None:
        near, far, labels = _cluster()
        reference = ood.fit_reference(near, labels)
        for method in ("mahalanobis", "cosine"):
            near_scores = ood.score_features(near, reference, method)
            far_scores = ood.score_features(far, reference, method)
            self.assertGreater(float(far_scores.min()), float(near_scores.max()))
            self.assertTrue(torch.isfinite(near_scores).all())
            self.assertTrue(torch.isfinite(far_scores).all())

    def test_score_direction_and_threshold_are_deterministic(self) -> None:
        near, far, labels = _cluster()
        first = ood.fit_reference(near, labels)
        second = ood.fit_reference(near.clone(), labels.clone())
        self.assertTrue(torch.equal(first["gaussian"]["mean"], second["gaussian"]["mean"]))
        self.assertTrue(torch.equal(first["gaussian"]["variance"], second["gaussian"]["variance"]))
        self.assertTrue(torch.equal(first["l2_centroid"], second["l2_centroid"]))
        scores = ood.score_features(torch.cat((near, far)), first, "mahalanobis")
        flags_a = scores >= 1.0
        flags_b = scores >= 1.0
        self.assertTrue(torch.equal(flags_a, flags_b))
        self.assertEqual(ood.SCORE_DIRECTION, "larger_is_more_ood")
        self.assertEqual(ood.POSITIVE_CLASS, "ood")

    def test_degenerate_covariance_stays_finite(self) -> None:
        constant = torch.tensor([[2.0, -3.0], [2.0, -3.0], [2.0, -3.0]])
        reference = ood.fit_diagonal_gaussian(constant)
        self.assertEqual(reference["dimensions_floored"], 2)
        self.assertTrue(torch.all(reference["stabilized_variance"] == ood.VARIANCE_FLOOR))
        distance = ood.mahalanobis_distance(torch.tensor([[2.0, 50.0]]), reference)
        self.assertTrue(torch.isfinite(distance).all())
        self.assertGreater(float(distance), 0.0)
        same = ood.mahalanobis_distance(constant[:1], reference)
        self.assertAlmostEqual(float(same), 0.0, places=5)

    def test_non_finite_and_empty_inputs_fail(self) -> None:
        with self.assertRaises(ValueError):
            ood.fit_diagonal_gaussian(torch.tensor([[1.0, float("nan")]]))
        with self.assertRaises(ValueError):
            ood.fit_diagonal_gaussian(torch.ones(1, 3))
        with self.assertRaises(ValueError):
            ood.l2_normalize(torch.zeros(1, 4))
        metrics = ood.ranking_metrics(torch.tensor([0.2, 0.1]), torch.tensor([]))
        self.assertIsNone(metrics["roc_auc"])
        self.assertIsNone(metrics["average_precision"])

    def test_rates_and_auroc_use_ood_as_the_positive_class(self) -> None:
        id_scores = torch.tensor([0.1, 0.6, 0.7, 0.2])
        ood_scores = torch.tensor([0.2, 0.8, 0.9])
        point = ood.operating_point(id_scores, ood_scores, 0.5)
        self.assertAlmostEqual(point["id_ood_flag_rate"], 0.5)
        self.assertEqual(point["id_flagged"], 2)
        self.assertAlmostEqual(point["ood_false_accept_rate"], 1 / 3)
        self.assertAlmostEqual(point["ood_detection_rate"], 2 / 3)
        self.assertAlmostEqual(point["precision"], 0.5)
        self.assertAlmostEqual(point["recall"], 2 / 3)
        perfect = ood.ranking_metrics(torch.tensor([0.1, 0.2]), torch.tensor([0.8, 0.9]))
        self.assertAlmostEqual(perfect["roc_auc"], 1.0)
        self.assertAlmostEqual(perfect["average_precision"], 1.0)
        self.assertEqual(perfect["positive_class"], "ood")
        reversed_scores = ood.ranking_metrics(torch.tensor([0.8, 0.9]), torch.tensor([0.1, 0.2]))
        self.assertAlmostEqual(reversed_scores["roc_auc"], 0.0)

    def test_candidates_are_research_rows_and_production_threshold_stays_null(self) -> None:
        id_scores = torch.linspace(0.0, 1.0, 21)
        ood_scores = torch.linspace(0.4, 2.0, 11)
        rows = ood.threshold_candidates(id_scores, ood_scores)
        self.assertEqual(
            [row["strategy"] for row in rows],
            ["id_percentile_95", "id_percentile_99", "id_flag_rate_0_05", "id_flag_rate_0_01"],
        )
        for row in rows:
            self.assertIsNone(row["production_ood_threshold"])
            self.assertGreaterEqual(row["threshold"], 0.0)
            self.assertLessEqual(row["id_ood_flag_rate"], 1.0)
            self.assertAlmostEqual(
                row["ood_detection_rate"] + row["ood_false_accept_rate"],
                1.0,
            )
        again = ood.threshold_candidates(id_scores, ood_scores)
        self.assertEqual([row["threshold"] for row in rows], [row["threshold"] for row in again])
        id_only = ood.threshold_candidates(id_scores, None)
        self.assertIsNone(id_only[0]["ood_false_accept_rate"])
        self.assertEqual(ood.threshold_candidates(torch.tensor([]), ood_scores), [])

    def test_quadrants_keep_authenticity_and_ood_separate(self) -> None:
        types = ["id", "id", "ood", "ood"]
        logits = torch.tensor([-4.0, 4.0, -4.0, 4.0])
        labeled = ood.labeled_quadrants(types, logits)
        counts = {(cell["dataset"], cell["authenticity_prediction"]): cell["count"] for cell in labeled["cells"]}
        self.assertEqual(counts[("id", "authentic")], 1)
        self.assertEqual(counts[("id", "fake")], 1)
        self.assertEqual(counts[("ood", "authentic")], 1)
        self.assertEqual(counts[("ood", "fake")], 1)
        self.assertIsNone(labeled["production_verdict"])
        danger = ood.strong_authentic_ood(torch.tensor([-4.0, 4.0]))
        self.assertEqual(danger["count"], 1)
        self.assertAlmostEqual(danger["percent_of_ood"], 0.5)
        self.assertFalse(danger["uses_ood_score"])
        flagged = ood.signal_quadrants(types, torch.tensor([0.1, 0.2, 0.8, 0.9]), logits, 0.5)
        self.assertEqual(flagged["basis"], "ood_flag")
        self.assertIsNone(flagged["production_verdict"])

    def test_reference_split_rejects_validation_and_test(self) -> None:
        for split in ("validation", "calibration", "test"):
            with self.assertRaises(ValueError):
                ood.assert_reference_split(split)
        self.assertEqual(ood.assert_reference_split("train"), "train")
        with self.assertRaises(ValueError):
            ood.assert_development_split("test")
        with self.assertRaises(ValueError):
            ood.assert_development_split("train")
        train = "/tmp/Label_0_Watches/Brand/a.jpg"
        with self.assertRaises(ValueError):
            ood.reject_ood_overlap([train], {train})

    def test_module_does_not_select_checkpoints_or_download_a_corpus(self) -> None:
        source = Path(ood.__file__).read_text()
        self.assertNotIn("select_checkpoints", source)
        self.assertNotIn("update_selection", source)
        self.assertNotIn("torch.hub", source)
        self.assertNotIn("torchvision.datasets", source)
        self.assertNotIn("fit_temperature", source)
        self.assertEqual(
            ood.NO_CORPUS_STATUS,
            "Real OOD corpus not available; OOD performance not production-validated.",
        )


class OodExtractionTests(unittest.TestCase):
    def test_cls_excludes_registers_and_leaves_weights_and_logits_unchanged(self) -> None:
        class _Backbone(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.scale = nn.Parameter(torch.tensor(1.0))

            def forward_features(self, images: torch.Tensor) -> torch.Tensor:
                batch = images.shape[0]
                tokens = torch.zeros(batch, 6, 4, device=images.device)
                tokens[:, 0, 0] = self.scale * 2
                tokens[:, 1:5] = 9
                tokens[:, 5] = 3
                return tokens

            def get_cls_token(self, features: torch.Tensor) -> torch.Tensor:
                return features[:, 0]

            def get_patch_tokens(self, features: torch.Tensor) -> torch.Tensor:
                return features[:, 1 + 4 :]

        class _Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.backbone = _Backbone()
                self.classifier_arch = "cls_only"
                self.trunk = nn.Linear(4, 1)

        model = _Model().eval()
        before = {name: value.detach().clone() for name, value in model.state_dict().items()}
        images = torch.zeros(2, 3, 8, 8)
        with torch.inference_mode():
            first_cls, first_logit = ood._cls_and_logit(model, images)
            second_cls, second_logit = ood._cls_and_logit(model, images)
        self.assertTrue(torch.equal(first_cls, second_cls))
        self.assertTrue(torch.equal(first_logit, second_logit))
        self.assertTrue(torch.all(first_cls[:, 0] == 2))
        self.assertFalse(torch.any(first_cls == 9))
        for name, value in model.state_dict().items():
            self.assertTrue(torch.equal(value, before[name]))
        self.assertFalse(first_cls.requires_grad)
        self.assertFalse(first_logit.requires_grad)


class OodCliTests(unittest.TestCase):
    def test_missing_corpus_is_not_validated_and_threshold_stays_null(self) -> None:
        root = Path(__file__).resolve().parents[1] / "ml_rtx5080" / "experiments"
        root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=root) as tmp:
            experiment = Path(tmp)
            authentic = experiment / "images" / "Label_0_Watches" / "CalBrand"
            fake = experiment / "images" / "Label_1_Watches" / "CalBrand"
            held_out = experiment / "images" / "Label_1_Watches" / "TestBrand"
            for directory in (authentic, fake, held_out):
                directory.mkdir(parents=True)
            Image.new("RGB", (24, 24), (30, 30, 30)).save(authentic / "a.png")
            Image.new("RGB", (24, 28), (180, 20, 20)).save(fake / "b.png")
            Image.new("RGB", (24, 24), (20, 20, 180)).save(held_out / "t.png")
            train_paths = [
                str((authentic / "a.png").resolve()),
                str((fake / "b.png").resolve()),
            ]
            test_path = str((held_out / "t.png").resolve())
            manifest = experiment / "split_manifest.json"
            manifest.write_text(
                json.dumps({"membership": {"train": train_paths, "validation": [], "test": [test_path]}})
            )
            manifest_bytes = manifest.read_bytes()
            model = build_authenticity_classifier("dinov2", CLS_ONLY, pretrained=False)
            checkpoint = experiment / "probe.pt"
            torch.save({"model_state": model.state_dict()}, checkpoint)
            checkpoint_bytes = checkpoint.read_bytes()
            with self.assertRaises(ValueError):
                ood_main(
                    [
                        "--checkpoint",
                        str(checkpoint),
                        "--split_manifest",
                        str(manifest),
                        "--output_dir",
                        str(experiment),
                        "--model",
                        "dinov2",
                        "--head",
                        "cls_only",
                        "--id_split",
                        "test",
                    ]
                )
            ood_main(
                [
                    "--checkpoint",
                    str(checkpoint),
                    "--split_manifest",
                    str(manifest),
                    "--output_dir",
                    str(experiment),
                    "--model",
                    "dinov2",
                    "--head",
                    "cls_only",
                    "--id_split",
                    "train",
                    "--method",
                    "both",
                    "--batch_size",
                    "1",
                    "--device",
                    "cpu",
                ]
            )
            self.assertEqual(manifest.read_bytes(), manifest_bytes)
            self.assertEqual(checkpoint.read_bytes(), checkpoint_bytes)
            summary = json.loads((experiment / "ood" / "ood_summary.json").read_text())
            config = json.loads((experiment / "ood" / "ood_config.json").read_text())
            self.assertEqual(summary["validation_status"], ood.NO_CORPUS_STATUS)
            self.assertFalse(summary["real_ood_corpus_available"])
            self.assertFalse(summary["ood_performance_production_validated"])
            self.assertIsNone(summary["production_ood_threshold"])
            self.assertFalse(summary["checkpoint_selection_rerun"])
            self.assertFalse(summary["scores_combined_with_authenticity"])
            self.assertFalse(summary["calibration_temperature_used"])
            self.assertEqual(summary["reference_split"], "train")
            self.assertFalse(summary["final_evaluation_enabled"])
            self.assertEqual(config["mahalanobis_representation"], "raw_cls")
            self.assertEqual(config["cosine_representation"], "l2_normalized_cls")
            self.assertEqual(config["covariance"], "diagonal")
            self.assertTrue(config["registers_excluded"])
            self.assertIsNone(summary["methods"]["mahalanobis"]["reported_ranking"])
            scores = (experiment / "ood" / "id_scores.csv").read_text()
            self.assertIn(train_paths[0], scores)
            self.assertNotIn(test_path, scores)
            self.assertFalse((experiment / "ood" / "ood_scores.csv").exists())
            forbidden = Path(__file__).resolve().parents[1] / "ml_rtx5080" / "checkpoints"
            self.assertFalse((forbidden / "ood").exists())


if __name__ == "__main__":
    unittest.main()
