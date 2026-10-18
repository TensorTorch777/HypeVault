"""Image-quality checks stay separate from authenticity and off the final test."""

from __future__ import annotations

import inspect
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ml_rtx5080"))

import image_quality as quality


def _thresholds() -> dict:
    return {
        "cutoffs": {
            "min_dimension_p5": 100.0,
            "sharpness_p5": 1.0,
            "mean_luminance_p5": 0.2,
            "mean_luminance_p95": 0.8,
            "luminance_std_p5": 0.05,
            "aspect_ratio_p5": 0.75,
            "aspect_ratio_p95": 1.5,
            "near_black_fraction_p95": 0.1,
            "near_white_fraction_p95": 0.1,
            "foreground_occupancy_p5": 0.05,
        }
    }


def _edge_image(size: int = 48) -> Image.Image:
    image = Image.new("RGB", (size, size), (0, 0, 0))
    image.paste(Image.new("RGB", (size // 2, size), (255, 255, 255)), (size // 2, 0))
    return image


class ImageQualityTests(unittest.TestCase):
    def test_feature_extraction_is_deterministic(self) -> None:
        image = _edge_image()
        first = quality.extract_quality_features(image, file_bytes=128, image_format="PNG")
        second = quality.extract_quality_features(image, file_bytes=128, image_format="PNG")
        self.assertEqual(first, second)
        self.assertEqual(set(first), set(quality.FEATURE_SCHEMA))

    def test_dimensions_drop_after_downsampling(self) -> None:
        image = Image.new("RGB", (40, 25), (30, 90, 140))
        features = quality.extract_quality_features(image)
        self.assertEqual(features["width"], 40)
        self.assertEqual(features["height"], 25)
        self.assertEqual(features["pixel_count"], 1000)
        self.assertEqual(features["min_dimension"], 25)
        small = image.resize((10, 6), Image.Resampling.BICUBIC)
        reduced = quality.extract_quality_features(small)
        self.assertLess(reduced["min_dimension"], features["min_dimension"])
        self.assertLess(reduced["pixel_count"], features["pixel_count"])

    def test_luminance_matches_bt601_and_falls_when_darkened(self) -> None:
        red = quality.extract_quality_features(Image.new("RGB", (12, 12), (255, 0, 0)))
        self.assertAlmostEqual(red["mean_luminance"], 0.299)
        black = quality.extract_quality_features(Image.new("RGB", (12, 12), (0, 0, 0)))
        self.assertEqual(black["mean_luminance"], 0.0)
        self.assertEqual(black["dark_pixel_fraction"], 1.0)
        bright = _edge_image()
        darker = ImageEnhance.Brightness(bright).enhance(0.25)
        self.assertLess(
            quality.extract_quality_features(darker)["mean_luminance"],
            quality.extract_quality_features(bright)["mean_luminance"],
        )

    def test_blur_lowers_sharpness(self) -> None:
        sharp = _edge_image()
        blurred = sharp.filter(ImageFilter.GaussianBlur(radius=4))
        sharp_score = quality.extract_quality_features(sharp)["sharpness_laplacian_variance"]
        blur_score = quality.extract_quality_features(blurred)["sharpness_laplacian_variance"]
        self.assertGreater(sharp_score, blur_score)
        flat = quality.extract_quality_features(Image.new("RGB", (16, 16), (40, 40, 40)))
        self.assertEqual(flat["sharpness_laplacian_variance"], 0.0)

    def test_contrast_is_luminance_spread(self) -> None:
        flat = quality.extract_quality_features(Image.new("RGB", (16, 16), (80, 80, 80)))
        varied = quality.extract_quality_features(_edge_image(32))
        self.assertEqual(flat["luminance_std"], 0.0)
        self.assertEqual(flat["luminance_percentile_spread"], 0.0)
        self.assertGreater(varied["luminance_std"], flat["luminance_std"])
        self.assertGreater(varied["luminance_percentile_spread"], 0.0)

    def test_clipping_fractions(self) -> None:
        image = Image.new("RGB", (10, 10), (0, 0, 0))
        image.paste(Image.new("RGB", (4, 10), (255, 255, 255)), (6, 0))
        features = quality.extract_quality_features(image)
        self.assertAlmostEqual(features["near_black_fraction"], 0.6)
        self.assertAlmostEqual(features["near_white_fraction"], 0.4)
        self.assertAlmostEqual(features["dark_pixel_fraction"], 0.6)
        self.assertAlmostEqual(features["bright_pixel_fraction"], 0.4)

    def test_aspect_ratio_and_crop_change_geometry(self) -> None:
        image = Image.new("RGB", (40, 25), (20, 20, 20))
        features = quality.extract_quality_features(image)
        self.assertAlmostEqual(features["aspect_ratio"], 40 / 25)
        cropped = image.crop((0, 0, 15, 25))
        cropped_features = quality.extract_quality_features(cropped)
        self.assertNotEqual(cropped_features["width"], features["width"])
        self.assertNotEqual(cropped_features["aspect_ratio"], features["aspect_ratio"])
        self.assertLess(cropped_features["pixel_count"], features["pixel_count"])

    def test_threshold_flags_are_deterministic(self) -> None:
        features = quality.extract_quality_features(Image.new("RGB", (80, 40), (5, 5, 5)))
        features["min_dimension"] = 40
        features["sharpness_laplacian_variance"] = 0.0
        features["mean_luminance"] = 0.01
        features["luminance_std"] = 0.0
        features["aspect_ratio"] = 2.0
        features["near_black_fraction"] = 0.9
        features["near_white_fraction"] = 0.0
        features["foreground_occupancy"] = 0.0
        first = quality.apply_quality_flags(features, _thresholds())
        second = quality.apply_quality_flags(features, _thresholds())
        self.assertEqual(first, second)
        self.assertEqual(
            first,
            [
                "LOW_RESOLUTION",
                "BLURRY",
                "TOO_DARK",
                "LOW_CONTRAST",
                "EXTREME_ASPECT_RATIO",
                "EXCESSIVE_CLIPPING",
                "SUSPICIOUS_FRAMING",
            ],
        )
        self.assertEqual(quality.quality_state(first), "degraded")
        self.assertEqual(quality.quality_state([]), "clean")
        clean = quality.extract_quality_features(Image.new("RGB", (120, 120), (100, 100, 100)))
        clean.update(
            {
                "min_dimension": 120,
                "sharpness_laplacian_variance": 5.0,
                "mean_luminance": 0.4,
                "luminance_std": 0.2,
                "aspect_ratio": 1.0,
                "near_black_fraction": 0.0,
                "near_white_fraction": 0.0,
                "foreground_occupancy": 0.5,
            }
        )
        self.assertEqual(quality.apply_quality_flags(clean, _thresholds()), [])
        only_authentic = [
            {
                "label": 0,
                "authenticity_prediction": "authentic",
                "quality_state": "clean",
                "quality_flags": [],
            }
        ]
        metrics = quality.classification_by_quality(only_authentic)["clean"]
        self.assertIsNone(metrics["false_authentic_rate"])
        self.assertEqual(metrics["false_fake_rate"], 0.0)

    def test_reference_stats_are_descriptive(self) -> None:
        rows = []
        for value in range(1, 21):
            rows.append({"min_dimension": value, "mean_luminance": value / 20.0, "file_bytes": None})
        for name in quality.NUMERIC_FEATURES:
            if name not in rows[0]:
                for row in rows:
                    row[name] = float(value_for(name, row))
        stats = quality.reference_distribution(rows)
        self.assertEqual(stats["min_dimension"]["count"], 20)
        self.assertAlmostEqual(stats["min_dimension"]["mean"], 10.5)
        self.assertAlmostEqual(stats["min_dimension"]["median"], float(np.percentile(np.arange(1, 21), 50)))
        self.assertEqual(stats["file_bytes"]["count"], 0)
        self.assertIsNone(stats["file_bytes"]["mean"])
        thresholds = quality.derive_thresholds(stats, 20)
        self.assertEqual(thresholds["cutoffs"]["min_dimension_p5"], stats["min_dimension"]["p5"])
        self.assertIsNone(thresholds["production_threshold"])
        self.assertFalse(thresholds["final_test_used"])
        with self.assertRaises(ValueError):
            quality.derive_thresholds(stats, 5)

    def test_final_test_paths_are_rejected(self) -> None:
        test_path = "/tmp/hypevault-quality/test/Label_1_Watches/Richard Mille/hidden.jpg"
        with self.assertRaises(RuntimeError):
            quality.reject_final_test_paths([test_path], {str(Path(test_path).resolve())})
        sample = {
            "path": test_path,
            "sample_id": "test/hidden.jpg",
            "split": "test",
            "label": 1,
            "brand": "Richard Mille",
        }
        manifest = {
            "membership": {"train": [test_path], "calibration": [], "validation": [], "test": [test_path]},
            "samples": [{**sample, "split": "train"}],
        }
        with self.assertRaises(RuntimeError):
            quality.prepare_source_records(manifest)

    def test_feature_schema_stays_separate_from_authenticity(self) -> None:
        features = quality.extract_quality_features(_edge_image())
        self.assertEqual(tuple(features), quality.FEATURE_SCHEMA)
        self.assertNotIn("authenticity_probability", features)
        self.assertNotIn("authenticity_logit", features)
        self.assertTrue(quality.THRESHOLD_RULES["not_fit_to_f1"])

    def test_transformed_comparison_records_geometry_change(self) -> None:
        image = _edge_image(64)
        before = quality.extract_quality_features(image)
        after = quality.extract_quality_features(quality.apply_challenge_transform(image, "mild_crop"))
        delta = quality.feature_delta(before, after)
        self.assertLess(delta["pixel_count"], 0)
        self.assertNotEqual(delta["width"], 0)
        table = quality.robustness_table(
            [
                _stress("lower_resolution", flip=1, probability=1.0, change=0.9),
                _stress("mild_crop", flip=0, probability=0.01, change=0.0),
            ]
        )
        ranked = quality.rank_failure_modes(table)
        self.assertEqual(ranked[0]["condition"], "lower_resolution")
        self.assertEqual(ranked[0]["false_fake_count"], 1)
        self.assertFalse(ranked[0]["model_updated"])

    def test_missing_and_corrupt_images_are_rejected(self) -> None:
        missing = Path("/tmp/hypevault-quality-missing/nope.jpg")
        with self.assertRaises(quality.QualityReadError):
            quality.extract_quality_features_from_path(missing)
        with tempfile.TemporaryDirectory() as tmp:
            corrupt = Path(tmp) / "broken.jpg"
            corrupt.write_bytes(b"this is not an image")
            with self.assertRaises(quality.QualityReadError):
                quality.extract_quality_features_from_path(corrupt)
            unsupported = Path(tmp) / "notes.txt"
            unsupported.write_text("hello")
            with self.assertRaises(quality.QualityReadError):
                quality.extract_quality_features_from_path(unsupported)

    def test_checkpoint_bytes_stay_unchanged(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "epoch_018.pt"
            checkpoint.write_bytes(b"frozen-quality-checkpoint")
            digest = quality.file_sha256(checkpoint)
            quality.extract_quality_features(_edge_image())
            self.assertEqual(checkpoint.read_bytes(), b"frozen-quality-checkpoint")
            self.assertEqual(quality.file_sha256(checkpoint), digest)
        source = inspect.getsource(quality.run_phase23)
        self.assertNotIn("torch.save", source)
        self.assertNotIn("fit_temperature", source)

    def test_temperature_is_not_mutated(self) -> None:
        self.assertEqual(quality.FROZEN_TEMPERATURE, 0.24038200410185356)
        self.assertNotIn("fit_temperature", inspect.getsource(quality))
        self.assertIsNone(quality.THRESHOLD_RULES.get("production_authenticity_threshold"))


def value_for(name: str, row: dict) -> float:
    if name == "min_dimension":
        return float(row["min_dimension"])
    if name == "mean_luminance":
        return float(row["mean_luminance"])
    return float(row["min_dimension"])


def _stress(condition: str, *, flip: int, probability: float, change: float) -> dict:
    return {
        "condition": condition,
        "prediction_flip": flip,
        "label": 0,
        "transformed_calibrated_probability": probability,
        "probability_change": change,
        "sample_id": condition,
    }


if __name__ == "__main__":
    unittest.main()
