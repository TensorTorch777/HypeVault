"""Phase 33 fast quality gate. The benchmark runner is not part of discovery."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

REPO = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(REPO / "ml_rtx5080"))
sys.path.insert(0, str(REPO / "backend"))


class QualityFastPathTests(unittest.TestCase):
    def test_reference_parity_and_decision(self) -> None:
        from image_quality import apply_quality_flags, extract_quality_features
        from policy_contract import INPUT_SIZE, PHASE23_CUTOFFS, PREPROCESSING_VERSION, PolicyEngine, frozen_shadow_configuration
        from quality_fastpath import decision_flags, fast_quality_gate, resolution_probe

        image = resolution_probe(480, 700)
        reference = extract_quality_features(image)
        reference_flags = decision_flags(apply_quality_flags(reference, {"cutoffs": dict(PHASE23_CUTOFFS)}))
        fast_features, fast_flags = fast_quality_gate(image)
        self.assertEqual(fast_flags, reference_flags)
        self.assertEqual(fast_features["min_dimension"], reference["min_dimension"])
        self.assertEqual(fast_features["luminance_std"], reference["luminance_std"])
        engine = PolicyEngine(frozen_shadow_configuration())
        kwargs = {
            "logit": -1.0,
            "preprocessing_version": PREPROCESSING_VERSION,
            "input_size": INPUT_SIZE,
            "checkpoint_sha256": "5a38c93fd442b03653c65d2a5ecc9c2687ef152f7c5c020763e4ce1fd9c7d28f",
        }
        old = engine.evaluate(quality_features=reference, quality_flags=reference_flags, **kwargs)
        new = engine.evaluate(quality_features=fast_features, quality_flags=fast_flags, **kwargs)
        self.assertEqual(old.decision, new.decision)
        self.assertEqual(old.decision, "REVIEW")
        self.assertEqual(old.ood_status, "unavailable")

    def test_resolution_boundaries(self) -> None:
        from quality_fastpath import fast_quality_gate, resolution_probe, strict_low_resolution

        self.assertTrue(strict_low_resolution(511))
        self.assertFalse(strict_low_resolution(512))
        self.assertFalse(strict_low_resolution(513))
        expectations = {
            (511, 511): True,
            (512, 512): False,
            (513, 513): False,
            (2000, 511): True,
            (2000, 512): False,
        }
        for (width, height), low_resolution in expectations.items():
            _features, flags = fast_quality_gate(resolution_probe(width, height))
            self.assertEqual("LOW_RESOLUTION" in flags, low_resolution, (width, height))
            self.assertNotIn("LOW_CONTRAST", flags)

    def test_contrast_threshold_and_exact_cutoff(self) -> None:
        from policy import LOW_CONTRAST_CUTOFF
        from quality_fastpath import contrast_boundary_images, fast_quality_gate, strict_low_contrast

        self.assertTrue(strict_low_contrast(LOW_CONTRAST_CUTOFF - 1e-12))
        self.assertFalse(strict_low_contrast(LOW_CONTRAST_CUTOFF))
        self.assertFalse(strict_low_contrast(LOW_CONTRAST_CUTOFF + 1e-12))
        self.assertEqual(LOW_CONTRAST_CUTOFF, 0.1933616647502347)
        boundary = contrast_boundary_images()
        below_features, below_flags = fast_quality_gate(boundary["below"])
        above_features, above_flags = fast_quality_gate(boundary["above"])
        self.assertLess(below_features["luminance_std"], LOW_CONTRAST_CUTOFF)
        self.assertGreater(above_features["luminance_std"], LOW_CONTRAST_CUTOFF)
        self.assertIn("LOW_CONTRAST", below_flags)
        self.assertNotIn("LOW_CONTRAST", above_flags)
        self.assertNotIn("LOW_RESOLUTION", below_flags)
        self.assertNotEqual(below_features["luminance_std"], LOW_CONTRAST_CUTOFF)

    def test_rgb_grayscale_dark_and_bright(self) -> None:
        from quality_fastpath import fast_quality_gate, resolution_probe

        color = resolution_probe(640, 480)
        gray = color.convert("L")
        self.assertEqual(fast_quality_gate(color)[1], fast_quality_gate(gray)[1])
        black_flags = fast_quality_gate(Image.new("RGB", (600, 600), (0, 0, 0)))[1]
        white_flags = fast_quality_gate(Image.new("RGB", (600, 600), (255, 255, 255)))[1]
        self.assertEqual(black_flags, ("LOW_CONTRAST",))
        self.assertEqual(white_flags, ("LOW_CONTRAST",))

    def test_missing_and_corrupt_files(self) -> None:
        from image_quality import QualityReadError
        from quality_fastpath import load_original_rgb

        with self.assertRaises(QualityReadError):
            load_original_rgb(REPO / "ml_rtx5080" / "experiments" / "quality_fastpath_v1" / "missing.jpg")
        with tempfile.TemporaryDirectory() as directory:
            corrupt = Path(directory) / "corrupt.jpg"
            corrupt.write_bytes(b"\x00\x01not-an-image")
            with self.assertRaises(QualityReadError):
                load_original_rgb(corrupt)

    def test_output_is_deterministic(self) -> None:
        from quality_fastpath import contrast_boundary_images, fast_quality_gate

        image = contrast_boundary_images()["below"]
        first = fast_quality_gate(image)
        second = fast_quality_gate(image)
        self.assertEqual(first, second)

    def test_stress_flags_match_the_reference(self) -> None:
        from hard_negative import apply_challenge_transform
        from image_quality import apply_quality_flags, extract_quality_features
        from policy_contract import PHASE23_CUTOFFS
        from quality_fastpath import decision_flags, fast_quality_gate

        source = REPO / "Label_0_Watches" / "Audemars Piguet" / "b1a40ed1b5b9.jpg"
        with Image.open(source) as handle:
            original = handle.convert("RGB")
            original.load()
        for name in ("mild_crop", "partial_crop", "lower_resolution", "darker", "resize_recompression"):
            transformed = apply_challenge_transform(original, name)
            reference = decision_flags(apply_quality_flags(extract_quality_features(transformed), {"cutoffs": dict(PHASE23_CUTOFFS)}))
            _features, fast_flags = fast_quality_gate(transformed)
            self.assertEqual(fast_flags, reference, name)
            if name == "lower_resolution":
                self.assertIn("LOW_RESOLUTION", fast_flags)

    def test_checkpoint_and_temperature_are_unchanged(self) -> None:
        from policy import FROZEN_TEMPERATURE

        checkpoint = REPO / "ml_rtx5080" / "experiments" / "v2_dinov3_cls_patch_attention" / "epoch_018.pt"
        self.assertEqual(checkpoint.stat().st_size, 1046865163)
        self.assertEqual(FROZEN_TEMPERATURE, 0.24038200410185356)
        self.assertFalse((REPO / "backend" / "inference" / "verdict.py").read_text().find("fast_quality_gate") >= 0)

    def test_final_test_is_rejected_before_open(self) -> None:
        from quality_fastpath import load_original_rgb

        manifest = json.loads((REPO / "ml_rtx5080" / "experiments" / "dataset_audit" / "split_manifest_v2.json").read_text())
        test_path = Path(manifest["membership"]["test"][0])
        blocked = {str(test_path.resolve())}
        opened = {"called": False}
        original = Image.open

        def guard(*_args, **_kwargs):
            opened["called"] = True
            raise AssertionError("final-test image was opened")

        Image.open = guard
        try:
            with self.assertRaises(RuntimeError):
                load_original_rgb(test_path, blocked)
        finally:
            Image.open = original
        self.assertFalse(opened["called"])

    def test_single_decode(self) -> None:
        from quality_fastpath import fast_quality_gate, load_original_rgb
        from reference_inference import preprocess_image

        path = REPO / "Label_0_Watches" / "Audemars Piguet" / "b1a40ed1b5b9.jpg"
        calls = {"count": 0}
        original = Image.open

        def counting(*args, **kwargs):
            calls["count"] += 1
            return original(*args, **kwargs)

        Image.open = counting
        try:
            image = load_original_rgb(path)
            fast_quality_gate(image)
            tensor = preprocess_image(image)
        finally:
            Image.open = original
        self.assertEqual(calls["count"], 1)
        self.assertEqual(tuple(tensor.shape), (3, 512, 512))

    def test_recorded_fastpath_contract(self) -> None:
        path = REPO / "ml_rtx5080" / "experiments" / "quality_fastpath_v1" / "fastpath_summary.json"
        if not path.is_file():
            self.skipTest("fast-path artifacts are produced by the phase runner, not by unit discovery")
        summary = json.loads(path.read_text())
        self.assertEqual(summary["flag_disagreements"], 0)
        self.assertEqual(summary["decision_mismatches"], 0)
        self.assertTrue(summary["stress_matches_expected"])
        self.assertTrue(summary["contract_passed"])
        self.assertEqual(summary["final_test_samples_used"], 0)
        self.assertEqual(summary["temperature"], 0.24038200410185356)
        self.assertEqual(summary["checkpoint_sha256"], "5a38c93fd442b03653c65d2a5ecc9c2687ef152f7c5c020763e4ce1fd9c7d28f")
        self.assertEqual(summary["production_policy_status"], "not_promoted")
        self.assertGreater(summary["quality_speedup_p50"], 1.0)
        self.assertFalse(summary["training_performed"])
        self.assertFalse(summary["calibration_performed"])
        self.assertFalse(summary["ood_evaluation_performed"])


if __name__ == "__main__":
    unittest.main()
