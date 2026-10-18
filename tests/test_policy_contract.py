"""Fail-closed shadow policy contract. Production promotion stays disabled."""

from __future__ import annotations

import inspect
import json
import sys
import tempfile
import unittest
from dataclasses import FrozenInstanceError
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ml_rtx5080"))

import image_quality
import policy_contract
import policy_eval
from policy_contract import (
    POLICY_VERSION,
    PREPROCESSING_VERSION,
    PolicyConfiguration,
    PolicyConfigurationError,
    PolicyEngine,
    frozen_shadow_configuration,
    preprocess_eval_images,
    results_equivalent,
)

EXPERIMENTS = Path(__file__).resolve().parents[1] / "ml_rtx5080" / "experiments"
PREPROCESS = PREPROCESSING_VERSION


class PolicyContractTests(unittest.TestCase):
    def _engine(self) -> PolicyEngine:
        events: list[dict] = []
        engine = PolicyEngine(event_logger=events.append)
        engine.events = events  # type: ignore[attr-defined]
        return engine

    def test_valid_authentic_result(self) -> None:
        engine = self._engine()
        result = engine.evaluate(probability=0.029, quality_flags=(), preprocessing_version=PREPROCESS, request_id="req-1")
        self.assertEqual(result.decision, "AUTHENTIC")
        self.assertEqual(result.system_status, "ok")
        self.assertEqual(result.reason_codes, ("AUTHENTICITY_BELOW_THRESHOLD",))
        self.assertEqual(result.policy_version, POLICY_VERSION)
        self.assertEqual(result.policy_status, "not_promoted")
        self.assertFalse(result.quality_review)
        self.assertEqual(result.ood_status, "unavailable")
        self.assertIsNone(result.ood_score)
        self.assertIsNone(result.ood_flag)
        self.assertFalse(result.in_distribution_assumed)
        self.assertEqual(engine.events[0]["request_id"], "req-1")
        self.assertEqual(engine.events[0]["decision"], "AUTHENTIC")

    def test_valid_fake_result(self) -> None:
        result = self._engine().evaluate(probability=0.694, quality_flags=("EXTREME_ASPECT_RATIO",), preprocessing_version=PREPROCESS)
        self.assertEqual(result.decision, "FAKE")
        self.assertEqual(result.reason_codes, ("AUTHENTICITY_THRESHOLD",))
        self.assertFalse(result.quality_review)

    def test_valid_review_result(self) -> None:
        result = self._engine().evaluate(probability=0.01, quality_flags=("LOW_RESOLUTION",), preprocessing_version=PREPROCESS)
        self.assertEqual(result.decision, "REVIEW")
        self.assertEqual(result.reason_codes, ("QUALITY_LOW_RESOLUTION",))
        self.assertTrue(result.quality_review)
        self.assertNotEqual(result.decision, "POLICY_ERROR")

    def test_quality_precedence(self) -> None:
        result = self._engine().evaluate(probability=0.99, quality_flags=("LOW_CONTRAST", "BLURRY"), preprocessing_version=PREPROCESS)
        self.assertEqual(result.decision, "REVIEW")
        self.assertEqual(result.reason_codes, ("QUALITY_LOW_CONTRAST",))
        diagnostic = self._engine().evaluate(probability=0.1, quality_flags=("BLURRY", "TOO_DARK"), preprocessing_version=PREPROCESS)
        self.assertEqual(diagnostic.decision, "AUTHENTIC")

    def test_threshold_boundary_is_fake(self) -> None:
        engine = self._engine()
        at = engine.evaluate(probability=0.50, quality_flags=(), preprocessing_version=PREPROCESS)
        below = engine.evaluate(probability=0.499999, quality_flags=(), preprocessing_version=PREPROCESS)
        self.assertEqual(at.decision, "FAKE")
        self.assertEqual(below.decision, "AUTHENTIC")
        self.assertEqual(at.authenticity_threshold, 0.50)

    def test_ood_unavailable_is_not_false(self) -> None:
        engine = self._engine()
        clean = engine.evaluate(probability=0.2, quality_flags=(), preprocessing_version=PREPROCESS)
        self.assertEqual(clean.ood_status, "unavailable")
        self.assertIsNone(clean.ood_score)
        self.assertIsNone(clean.ood_flag)
        self.assertFalse(clean.in_distribution_assumed)
        flagged = engine.evaluate(probability=0.2, quality_flags=(), preprocessing_version=PREPROCESS, ood_flag=False)
        scored = engine.evaluate(probability=0.2, quality_flags=(), preprocessing_version=PREPROCESS, ood_score=0.0)
        self.assertIsNone(flagged.decision)
        self.assertIsNone(scored.decision)
        self.assertIsNone(flagged.ood_flag)
        self.assertIsNone(scored.ood_score)
        self.assertEqual(flagged.ood_status, "unavailable")
        self.assertEqual(flagged.system_status, "POLICY_ERROR")

    def test_invalid_nan_and_inf_fail_closed(self) -> None:
        engine = self._engine()
        for kwargs in (
            {"probability": float("nan"), "quality_flags": ()},
            {"logit": float("inf"), "quality_flags": ()},
            {"logit": float("-inf"), "quality_flags": ("LOW_RESOLUTION",)},
            {"probability": 1.2, "quality_flags": ()},
        ):
            result = engine.evaluate(preprocessing_version=PREPROCESS, **kwargs)
            self.assertIsNone(result.decision)
            self.assertEqual(result.system_status, "POLICY_ERROR")
            self.assertIn("POLICY_INVALID_INPUT", result.reason_codes)
            self.assertNotIn(result.decision, {"AUTHENTIC", "FAKE", "REVIEW"})

    def test_invalid_temperature_threshold_and_policy_version_fail_initialization(self) -> None:
        for overrides in (
            {"temperature": float("nan")},
            {"temperature": 0.5},
            {"temperature": -1.0},
            {"authenticity_threshold": 0.0},
            {"authenticity_threshold": 1.0},
            {"policy_version": "production_v1"},
            {"policy_version": "unknown"},
            {"ood_status": "available"},
            {"ood_status": "false"},
            {"production_policy_status": "promoted"},
            {"production_authenticity_threshold": 0.5},
        ):
            with self.assertRaises(PolicyConfigurationError):
                PolicyEngine(PolicyConfiguration(**overrides))

    def test_model_and_checkpoint_mismatch_fail_initialization(self) -> None:
        cases = (
            {"model_family": "dinov2"},
            {"head": "cls_only"},
            {"architecture": "vit_base_patch14_dinov2.lvd142m"},
            {"input_size": 504},
            {"checkpoint_sha256": "0" * 64},
            {"preprocessing_version": "resize_center_crop_eval_v1"},
        )
        for overrides in cases:
            with self.assertRaises(PolicyConfigurationError) as caught:
                PolicyEngine(PolicyConfiguration(**overrides))
            self.assertIn("POLICY_CONFIGURATION_ERROR", caught.exception.reason_codes)
            self.assertNotIn("AUTHENTIC", caught.exception.reason_codes)

    def test_preprocessing_mismatch_and_contract(self) -> None:
        from PIL import Image
        from augmentations import ResizePadToSquare, imagenet_mean_fill

        engine = self._engine()
        for version in ("resize_center_crop_eval_v1", "conservative_train_v1", "current_legacy_train_v1"):
            result = engine.evaluate(probability=0.1, quality_flags=(), preprocessing_version=version)
            self.assertIsNone(result.decision)
            self.assertIn("POLICY_CONFIGURATION_ERROR", result.reason_codes)
        size = engine.evaluate(probability=0.1, quality_flags=(), preprocessing_version=PREPROCESS, input_size=504)
        self.assertIsNone(size.decision)
        self.assertIn("POLICY_MODEL_MISMATCH", size.reason_codes)
        missing = engine.evaluate(probability=0.1, quality_flags=())
        self.assertIsNone(missing.decision)
        self.assertIn("POLICY_INVALID_INPUT", missing.reason_codes)
        wide = Image.new("RGB", (600, 100), (255, 0, 0))
        canvas = ResizePadToSquare(512)(wide)
        self.assertEqual(canvas.size, (512, 512))
        self.assertEqual(canvas.getpixel((0, 0)), imagenet_mean_fill())
        batch = preprocess_eval_images([wide])
        self.assertEqual(tuple(batch.shape), (1, 3, 512, 512))
        with self.assertRaises(PolicyConfigurationError):
            preprocess_eval_images([wide], train=True)
        with self.assertRaises(PolicyConfigurationError):
            preprocess_eval_images([wide], family="dinov2")
        with self.assertRaises(PolicyConfigurationError):
            preprocess_eval_images([wide], profile="current_legacy")

    def test_corrupted_image_and_missing_file(self) -> None:
        engine = self._engine()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            corrupt = root / "broken.jpg"
            corrupt.write_bytes(b"this is not an image")
            missing = root / "missing.jpg"
            before = corrupt.read_bytes()
            rows = engine.evaluate_paths([corrupt, missing], logits={str(corrupt.resolve()): -1.0}, batch_size=2)
            self.assertEqual(corrupt.read_bytes(), before)
        self.assertEqual(len(rows), 2)
        self.assertIsNone(rows[0]["decision"])
        self.assertIsNone(rows[1]["decision"])
        self.assertEqual(rows[0]["system_status"], "POLICY_ERROR")
        self.assertEqual(rows[1]["system_status"], "POLICY_ERROR")
        self.assertIn("POLICY_INVALID_INPUT", rows[0]["reason_codes"])
        self.assertIn("corrupted", rows[0]["error_reason"])
        self.assertIn("missing", rows[1]["error_reason"])
        self.assertIsNone(rows[0]["ood_flag"])

    def test_missing_quality_feature(self) -> None:
        result = self._engine().evaluate(probability=0.4, preprocessing_version=PREPROCESS)
        self.assertIsNone(result.decision)
        self.assertIn("POLICY_INVALID_INPUT", result.reason_codes)
        self.assertIn("missing quality feature", result.error_reason or "")

    def test_reason_codes_are_deterministic(self) -> None:
        engine = self._engine()
        kwargs = {
            "probability": 0.2,
            "quality_flags": ("LOW_CONTRAST", "LOW_RESOLUTION", "EXCESSIVE_CLIPPING"),
            "preprocessing_version": PREPROCESS,
        }
        first = engine.evaluate(**kwargs)
        second = engine.evaluate(**kwargs)
        self.assertEqual(first.reason_codes, second.reason_codes)
        self.assertEqual(
            first.reason_codes,
            ("QUALITY_MULTIPLE_FLAGS", "QUALITY_LOW_RESOLUTION", "QUALITY_LOW_CONTRAST"),
        )
        self.assertEqual(first.to_dict()["decision"], second.to_dict()["decision"])
        self.assertGreaterEqual(len(first.reason_codes), 1)

    def test_batch_matches_single_image(self) -> None:
        import torch
        from PIL import Image

        images = [
            Image.new("RGB", (600, 100), (255, 0, 0)),
            Image.new("RGB", (128, 128), (0, 0, 0)),
            Image.new("RGB", (800, 900), (30, 40, 50)),
        ]

        def logit_fn(batch: torch.Tensor) -> list[float]:
            return batch.detach().float().mean(dim=(1, 2, 3)).tolist()

        single_batch = preprocess_eval_images(images[:1])
        grouped_batch = preprocess_eval_images(images)
        self.assertTrue(torch.equal(single_batch[0], grouped_batch[0]))
        alone = PolicyEngine().evaluate_images(images, logit_fn, batch_size=1)
        paired = PolicyEngine().evaluate_images(images, logit_fn, batch_size=2)
        whole = PolicyEngine().evaluate_images(images, logit_fn, batch_size=3)
        for left, middle, right in zip(alone, paired, whole, strict=True):
            self.assertTrue(results_equivalent(left, middle))
            self.assertTrue(results_equivalent(left, right))
            self.assertIsNotNone(left.decision)

    def test_final_test_samples_are_rejected(self) -> None:
        engine = self._engine()
        with self.assertRaises(RuntimeError):
            engine.evaluate(probability=0.1, quality_flags=(), preprocessing_version=PREPROCESS, split="test")
        with self.assertRaises(RuntimeError):
            engine.evaluate(
                probability=0.1,
                quality_flags=(),
                preprocessing_version=PREPROCESS,
                split="train",
                sample_id="hidden",
                test_sample_ids={"hidden"},
            )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "watch.jpg"
            from PIL import Image

            Image.new("RGB", (32, 32), (1, 2, 3)).save(path)
            with self.assertRaises(RuntimeError):
                engine.evaluate_paths([path], test_paths={str(path.resolve())})
        with self.assertRaises(RuntimeError):
            policy_eval.simulate_contract(
                Path(tmp),
                Path(tmp) / "missing.json",
                input_splits=("test",),
                enforce_expected_sha=False,
            )

    def test_shadow_status_remains_not_promoted(self) -> None:
        config = frozen_shadow_configuration()
        self.assertEqual(config.policy_version, "shadow_v1")
        self.assertNotEqual(config.policy_version, "production_v1")
        self.assertEqual(config.policy_status, "not_promoted")
        self.assertEqual(config.production_policy_status, "not_promoted")
        self.assertIsNone(config.production_authenticity_threshold)
        self.assertIsNone(config.production_quality_gate)
        self.assertIsNone(config.production_ood_threshold)
        self.assertEqual(config.temperature, 0.24038200410185356)
        with self.assertRaises(FrozenInstanceError):
            config.temperature = 1.0  # type: ignore[misc]
        result = self._engine().evaluate(probability=0.1, quality_flags=(), preprocessing_version=PREPROCESS)
        self.assertEqual(result.production_policy_status, "not_promoted")
        self.assertEqual(policy_contract.DIAGNOSTIC_FLAGS, image_quality.FLAG_NAMES)
        cutoffs = json.loads((EXPERIMENTS / "image_quality_v1" / "quality_thresholds.json").read_text())["cutoffs"]
        self.assertEqual(cutoffs["min_dimension_p5"], policy_contract.LOW_RESOLUTION_CUTOFF)
        self.assertEqual(cutoffs["luminance_std_p5"], policy_contract.LOW_CONTRAST_CUTOFF)

    def test_events_and_counters_omit_pixels(self) -> None:
        engine = self._engine()
        engine.evaluate(probability=0.1, quality_flags=(), preprocessing_version=PREPROCESS, request_id="r")
        engine.evaluate(probability=0.9, quality_flags=(), preprocessing_version=PREPROCESS)
        engine.evaluate(probability=0.9, quality_flags=("LOW_RESOLUTION", "LOW_CONTRAST"), preprocessing_version=PREPROCESS)
        engine.evaluate(probability=float("nan"), quality_flags=(), preprocessing_version=PREPROCESS)
        event = engine.events[0]
        self.assertEqual(event["request_id"], "r")
        self.assertEqual(event["decision"], "AUTHENTIC")
        self.assertFalse({"pixels", "image_bytes", "raw_image", "pixel_values"} & set(event))
        counts = engine.counters.to_dict()
        self.assertEqual(counts["authentic"], 1)
        self.assertEqual(counts["fake"], 1)
        self.assertEqual(counts["review"], 1)
        self.assertEqual(counts["policy_error"], 1)
        self.assertEqual(counts["quality_review"], 1)
        self.assertEqual(counts["low_resolution"], 1)
        self.assertEqual(counts["low_contrast"], 1)
        self.assertEqual(counts["ood_unavailable"], 4)

    def test_runtime_temperature_replacement_is_rejected(self) -> None:
        result = self._engine().evaluate(
            probability=0.1,
            quality_flags=(),
            preprocessing_version=PREPROCESS,
            temperature=0.56,
        )
        self.assertIsNone(result.decision)
        self.assertIn("POLICY_CONFIGURATION_ERROR", result.reason_codes)

    def test_cli_rejects_a_different_checkpoint_and_backend_files_stay_put(self) -> None:
        repo = Path(__file__).resolve().parents[1]
        before = policy_eval._backend_hashes(repo)
        source = inspect.getsource(policy_contract) + inspect.getsource(policy_eval)
        self.assertNotIn("torch.save", source)
        self.assertNotIn("fit_temperature", source)
        self.assertNotIn(".backward(", source)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image = root / "watch.jpg"
            from PIL import Image

            Image.new("RGB", (40, 40), (8, 8, 8)).save(image)
            original = image.read_bytes()
            checkpoint = root / "other.pt"
            checkpoint.write_bytes(b"not-the-frozen-checkpoint")
            with self.assertRaises(RuntimeError):
                policy_eval.main(
                    [
                        "--mode",
                        "shadow",
                        "--image",
                        str(image),
                        "--logit",
                        "-2.0",
                        "--checkpoint",
                        str(checkpoint),
                        "--experiment",
                        str(root),
                    ]
                )
            self.assertEqual(image.read_bytes(), original)
        self.assertEqual(policy_eval._backend_hashes(repo), before)

    def test_shadow_counts_reproduce_phase26(self) -> None:
        manifest = EXPERIMENTS / "dataset_audit" / "split_manifest_v2.json"
        observed = policy_eval.simulate_contract(EXPERIMENTS, manifest)
        counts = observed["observed"]
        self.assertEqual(counts["authentic_count"], 11917)
        self.assertEqual(counts["review_count"], 1855)
        self.assertEqual(counts["fake_count"], 13222)
        self.assertEqual(counts["false_authentic_count"], 0)
        self.assertEqual(counts["false_fake_count"], 0)
        self.assertEqual(counts["sample_count"], 26994)
        self.assertAlmostEqual(counts["review_rate"], 1855 / 26994)
        self.assertEqual(observed["stress"]["caught_by_quality_policy"], 8)
        self.assertEqual(observed["stress"]["demonstrated_false_fake_events"], 9)
        self.assertEqual(observed["stress"]["missed_ids"], ["Label_0_Watches/Richard Mille/9eda705bd8c9.jpg|resize_recompression"])
        self.assertEqual(observed["stress"]["false_authentic_count"], 0)
        self.assertEqual(observed["final_test_samples_used"], 0)
        self.assertEqual(observed["production_policy_status"], "not_promoted")
        self.assertIsNone(observed["production_authenticity_threshold"])
        self.assertIsNone(observed["production_quality_gate"])
        self.assertIsNone(observed["production_ood_threshold"])
        self.assertEqual(observed["ood_status"], "unavailable")
        self.assertEqual(observed["policy_error_count"], 0)
        self.assertFalse(observed["training_performed"])
        self.assertFalse(observed["live_backend_modified"])
        destination = EXPERIMENTS / "policy_contract_v1" / "policy_simulation_summary.json"
        if not destination.is_file():
            written = policy_eval.run_phase27(EXPERIMENTS, manifest)
        else:
            written = json.loads(destination.read_text())
        self.assertEqual(written["observed"]["authentic_count"], 11917)
        self.assertEqual(written["observed"]["review_count"], 1855)
        self.assertEqual(written["observed"]["fake_count"], 13222)
        audit = json.loads((destination.parent / "policy_contract_audit.json").read_text())
        self.assertEqual(audit["final_test_samples_used"], 0)
        self.assertEqual(audit["preprocessing_version"], "resize_pad_square_eval_v1")
        self.assertIn("backend/inference/verdict.py", audit["backend_sha256"])
        self.assertIn("backend/inference/triton_client.py", audit["backend_sha256"])


if __name__ == "__main__":
    unittest.main()
