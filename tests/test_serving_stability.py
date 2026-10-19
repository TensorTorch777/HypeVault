"""Phase 32 serving-stability checks. The 1,000-request stress run stays outside discovery."""

from __future__ import annotations

import json
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

REPO = Path(__file__).resolve().parents[1]
_FROZEN_CHECKPOINT = REPO / "ml_rtx5080" / "experiments" / "v2_dinov3_cls_patch_attention" / "epoch_018.pt"
sys.path.insert(0, str(REPO / "ml_rtx5080"))
sys.path.insert(0, str(REPO / "backend"))

AUTHENTIC = REPO / "Label_0_Watches" / "Audemars Piguet" / "b1a40ed1b5b9.jpg"


def _backend():
    from inference.dinov3_model import shared_backend

    backend = shared_backend()
    backend.load()
    return backend


def _open_rgb(path: Path) -> Image.Image:
    with Image.open(path) as handle:
        image = handle.convert("RGB")
        image.load()
    return image


def _one_fake() -> Path:
    manifest = json.loads((REPO / "ml_rtx5080" / "experiments" / "dataset_audit" / "split_manifest_v2.json").read_text())
    blocked = {str(Path(item).resolve()) for item in manifest["membership"]["test"]}
    quality = REPO / "ml_rtx5080" / "experiments" / "image_quality_v1" / "quality_features_nontest.csv"
    import csv

    with quality.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row["source_split"] == "test":
                raise RuntimeError("quality table contains a final-test row")
            if row["label"] != "1":
                continue
            path = (REPO / row["sample_id"]).resolve()
            if str(path) in blocked:
                raise RuntimeError("final-test path reached the stability test")
            return path
    raise FileNotFoundError("no non-test fake image was found")


class ServingStabilityTests(unittest.TestCase):
    def setUp(self) -> None:
        os.environ["HYPEVAULT_DEPLOYMENT_MODE"] = "research"

    def tearDown(self) -> None:
        os.environ.pop("HYPEVAULT_SHADOW_POLICY", None)
        os.environ.pop("HYPEVAULT_SHADOW_SERVING", None)
        os.environ.pop("HYPEVAULT_DEPLOYMENT_MODE", None)

    @unittest.skipUnless(_FROZEN_CHECKPOINT.is_file(), "frozen checkpoint is local, not in git")
    def test_checkpoint_identity(self) -> None:
        from reference_inference import file_sha256

        digest = file_sha256(REPO / "ml_rtx5080" / "experiments" / "v2_dinov3_cls_patch_attention" / "epoch_018.pt")
        self.assertEqual(digest, "5a38c93fd442b03653c65d2a5ecc9c2687ef152f7c5c020763e4ce1fd9c7d28f")

    def test_temperature_is_immutable(self) -> None:
        import math
        from policy import FROZEN_TEMPERATURE, calibrated_probability_from_logit

        self.assertEqual(FROZEN_TEMPERATURE, 0.24038200410185356)
        logit = 0.2
        self.assertAlmostEqual(
            calibrated_probability_from_logit(logit),
            1.0 / (1.0 + math.exp(-logit / FROZEN_TEMPERATURE)),
        )

    def test_concurrency_configuration(self) -> None:
        from serving_stability import validate_worker_count

        for workers in (1, 2, 4, 8, 16):
            self.assertEqual(validate_worker_count(workers), workers)
        for workers in (0, 3, 7, 32, -1):
            with self.assertRaises(ValueError):
                validate_worker_count(workers)

    def test_shadow_default_off(self) -> None:
        from inference.shadow import compose_shadow_record, shadow_mode_enabled, shadow_serving_backend

        os.environ.pop("HYPEVAULT_SHADOW_POLICY", None)
        self.assertFalse(shadow_mode_enabled())
        self.assertEqual(shadow_serving_backend(), "pytorch")
        record = compose_shadow_record("AUTHENTIC", 0.9, enabled=False)
        self.assertEqual(set(record), {"verdict", "confidence"})

    def test_final_test_rejection(self) -> None:
        from inference.shadow import run_shadow_image

        manifest = json.loads((REPO / "ml_rtx5080" / "experiments" / "dataset_audit" / "split_manifest_v2.json").read_text())
        test_path = Path(manifest["membership"]["test"][0])
        import PIL.Image as image_module

        original = image_module.open
        image_module.open = lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("final-test image was opened"))
        try:
            with self.assertRaises(RuntimeError):
                run_shadow_image(test_path)
        finally:
            image_module.open = original

    def test_no_fallback(self) -> None:
        import dinov3_serving
        import inference.local_torch as local_torch
        from inference.dinov3_model import shared_backend
        from inference.shadow import compose_shadow_record, reset_shadow_for_tests
        from policy_contract import PolicyConfigurationError

        reset_shadow_for_tests()
        calls = {"dinov2": 0, "onnx": 0}
        original_load = local_torch._load_bundle_sync
        original_onnx = dinov3_serving.forward_onnx_logits
        local_torch._load_bundle_sync = lambda *_args, **_kwargs: calls.__setitem__("dinov2", calls["dinov2"] + 1)
        dinov3_serving.forward_onnx_logits = lambda *_args, **_kwargs: calls.__setitem__("onnx", calls["onnx"] + 1)
        try:
            with patch("torch.cuda.is_available", return_value=False):
                with self.assertRaises(PolicyConfigurationError) as caught:
                    shared_backend().load()
                record = compose_shadow_record("FAKE", 0.2, image=Image.new("RGB", (32, 32), 3), enabled=True)
        finally:
            local_torch._load_bundle_sync = original_load
            dinov3_serving.forward_onnx_logits = original_onnx
        self.assertIn("POLICY_INFRASTRUCTURE_ERROR", caught.exception.reason_codes)
        self.assertEqual(calls, {"dinov2": 0, "onnx": 0})
        self.assertIsNone(record["shadow_decision"])
        self.assertEqual(record["shadow_system_status"], "POLICY_ERROR")
        self.assertEqual(record["verdict"], "FAKE")

    @unittest.skipUnless(_FROZEN_CHECKPOINT.is_file(), "frozen checkpoint is local, not in git")
    def test_determinism(self) -> None:
        from policy import calibrated_probability_from_logit

        backend = _backend()
        image = _open_rgb(AUTHENTIC)
        tensor = backend.preprocess(image).unsqueeze(0)
        logits = [float(backend.predict_logits(tensor)[0]) for _ in range(20)]
        probabilities = [calibrated_probability_from_logit(value) for value in logits]
        self.assertLessEqual(max(logits) - min(logits), 1e-4)
        self.assertLessEqual(max(probabilities) - min(probabilities), 1e-5)

    @unittest.skipUnless(_FROZEN_CHECKPOINT.is_file(), "frozen checkpoint is local, not in git")
    def test_request_isolation(self) -> None:
        backend = _backend()
        authentic = _open_rgb(AUTHENTIC)
        fake = _open_rgb(_one_fake())
        authentic_tensor = backend.preprocess(authentic).unsqueeze(0)
        fake_tensor = backend.preprocess(fake).unsqueeze(0)
        authentic_logit = float(backend.predict_logits(authentic_tensor)[0])
        fake_logit = float(backend.predict_logits(fake_tensor)[0])
        forward = backend.predict_logits(torch_cat(authentic_tensor, fake_tensor))
        reverse = backend.predict_logits(torch_cat(fake_tensor, authentic_tensor))
        self.assertLessEqual(abs(float(forward[0]) - authentic_logit), 1e-4)
        self.assertLessEqual(abs(float(forward[1]) - fake_logit), 1e-4)
        self.assertLessEqual(abs(float(reverse[0]) - fake_logit), 1e-4)
        self.assertLessEqual(abs(float(reverse[1]) - authentic_logit), 1e-4)

    @unittest.skipUnless(_FROZEN_CHECKPOINT.is_file(), "frozen checkpoint is local, not in git")
    def test_batch_single_parity(self) -> None:
        from policy_contract import INPUT_SIZE, PREPROCESSING_VERSION, PolicyEngine, frozen_shadow_configuration
        from reference_inference import quality_flags_for_image

        backend = _backend()
        image = _open_rgb(AUTHENTIC)
        tensor = backend.preprocess(image).unsqueeze(0)
        single = float(backend.predict_logits(tensor)[0])
        batched = [float(value) for value in backend.predict_logits(tensor.repeat(4, 1, 1, 1)).reshape(-1).tolist()]
        features, flags = quality_flags_for_image(image)
        engine = PolicyEngine(frozen_shadow_configuration())
        single_decision = engine.evaluate(
            logit=single,
            quality_features=features,
            quality_flags=flags,
            preprocessing_version=PREPROCESSING_VERSION,
            input_size=INPUT_SIZE,
            checkpoint_sha256=backend.checkpoint_sha256,
        ).decision
        for logit in batched:
            self.assertLessEqual(abs(logit - single), 1e-4)
            decision = engine.evaluate(
                logit=logit,
                quality_features=features,
                quality_flags=flags,
                preprocessing_version=PREPROCESSING_VERSION,
                input_size=INPUT_SIZE,
                checkpoint_sha256=backend.checkpoint_sha256,
            ).decision
            self.assertEqual(decision, single_decision)

    @unittest.skipUnless(_FROZEN_CHECKPOINT.is_file(), "frozen checkpoint is local, not in git")
    def test_policy_parity(self) -> None:
        from policy import calibrated_probability_from_logit
        from policy_contract import INPUT_SIZE, PREPROCESSING_VERSION, PolicyEngine, frozen_shadow_configuration
        from reference_inference import forward_logits, quality_flags_for_image

        backend = _backend()
        image = _open_rgb(AUTHENTIC)
        tensor = backend.preprocess(image).unsqueeze(0)
        reference = float(forward_logits(backend.model, tensor, backend.device)[0])
        shadow = float(backend.predict_logits(tensor)[0])
        self.assertLessEqual(abs(reference - shadow), 1e-4)
        self.assertLessEqual(
            abs(calibrated_probability_from_logit(reference) - calibrated_probability_from_logit(shadow)),
            1e-5,
        )
        features, flags = quality_flags_for_image(image)
        engine = PolicyEngine(frozen_shadow_configuration())
        kwargs = {
            "quality_features": features,
            "quality_flags": flags,
            "preprocessing_version": PREPROCESSING_VERSION,
            "input_size": INPUT_SIZE,
            "checkpoint_sha256": backend.checkpoint_sha256,
        }
        self.assertEqual(engine.evaluate(logit=reference, **kwargs).decision, engine.evaluate(logit=shadow, **kwargs).decision)
        self.assertEqual(engine.evaluate(logit=shadow, **kwargs).ood_status, "unavailable")

    @unittest.skipUnless(_FROZEN_CHECKPOINT.is_file(), "frozen checkpoint is local, not in git")
    def test_quality_failure_isolation(self) -> None:
        import numpy as np
        from inference.shadow import run_shadow_image

        backend = _backend()
        image = _open_rgb(AUTHENTIC)
        tensor = backend.preprocess(image).unsqueeze(0)
        golden = float(backend.predict_logits(tensor)[0])
        noisy = Image.fromarray(np.random.default_rng(7).integers(0, 256, size=(180, 200, 3), dtype=np.uint8), mode="RGB")
        probes = (
            noisy,
            Image.new("RGB", (800, 800), (128, 128, 128)),
            Image.new("RGB", (96, 128), (140, 140, 140)),
        )
        for probe in probes:
            result, _timings = run_shadow_image(probe)
            self.assertEqual(result.decision, "REVIEW")
            follow = float(backend.predict_logits(tensor)[0])
            self.assertLessEqual(abs(follow - golden), 1e-4)

    @unittest.skipUnless(_FROZEN_CHECKPOINT.is_file(), "frozen checkpoint is local, not in git")
    def test_malformed_input(self) -> None:
        from inference.shadow import run_shadow_image

        backend = _backend()
        image = _open_rgb(AUTHENTIC)
        tensor = backend.preprocess(image).unsqueeze(0)
        golden = float(backend.predict_logits(tensor)[0])
        result, _timings = run_shadow_image(b"\x00\x01not-an-image")
        self.assertIsNone(result.decision)
        self.assertEqual(result.system_status, "POLICY_ERROR")
        self.assertIn("POLICY_INVALID_INPUT", result.reason_codes)
        self.assertLessEqual(abs(float(backend.predict_logits(tensor)[0]) - golden), 1e-4)

    @unittest.skipUnless(_FROZEN_CHECKPOINT.is_file(), "frozen checkpoint is local, not in git")
    def test_missing_file(self) -> None:
        from inference.shadow import run_shadow_image

        backend = _backend()
        image = _open_rgb(AUTHENTIC)
        tensor = backend.preprocess(image).unsqueeze(0)
        golden = float(backend.predict_logits(tensor)[0])
        missing = REPO / "ml_rtx5080" / "experiments" / "serving_stability_v1" / "missing.jpg"
        result, _timings = run_shadow_image(missing)
        self.assertIsNone(result.decision)
        self.assertEqual(result.system_status, "POLICY_ERROR")
        self.assertLessEqual(abs(float(backend.predict_logits(tensor)[0]) - golden), 1e-4)

    @unittest.skipUnless(_FROZEN_CHECKPOINT.is_file(), "frozen checkpoint is local, not in git")
    def test_z_release_frees_gpu_memory(self) -> None:
        import gc
        import torch
        from inference.dinov3_model import reset_backend_for_tests, shared_backend
        from inference.shadow import reset_shadow_for_tests

        backend = _backend()
        held = torch.cuda.memory_allocated()
        self.assertIsNotNone(backend.model)
        self.assertGreater(held, 100_000_000)
        reset_shadow_for_tests()
        backend.release()
        gc.collect()
        torch.cuda.synchronize()
        self.assertIsNone(backend.model)
        self.assertLess(torch.cuda.memory_allocated(), held)
        reset_backend_for_tests()

    def test_recorded_stress_contract(self) -> None:
        path = REPO / "ml_rtx5080" / "experiments" / "serving_stability_v1" / "serving_stability_summary.json"
        if not path.is_file():
            self.skipTest("stress artifacts are produced by the phase runner, not by unit discovery")
        summary = json.loads(path.read_text())
        self.assertEqual(summary["checkpoint_sha256"], "5a38c93fd442b03653c65d2a5ecc9c2687ef152f7c5c020763e4ce1fd9c7d28f")
        self.assertEqual(summary["temperature"], 0.24038200410185356)
        self.assertEqual(summary["precision"], "fp32")
        self.assertFalse(summary["bf16_authoritative"])
        self.assertEqual(summary["bf16_label"], "research_only")
        self.assertEqual(summary["production_policy_status"], "not_promoted")
        self.assertFalse(summary["shadow_mode_default"])
        self.assertEqual(summary["final_test_samples_used"], 0)
        self.assertGreaterEqual(summary["sequential_requests"], 1000)
        self.assertTrue(summary["memory_stable"])
        self.assertTrue(summary["determinism_passed"])
        self.assertTrue(summary["request_isolation_passed"])
        self.assertTrue(summary["failure_isolation_passed"])
        self.assertTrue(summary["reload_passed"])
        self.assertTrue(summary["contract_passed"])
        self.assertFalse(summary["training_performed"])
        self.assertFalse(summary["calibration_performed"])
        self.assertFalse(summary["ood_evaluation_performed"])


def torch_cat(left, right):
    import torch

    return torch.cat((left, right), dim=0)


if __name__ == "__main__":
    unittest.main()
