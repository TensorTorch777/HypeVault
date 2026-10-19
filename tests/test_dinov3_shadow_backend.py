"""Phase 31 CUDA PyTorch DINOv3 shadow backend."""

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


class DinoV3ShadowBackendTests(unittest.TestCase):
    def setUp(self) -> None:
        os.environ["HYPEVAULT_DEPLOYMENT_MODE"] = "research"

    def tearDown(self) -> None:
        os.environ.pop("HYPEVAULT_SHADOW_POLICY", None)
        os.environ.pop("HYPEVAULT_SHADOW_SERVING", None)
        os.environ.pop("HYPEVAULT_DEPLOYMENT_MODE", None)

    @unittest.skipUnless(_FROZEN_CHECKPOINT.is_file(), "frozen checkpoint is local, not in git")
    def test_checkpoint_and_metadata_identity(self) -> None:
        from inference.dinov3_model import CHECKPOINT, EXPECTED_GPU, DinoV3ShadowBackend
        from reference_inference import file_sha256

        self.assertEqual(
            file_sha256(CHECKPOINT),
            "5a38c93fd442b03653c65d2a5ecc9c2687ef152f7c5c020763e4ce1fd9c7d28f",
        )
        metadata = DinoV3ShadowBackend().metadata()
        self.assertEqual(metadata["model_family"], "dinov3")
        self.assertEqual(metadata["backbone"], "vit_base_patch16_dinov3.lvd1689m")
        self.assertEqual(metadata["head"], "cls_patch_attention")
        self.assertEqual(metadata["input_size"], 512)
        self.assertEqual(metadata["preprocessing_version"], "resize_pad_square_eval_v1")
        self.assertEqual(metadata["authoritative_precision"], "fp32")
        self.assertEqual(metadata["temperature"], 0.24038200410185356)
        self.assertEqual(EXPECTED_GPU, "NVIDIA GeForce RTX 5080")

    def test_preprocessing_is_512_and_matches_the_reference(self) -> None:
        import torch
        from inference.dinov3_model import DinoV3ShadowBackend
        from reference_inference import preprocess_image

        backend = DinoV3ShadowBackend()
        images = [
            Image.new("RGB", (900, 200), (180, 20, 20)),
            Image.new("RGB", (180, 900), (20, 20, 180)),
            Image.new("RGB", (512, 512), (8, 8, 8)),
            Image.new("L", (640, 240), 30),
        ]
        for image in images:
            reference = preprocess_image(image)
            serving = backend.preprocess(image)
            self.assertEqual(tuple(serving.shape), (3, 512, 512))
            self.assertLessEqual(float(torch.max(torch.abs(reference - serving)).item()), 1e-5)

    def test_temperature_is_applied_once_to_the_raw_logit(self) -> None:
        import math
        from policy import FROZEN_TEMPERATURE, calibrated_probability_from_logit

        logit = -1.25
        calibrated = calibrated_probability_from_logit(logit)
        self.assertEqual(FROZEN_TEMPERATURE, 0.24038200410185356)
        self.assertAlmostEqual(calibrated, 1.0 / (1.0 + math.exp(-logit / FROZEN_TEMPERATURE)))
        doubled = 1.0 / (1.0 + math.exp(-(1.0 / (1.0 + math.exp(-logit))) / FROZEN_TEMPERATURE))
        self.assertNotAlmostEqual(calibrated, doubled, places=3)

    def test_cuda_unavailable_is_an_infrastructure_error(self) -> None:
        from inference.dinov3_model import reset_backend_for_tests, shared_backend
        from inference.shadow import compose_shadow_record, reset_shadow_for_tests
        from policy_contract import PolicyConfigurationError

        reset_shadow_for_tests()
        reset_backend_for_tests()
        with patch("torch.cuda.is_available", return_value=False):
            with self.assertRaises(PolicyConfigurationError) as caught:
                shared_backend().load()
            self.assertIn("POLICY_INFRASTRUCTURE_ERROR", caught.exception.reason_codes)
            record = compose_shadow_record("AUTHENTIC", 0.94, image=Image.new("RGB", (32, 32), 1), enabled=True)
        self.assertEqual(record["verdict"], "AUTHENTIC")
        self.assertIsNone(record["shadow_decision"])
        self.assertEqual(record["shadow_system_status"], "POLICY_ERROR")
        self.assertIn("POLICY_INFRASTRUCTURE_ERROR", record["shadow_reason_codes"])

    def test_cuda_failure_does_not_call_dinov2_or_cpu_onnx(self) -> None:
        import inference.local_torch as local_torch
        from dinov3_serving import forward_onnx_logits
        from inference.dinov3_model import reset_backend_for_tests
        from inference.shadow import compose_shadow_record, reset_shadow_for_tests

        reset_shadow_for_tests()
        reset_backend_for_tests()
        calls = {"dinov2": 0, "onnx": 0}
        original_load = local_torch._load_bundle_sync
        original_onnx = forward_onnx_logits

        def refuse_dinov2(*_args, **_kwargs):
            calls["dinov2"] += 1
            raise AssertionError("DINOv2 loader was called")

        def refuse_onnx(*_args, **_kwargs):
            calls["onnx"] += 1
            raise AssertionError("CPU ONNX was called")

        local_torch._load_bundle_sync = refuse_dinov2
        import dinov3_serving

        dinov3_serving.forward_onnx_logits = refuse_onnx
        try:
            with patch("torch.cuda.is_available", return_value=False):
                record = compose_shadow_record("FAKE", 0.7, image=Image.new("RGB", (40, 40), 2), enabled=True)
        finally:
            local_torch._load_bundle_sync = original_load
            dinov3_serving.forward_onnx_logits = original_onnx
        self.assertEqual(calls, {"dinov2": 0, "onnx": 0})
        self.assertIsNone(record["shadow_decision"])
        self.assertEqual(record["verdict"], "FAKE")

    def test_final_test_is_rejected_and_shadow_defaults_off(self) -> None:
        from inference.shadow import run_shadow_image, shadow_mode_enabled, shadow_serving_backend

        os.environ.pop("HYPEVAULT_SHADOW_POLICY", None)
        os.environ.pop("HYPEVAULT_SHADOW_SERVING", None)
        self.assertFalse(shadow_mode_enabled())
        self.assertEqual(shadow_serving_backend(), "pytorch")
        manifest = json.loads((REPO / "ml_rtx5080" / "experiments" / "dataset_audit" / "split_manifest_v2.json").read_text())
        test_path = Path(manifest["membership"]["test"][0])
        opened = {"called": False}

        class Guard:
            @staticmethod
            def open(*_args, **_kwargs):
                opened["called"] = True
                raise AssertionError("final-test image was opened")

        import PIL.Image as image_module

        original = image_module.open
        image_module.open = Guard.open
        try:
            with self.assertRaises(RuntimeError):
                run_shadow_image(test_path)
        finally:
            image_module.open = original
        self.assertFalse(opened["called"])

    def test_production_modules_do_not_import_the_shadow_backend(self) -> None:
        for relative in (
            "backend/inference/verdict.py",
            "backend/inference/dinov2_model.py",
            "backend/inference/local_torch.py",
            "backend/inference/triton_client.py",
            "backend/config.py",
        ):
            text = (REPO / relative).read_text()
            self.assertNotIn("dinov3_model", text)
            self.assertNotIn("cls_patch_attention", text)

    @unittest.skipUnless(_FROZEN_CHECKPOINT.is_file(), "frozen checkpoint is local, not in git")
    def test_phase31_gpu_artifacts(self) -> None:
        import serving_v2

        destination = REPO / "ml_rtx5080" / "experiments" / "serving_v2" / "shadow_backend_summary.json"
        if not destination.is_file():
            serving_v2.run_phase31()
        summary = json.loads(destination.read_text())
        self.assertEqual(summary["checkpoint_sha256"], "5a38c93fd442b03653c65d2a5ecc9c2687ef152f7c5c020763e4ce1fd9c7d28f")
        self.assertEqual(summary["membership_hash"], "78a38ebac1102977ec5598577126ade77fed8edd6563e13dbdee73dba389f678")
        self.assertEqual(summary["temperature"], 0.24038200410185356)
        self.assertEqual(summary["policy_version"], "shadow_v1")
        self.assertEqual(summary["precision"], "fp32")
        self.assertFalse(summary["bf16_authoritative"])
        self.assertIn("RTX 5080", summary["device"])
        self.assertGreaterEqual(summary["images"], 50)
        self.assertLessEqual(summary["max_abs_logit_error"], summary["logit_atol"])
        self.assertLessEqual(summary["max_abs_probability_error"], summary["probability_atol"])
        self.assertEqual(summary["decision_mismatches"], 0)
        self.assertEqual(summary["batch_decision_mismatches"], 0)
        parity = (destination.parent / "pytorch_cuda_parity.csv").read_text().strip().splitlines()
        self.assertGreaterEqual(sum("REVIEW" in row for row in parity), 1)
        self.assertNotIn("False", "\n".join(row.split(",")[-2] for row in parity[1:]))
        self.assertLessEqual(summary["batch_max_logit_deviation"], summary["logit_atol"])
        self.assertEqual(summary["production_policy_status"], "not_promoted")
        self.assertFalse(summary["shadow_mode_default"])
        self.assertEqual(summary["shadow_serving_default"], "pytorch")
        self.assertEqual(summary["final_test_samples_used"], 0)
        self.assertFalse(summary["training_performed"])
        self.assertFalse(summary["calibration_performed"])
        self.assertFalse(summary["ood_evaluation_performed"])
        self.assertFalse(summary["live_backend_modified"])
        self.assertFalse(summary["cuda_library_remap"])
        benchmark = (destination.parent / "pytorch_cuda_benchmark.csv").read_text().strip().splitlines()
        self.assertGreaterEqual(len(benchmark), 11)
        self.assertIn("fp32", benchmark[1])
        self.assertIn("bf16", benchmark[-1])
        memory = (destination.parent / "memory_benchmark.csv").read_text()
        self.assertIn("peak_allocated_bytes", memory)
        environment = json.loads((destination.parent / "environment_check.json").read_text())
        self.assertEqual(environment["warmup_iterations"], 20)
        self.assertEqual(environment["timed_iterations"], 100)
        self.assertTrue(environment["bf16_supported"])
        self.assertFalse(environment["onnx_cuda_used"])
        checkpoint = REPO / "ml_rtx5080" / "experiments" / "v2_dinov3_cls_patch_attention" / "epoch_018.pt"
        self.assertEqual(checkpoint.stat().st_size, 1046865163)


if __name__ == "__main__":
    unittest.main()
