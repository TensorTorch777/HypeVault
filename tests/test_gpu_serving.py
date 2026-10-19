"""Phase 30 GPU serving checks. Unavailable runtimes are skipped, not passed."""

from __future__ import annotations

import json
import os
import sys
import importlib.util
import unittest
from pathlib import Path

from PIL import Image

REPO = Path(__file__).resolve().parents[1]
_HAS_ONNXRUNTIME = importlib.util.find_spec("onnxruntime") is not None
_DATASET = REPO / "Label_0_Watches"
_FROZEN_CHECKPOINT = REPO / "ml_rtx5080" / "experiments" / "v2_dinov3_cls_patch_attention" / "epoch_018.pt"
_DINOV3_ONNX = REPO / "models" / "dinov3_authenticity_classifier" / "1" / "model.onnx"
sys.path.insert(0, str(REPO / "ml_rtx5080"))
sys.path.insert(0, str(REPO / "backend"))


class GpuServingTests(unittest.TestCase):
    def setUp(self) -> None:
        os.environ["HYPEVAULT_DEPLOYMENT_MODE"] = "research"

    def tearDown(self) -> None:
        os.environ.pop("HYPEVAULT_SHADOW_POLICY", None)
        os.environ.pop("HYPEVAULT_SHADOW_SERVING", None)
        os.environ.pop("HYPEVAULT_DEPLOYMENT_MODE", None)

    @unittest.skipUnless(_HAS_ONNXRUNTIME, "onnxruntime is not installed (GPU serving extra)")
    def test_execution_provider_names_are_not_treated_as_loaded(self) -> None:
        import onnxruntime as ort
        from dinov3_serving import probe_cuda_execution_provider, reset_onnx_session

        reset_onnx_session()
        listed = ort.get_available_providers()
        self.assertIn("CUDAExecutionProvider", listed)
        probe = probe_cuda_execution_provider()
        self.assertTrue(probe["listed"])
        self.assertFalse(probe["available"])
        self.assertTrue(probe["error"])

    @unittest.skipUnless(_DINOV3_ONNX.is_file(), "ONNX artifact is local, not in git")
    def test_cuda_provider_unavailable_is_explicit(self) -> None:
        from dinov3_serving import probe_cuda_execution_provider

        probe = probe_cuda_execution_provider()
        if probe["available"]:
            self.skipTest("CUDA provider loaded; parity is covered by the serving run")
        self.assertEqual(probe["available"], False)
        self.assertIn("libcublasLt.so.12", probe["error"])

    @unittest.skipUnless(_HAS_ONNXRUNTIME, "onnxruntime is not installed (GPU serving extra)")
    def test_onnx_cuda_and_triton_fail_closed(self) -> None:
        from inference.shadow import compose_shadow_record
        from policy_contract import SYSTEM_STATUS_ERROR

        image = Image.new("RGB", (64, 48), (20, 30, 40))
        for backend in ("onnx_cuda", "triton"):
            os.environ["HYPEVAULT_SHADOW_SERVING"] = backend
            record = compose_shadow_record("AUTHENTIC", 0.91, image=image, enabled=True)
            self.assertEqual(record["verdict"], "AUTHENTIC")
            self.assertEqual(record["confidence"], 0.91)
            self.assertIsNone(record["shadow_decision"])
            self.assertEqual(record["shadow_system_status"], SYSTEM_STATUS_ERROR)
            self.assertIn("POLICY_INFRASTRUCTURE_ERROR", record["shadow_reason_codes"])
            self.assertNotIn(record["shadow_decision"], {"AUTHENTIC", "FAKE", "REVIEW"})

    @unittest.skipUnless(_HAS_ONNXRUNTIME, "onnxruntime is not installed (GPU serving extra)")
    def test_gpu_request_does_not_call_the_dinov2_loader(self) -> None:
        import inference.local_torch as local_torch
        from inference.shadow import compose_shadow_record

        called = {"count": 0}
        original = local_torch._load_bundle_sync

        def refuse(*_args, **_kwargs):
            called["count"] += 1
            raise AssertionError("DINOv2 loader was called")

        local_torch._load_bundle_sync = refuse
        os.environ["HYPEVAULT_SHADOW_SERVING"] = "onnx_cuda"
        try:
            record = compose_shadow_record("FAKE", 0.8, image=Image.new("RGB", (32, 32), 0), enabled=True)
        finally:
            local_torch._load_bundle_sync = original
        self.assertEqual(called["count"], 0)
        self.assertIsNone(record["shadow_decision"])
        self.assertEqual(record["verdict"], "FAKE")

    @unittest.skipUnless(_FROZEN_CHECKPOINT.is_file(), "frozen checkpoint is local, not in git")
    def test_checkpoint_identity_and_input_shape(self) -> None:
        from dinov3_serving import INPUT_SIZE, MODEL_NAME
        from reference_inference import file_sha256

        digest = file_sha256(REPO / "ml_rtx5080" / "experiments" / "v2_dinov3_cls_patch_attention" / "epoch_018.pt")
        self.assertEqual(digest, "5a38c93fd442b03653c65d2a5ecc9c2687ef152f7c5c020763e4ce1fd9c7d28f")
        self.assertEqual(INPUT_SIZE, 512)
        self.assertEqual(MODEL_NAME, "dinov3_authenticity_classifier")
        config = (REPO / "models" / "dinov2_classifier" / "config.pbtxt").read_text()
        self.assertIn('name: "dinov2_classifier"', config)
        self.assertIn("518", config)

    @unittest.skipUnless(_DATASET.is_dir(), "local dataset, not in git")
    def test_preprocessing_and_quality_flags_match_after_single_decode(self) -> None:
        import csv
        import torch
        from dinov3_serving import single_decode_inputs
        from image_quality import apply_quality_flags, extract_quality_features_from_path
        from policy_contract import PHASE23_CUTOFFS
        from reference_inference import preprocess_image

        golden = REPO / "ml_rtx5080" / "experiments" / "backend_parity_v2" / "onnx_parity.csv"
        sample_ids = []
        with golden.open(newline="") as handle:
            for row in csv.DictReader(handle):
                if row["sample_id"] not in sample_ids:
                    sample_ids.append(row["sample_id"])
        self.assertGreaterEqual(len(sample_ids), 20)
        thresholds = {"cutoffs": dict(PHASE23_CUTOFFS)}
        for sample_id in sample_ids:
            path = REPO / sample_id
            before = tuple(apply_quality_flags(extract_quality_features_from_path(path), thresholds))
            image, _features, after, tensor = single_decode_inputs(path)
            reference = preprocess_image(image)
            self.assertEqual(before, after)
            self.assertEqual(tuple(tensor.shape), (3, 512, 512))
            self.assertLessEqual(float(torch.max(torch.abs(tensor - reference)).item()), 1e-5)

    def test_final_test_is_rejected_and_shadow_defaults_off(self) -> None:
        from inference.shadow import run_shadow_image, shadow_mode_enabled, shadow_serving_backend

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

    @unittest.skipUnless(_HAS_ONNXRUNTIME, "onnxruntime is not installed (GPU serving extra)")
    def test_gpu_batch_consistency_when_cuda_provider_loads(self) -> None:
        from dinov3_serving import probe_cuda_execution_provider

        probe = probe_cuda_execution_provider()
        if not probe["available"]:
            self.skipTest(probe["error"] or "CUDA ONNX Runtime did not load")

    def test_tensorrt_and_triton_parity_when_available(self) -> None:
        import importlib.util
        import shutil
        import socket

        if importlib.util.find_spec("tensorrt") is None:
            tensorrt_available = False
        else:
            tensorrt_available = True
        reachable = False
        try:
            with socket.create_connection(("127.0.0.1", 8001), 0.3):
                reachable = True
        except OSError:
            reachable = False
        if not tensorrt_available and shutil.which("tritonserver") is None and not reachable:
            self.skipTest("TensorRT and Triton are not available")

    @unittest.skipUnless(_FROZEN_CHECKPOINT.is_file(), "frozen checkpoint is local, not in git")
    def test_phase30_artifacts(self) -> None:
        import serving_readiness

        destination = REPO / "ml_rtx5080" / "experiments" / "serving_v1" / "serving_summary.json"
        if not destination.is_file():
            serving_readiness.run_phase30()
        summary = json.loads(destination.read_text())
        self.assertEqual(summary["checkpoint_sha256"], "5a38c93fd442b03653c65d2a5ecc9c2687ef152f7c5c020763e4ce1fd9c7d28f")
        self.assertEqual(summary["input_dims"], [3, 512, 512])
        self.assertEqual(summary["production_policy_status"], "not_promoted")
        self.assertFalse(summary["shadow_mode_default"])
        self.assertEqual(summary["final_test_samples_used"], 0)
        self.assertFalse(summary["training_performed"])
        self.assertFalse(summary["calibration_performed"])
        self.assertFalse(summary["ood_evaluation_performed"])
        self.assertFalse(summary["live_backend_modified"])
        self.assertTrue(summary["quality_flags_match"])
        self.assertGreaterEqual(summary["quality_images"], 20)
        audit = json.loads((destination.parent / "environment_audit.json").read_text())
        self.assertIn("libcublasLt.so.12", " ".join(audit["ort_cuda_missing_libraries"]))
        self.assertFalse(audit["ollama_library_used"])
        self.assertFalse(audit["cuda_provider_loaded"])
        if summary["onnx_cuda_provider"] != "available":
            self.assertEqual(summary["gpu_onnx_parity"], "not_measured")
            self.assertFalse((destination.parent / "onnx_cuda_parity.csv").exists())
        self.assertEqual(summary["tensorrt_status"], "unavailable")
        self.assertEqual(summary["triton_status"], "unavailable")
        self.assertFalse((destination.parent / "tensorrt_parity.csv").exists())
        checkpoint = REPO / "ml_rtx5080" / "experiments" / "v2_dinov3_cls_patch_attention" / "epoch_018.pt"
        self.assertEqual(checkpoint.stat().st_size, 1046865163)


if __name__ == "__main__":
    unittest.main()
