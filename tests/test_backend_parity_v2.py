"""Phase 29 parallel DINOv3 serving artifact. Triton is skipped when the server is down."""

from __future__ import annotations

import json
import os
import sys
import unittest
from pathlib import Path

from PIL import Image

REPO = Path(__file__).resolve().parents[1]
_DATASET = REPO / "Label_0_Watches"
_DINOV2_TRITON_CONFIG = REPO / "models" / "dinov2_classifier" / "config.pbtxt"
_DINOV3_TRITON_CONFIG = REPO / "models" / "dinov3_authenticity_classifier" / "config.pbtxt"
_FROZEN_CHECKPOINT = REPO / "ml_rtx5080" / "experiments" / "v2_dinov3_cls_patch_attention" / "epoch_018.pt"
sys.path.insert(0, str(REPO / "ml_rtx5080"))
sys.path.insert(0, str(REPO / "backend"))


class BackendParityV2Tests(unittest.TestCase):
    def test_dinov3_metadata_and_input_shape(self) -> None:
        from classifier import build_authenticity_classifier
        from dinov3_serving import INPUT_SIZE, MODEL_NAME, architecture_record
        from policy_contract import MODEL_ARCHITECTURE, MODEL_FAMILY, MODEL_HEAD

        self.assertEqual(MODEL_NAME, "dinov3_authenticity_classifier")
        self.assertEqual(MODEL_FAMILY, "dinov3")
        self.assertEqual(MODEL_ARCHITECTURE, "vit_base_patch16_dinov3.lvd1689m")
        self.assertEqual(MODEL_HEAD, "cls_patch_attention")
        self.assertEqual(INPUT_SIZE, 512)
        model = build_authenticity_classifier("dinov3", "cls_patch_attention", image_size=512, pretrained=False)
        record = architecture_record(model)
        self.assertEqual(record["image_size"], 512)
        self.assertEqual(record["num_register_tokens"], 4)
        self.assertTrue(record["attention_pool"])
        self.assertTrue(record["fusion"])
        self.assertTrue(record["trunk"])
        self.assertFalse(record["temperature_in_graph"])
        self.assertEqual(record["output"], "raw_logit")

    @unittest.skipUnless(_FROZEN_CHECKPOINT.is_file(), "frozen checkpoint is local, not in git")
    def test_checkpoint_hash(self) -> None:
        from reference_inference import file_sha256

        digest = file_sha256(REPO / "ml_rtx5080" / "experiments" / "v2_dinov3_cls_patch_attention" / "epoch_018.pt")
        self.assertEqual(digest, "5a38c93fd442b03653c65d2a5ecc9c2687ef152f7c5c020763e4ce1fd9c7d28f")

    def test_preprocessing_matches_the_reference(self) -> None:
        import torch
        from dinov3_serving import TENSOR_ATOL, preprocess_serving_image
        from reference_inference import preprocess_image

        images = {
            "square": Image.new("RGB", (512, 512), (1, 2, 3)),
            "wide": Image.new("RGB", (800, 100), (9, 9, 9)),
            "tall": Image.new("RGB", (100, 800), (4, 5, 6)),
            "tiny": Image.new("RGB", (8, 8), (255, 0, 0)),
            "large": Image.new("RGB", (1500, 900), (7, 8, 9)),
            "rgb": Image.new("RGB", (640, 480), (12, 13, 14)),
            "grayscale_convertible": Image.new("L", (640, 200), 30),
        }
        for image in images.values():
            reference = preprocess_image(image)
            serving = preprocess_serving_image(image)
            self.assertEqual(tuple(serving.shape), (3, 512, 512))
            self.assertLessEqual(float(torch.max(torch.abs(reference - serving)).item()), TENSOR_ATOL)
        wide = preprocess_serving_image(images["wide"])
        self.assertLess(float(wide[:, 0, :].abs().mean()), float(wide[:, 256, :].abs().mean()))

    def test_temperature_is_outside_the_logit(self) -> None:
        import math
        from policy import FROZEN_TEMPERATURE, calibrated_probability_from_logit

        logit = -1.2462576627731323
        calibrated = calibrated_probability_from_logit(logit)
        self.assertAlmostEqual(calibrated, 1.0 / (1.0 + math.exp(-logit / FROZEN_TEMPERATURE)))
        self.assertEqual(FROZEN_TEMPERATURE, 0.24038200410185356)

    def test_shadow_flags_default_off(self) -> None:
        os.environ.pop("HYPEVAULT_SHADOW_POLICY", None)
        os.environ.pop("HYPEVAULT_SHADOW_SERVING", None)
        from inference.shadow import compose_shadow_record, shadow_mode_enabled, shadow_serving_backend

        self.assertFalse(shadow_mode_enabled())
        self.assertEqual(shadow_serving_backend(), "pytorch")
        record = compose_shadow_record("FAKE", 0.91, image=None, enabled=False)
        self.assertEqual(record, {"verdict": "FAKE", "confidence": 0.91})

    @unittest.skipUnless(_DINOV2_TRITON_CONFIG.is_file(), "Triton model repository is local, not in git")
    def test_old_production_and_triton_config_stay_dinov2(self) -> None:
        config = (REPO / "models" / "dinov2_classifier" / "config.pbtxt").read_text()
        self.assertIn('name: "dinov2_classifier"', config)
        self.assertIn("518", config)
        self.assertNotIn("dinov3_authenticity_classifier", config)
        verdict = (REPO / "backend" / "inference" / "verdict.py").read_text()
        routes = (REPO / "backend" / "inference" / "routes.py").read_text()
        self.assertNotIn("dinov3_authenticity_classifier", verdict)
        self.assertNotIn("HYPEVAULT_SHADOW_SERVING", routes)

    def test_final_test_sample_is_rejected(self) -> None:
        from inference.shadow import run_shadow_image

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

    @unittest.skipUnless(_DATASET.is_dir(), "local dataset, not in git")
    def test_quality_shared_decode_preserves_flags(self) -> None:
        from image_quality import apply_quality_flags, extract_quality_features, extract_quality_features_from_path
        from policy_contract import PHASE23_CUTOFFS

        path = REPO / "Label_0_Watches" / "A. Lange & Söhne" / "0003c0bd6ebe.jpg"
        from_path = extract_quality_features_from_path(path)
        with Image.open(path) as handle:
            rgb = handle.convert("RGB")
            rgb.load()
            owned = rgb.copy()
        shared = extract_quality_features(
            owned,
            file_bytes=path.stat().st_size,
            image_format=from_path["image_format"],
            jpeg_quantization_tables_present=from_path["jpeg_quantization_tables_present"],
        )
        thresholds = {"cutoffs": dict(PHASE23_CUTOFFS)}
        self.assertEqual(apply_quality_flags(from_path, thresholds), apply_quality_flags(shared, thresholds))

    @unittest.skipUnless(_DINOV3_TRITON_CONFIG.is_file(), "Triton model repository is local, not in git")
    def test_phase29_artifacts(self) -> None:
        import backend_parity_v2

        destination = REPO / "ml_rtx5080" / "experiments" / "backend_parity_v2" / "shadow_model_summary.json"
        if not destination.is_file():
            backend_parity_v2.run_phase29()
        summary = json.loads(destination.read_text())
        self.assertEqual(summary["checkpoint_sha256"], "5a38c93fd442b03653c65d2a5ecc9c2687ef152f7c5c020763e4ce1fd9c7d28f")
        self.assertEqual(summary["input_dims"], [3, 512, 512])
        self.assertFalse(summary["temperature_in_graph"])
        self.assertEqual(summary["temperature"], 0.24038200410185356)
        self.assertEqual(summary["onnx_parity"]["status"], "passed")
        self.assertGreaterEqual(summary["onnx_parity"]["images"], 20)
        self.assertLessEqual(summary["onnx_parity"]["max_abs_logit_error"], summary["tolerances"]["logit_atol"])
        self.assertEqual(summary["onnx_parity"]["decision_disagreements"], 0)
        self.assertEqual(summary["batch_parity"]["decision_mismatches"], 0)
        self.assertLessEqual(summary["batch_parity"]["max_logit_deviation"], summary["tolerances"]["logit_atol"])
        self.assertLessEqual(summary["determinism"]["pytorch_span"], summary["tolerances"]["determinism_atol"])
        self.assertLessEqual(summary["determinism"]["onnx_span"], summary["tolerances"]["determinism_atol"])
        self.assertEqual(summary["production_policy_status"], "not_promoted")
        self.assertFalse(summary["shadow_mode_default"])
        self.assertEqual(summary["shadow_serving_default"], "pytorch")
        self.assertTrue(summary["old_model_untouched"])
        self.assertEqual(summary["final_test_samples_used"], 0)
        self.assertFalse(summary["training_performed"])
        self.assertFalse(summary["calibration_performed"])
        self.assertFalse(summary["ood_evaluation_performed"])
        audit = json.loads((destination.parent / "numerical_reproducibility_audit.json").read_text())
        for field in (
            "authoritative_path",
            "stored_phase22_bf16_logit",
            "fresh_fp32_forward",
            "fresh_bf16_model_forward",
            "image_bytes_match_manifest",
            "reproduced_stored_logit",
        ):
            self.assertIn(field, audit)
        self.assertFalse(audit["reproduced_stored_logit"])
        self.assertIn("fp32 reference", audit["authoritative_path"])
        quality = json.loads((destination.parent / "quality_latency_profile.json").read_text())
        self.assertFalse(quality["rule_changed"])
        self.assertTrue(quality["shared_decode"]["flags_match"])
        self.assertFalse(quality["resized_model_buffer"]["used_for_decisions"])
        self.assertFalse(quality["resized_model_buffer"]["small_flags_match"])
        self.assertIn("LOW_RESOLUTION", quality["resized_model_buffer"]["small_original_flags"])
        config = (REPO / "models" / "dinov3_authenticity_classifier" / "config.pbtxt").read_text()
        self.assertIn('name: "dinov3_authenticity_classifier"', config)
        self.assertIn("512", config)
        self.assertNotIn("518", config)
        old = (REPO / "models" / "dinov2_classifier" / "config.pbtxt").read_text()
        self.assertIn("518", old)
        checkpoint = REPO / "ml_rtx5080" / "experiments" / "v2_dinov3_cls_patch_attention" / "epoch_018.pt"
        self.assertEqual(checkpoint.stat().st_size, 1046865163)

    def test_triton_parity_is_not_marked_passed_when_unavailable(self) -> None:
        destination = REPO / "ml_rtx5080" / "experiments" / "backend_parity_v2" / "shadow_model_summary.json"
        if not destination.is_file():
            import backend_parity_v2

            backend_parity_v2.run_phase29()
        summary = json.loads(destination.read_text())
        if summary["triton_parity_status"] != "passed":
            self.skipTest(summary["triton_reason"])
        self.fail("Triton parity was marked passed without a live comparison in this test")


if __name__ == "__main__":
    unittest.main()
