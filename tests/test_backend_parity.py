"""Inference-path audit and shadow-only parity. Production verdicts stay in place."""

from __future__ import annotations

import json
import os
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
_FROZEN_CHECKPOINT = REPO / "ml_rtx5080" / "experiments" / "v2_dinov3_cls_patch_attention" / "epoch_018.pt"
sys.path.insert(0, str(REPO / "ml_rtx5080"))
sys.path.insert(0, str(REPO / "backend"))

import backend_parity
import inference.shadow as shadow
from inference.schemas import AuthenticateResponse
from inference.verdict import apply_min_authentic_confidence, logits_to_verdict
from policy_contract import POLICY_VERSION, PolicyConfigurationError
from reference_inference import validate_reference_identity


class BackendParityTests(unittest.TestCase):
    def test_backend_audit_records_the_production_mismatch(self) -> None:
        audit = backend_parity.build_backend_audit(hash_artifacts=False)
        from config import Settings, settings

        self.assertIn("dinov2_vitg14_reg", audit["answers"]["backbone"])
        self.assertIn(settings.dinov2_model_name, audit["answers"]["backbone"])
        self.assertIn("Not cls_patch_attention", audit["answers"]["head"])
        self.assertEqual(audit["answers"]["code_default_image_size"], 518)
        self.assertEqual(audit["answers"]["code_default_image_size"], Settings.model_fields["inference_img_size"].default)
        self.assertEqual(audit["answers"]["effective_image_size"], settings.inference_img_size)
        self.assertNotEqual(audit["answers"]["effective_image_size"], 512)
        self.assertFalse(audit["answers"]["configured_checkpoint_matches_frozen"])
        self.assertTrue(audit["answers"]["applies_sigmoid"])
        self.assertEqual(audit["answers"]["sigmoid_definition"], "sigmoid(raw_logit), with no temperature")
        self.assertEqual(audit["answers"]["threshold"], 0.50)
        self.assertTrue(audit["answers"]["assumes_dinov2"])
        self.assertFalse(audit["answers"]["local_and_triton_same_model"])
        self.assertFalse(audit["model_identity"]["identity_match"])
        self.assertTrue(audit["preprocessing"]["material_difference"])
        self.assertEqual(audit["preprocessing"]["reference"], "resize_pad_square_eval_v1")
        self.assertEqual(audit["preprocessing"]["reference_shape"], [3, 512, 512])
        self.assertEqual(audit["preprocessing"]["production_shape"], [1, 3, settings.inference_img_size, settings.inference_img_size])
        self.assertEqual(audit["preprocessing"]["integration_of_production_preprocess"], "stopped")
        self.assertFalse(audit["preprocessing"]["shadow_uses_production_preprocess"])
        self.assertFalse(audit["triton"]["corresponds_to_frozen_checkpoint"])
        self.assertEqual(audit["triton"]["parity_status"], "skipped")
        self.assertFalse(audit["shadow_mode_default"])
        self.assertFalse(audit["customer_traffic_used"])

    def test_checkpoint_and_model_metadata_identity(self) -> None:
        metadata = shadow.shadow_contract_metadata()
        self.assertEqual(metadata["model_family"], "dinov3")
        self.assertEqual(metadata["backbone"], "vit_base_patch16_dinov3.lvd1689m")
        self.assertEqual(metadata["head"], "cls_patch_attention")
        self.assertEqual(metadata["input_size"], 512)
        self.assertEqual(metadata["preprocessing_version"], "resize_pad_square_eval_v1")
        self.assertEqual(metadata["policy_version"], POLICY_VERSION)
        self.assertFalse(metadata["shadow_mode_default"])

    def test_invalid_config_fails_closed(self) -> None:
        for kwargs in (
            {"family": "dinov2"},
            {"head": "cls_only"},
            {"input_size": 504},
            {"input_size": 518},
            {"preprocessing_version": "resize_center_crop_eval_v1"},
            {"checkpoint_sha256": "0" * 64},
        ):
            with self.assertRaises(PolicyConfigurationError) as caught:
                validate_reference_identity(**kwargs)
            self.assertTrue(caught.exception.reason_codes)
            self.assertNotIn("AUTHENTIC", caught.exception.reason_codes)

    def test_invalid_input_fails_closed(self) -> None:
        result, _timings = shadow.run_shadow_image(b"this is not an image")
        self.assertIsNone(result.decision)
        self.assertEqual(result.system_status, "POLICY_ERROR")
        self.assertIn("POLICY_INVALID_INPUT", result.reason_codes)
        self.assertIsNone(result.ood_score)
        self.assertIsNone(result.ood_flag)
        missing, _timings = shadow.run_shadow_image(None)
        self.assertIsNone(missing.decision)
        self.assertEqual(missing.system_status, "POLICY_ERROR")

    def test_ood_unavailable_semantics(self) -> None:
        result, _timings = shadow.run_shadow_image(b"not-an-image")
        self.assertEqual(result.ood_status, "unavailable")
        self.assertIsNone(result.ood_score)
        self.assertIsNone(result.ood_flag)
        self.assertFalse(result.in_distribution_assumed)

    def test_shadow_mode_defaults_off_and_preserves_the_verdict(self) -> None:
        os.environ.pop("HYPEVAULT_SHADOW_POLICY", None)
        self.assertFalse(shadow.shadow_mode_enabled())
        record = shadow.compose_shadow_record("AUTHENTIC", 0.97, image=None, enabled=False)
        self.assertEqual(record, {"verdict": "AUTHENTIC", "confidence": 0.97})
        self.assertNotIn("shadow_decision", record)

    def test_existing_production_response_is_unchanged(self) -> None:
        self.assertEqual(
            set(AuthenticateResponse.model_fields),
            {
                "verdict",
                "confidence",
                "s3_url",
                "listing_id",
                "listing_status",
                "status",
                "model_decision",
                "screening_code",
                "screening_reason",
                "publication_decision",
                "declared_brand",
                "brand_verification",
                "model",
                "model_status",
                "research_candidate",
                "production_validation",
            },
        )
        self.assertEqual(AuthenticateResponse.model_fields["model"].default, "LEGACY_DINOV2")
        self.assertIs(AuthenticateResponse.model_fields["research_candidate"].default, False)
        self.assertNotIn("checkpoint_sha", AuthenticateResponse.model_fields)
        self.assertNotIn("verified_brand", AuthenticateResponse.model_fields)
        routes = (REPO / "backend" / "inference" / "routes.py").read_text()
        verdict = (REPO / "backend" / "inference" / "verdict.py").read_text()
        local_torch = (REPO / "backend" / "inference" / "local_torch.py").read_text()
        self.assertNotIn("shadow_v1", routes)
        self.assertNotIn("HYPEVAULT_SHADOW_POLICY", routes)
        self.assertNotIn("resize_pad_square_eval_v1", verdict)
        self.assertIn("1.0 / (1.0 + np.exp(-raw))", verdict)
        self.assertNotIn("cls_patch_attention", local_torch)
        import numpy as np

        raw_verdict, confidence = logits_to_verdict(np.asarray([-4.0], dtype=np.float32))
        self.assertEqual(raw_verdict, "AUTHENTIC")
        remapped, remapped_confidence = apply_min_authentic_confidence(raw_verdict, 0.70, 0.88)
        self.assertEqual(remapped, "FAKE")
        self.assertAlmostEqual(remapped_confidence, 0.30)

    def test_final_test_sample_is_rejected_without_opening_it(self) -> None:
        manifest = json.loads((REPO / "ml_rtx5080" / "experiments" / "dataset_audit" / "split_manifest_v2.json").read_text())
        test_path = Path(manifest["membership"]["test"][0])
        self.assertTrue(test_path.is_file())
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
                shadow.run_shadow_image(test_path)
        finally:
            image_module.open = original
        self.assertFalse(opened["called"])

    def test_temperature_applies_to_the_raw_logit(self) -> None:
        from policy import calibrated_probability_from_logit
        import math

        logit = -0.84
        calibrated = calibrated_probability_from_logit(logit)
        raw = 1.0 / (1.0 + math.exp(-logit))
        double = 1.0 / (1.0 + math.exp(-(raw / 0.24038200410185356)))
        self.assertNotAlmostEqual(calibrated, double, places=3)
        self.assertAlmostEqual(calibrated, 1.0 / (1.0 + math.exp(-logit / 0.24038200410185356)))

    @unittest.skipUnless(_FROZEN_CHECKPOINT.is_file(), "frozen checkpoint is local, not in git")
    def test_phase28_parity_artifacts(self) -> None:
        destination = REPO / "ml_rtx5080" / "experiments" / "backend_parity_v1" / "parity_summary.json"
        if not destination.is_file():
            backend_parity.run_phase28()
        summary = json.loads(destination.read_text())
        self.assertEqual(summary["checkpoint_sha256"], "5a38c93fd442b03653c65d2a5ecc9c2687ef152f7c5c020763e4ce1fd9c7d28f")
        self.assertEqual(summary["temperature"], 0.24038200410185356)
        self.assertEqual(summary["policy_version"], "shadow_v1")
        self.assertLessEqual(summary["local_parity"]["max_abs_logit_delta"], summary["tolerances"]["logit_atol"])
        self.assertLessEqual(summary["local_parity"]["max_abs_probability_delta"], summary["tolerances"]["probability_atol"])
        self.assertTrue(summary["local_parity"]["decisions_match"])
        self.assertTrue(summary["local_parity"]["quality_flags_match"])
        self.assertFalse(summary["local_parity"]["production_local_torch_executed"])
        self.assertEqual(summary["triton_parity_status"], "skipped")
        self.assertEqual(summary["phase22_regression"]["review"], 8)
        self.assertEqual(summary["phase22_regression"]["fake"], 1)
        self.assertEqual(summary["phase22_regression"]["false_authentic_count"], 0)
        self.assertEqual(summary["phase22_regression"]["named_cases"]["rm_resize_recompression"], "FAKE")
        self.assertEqual(summary["phase22_regression"]["named_cases"]["ap_mild_crop"], "REVIEW")
        self.assertLessEqual(summary["batch_parity"]["max_abs_logit_delta"], summary["tolerances"]["batch_logit_atol"])
        self.assertEqual(summary["final_test_samples_used"], 0)
        self.assertFalse(summary["training_performed"])
        self.assertFalse(summary["calibration_performed"])
        self.assertFalse(summary["ood_evaluation_performed"])
        self.assertFalse(summary["customer_traffic_used"])
        self.assertFalse(summary["live_backend_modified"])
        self.assertFalse(summary["shadow_mode_default"])
        self.assertEqual(summary["production_policy_status"], "not_promoted")
        latency = summary["latency"]["total_ms"]
        self.assertIn("p50", latency)
        self.assertIn("p95", latency)
        self.assertIn("p99", latency)
        golden = json.loads((destination.parent / "golden_cases.json").read_text())
        self.assertEqual(golden["final_test_samples_used"], 0)
        case_ids = [row["case_id"] for row in golden["cases"]]
        for required in (
            "clean_authentic",
            "clean_fake",
            "low_resolution",
            "low_contrast",
            "both_quality_flags",
            "strong_authentic_probability",
            "strong_fake_probability",
            "phase22_resize_source",
            "malformed_input",
        ):
            self.assertIn(required, case_ids)
        malformed = next(row for row in golden["cases"] if row["case_id"] == "malformed_input")
        self.assertIsNone(malformed["expected_decision"])
        self.assertEqual(malformed["expected_system_status"], "POLICY_ERROR")
        checkpoint = REPO / "ml_rtx5080" / "experiments" / "v2_dinov3_cls_patch_attention" / "epoch_018.pt"
        self.assertEqual(checkpoint.stat().st_size, 1046865163)


if __name__ == "__main__":
    unittest.main()
