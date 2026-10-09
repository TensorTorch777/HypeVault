"""Phase 40 declaration semantics and DINOv3 isolation. No model scoring."""

from __future__ import annotations

import json
import os
import sys
import unittest
from pathlib import Path

from PIL import Image

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "backend"))

LOCK = REPO / "ml_rtx5080" / "experiments" / "scope_lock_v1"
HARDENING = REPO / "ml_rtx5080" / "experiments" / "research_demo_hardening_v1"


class ResearchDemoHardeningTests(unittest.TestCase):
    def tearDown(self) -> None:
        os.environ.pop("HYPEVAULT_DEPLOYMENT_MODE", None)

    def test_contract_is_declaration_only(self) -> None:
        contract = json.loads((LOCK / "scope_contract.json").read_text())
        self.assertEqual(contract["brand_input_mode"], "USER_DECLARED")
        self.assertFalse(contract["brand_identity_verified_by_model"])
        self.assertFalse(contract["brand_identity_verified_by_image"])
        self.assertEqual(contract["brand_scope_gate"], "DECLARATION_ONLY")
        self.assertIn("not independently verified from the image", contract["brand_declaration_warning"])
        self.assertIn("outside the current model scope", contract["brand_verification"])

    def test_claims_do_not_become_brand_identification(self) -> None:
        claims = json.loads((LOCK / "claims_matrix.json").read_text())
        named = claims["named_capabilities"]
        self.assertEqual(named["GENERAL_LUXURY_WATCH_DETECTOR"], "NOT_SUPPORTED")
        self.assertEqual(named["UNSEEN_BRAND_REJECTION"], "NOT_SUPPORTED")
        self.assertEqual(named["OOD_PROTECTION"], "NOT_SUPPORTED")
        self.assertEqual(named["PRODUCTION_READY"], "NOT_SUPPORTED")
        self.assertEqual(named["FIVE_BRAND_CLASSIFICATION"], "SUPPORTED_WITH_SCOPE")
        self.assertEqual(named["BRAND_IDENTITY_FROM_IMAGE"], "NOT_SUPPORTED")
        self.assertIn("not brand identification", claims["five_brand_classification_meaning"])

    def test_unsupported_partial_and_missing_brands_fail_closed(self) -> None:
        from inference.scope_gate import evaluate_declared_brand, unsupported_scope_body

        for raw in (None, "", "   ", "Patek", "AP", "Rolex", "Omega", "Cartier", "Tudor", "Patek Philippe Nautilus"):
            result = evaluate_declared_brand(raw)
            self.assertEqual(result.scope_status, "UNSUPPORTED_SCOPE")
            self.assertIsNone(result.decision)
        body = unsupported_scope_body()
        self.assertEqual(body["status"], "UNSUPPORTED_SCOPE")
        self.assertIsNone(body["decision"])
        self.assertIn("five-brand research scope", body["reason"])

    def test_errors_are_not_authenticity_verdicts(self) -> None:
        from inference.schemas import AuthenticateResponse
        from inference.scope_gate import failure_body

        for status, reason in (
            ("INVALID_INPUT", "Invalid image file"),
            ("MODEL_ERROR", "The authenticity model did not return a result."),
            ("POLICY_ERROR", "DINOv3 candidate is blocked in production mode."),
        ):
            body = failure_body(status, reason)
            self.assertEqual(body["status"], status)
            self.assertIsNone(body["decision"])
            self.assertNotIn(body["status"], {"AUTHENTIC", "FAKE", "REVIEW"})
        self.assertNotIn("verified_brand", AuthenticateResponse.model_fields)
        self.assertEqual(AuthenticateResponse.model_fields["brand_verification"].default, "NOT_PERFORMED")
        self.assertEqual(AuthenticateResponse.model_fields["model"].default, "LEGACY_DINOV2")
        self.assertIs(AuthenticateResponse.model_fields["research_candidate"].default, False)
        self.assertEqual(AuthenticateResponse.model_fields["production_validation"].default, "NOT_ESTABLISHED")

    def test_live_defaults_stay_on_dinov2(self) -> None:
        from inference.research_guard import deployment_mode, dinov3_research_allowed, live_path_targets_dinov3

        config = (REPO / "backend" / "config.py").read_text()
        self.assertIn('inference_backend: str = "triton"', config)
        self.assertIn('triton_model_name: str = "dinov2_classifier"', config)
        self.assertIn('local_model_path: str = ""', config)
        os.environ.pop("HYPEVAULT_DEPLOYMENT_MODE", None)
        self.assertEqual(deployment_mode(), "LEGACY_PRODUCTION")
        self.assertFalse(dinov3_research_allowed())
        self.assertFalse(live_path_targets_dinov3("", "dinov2_classifier", "dinov2_vitg14_reg"))
        self.assertTrue(
            live_path_targets_dinov3(
                "ml_rtx5080/experiments/v2_dinov3_cls_patch_attention/epoch_018.pt",
                "dinov2_classifier",
                "dinov2_vitg14_reg",
            )
        )
        routes = (REPO / "backend" / "inference" / "routes.py").read_text()
        self.assertLess(routes.index("live_path_targets_dinov3"), routes.index("classify_image("))
        self.assertNotIn("verified_brand", routes)
        self.assertNotIn("shadow_v1", routes)
        for name in ("verdict.py", "dinov2_model.py", "local_torch.py", "triton_client.py"):
            text = (REPO / "backend" / "inference" / name).read_text()
            self.assertNotIn("HYPEVAULT_DEPLOYMENT_MODE", text)
            self.assertNotIn("dinov3", text.lower())

    def test_production_mode_blocks_dinov3_before_a_verdict(self) -> None:
        from inference.shadow import PolicyConfigurationError, _shadow_logit, compose_shadow_record

        os.environ["HYPEVAULT_DEPLOYMENT_MODE"] = "production"
        with self.assertRaises(PolicyConfigurationError) as caught:
            _shadow_logit(None)
        self.assertIn("blocked outside research mode", str(caught.exception))
        record = compose_shadow_record("AUTHENTIC", 0.99, image=Image.new("RGB", (24, 24), (1, 2, 3)), enabled=True)
        self.assertIsNone(record["shadow_decision"])
        self.assertEqual(record["shadow_system_status"], "POLICY_ERROR")
        self.assertNotIn(record["shadow_decision"], {"AUTHENTIC", "FAKE"})
        self.assertEqual(record["shadow_model_role"], "RESEARCH MODEL — NOT FOR PRODUCTION")

    def test_research_mode_is_the_only_dinov3_allowance(self) -> None:
        from inference.research_guard import dinov3_research_allowed

        os.environ["HYPEVAULT_DEPLOYMENT_MODE"] = "research"
        self.assertTrue(dinov3_research_allowed())
        os.environ["HYPEVAULT_DEPLOYMENT_MODE"] = "shadow"
        self.assertTrue(dinov3_research_allowed())
        os.environ["HYPEVAULT_DEPLOYMENT_MODE"] = "production"
        self.assertFalse(dinov3_research_allowed())

    def test_scope_matrix_and_routing_audit_exist(self) -> None:
        matrix = json.loads((HARDENING / "scope_matrix.json").read_text())
        routing = json.loads((HARDENING / "routing_audit.json").read_text())
        errors = json.loads((HARDENING / "error_semantics.json").read_text())
        self.assertEqual(matrix["brand_identity_from_image"], "not_available")
        self.assertEqual(matrix["unsupported_brand"], "unsupported_scope")
        self.assertFalse(routing["dinov3_production_traffic_allowed"])
        self.assertFalse(routing["automatic_fallback_between_dinov2_and_dinov3"])
        self.assertIn("UNSUPPORTED_SCOPE", errors["statuses"])
        self.assertIn("MODEL_ERROR", errors["statuses"])
        self.assertIn("INVALID_INPUT", errors["statuses"])


if __name__ == "__main__":
    unittest.main()
