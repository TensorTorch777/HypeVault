"""Phase 41 customer-boundary checks. No model load and no final-test access."""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "backend"))

FROZEN_SHA = "5a38c93fd442b03653c65d2a5ecc9c2687ef152f7c5c020763e4ce1fd9c7d28f"
FROZEN_TEMPERATURE = 0.24038200410185356


class CustomerBoundaryTests(unittest.TestCase):
    def tearDown(self) -> None:
        os.environ.pop("HYPEVAULT_DEPLOYMENT_MODE", None)

    def test_state_machine_fail_closed(self) -> None:
        from inference.research_guard import deployment_state, dinov3_research_allowed

        os.environ.pop("HYPEVAULT_DEPLOYMENT_MODE", None)
        self.assertEqual(deployment_state(), "LEGACY_PRODUCTION")
        self.assertFalse(dinov3_research_allowed())

        os.environ["HYPEVAULT_DEPLOYMENT_MODE"] = "production"
        self.assertEqual(deployment_state(), "LEGACY_PRODUCTION")
        self.assertFalse(dinov3_research_allowed())

        os.environ["HYPEVAULT_DEPLOYMENT_MODE"] = "research"
        self.assertEqual(deployment_state(), "RESEARCH")
        self.assertTrue(dinov3_research_allowed())

        os.environ["HYPEVAULT_DEPLOYMENT_MODE"] = "shadow"
        self.assertEqual(deployment_state(), "SHADOW")
        self.assertTrue(dinov3_research_allowed())

        for raw in ("live", "prod", "RESEARCHING", "dinov3", "true", " "):
            os.environ["HYPEVAULT_DEPLOYMENT_MODE"] = raw
            self.assertEqual(deployment_state(), "PRODUCTION_BLOCKED", raw)
            self.assertFalse(dinov3_research_allowed())

    def test_dinov3_blocks_never_return_a_verdict(self) -> None:
        from inference.research_access import checkpoint_block_body, mode_block_body, temperature_block_body
        from inference.research_guard import dinov3_research_allowed

        for raw in (None, "production", "unknown-mode", ""):
            if raw is None:
                os.environ.pop("HYPEVAULT_DEPLOYMENT_MODE", None)
            else:
                os.environ["HYPEVAULT_DEPLOYMENT_MODE"] = raw
            self.assertFalse(dinov3_research_allowed())
            body = mode_block_body()
            self.assertIsNotNone(body)
            assert body is not None
            self.assertEqual(body["status"], "POLICY_ERROR")
            self.assertIsNone(body["decision"])
            self.assertNotIn(body["status"], {"AUTHENTIC", "FAKE", "REVIEW"})

        mismatch = checkpoint_block_body("0" * 64)
        assert mismatch is not None
        self.assertEqual(mismatch["status"], "POLICY_ERROR")
        self.assertIsNone(mismatch["decision"])
        self.assertIsNone(checkpoint_block_body(FROZEN_SHA))
        drifted = temperature_block_body(1.0)
        assert drifted is not None
        self.assertEqual(drifted["status"], "POLICY_ERROR")
        self.assertIsNone(drifted["decision"])
        self.assertIsNone(temperature_block_body(FROZEN_TEMPERATURE))

    def test_unknown_mode_blocks_the_shadow_forward(self) -> None:
        from inference.shadow import PolicyConfigurationError, _shadow_logit

        os.environ["HYPEVAULT_DEPLOYMENT_MODE"] = "not-a-mode"
        with self.assertRaises(PolicyConfigurationError) as caught:
            _shadow_logit(None)
        self.assertIn("blocked outside research mode", str(caught.exception))

    def test_research_success_names_the_frozen_candidate(self) -> None:
        from inference.schemas import AuthenticateResponse, ResearchVerifyResponse
        from inference.research_access import research_success_body

        body = research_success_body(
            decision="REVIEW",
            declared_brand="Patek Philippe",
            checkpoint_sha=FROZEN_SHA,
            policy_version="shadow_v1",
        )
        parsed = ResearchVerifyResponse(**body)
        self.assertEqual(parsed.brand_verification, "NOT_PERFORMED")
        self.assertEqual(parsed.model_scope, "FIVE_BRAND_RESEARCH_PROTOTYPE")
        self.assertTrue(parsed.research_only)
        self.assertFalse(parsed.production_ready)
        self.assertEqual(parsed.checkpoint_sha, FROZEN_SHA)
        self.assertEqual(parsed.model, "DINOV3_RESEARCH_PROTOTYPE")
        self.assertEqual(parsed.publication_decision, "BLOCKED")
        self.assertNotIn("verified_brand", ResearchVerifyResponse.model_fields)
        legacy = AuthenticateResponse(
            verdict="FAKE",
            confidence=0.2,
            s3_url="s3://example",
            listing_id="listing",
            listing_status="rejected",
            status="FAKE",
            model_decision="FAKE",
            declared_brand="Patek Philippe",
        )
        self.assertEqual(legacy.model, "LEGACY_DINOV2")
        self.assertEqual(legacy.model_status, "LEGACY")
        self.assertFalse(legacy.research_candidate)
        self.assertEqual(legacy.production_validation, "NOT_ESTABLISHED")
        self.assertNotIn("checkpoint_sha", legacy.model_dump())

    def test_routes_stay_on_separate_models(self) -> None:
        legacy = (REPO / "backend" / "inference" / "routes.py").read_text()
        research = (REPO / "backend" / "inference" / "research_routes.py").read_text()
        main = (REPO / "backend" / "main.py").read_text()
        self.assertIn("LEGACY_DINOV2", legacy)
        self.assertNotIn("run_shadow_image", legacy)
        self.assertNotIn(FROZEN_SHA, legacy)
        self.assertLess(research.index("mode_block_body"), research.index("infer_allowlisted_model"))
        self.assertLess(research.index("checkpoint_block_body"), research.index("infer_allowlisted_model"))
        self.assertNotIn("run_shadow_image", research)
        self.assertIn('prefix="/research"', main)
        self.assertIn('prefix="/verify"', main)
        compose = (REPO / "infra" / "docker-compose.yml").read_text()
        self.assertNotIn("HYPEVAULT_DEPLOYMENT_MODE", compose)
        self.assertNotIn("dinov3", compose.lower())

    def test_boundary_artifacts_exist(self) -> None:
        root = REPO / "ml_rtx5080" / "experiments" / "customer_boundary_v1"
        for name in (
            "route_audit.json",
            "deployment_state_machine.json",
            "research_api_contract.json",
            "legacy_route_contract.json",
            "production_block_matrix.json",
            "claim_audit.json",
            "customer_boundary_report.md",
        ):
            self.assertTrue((root / name).is_file(), name)


if __name__ == "__main__":
    unittest.main()
