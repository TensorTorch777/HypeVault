"""Occlusion evidence and the research explanation contract. No checkpoint or threshold edits."""

from __future__ import annotations

import asyncio
import io
import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from PIL import Image

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "backend"))


def _jpeg() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (32, 24), (20, 30, 40)).save(buffer, format="JPEG")
    return buffer.getvalue()


def _client():
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from auth.deps import get_current_user
    from inference.research_routes import router

    app = FastAPI()
    app.include_router(router, prefix="/research")
    app.dependency_overrides[get_current_user] = lambda: SimpleNamespace(
        email="researcher@example.com", role="buyer"
    )
    return TestClient(app)


def _route(logit: float, model: str = "LEGACY_DINOV2"):
    async def infer(model_id, array):
        del model_id, array
        return {"logit": logit, "response_model": model, "version": "1"}

    return infer


class OcclusionTests(unittest.TestCase):
    def test_grid_covers_the_image_without_gaps(self) -> None:
        from inference.occlusion import FILL_RGB, mask_region, occlusion_boxes

        boxes = occlusion_boxes(32, 24)
        self.assertEqual(len(boxes), 16)
        covered = Image.new("L", (32, 24), 0)
        for box in boxes:
            covered.paste(255, (box["x"], box["y"], box["x"] + box["width"], box["y"] + box["height"]))
        self.assertEqual(covered.getextrema(), (255, 255))
        masked = mask_region(Image.new("RGB", (32, 24), (1, 2, 3)), boxes[0])
        self.assertEqual(masked.getpixel((0, 0)), FILL_RGB)

    def test_empty_or_misaligned_evidence_is_rejected(self) -> None:
        from inference.occlusion import sensitivity_record

        with self.assertRaises(ValueError):
            sensitivity_record(
                baseline_logit=1.0,
                masked_logits=[],
                boxes=[],
                model_id="LEGACY_DINOV2",
                model_version="1",
                preprocessing_id="legacy_square_resize_504_imagenet",
                decision="FAKE",
            )

    def test_equal_logits_do_not_support_a_visual_summary(self) -> None:
        from inference.occlusion import sensitivity_record

        boxes = [{"row": 0, "col": 0, "x": 0, "y": 0, "width": 1, "height": 1}]
        record = sensitivity_record(
            baseline_logit=1.25,
            masked_logits=[1.25],
            boxes=boxes,
            model_id="LEGACY_DINOV2",
            model_version="1",
            preprocessing_id="legacy_square_resize_504_imagenet",
            decision="FAKE",
        )
        self.assertFalse(record["evidence_supports_visual_summary"])
        self.assertEqual(record["patches"][0]["delta_logit"], 0.0)

    def test_timeout_and_inference_failure_stop_without_a_partial_map(self) -> None:
        from PIL import Image as PilImage

        from inference.occlusion import measure_regions

        image = PilImage.new("RGB", (8, 8), (4, 5, 6))

        async def slow(sample):
            del sample
            return 1.0

        with self.assertRaises(TimeoutError):
            asyncio.run(measure_regions(image, slow, timeout_s=-1))

        async def broken(sample):
            del sample
            raise RuntimeError("inference failed")

        with self.assertRaises(RuntimeError):
            asyncio.run(measure_regions(image, broken, timeout_s=5))

    def test_a_slow_final_inference_cannot_return_success_after_the_deadline(self) -> None:
        from PIL import Image as PilImage

        from inference.occlusion import measure_regions

        image = PilImage.new("RGB", (8, 8), (4, 5, 6))
        calls = {"n": 0}

        async def late_final(sample):
            del sample
            calls["n"] += 1
            if calls["n"] == 17:
                await asyncio.sleep(0.2)
            return 1.0

        with self.assertRaises(TimeoutError):
            asyncio.run(measure_regions(image, late_final, timeout_s=0.05))

    def test_display_cutoff_is_a_provisional_heuristic(self) -> None:
        from inference.occlusion import WEAK_ABS_DELTA, WEAK_ABS_DELTA_ROLE, sensitivity_record

        boxes = [{"row": 0, "col": 0, "x": 0, "y": 0, "width": 1, "height": 1}]
        record = sensitivity_record(
            baseline_logit=0.0,
            masked_logits=[0.002],
            boxes=boxes,
            model_id="LEGACY_DINOV2",
            model_version="1",
            preprocessing_id="legacy_square_resize_504_imagenet",
            decision="AUTHENTIC",
        )
        self.assertEqual(WEAK_ABS_DELTA, 0.001)
        self.assertEqual(WEAK_ABS_DELTA_ROLE, "provisional_heuristic")
        self.assertIn("not tuned on the locked final test set", record["method"]["weak_abs_delta_note"])
        self.assertNotIn("component", " ".join(str(item) for item in record["patches"][0]))


class ExplanationTextTests(unittest.TestCase):
    def _evidence(self, decision: str = "FAKE", delta: float = 0.0) -> dict:
        return {
            "model_id": "LEGACY_DINOV2",
            "model_version": "1",
            "preprocessing_id": "legacy_square_resize_504_imagenet",
            "baseline_logit": 1.0,
            "baseline_decision": decision,
            "patches": [{"row": 0, "col": 1, "x": 0, "y": 0, "width": 1, "height": 1, "delta_logit": delta}],
            "max_abs_delta_logit": abs(delta),
            "evidence_supports_visual_summary": abs(delta) >= 1e-3,
        }

    def test_fallback_states_when_evidence_is_too_weak(self) -> None:
        from inference.explanation_text import deterministic_explanation

        text = deterministic_explanation(self._evidence())
        self.assertIn("FAKE", text["observation"])
        self.assertIn("did not change the raw logit enough", text["observation"])
        self.assertEqual(text["source"], "deterministic_fallback")
        self.assertIn("not proof of authenticity", text["limitations"])

    def test_llm_timeout_invalid_output_and_unsupported_claims_fall_back(self) -> None:
        from inference.explanation_text import narrate

        evidence = self._evidence(delta=0.5)
        os.environ["HYPEVAULT_EXPLANATION_LLM_URL"] = "http://llm.invalid/explain"

        async def timed_out(payload):
            del payload
            raise TimeoutError("llm timeout")

        async def invalid(payload):
            del payload
            return {"observation": ""}

        async def unsupported(payload):
            del payload
            return {
                "observation": "The model classified this as FAKE because the dial is a counterfeit defect.",
                "hypothesis": "The serial number proves fake.",
                "source": "llm",
                "model_id": "LEGACY_DINOV2",
                "baseline_decision": "FAKE",
            }

        async def mismatched(payload):
            del payload
            return {
                "observation": "The model classified this as AUTHENTIC.",
                "hypothesis": "Only measured changes.",
                "source": "llm",
                "model_id": "OTHER",
                "baseline_decision": "AUTHENTIC",
            }

        try:
            for generator in (timed_out, invalid, unsupported, mismatched):
                text = asyncio.run(narrate(evidence, generator))
                self.assertEqual(text["source"], "deterministic_fallback")
                self.assertEqual(text["llm_status"], "rejected")
                self.assertEqual(text["baseline_decision"], "FAKE")
                self.assertNotIn("counterfeit defect", text["observation"])
        finally:
            os.environ.pop("HYPEVAULT_EXPLANATION_LLM_URL", None)

    def test_model_id_mismatch_is_rejected(self) -> None:
        from inference.explanation_text import validate_explanation

        evidence = self._evidence()
        with self.assertRaises(ValueError):
            validate_explanation(
                {
                    "observation": "The model classified this as FAKE.",
                    "hypothesis": "Measured only.",
                    "source": "llm",
                    "model_id": "DINOV3_RESEARCH_PROTOTYPE",
                    "baseline_decision": "FAKE",
                },
                evidence,
            )


class ResearchExplainRouteTests(unittest.TestCase):
    def tearDown(self) -> None:
        os.environ.pop("HYPEVAULT_DEPLOYMENT_MODE", None)

    def test_explain_keeps_classification_publication_and_score_separate(self) -> None:
        os.environ["HYPEVAULT_DEPLOYMENT_MODE"] = "research"
        with (
            patch("config.settings.research_user_emails", "researcher@example.com"),
            patch("inference.research_routes.infer_allowlisted_model", new=_route(2.0)),
        ):
            response = _client().post(
                "/research/explain",
                files={"image": ("a.jpg", _jpeg(), "image/jpeg")},
                data={"brand": "Patek Philippe", "logical_model": "dinov2_legacy"},
            )
        body = response.json()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(body["status"], "ok")
        self.assertEqual(body["classification"]["decision"], "FAKE")
        self.assertEqual(body["publication_decision"], "BLOCKED")
        self.assertFalse(body["independent_authentication"])
        self.assertEqual(body["sensitivity"]["model_id"], "LEGACY_DINOV2")
        self.assertEqual(body["sensitivity"]["method"]["weak_abs_delta_role"], "provisional_heuristic")
        self.assertEqual(body["cost"]["inferences"], 17)
        self.assertGreaterEqual(body["cost"]["elapsed_ms"], 0)
        self.assertNotIn("probability that the watch is genuinely authentic", body["explanation"]["observation"])

    def test_timeout_returns_unavailable_without_invented_evidence(self) -> None:
        os.environ["HYPEVAULT_DEPLOYMENT_MODE"] = "research"
        with (
            patch("config.settings.research_user_emails", "researcher@example.com"),
            patch("inference.occlusion.measure_regions", new=AsyncMock(side_effect=TimeoutError("budget"))),
        ):
            response = _client().post(
                "/research/explain",
                files={"image": ("a.jpg", _jpeg(), "image/jpeg")},
                data={"brand": "Patek Philippe", "logical_model": "dinov2_legacy"},
            )
        self.assertEqual(response.json()["status"], "unavailable")
        self.assertIsNone(response.json()["sensitivity"])

    def test_inference_failure_returns_unavailable_without_invented_evidence(self) -> None:
        os.environ["HYPEVAULT_DEPLOYMENT_MODE"] = "research"
        infer = AsyncMock(side_effect=RuntimeError("triton down"))
        with (
            patch("config.settings.research_user_emails", "researcher@example.com"),
            patch("inference.research_routes.infer_allowlisted_model", new=infer),
        ):
            response = _client().post(
                "/research/explain",
                files={"image": ("a.jpg", _jpeg(), "image/jpeg")},
                data={"brand": "Patek Philippe", "logical_model": "dinov2_legacy"},
            )
        body = response.json()
        self.assertEqual(body["status"], "unavailable")
        self.assertIsNone(body["sensitivity"])
        self.assertIsNone(body["explanation"])
        self.assertEqual(body["publication_decision"], "BLOCKED")

    def test_model_identity_mismatch_discards_the_explanation(self) -> None:
        os.environ["HYPEVAULT_DEPLOYMENT_MODE"] = "research"
        with (
            patch("config.settings.research_user_emails", "researcher@example.com"),
            patch(
                "inference.research_routes.infer_allowlisted_model",
                new=_route(1.0, model="DINOV3_RESEARCH_PROTOTYPE"),
            ),
        ):
            response = _client().post(
                "/research/explain",
                files={"image": ("a.jpg", _jpeg(), "image/jpeg")},
                data={"brand": "Patek Philippe", "logical_model": "dinov2_legacy"},
            )
        self.assertEqual(response.json()["status"], "unavailable")
        self.assertIsNone(response.json()["sensitivity"])

    def test_frozen_model_policy_constants_are_unchanged(self) -> None:
        from inference.model_router import FROZEN_DINOV3_SHA256, FROZEN_DINOV3_TEMPERATURE, FROZEN_DINOV3_THRESHOLD
        from inference.publication_gate import AUTHENTICITY_MODEL_PRODUCTION_APPROVED

        self.assertEqual(
            FROZEN_DINOV3_SHA256,
            "5a38c93fd442b03653c65d2a5ecc9c2687ef152f7c5c020763e4ce1fd9c7d28f",
        )
        self.assertEqual(FROZEN_DINOV3_TEMPERATURE, 0.24038200410185356)
        self.assertEqual(FROZEN_DINOV3_THRESHOLD, 0.50)
        self.assertFalse(AUTHENTICITY_MODEL_PRODUCTION_APPROVED)
        router = (REPO / "backend" / "inference" / "model_router.py").read_text()
        self.assertIn("fe1daa0bf71c5e9b73267d40784442748b8fd1999a8d107979f1338c52f0fa66", router)


if __name__ == "__main__":
    unittest.main()
