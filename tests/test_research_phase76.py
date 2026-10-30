"""The Phase 76 DINOv3 + Perception Encoder head as a third research model. Mocks are not GPU serving evidence."""

from __future__ import annotations

import io
import json
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from PIL import Image

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "backend"))
sys.path.insert(0, str(REPO / "ml_rtx5080"))

HEAD_SHA = json.loads((REPO / "ml_rtx5080/experiments/phase76_fusion/frozen_rule.json").read_text())["model"]["sha256"]
CPU_PROFILE = {"kinds": frozenset({"KIND_CPU"}), "cuda_parameters": {}}


class _Upload:
    def __init__(self, raw: bytes, content_type: str = "image/jpeg") -> None:
        self._raw = raw
        self.content_type = content_type

    async def read(self) -> bytes:
        return self._raw


def _jpeg() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (64, 48), (8, 9, 10)).save(buffer, format="JPEG")
    return buffer.getvalue()


def _phase76_route(margin: float):
    async def infer(model_id, model_input):
        assert model_id == "phase76_fusion", model_id
        assert isinstance(model_input, Image.Image), type(model_input)
        return {
            "logit": margin,
            "logical_id": model_id,
            "triton_name": None,
            "version": "1",
            "response_model": "PHASE76_DINOV3_PE",
            "temperature": None,
        }

    return infer


class Phase76ResearchModelTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        os.environ["HYPEVAULT_DEPLOYMENT_MODE"] = "research"

    def tearDown(self) -> None:
        os.environ.pop("HYPEVAULT_DEPLOYMENT_MODE", None)

    def test_phase76_is_allowlisted_in_process(self) -> None:
        from inference.model_router import IN_PROCESS_GPU, PHASE76_FUSION, allowlist, resolve_model

        self.assertEqual(allowlist(), ("dinov2_legacy", "dinov3_experimental", "phase76_fusion"))
        spec = resolve_model(PHASE76_FUSION)
        self.assertIsNone(spec["triton_name"])
        self.assertEqual(spec["response_model"], "PHASE76_DINOV3_PE")
        self.assertEqual(spec["output_meaning"], "score_minus_t_fake")
        self.assertEqual(spec["decision_threshold"], 0.0)
        self.assertEqual(spec["validated_serving"], {IN_PROCESS_GPU: {}})

    async def test_readiness_follows_the_in_process_head_and_the_live_setting(self) -> None:
        from inference.model_router import readiness_report

        common = (
            patch("inference.model_router.named_model_ready", new=AsyncMock(return_value=True)),
            patch("inference.model_router.named_model_serving_profile", new=AsyncMock(return_value=CPU_PROFILE)),
            patch("inference.model_router.triton_server_status", new=AsyncMock(return_value={"live": True, "ready": True})),
        )
        with common[0], common[1], common[2], patch("inference.live_screening.phase76_unavailable_reason", return_value=None), patch(
            "inference.model_router.settings.live_screening_model", "phase76_fusion"
        ):
            rows = {row["logical_id"]: row for row in (await readiness_report())["models"]}
        self.assertTrue(rows["phase76_fusion"]["ready"])
        self.assertEqual(rows["phase76_fusion"]["served_instance_kinds"], ["IN_PROCESS_GPU"])
        self.assertTrue(rows["phase76_fusion"]["same_model_as_live_route"])
        self.assertFalse(rows["dinov3_experimental"]["same_model_as_live_route"])
        self.assertFalse(rows["dinov2_legacy"]["same_model_as_live_route"])

        with common[0], common[1], common[2], patch(
            "inference.live_screening.phase76_unavailable_reason", return_value="CUDA_UNAVAILABLE"
        ), patch("inference.model_router.settings.live_screening_model", "dinov3_frozen"):
            rows = {row["logical_id"]: row for row in (await readiness_report())["models"]}
        self.assertFalse(rows["phase76_fusion"]["ready"])
        self.assertEqual(rows["phase76_fusion"]["unavailable_reason"], "CUDA_UNAVAILABLE")
        self.assertFalse(rows["phase76_fusion"]["same_model_as_live_route"])

    async def test_phase76_never_reaches_triton(self) -> None:
        from inference.model_router import PHASE76_FUSION, ModelRoutingError, infer_allowlisted_model

        triton = AsyncMock(side_effect=AssertionError("Phase 76 reached Triton"))
        pil = Image.new("RGB", (64, 48))
        with (
            patch("inference.model_router.infer_named_model", triton),
            patch("inference.live_screening.phase76_unavailable_reason", return_value=None),
            patch("inference.live_screening.phase76_margin", return_value=0.75) as margin,
        ):
            result = await infer_allowlisted_model(PHASE76_FUSION, pil)
        self.assertEqual(result["logit"], 0.75)
        self.assertEqual(result["response_model"], "PHASE76_DINOV3_PE")
        margin.assert_called_once_with(pil)

        with (
            patch("inference.model_router.infer_named_model", triton),
            patch("inference.live_screening.phase76_unavailable_reason", return_value="HEAD_FILES_MISSING"),
            patch("inference.live_screening.phase76_margin", side_effect=AssertionError("ran while unavailable")),
        ):
            with self.assertRaises(ModelRoutingError) as caught:
                await infer_allowlisted_model(PHASE76_FUSION, pil)
        self.assertEqual(caught.exception.status, "MODEL_UNAVAILABLE")
        self.assertIn("HEAD_FILES_MISSING", caught.exception.message)

        with (
            patch("inference.live_screening.phase76_unavailable_reason", return_value=None),
            patch("inference.live_screening.phase76_margin", return_value=float("nan")),
        ):
            with self.assertRaises(ModelRoutingError) as caught:
                await infer_allowlisted_model(PHASE76_FUSION, pil)
        self.assertEqual(caught.exception.status, "INFERENCE_ERROR")
        triton.assert_not_awaited()

    async def test_verify_applies_the_frozen_cut_and_reports_the_head(self) -> None:
        from inference.research_routes import research_verify

        for margin, expected in ((0.3, "FAKE"), (0.0, "FAKE"), (-0.3, "AUTHENTIC")):
            with (
                patch("inference.research_routes.infer_allowlisted_model", new=AsyncMock(side_effect=_phase76_route(margin))),
                patch("inference.live_screening.phase76_head_sha256", return_value=HEAD_SHA),
            ):
                body = await research_verify(
                    current_user=object(), image=_Upload(_jpeg()), brand="Patek Philippe", logical_model="phase76_fusion"
                )
            self.assertEqual(body.decision, expected, margin)
            self.assertEqual(body.model, "PHASE76_DINOV3_PE")
            self.assertEqual(body.model_scope, "PHASE76_DINOV3_PE")
            self.assertEqual(body.checkpoint_sha, HEAD_SHA)
            self.assertEqual(body.policy_version, "phase76_frozen_cut")
            self.assertEqual(body.publication_decision, "BLOCKED")
            self.assertFalse(body.production_ready)

    async def test_explanation_matches_the_verified_phase76_class(self) -> None:
        from inference.research_routes import research_explain

        with patch("inference.research_routes.infer_allowlisted_model", new=AsyncMock(side_effect=_phase76_route(0.4))):
            response = await research_explain(
                current_user=object(), image=_Upload(_jpeg()), brand="Patek Philippe", logical_model="phase76_fusion"
            )
        payload = json.loads(response.body)
        self.assertEqual(payload["status"], "ok")
        self.assertEqual(payload["classification"]["model"], "PHASE76_DINOV3_PE")
        self.assertEqual(payload["classification"]["decision"], "FAKE")
        self.assertEqual(payload["sensitivity"]["baseline_decision"], "FAKE")
        self.assertIn("frozen cut", payload["explanation"]["limitations"])
        self.assertEqual(payload["publication_decision"], "BLOCKED")

    def test_success_body_requires_the_head_digest(self) -> None:
        from inference.research_access import research_success_body

        with self.assertRaises(ValueError):
            research_success_body(
                decision="FAKE", declared_brand="Patek Philippe", checkpoint_sha=None,
                policy_version="phase76_frozen_cut", model="PHASE76_DINOV3_PE", model_scope="PHASE76_DINOV3_PE",
            )
        with self.assertRaises(ValueError):
            research_success_body(
                decision="REVIEW", declared_brand="Patek Philippe", checkpoint_sha=HEAD_SHA,
                policy_version="phase76_frozen_cut", model="PHASE76_DINOV3_PE", model_scope="PHASE76_DINOV3_PE",
            )


if __name__ == "__main__":
    unittest.main()
