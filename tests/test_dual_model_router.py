"""Phase 50 allowlisted dual-model routing. Mocks are not GPU serving evidence."""

from __future__ import annotations

import io
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from PIL import Image

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "backend"))
sys.path.insert(0, str(REPO / "ml_rtx5080"))

FROZEN_SHA = "5a38c93fd442b03653c65d2a5ecc9c2687ef152f7c5c020763e4ce1fd9c7d28f"
FROZEN_TEMPERATURE = 0.24038200410185356
LIVE_DINOV2_SHA = "fe1daa0bf71c5e9b73267d40784442748b8fd1999a8d107979f1338c52f0fa66"


class _Upload:
    def __init__(self, raw: bytes, content_type: str = "image/jpeg") -> None:
        self._raw = raw
        self.content_type = content_type

    async def read(self) -> bytes:
        return self._raw


def _jpeg() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (32, 24), (8, 9, 10)).save(buffer, format="JPEG")
    return buffer.getvalue()


class DualModelRouterTests(unittest.IsolatedAsyncioTestCase):
    def tearDown(self) -> None:
        os.environ.pop("HYPEVAULT_DEPLOYMENT_MODE", None)

    def test_allowlist_keeps_separate_contracts(self) -> None:
        from inference.model_router import (
            DINOV2_LEGACY,
            DINOV3_EXPERIMENTAL,
            FROZEN_DINOV3_TEMPERATURE,
            PARITY_LOGIT_ATOL,
            PARITY_PROBABILITY_ATOL,
            resolve_model,
        )

        dinov2 = resolve_model(DINOV2_LEGACY)
        dinov3 = resolve_model(DINOV3_EXPERIMENTAL)
        self.assertEqual(dinov2["triton_name"], "dinov2_vitb14_live")
        self.assertEqual(dinov3["triton_name"], "dinov3_authenticity_candidate")
        self.assertNotEqual(dinov2["triton_name"], dinov3["triton_name"])
        self.assertEqual(dinov2["input_dims"], [3, 504, 504])
        self.assertTrue(dinov2["same_model_as_live_route"])
        self.assertEqual(dinov2["source_checkpoint_sha256"], LIVE_DINOV2_SHA)
        self.assertEqual(dinov2["validated_instance_kinds"], frozenset())
        self.assertEqual(dinov3["validated_instance_kinds"], frozenset({"KIND_CPU"}))
        self.assertEqual(dinov3["input_dims"], [3, 512, 512])
        self.assertIsNone(dinov2["temperature"])
        self.assertEqual(dinov3["temperature"], FROZEN_TEMPERATURE)
        self.assertEqual(FROZEN_DINOV3_TEMPERATURE, FROZEN_TEMPERATURE)
        self.assertEqual(dinov3["checkpoint_sha256"], FROZEN_SHA)
        self.assertEqual(dinov2["output_meaning"], "raw_logit")
        self.assertEqual(dinov3["output_meaning"], "raw_logit")
        self.assertEqual(PARITY_LOGIT_ATOL, 1e-4)
        self.assertEqual(PARITY_PROBABILITY_ATOL, 1e-5)
        with self.assertRaises(Exception):
            resolve_model("dinov3_authenticity_candidate")
        with self.assertRaises(Exception):
            resolve_model("../epoch_018.pt")

    async def test_missing_model_does_not_call_the_other_model(self) -> None:
        from inference.model_router import DINOV3_EXPERIMENTAL, ModelRoutingError, infer_allowlisted_model

        ready = AsyncMock(return_value=False)
        infer = AsyncMock(side_effect=AssertionError("silent fallback"))
        with patch("inference.model_router.named_model_ready", ready), patch(
            "inference.model_router.infer_named_model", infer
        ):
            with self.assertRaises(ModelRoutingError) as caught:
                await infer_allowlisted_model(DINOV3_EXPERIMENTAL, None)
        self.assertEqual(caught.exception.status, "MODEL_UNAVAILABLE")
        ready.assert_awaited_once_with("dinov3_authenticity_candidate", "1")
        infer.assert_not_awaited()

    async def test_non_finite_or_wrong_size_output_fails_closed(self) -> None:
        from inference.model_router import DINOV3_EXPERIMENTAL, ModelRoutingError, infer_allowlisted_model
        import numpy as np

        batch = np.zeros((1, 3, 512, 512), dtype=np.float32)
        for output in (
            np.asarray([[float("nan")]], dtype=np.float32),
            np.asarray([[float("inf")]], dtype=np.float32),
            np.asarray([[0.1], [0.2]], dtype=np.float32),
        ):
            with (
                patch("inference.model_router.named_model_ready", AsyncMock(return_value=True)),
                patch("inference.model_router.named_model_instance_kinds", AsyncMock(return_value=frozenset({"KIND_CPU"}))),
                patch("inference.model_router.infer_named_model", AsyncMock(return_value=output)),
            ):
                with self.assertRaises(ModelRoutingError) as caught:
                    await infer_allowlisted_model(DINOV3_EXPERIMENTAL, batch)
            self.assertEqual(caught.exception.status, "INFERENCE_ERROR")

    async def test_readiness_is_per_model_and_separate_from_server_ready(self) -> None:
        from inference.model_router import readiness_report

        async def ready(name, version):
            self.assertEqual(version, "1")
            return True

        with (
            patch("inference.model_router.named_model_ready", new=AsyncMock(side_effect=ready)),
            patch("inference.model_router.named_model_instance_kinds", new=AsyncMock(return_value=frozenset({"KIND_CPU"}))),
            patch("inference.model_router.triton_server_status", new=AsyncMock(return_value={"live": True, "ready": False})),
        ):
            report = await readiness_report()
        rows = {row["logical_id"]: row for row in report["models"]}
        self.assertTrue(rows["dinov3_experimental"]["ready"])
        self.assertIsNone(rows["dinov3_experimental"]["unavailable_reason"])
        self.assertFalse(rows["dinov2_legacy"]["ready"])
        self.assertEqual(rows["dinov2_legacy"]["unavailable_reason"], "PARITY_NOT_VALIDATED_FOR_SERVED_INSTANCE")
        self.assertTrue(rows["dinov2_legacy"]["same_model_as_live_route"])
        self.assertEqual(report["server"], {"live": True, "ready": False})
        self.assertEqual(report["publication_decision"], "BLOCKED")
        self.assertNotIn("decision", report)

    async def test_unvalidated_instance_kind_is_never_routed(self) -> None:
        from inference.model_router import DINOV2_LEGACY, DINOV3_EXPERIMENTAL, ModelRoutingError, infer_allowlisted_model
        import numpy as np

        infer = AsyncMock(side_effect=AssertionError("an unvalidated model reached inference"))
        cases = (
            (DINOV2_LEGACY, frozenset({"KIND_CPU"}), "PARITY_NOT_VALIDATED_FOR_SERVED_INSTANCE"),
            (DINOV2_LEGACY, frozenset({"KIND_GPU"}), "PARITY_NOT_VALIDATED_FOR_SERVED_INSTANCE"),
            (DINOV3_EXPERIMENTAL, frozenset({"KIND_GPU"}), "PARITY_NOT_VALIDATED_FOR_SERVED_INSTANCE"),
            (DINOV3_EXPERIMENTAL, frozenset({"KIND_CPU", "KIND_GPU"}), "PARITY_NOT_VALIDATED_FOR_SERVED_INSTANCE"),
            (DINOV3_EXPERIMENTAL, None, "INSTANCE_KIND_UNKNOWN"),
        )
        for model_id, kinds, reason in cases:
            with (
                patch("inference.model_router.named_model_ready", AsyncMock(return_value=True)),
                patch("inference.model_router.named_model_instance_kinds", AsyncMock(return_value=kinds)),
                patch("inference.model_router.infer_named_model", infer),
            ):
                with self.assertRaises(ModelRoutingError) as caught:
                    await infer_allowlisted_model(model_id, np.zeros((1, 3, 504, 504), dtype=np.float32))
            self.assertEqual(caught.exception.status, "MODEL_UNAVAILABLE", (model_id, kinds))
            self.assertIn(reason, caught.exception.message)
        infer.assert_not_awaited()

    async def test_selected_model_is_the_only_infer_target(self) -> None:
        from inference.model_router import DINOV3_EXPERIMENTAL, infer_allowlisted_model
        import numpy as np

        ready = AsyncMock(return_value=True)
        infer = AsyncMock(return_value=np.asarray([[0.25]], dtype=np.float32))
        with (
            patch("inference.model_router.named_model_ready", ready),
            patch("inference.model_router.named_model_instance_kinds", AsyncMock(return_value=frozenset({"KIND_CPU"}))),
            patch("inference.model_router.infer_named_model", infer),
        ):
            result = await infer_allowlisted_model(DINOV3_EXPERIMENTAL, np.zeros((1, 3, 512, 512), dtype=np.float32))
        self.assertEqual(result["triton_name"], "dinov3_authenticity_candidate")
        self.assertEqual(result["response_model"], "DINOV3_RESEARCH_PROTOTYPE")
        self.assertEqual(result["logit"], 0.25)
        infer.assert_awaited_once()
        self.assertEqual(infer.await_args.args[0], "dinov3_authenticity_candidate")
        self.assertEqual(infer.await_args.args[1], "1")
        ready.assert_awaited_once_with("dinov3_authenticity_candidate", "1")

    async def test_research_routes_and_fail_closed(self) -> None:
        from fastapi.responses import JSONResponse

        from inference.publication_gate import AUTHENTICITY_MODEL_PRODUCTION_APPROVED
        from inference.research_routes import research_verify

        self.assertFalse(AUTHENTICITY_MODEL_PRODUCTION_APPROVED)
        image = _Upload(_jpeg())

        os.environ["HYPEVAULT_DEPLOYMENT_MODE"] = "production"
        blocked = await research_verify(current_user=object(), image=image, brand="Patek Philippe", logical_model="dinov3_experimental")
        self.assertIsInstance(blocked, JSONResponse)
        self.assertEqual(blocked.status_code, 403)
        self.assertIn(b'"decision":null', blocked.body)

        os.environ["HYPEVAULT_DEPLOYMENT_MODE"] = "not-a-mode"
        unknown_mode = await research_verify(current_user=object(), image=image, brand="Patek Philippe", logical_model="dinov2_legacy")
        self.assertEqual(unknown_mode.status_code, 403)
        self.assertIn(b'"decision":null', unknown_mode.body)

        os.environ["HYPEVAULT_DEPLOYMENT_MODE"] = "research"
        unknown = await research_verify(current_user=object(), image=image, brand="Patek Philippe", logical_model="dinov2_classifier")
        self.assertEqual(unknown.status_code, 422)
        self.assertIn(b"Unknown model identifier", unknown.body)
        self.assertIn(b'"decision":null', unknown.body)

        missing_brand = await research_verify(current_user=object(), image=image, brand="", logical_model="dinov2_legacy")
        self.assertEqual(missing_brand.status_code, 422)
        self.assertIn(b'"decision":null', missing_brand.body)

        unsupported = await research_verify(current_user=object(), image=image, brand="Rolex", logical_model="dinov3_experimental")
        self.assertEqual(unsupported.status_code, 422)
        self.assertIn(b'"decision":null', unsupported.body)

        invalid = await research_verify(
            current_user=object(),
            image=_Upload(b"not-an-image"),
            brand="Patek Philippe",
            logical_model="dinov3_experimental",
        )
        self.assertEqual(invalid.status_code, 400)
        self.assertIn(b'"decision":null', invalid.body)

        skipped = AsyncMock(side_effect=AssertionError("checkpoint mismatch must not infer"))
        with (
            patch("inference.research_routes.frozen_checkpoint_digest", return_value="0" * 64),
            patch("inference.research_routes.infer_allowlisted_model", new=skipped),
        ):
            mismatch = await research_verify(
                current_user=object(),
                image=image,
                brand="Patek Philippe",
                logical_model="dinov3_experimental",
            )
        self.assertEqual(mismatch.status_code, 403)
        self.assertIn(b"POLICY_ERROR", mismatch.body)
        self.assertIn(b'"decision":null', mismatch.body)
        skipped.assert_not_awaited()

    async def test_successful_selection_identifies_the_model_and_blocks_publication(self) -> None:
        from inference.research_routes import research_verify
        import numpy as np

        os.environ["HYPEVAULT_DEPLOYMENT_MODE"] = "research"
        image = _Upload(_jpeg())

        def dinov2_infer_for(logit):
            async def dinov2_infer(model_id, array):
                self.assertEqual(model_id, "dinov2_legacy")
                self.assertEqual(tuple(array.shape), (1, 3, 504, 504))
                return {
                    "logit": logit,
                    "logical_id": model_id,
                    "triton_name": "dinov2_vitb14_live",
                    "version": "1",
                    "response_model": "LEGACY_DINOV2",
                    "temperature": None,
                }

            return dinov2_infer

        with (
            patch("inference.research_routes.settings.inference_img_size", 518),
            patch("inference.research_routes.infer_allowlisted_model", new=AsyncMock(side_effect=dinov2_infer_for(-3.0))),
        ):
            dinov2 = await research_verify(current_user=object(), image=image, brand="Patek Philippe", logical_model="dinov2_legacy")
        self.assertEqual(dinov2.model, "LEGACY_DINOV2")
        self.assertEqual(dinov2.model_version, "1")
        self.assertEqual(dinov2.decision, "AUTHENTIC")
        self.assertIsNone(dinov2.checkpoint_sha)
        self.assertNotEqual(dinov2.policy_version, "shadow_v1")
        self.assertEqual(dinov2.publication_decision, "BLOCKED")
        self.assertFalse(dinov2.production_ready)
        self.assertEqual(dinov2.brand_verification, "NOT_PERFORMED")

        # sigmoid(-1) gives P(authentic) 0.731, below the live route's 0.88 authentic floor.
        with patch("inference.research_routes.infer_allowlisted_model", new=AsyncMock(side_effect=dinov2_infer_for(-1.0))):
            floored = await research_verify(current_user=object(), image=image, brand="Patek Philippe", logical_model="dinov2_legacy")
        self.assertEqual(floored.decision, "FAKE")

        class _Result:
            system_status = "ok"
            decision = "REVIEW"
            checkpoint_sha256 = FROZEN_SHA
            temperature = FROZEN_TEMPERATURE
            policy_version = "shadow_v1"

        async def dinov3_infer(model_id, array):
            self.assertEqual(model_id, "dinov3_experimental")
            self.assertEqual(tuple(array.shape), (1, 3, 512, 512))
            return {
                "logit": -0.1,
                "logical_id": model_id,
                "triton_name": "dinov3_authenticity_candidate",
                "version": "1",
                "response_model": "DINOV3_RESEARCH_PROTOTYPE",
                "temperature": FROZEN_TEMPERATURE,
            }

        with (
            patch("inference.research_routes.frozen_checkpoint_digest", return_value=FROZEN_SHA),
            patch("inference.research_routes.infer_allowlisted_model", new=AsyncMock(side_effect=dinov3_infer)),
            patch("inference.shadow._policy_batch", return_value=[_Result()]),
        ):
            dinov3 = await research_verify(
                current_user=object(),
                image=image,
                brand="Audemars Piguet",
                logical_model="dinov3_experimental",
            )
        self.assertEqual(dinov3.model, "DINOV3_RESEARCH_PROTOTYPE")
        self.assertEqual(dinov3.checkpoint_sha, FROZEN_SHA)
        self.assertEqual(dinov3.model_scope, "FIVE_BRAND_RESEARCH_PROTOTYPE")
        self.assertTrue(dinov3.research_only)
        self.assertFalse(dinov3.production_ready)
        self.assertEqual(dinov3.publication_decision, "BLOCKED")
        self.assertEqual(dinov3.brand_verification, "NOT_PERFORMED")
        self.assertEqual(dinov3.decision, "REVIEW")
        del np

    async def test_inference_failure_has_no_decision(self) -> None:
        from fastapi.responses import JSONResponse

        from inference.model_router import ModelRoutingError
        from inference.research_routes import research_verify

        os.environ["HYPEVAULT_DEPLOYMENT_MODE"] = "research"
        with (
            patch("inference.research_routes.frozen_checkpoint_digest", return_value=FROZEN_SHA),
            patch(
                "inference.research_routes.infer_allowlisted_model",
                new=AsyncMock(side_effect=ModelRoutingError("INFERENCE_ERROR", "timeout")),
            ),
        ):
            failed = await research_verify(
                current_user=object(),
                image=_Upload(_jpeg()),
                brand="Vacheron Constantin",
                logical_model="dinov3_experimental",
            )
        self.assertIsInstance(failed, JSONResponse)
        self.assertEqual(failed.status_code, 503)
        self.assertIn(b'"decision":null', failed.body)
        self.assertIn(b"MODEL_ERROR", failed.body)

    def test_live_route_rejects_a_dinov3_selector_before_inference(self) -> None:
        routes = (REPO / "backend" / "inference" / "routes.py").read_text()
        self.assertLess(routes.index("DINOv3 candidate is blocked"), routes.index("classify_image("))
        research = (REPO / "frontend" / "src" / "app" / "research" / "page.tsx").read_text()
        seller = (REPO / "frontend" / "src" / "app" / "seller" / "upload" / "page.tsx").read_text()
        self.assertIn("EXPERIMENTAL — NOT APPROVED FOR PRODUCTION", research)
        self.assertIn("dinov2_legacy", research)
        self.assertIn("dinov3_experimental", research)
        self.assertNotIn("dinov3_experimental", seller)
        self.assertNotIn("dinov3_authenticity_candidate", research)

    def test_candidate_config_does_not_replace_dinov2(self) -> None:
        candidate = (REPO / "infra" / "triton" / "dinov3_authenticity_candidate" / "config.pbtxt").read_text()
        dinov2 = (REPO / "infra" / "triton" / "dinov2_classifier" / "config.pbtxt").read_text()
        self.assertIn('name: "dinov3_authenticity_candidate"', candidate)
        self.assertIn('name: "dinov2_classifier"', dinov2)
        self.assertIn("dims: [ 3, 518, 518 ]", dinov2)
        self.assertIn("dims: [ 3, 512, 512 ]", candidate)
        self.assertNotIn("dinov3", dinov2.lower())

    def test_live_dinov2_is_a_separate_triton_model(self) -> None:
        live = (REPO / "infra" / "triton" / "dinov2_vitb14_live" / "config.pbtxt").read_text()
        legacy = (REPO / "infra" / "triton" / "dinov2_classifier" / "config.pbtxt").read_text()
        self.assertIn('name: "dinov2_vitb14_live"', live)
        self.assertIn("dims: [ 3, 504, 504 ]", live)
        self.assertIn("reshape: { shape: [ ] }", live)
        self.assertIn('name: "dinov2_classifier"', legacy)
        self.assertIn("dims: [ 3, 518, 518 ]", legacy)
        router = (REPO / "backend" / "inference" / "model_router.py").read_text()
        self.assertNotIn('"dinov2_classifier"', router)

    @unittest.skipUnless(
        (REPO / "models" / "dinov2_classifier" / "config.pbtxt").is_file(),
        "requires the local gitignored Triton repository models/dinov2_classifier",
    )
    def test_tracked_dinov2_config_matches_the_local_repository(self) -> None:
        tracked = (REPO / "infra" / "triton" / "dinov2_classifier" / "config.pbtxt").read_bytes()
        deployed = (REPO / "models" / "dinov2_classifier" / "config.pbtxt").read_bytes()
        self.assertEqual(tracked, deployed)


if __name__ == "__main__":
    unittest.main()
