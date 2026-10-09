"""DINOv3 checkpoint identity is verified at startup and cached. No GPU and no final-test access."""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "backend"))
sys.path.insert(0, str(REPO / "ml_rtx5080"))

FROZEN_SHA = "5a38c93fd442b03653c65d2a5ecc9c2687ef152f7c5c020763e4ce1fd9c7d28f"
FROZEN_TEMPERATURE = 0.24038200410185356


def _jpeg() -> bytes:
    import io

    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", (32, 24), (8, 9, 10)).save(buffer, format="JPEG")
    return buffer.getvalue()


class _Upload:
    def __init__(self, raw: bytes, content_type: str = "image/jpeg") -> None:
        self._raw = raw
        self.content_type = content_type

    async def read(self) -> bytes:
        return self._raw


class CheckpointIdentityTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        from inference.checkpoint_identity import reset_checkpoint_identity_for_tests

        reset_checkpoint_identity_for_tests()

    def tearDown(self) -> None:
        from inference.checkpoint_identity import reset_checkpoint_identity_for_tests

        reset_checkpoint_identity_for_tests()
        os.environ.pop("HYPEVAULT_DEPLOYMENT_MODE", None)

    def test_correct_identity_is_cached_and_repeated_reads_do_not_rehash(self) -> None:
        from inference.checkpoint_identity import (
            hash_call_count,
            reload_checkpoint_identity,
            verified_checkpoint_digest,
            verify_frozen_checkpoint,
        )
        from inference.research_routes import frozen_checkpoint_digest

        with patch("inference.checkpoint_identity._compute_digest", return_value=FROZEN_SHA) as hashed:
            digest = verify_frozen_checkpoint()
            self.assertEqual(digest, FROZEN_SHA)
            self.assertEqual(hash_call_count(), 1)
            self.assertEqual(verified_checkpoint_digest(), FROZEN_SHA)
            self.assertEqual(frozen_checkpoint_digest(), FROZEN_SHA)
            self.assertEqual(frozen_checkpoint_digest(), FROZEN_SHA)
            verify_frozen_checkpoint()
            self.assertEqual(hash_call_count(), 1)
            hashed.assert_called_once()

            reload_checkpoint_identity()
            self.assertEqual(hash_call_count(), 2)
            self.assertEqual(verified_checkpoint_digest(), FROZEN_SHA)
            self.assertEqual(hashed.call_count, 2)

    def test_mismatched_checkpoint_fails_closed_and_is_not_cached(self) -> None:
        from inference.checkpoint_identity import (
            CheckpointIdentityError,
            checkpoint_identity_ready,
            checkpoint_identity_status,
            verified_checkpoint_digest,
            verify_frozen_checkpoint,
        )

        with patch("inference.checkpoint_identity._compute_digest", return_value="0" * 64):
            with self.assertRaises(CheckpointIdentityError) as caught:
                verify_frozen_checkpoint()
        self.assertEqual(caught.exception.status, "POLICY_ERROR")
        self.assertFalse(checkpoint_identity_ready())
        self.assertIsNone(checkpoint_identity_status()["sha256"])
        with self.assertRaises(CheckpointIdentityError):
            verified_checkpoint_digest()

    def test_reload_of_a_bad_replacement_clears_the_cache(self) -> None:
        from inference.checkpoint_identity import (
            CheckpointIdentityError,
            checkpoint_identity_ready,
            hash_call_count,
            reload_checkpoint_identity,
            verify_frozen_checkpoint,
        )

        with patch("inference.checkpoint_identity._compute_digest", side_effect=[FROZEN_SHA, "0" * 64, FROZEN_SHA]):
            verify_frozen_checkpoint()
            self.assertTrue(checkpoint_identity_ready())
            with self.assertRaises(CheckpointIdentityError):
                reload_checkpoint_identity()
            self.assertFalse(checkpoint_identity_ready())
            self.assertEqual(hash_call_count(), 2)
            digest = reload_checkpoint_identity()
        self.assertEqual(digest, FROZEN_SHA)
        self.assertTrue(checkpoint_identity_ready())
        self.assertEqual(hash_call_count(), 3)

    def test_hashing_error_is_model_error_and_does_not_cache(self) -> None:
        from inference.checkpoint_identity import (
            CheckpointIdentityError,
            checkpoint_identity_ready,
            verify_frozen_checkpoint,
        )

        with patch("inference.checkpoint_identity._compute_digest", side_effect=OSError("read failed")):
            with self.assertRaises(CheckpointIdentityError) as caught:
                verify_frozen_checkpoint()
        self.assertEqual(caught.exception.status, "MODEL_ERROR")
        self.assertFalse(checkpoint_identity_ready())

    def test_unverified_digest_is_never_cached_from_a_local_file(self) -> None:
        from inference.checkpoint_identity import (
            CheckpointIdentityError,
            checkpoint_identity_status,
            verify_frozen_checkpoint,
        )

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as handle:
            handle.write(b"not-the-frozen-checkpoint")
            path = Path(handle.name)
        try:
            with self.assertRaises(CheckpointIdentityError):
                verify_frozen_checkpoint(path=path)
            self.assertIsNone(checkpoint_identity_status()["sha256"])
        finally:
            path.unlink(missing_ok=True)

    async def test_repeated_research_requests_do_not_rehash(self) -> None:
        from inference.checkpoint_identity import hash_call_count, verify_frozen_checkpoint
        from inference.research_routes import research_verify

        os.environ["HYPEVAULT_DEPLOYMENT_MODE"] = "research"

        class _Result:
            system_status = "ok"
            decision = "REVIEW"
            checkpoint_sha256 = FROZEN_SHA
            temperature = FROZEN_TEMPERATURE
            policy_version = "shadow_v1"

        async def dinov3_infer(model_id, array):
            del array
            self.assertEqual(model_id, "dinov3_experimental")
            return {
                "logit": -0.1,
                "logical_id": model_id,
                "triton_name": "dinov3_authenticity_candidate",
                "version": "1",
                "response_model": "DINOV3_RESEARCH_PROTOTYPE",
                "temperature": FROZEN_TEMPERATURE,
            }

        with patch("inference.checkpoint_identity._compute_digest", return_value=FROZEN_SHA) as hashed:
            verify_frozen_checkpoint()
            with (
                patch("inference.research_routes.infer_allowlisted_model", new=AsyncMock(side_effect=dinov3_infer)),
                patch("inference.shadow._policy_batch", return_value=[_Result()]),
            ):
                first = await research_verify(
                    current_user=object(),
                    image=_Upload(_jpeg()),
                    brand="Audemars Piguet",
                    logical_model="dinov3_experimental",
                )
                second = await research_verify(
                    current_user=object(),
                    image=_Upload(_jpeg()),
                    brand="Audemars Piguet",
                    logical_model="dinov3_experimental",
                )
        self.assertEqual(first.checkpoint_sha, FROZEN_SHA)
        self.assertEqual(second.checkpoint_sha, FROZEN_SHA)
        self.assertEqual(first.decision, "REVIEW")
        self.assertEqual(hash_call_count(), 1)
        hashed.assert_called_once()

    async def test_mismatch_does_not_infer_or_return_a_decision(self) -> None:
        from fastapi.responses import JSONResponse

        from inference.checkpoint_identity import verify_frozen_checkpoint
        from inference.research_routes import research_verify

        os.environ["HYPEVAULT_DEPLOYMENT_MODE"] = "research"
        skipped = AsyncMock(side_effect=AssertionError("mismatch must not infer"))
        with patch("inference.checkpoint_identity._compute_digest", return_value="0" * 64):
            with self.assertRaises(Exception):
                verify_frozen_checkpoint()
            with patch("inference.research_routes.infer_allowlisted_model", new=skipped):
                blocked = await research_verify(
                    current_user=object(),
                    image=_Upload(_jpeg()),
                    brand="Patek Philippe",
                    logical_model="dinov3_experimental",
                )
        self.assertIsInstance(blocked, JSONResponse)
        self.assertEqual(blocked.status_code, 403)
        self.assertIn(b'"decision":null', blocked.body)
        self.assertIn(b"POLICY_ERROR", blocked.body)
        skipped.assert_not_awaited()

    async def test_hashing_error_on_the_request_path_has_no_decision(self) -> None:
        from fastapi.responses import JSONResponse

        from inference.research_routes import research_verify

        os.environ["HYPEVAULT_DEPLOYMENT_MODE"] = "research"
        skipped = AsyncMock(side_effect=AssertionError("hashing error must not infer"))
        with patch("inference.research_routes.infer_allowlisted_model", new=skipped):
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
        skipped.assert_not_awaited()

    def test_real_frozen_checkpoint_matches_when_present(self) -> None:
        from inference.checkpoint_identity import CHECKPOINT, verify_frozen_checkpoint

        self.assertEqual(
            CHECKPOINT,
            REPO / "ml_rtx5080" / "experiments" / "v2_dinov3_cls_patch_attention" / "epoch_018.pt",
        )
        if not CHECKPOINT.is_file():
            self.skipTest("frozen DINOv3 checkpoint is not present")
        digest = verify_frozen_checkpoint()
        self.assertEqual(digest, FROZEN_SHA)


if __name__ == "__main__":
    unittest.main()
