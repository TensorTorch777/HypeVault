"""Live origin check: model selection, calibrated decision and the 0.88 listing rule."""

from __future__ import annotations

import asyncio
import sys
import unittest
from pathlib import Path
from unittest import mock

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "backend"))

from PIL import Image  # noqa: E402

from config import settings  # noqa: E402
from inference import live_screening  # noqa: E402
from inference.verdict import apply_min_authentic_confidence  # noqa: E402


class LiveScreeningTests(unittest.TestCase):
    def test_default_stays_on_dinov2(self) -> None:
        # benchmark_v1 named no frontier model, so DINOv3 is opt-in only.
        config = (REPO / "backend" / "config.py").read_text()
        self.assertIn('live_screening_model: str = "dinov2_legacy"', config)

    def test_unknown_model_fails_closed(self) -> None:
        with mock.patch.object(settings, "live_screening_model", "dinov3_experimental"):
            with self.assertRaises(live_screening.LiveModelConfigurationError):
                live_screening.live_model()

    def test_decision_matches_the_legacy_verdict_pair(self) -> None:
        self.assertEqual(live_screening.verdict_from_probability(0.5), ("FAKE", 0.5))
        self.assertEqual(live_screening.verdict_from_probability(0.97), ("FAKE", 0.97))
        self.assertEqual(live_screening.verdict_from_probability(0.05), ("AUTHENTIC", 0.95))

    def test_listing_needs_088_confidence_in_an_original_photo(self) -> None:
        listed = apply_min_authentic_confidence(*live_screening.verdict_from_probability(0.10), 0.88)
        held = apply_min_authentic_confidence(*live_screening.verdict_from_probability(0.30), 0.88)
        self.assertEqual(listed[0], "AUTHENTIC")
        self.assertEqual(held[0], "FAKE")

    def test_dinov3_path_uses_the_calibrated_probability(self) -> None:
        image = Image.new("RGB", (600, 600), "white")
        with mock.patch.object(settings, "live_screening_model", "dinov3_frozen"), \
                mock.patch.object(live_screening, "_dinov3_probability", return_value=0.02) as probability:
            verdict, confidence, model = asyncio.run(live_screening.classify_listing_photo(image))
        probability.assert_called_once_with(image)
        self.assertEqual((verdict, confidence, model), ("AUTHENTIC", 0.98, "dinov3_frozen"))
        self.assertEqual(live_screening.RESPONSE_MODEL[model], "DINOV3_FROZEN")

    def test_dinov3_loader_is_the_hash_checked_frozen_backend(self) -> None:
        source = (REPO / "backend" / "inference" / "live_screening.py").read_text()
        self.assertIn("from inference.dinov3_model import shared_backend", source)
        self.assertIn("predict_probability", source)
        backend = (REPO / "backend" / "inference" / "dinov3_model.py").read_text()
        self.assertIn("EXPECTED_CHECKPOINT_SHA256", backend)

    def test_legacy_route_refuses_a_dinov3_checkpoint(self) -> None:
        image = Image.new("RGB", (600, 600), "white")
        with mock.patch.object(settings, "live_screening_model", "dinov2_legacy"), \
                mock.patch.object(settings, "local_model_path", "ml_rtx5080/experiments/v2_dinov3_cls_patch_attention/epoch_018.pt"):
            with self.assertRaises(live_screening.LiveModelConfigurationError):
                asyncio.run(live_screening.classify_listing_photo(image))


if __name__ == "__main__":
    unittest.main()
