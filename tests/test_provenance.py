"""Content Credentials (C2PA) AI labels: signed "made by AI" files are rejected; unlabelled files are not judged."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "backend"))

from inference.provenance import ai_label_from_store, read_ai_label  # noqa: E402
from inference.screening import AI_LABEL, check_label  # noqa: E402

IPTC = "http://cv.iptc.org/newscodes/digitalsourcetype/"


def store(*manifests: dict, state: str = "Valid") -> dict:
    ids = [f"urn:{i}" for i in range(len(manifests))]
    return {"active_manifest": ids[0], "manifests": dict(zip(ids, manifests)), "validation_state": state}


def manifest(actions: list[dict], *, generator: str, issuer: str, common_name: str = "") -> dict:
    return {
        "claim_generator_info": [{"name": generator}],
        "signature_info": {"issuer": issuer, "common_name": common_name},
        "assertions": [{"label": "c2pa.actions.v2", "data": {"actions": actions}}],
    }


CHATGPT = store(manifest(
    [{"action": "c2pa.created", "softwareAgent": {"name": "ChatGPT", "version": "gpt-image"},
      "digitalSourceType": IPTC + "trainedAlgorithmicMedia"}, {"action": "c2pa.converted"}],
    generator="OpenAI Media Service API", issuer="OpenAI OpCo, LLC", common_name="OpenAI Media Service",
), state="Invalid")
# Gemini: the outer manifest only says "enhanced"; the AI source type is on an inner manifest.
GEMINI = store(
    manifest([{"action": "c2pa.transcoded", "digitalSourceType": IPTC + "algorithmicallyEnhanced"}],
             generator="Google C2PA Core Generator Library", issuer="Google LLC"),
    manifest([{"action": "c2pa.created", "digitalSourceType": IPTC + "trainedAlgorithmicMedia"}],
             generator="Google C2PA Core Generator Library", issuer="Google LLC"),
)
OLD_GEMINI = store(manifest(
    [{"action": "c2pa.opened"}, {"action": "c2pa.edited", "digitalSourceType": IPTC + "composite",
                                 "description": "Added visible watermark"}],
    generator="Google C2PA Core Generator Library", issuer="Google LLC",
))
PIXEL_PHOTO = store(manifest(
    [{"action": "c2pa.created", "digitalSourceType": IPTC + "computationalCapture"}],
    generator="Google C2PA Core Generator Library", issuer="Google LLC",
))
CAMERA_EDIT = store(manifest(
    [{"action": "c2pa.created", "digitalSourceType": IPTC + "digitalCapture"}, {"action": "c2pa.cropped"}],
    generator="Adobe Lightroom", issuer="Adobe Inc.",
))


class ProvenanceTests(unittest.TestCase):
    def test_a_chatgpt_label_is_ai_even_when_its_certificate_fails_trust_checks(self) -> None:
        label = ai_label_from_store(CHATGPT)
        assert label is not None
        self.assertEqual((label.maker, label.tool, label.source_type), ("OpenAI OpCo, LLC", "ChatGPT gpt-image", "trainedAlgorithmicMedia"))

    def test_gemini_is_found_through_its_inner_manifest(self) -> None:
        label = ai_label_from_store(GEMINI)
        assert label is not None
        self.assertEqual((label.maker, label.source_type), ("Google LLC", "trainedAlgorithmicMedia"))

    def test_an_ai_service_label_without_a_camera_capture_is_ai(self) -> None:
        label = ai_label_from_store(OLD_GEMINI)
        assert label is not None
        self.assertIsNone(label.source_type)

    def test_camera_photos_with_labels_are_not_ai(self) -> None:
        self.assertIsNone(ai_label_from_store(PIXEL_PHOTO))
        self.assertIsNone(ai_label_from_store(CAMERA_EDIT))

    def test_files_without_a_label_are_not_judged(self) -> None:
        self.assertIsNone(read_ai_label(b"\xff\xd8\xff\xe0 not really a jpeg", "image/jpeg"))
        self.assertIsNone(check_label(None))

    def test_a_labelled_file_is_rejected_with_who_signed_it(self) -> None:
        out = check_label(ai_label_from_store(CHATGPT))
        assert out is not None
        self.assertEqual((out.status, out.code), ("rejected", AI_LABEL))
        self.assertIn("OpenAI OpCo, LLC", out.reason)
        self.assertIn("ChatGPT gpt-image", out.reason)
        self.assertEqual(out.details["label_validation"], "Invalid")


if __name__ == "__main__":
    unittest.main()
