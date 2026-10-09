"""Phase 39 scope lock. Unsupported brands fail closed before any verdict."""

from __future__ import annotations

import hashlib
import json
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
LOCK = REPO / "ml_rtx5080" / "experiments" / "scope_lock_v1"
CHECKPOINT = REPO / "ml_rtx5080" / "experiments" / "v2_dinov3_cls_patch_attention" / "epoch_018.pt"
CHECKPOINT_SHA = "5a38c93fd442b03653c65d2a5ecc9c2687ef152f7c5c020763e4ce1fd9c7d28f"
TEMPERATURE = 0.24038200410185356


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class ScopeLockTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        backend = str(REPO / "backend")
        if backend not in sys.path:
            sys.path.insert(0, backend)

    def test_supported_brands_are_explicit_and_not_verdicts(self) -> None:
        from inference.scope_gate import SUPPORTED_BRANDS, evaluate_declared_brand

        for brand in (
            "A. Lange & Söhne",
            "A. Lange & Sohne",
            "a. lange and sohne",
            "Audemars Piguet",
            "Patek Philippe",
            "Richard Mille",
            "Vacheron Constantin",
        ):
            result = evaluate_declared_brand(brand)
            self.assertEqual(result.scope_status, "SUPPORTED", brand)
            self.assertIsNone(result.decision)
            self.assertIn(result.canonical_brand, SUPPORTED_BRANDS)

    def test_unsupported_brand_fails_closed_without_a_verdict(self) -> None:
        from inference.scope_gate import evaluate_declared_brand, unsupported_scope_body

        for brand in (None, "", "   ", "Rolex", "Omega", "Patek", "AP", "unknown", "watch", "Patek Philippe Nautilus"):
            result = evaluate_declared_brand(brand)
            self.assertEqual(result.scope_status, "UNSUPPORTED_SCOPE", brand)
            self.assertIsNone(result.decision)
            self.assertNotIn(result.decision, ("AUTHENTIC", "FAKE"))
        body = unsupported_scope_body()
        self.assertEqual(body["status"], "UNSUPPORTED_SCOPE")
        self.assertIsNone(body["decision"])
        self.assertNotIn(body["decision"], ("AUTHENTIC", "FAKE"))
        self.assertIn("five-brand", body["reason"])

    def test_gate_does_not_read_images_or_the_final_test(self) -> None:
        source = (REPO / "backend" / "inference" / "scope_gate.py").read_text()
        self.assertNotIn("Image.open", source)
        self.assertNotIn("final_test_v1", source)
        self.assertNotIn("dinov2_model", source)
        self.assertNotIn("sigmoid", source)
        routes = (REPO / "backend" / "inference" / "routes.py").read_text()
        self.assertLess(routes.index("evaluate_declared_brand"), routes.index("classify_image("))
        self.assertNotIn("UNSUPPORTED_SCOPE", (REPO / "backend" / "inference" / "dinov2_model.py").read_text())

    def test_frozen_checkpoint_and_temperature(self) -> None:
        from inference.scope_gate import CHECKPOINT_SHA256, TEMPERATURE

        contract = json.loads((LOCK / "scope_contract.json").read_text())
        self.assertEqual(CHECKPOINT_SHA256, CHECKPOINT_SHA)
        self.assertEqual(contract["checkpoint_sha256"], CHECKPOINT_SHA)
        self.assertEqual(TEMPERATURE, TEMPERATURE)
        self.assertEqual(contract["temperature"], TEMPERATURE)

    @unittest.skipUnless(CHECKPOINT.is_file(), "requires the local frozen checkpoint epoch_018.pt, which is not in git")
    def test_local_checkpoint_matches_the_frozen_sha(self) -> None:
        self.assertEqual(_sha256(CHECKPOINT), CHECKPOINT_SHA)

    def test_claims_and_promotion_gate_agree(self) -> None:
        claims = json.loads((LOCK / "claims_matrix.json").read_text())
        labels = {row["id"]: row["label"] for row in claims["claims"]}
        self.assertEqual(labels["A"], "SUPPORTED_WITH_SCOPE")
        self.assertEqual(labels["B"], "NOT_SUPPORTED")
        self.assertEqual(labels["C"], "NOT_SUPPORTED")
        self.assertEqual(labels["D"], "NOT_SUPPORTED")
        self.assertEqual(labels["G"], "NOT_SUPPORTED")
        self.assertEqual(labels["J"], "NOT_SUPPORTED")
        gate = json.loads((LOCK / "promotion_gate_v1.json").read_text())
        self.assertEqual(gate["OOD"], "FAIL")
        self.assertEqual(gate["FINAL_TEST"], "PASS")
        self.assertFalse(gate["PROMOTION_ALLOWED"])
        self.assertIsNone(gate["production_ood_threshold"])
        card = (LOCK / "model_card_v1.md").read_text()
        self.assertIn(CHECKPOINT_SHA, card)
        self.assertIn("0.24038200410185356", card)
        self.assertIn(
            "This is strong in-distribution evidence and is not evidence of universal luxury-watch authenticity.",
            card,
        )
        self.assertIn("The current system does not have validated open-set protection.", card)


if __name__ == "__main__":
    unittest.main()
