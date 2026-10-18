"""Phase 52 protocol lock. No Triton call and no final-test decode."""

from __future__ import annotations

import hashlib
import io
import sys
import unittest
from pathlib import Path

from PIL import Image

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "ml_rtx5080"))

FROZEN_SHA = "5a38c93fd442b03653c65d2a5ecc9c2687ef152f7c5c020763e4ce1fd9c7d28f"


class ShortcutSensitivityProtocolTests(unittest.TestCase):
    def test_registered_protocol_matches_the_document_and_frozen_contract(self) -> None:
        import shortcut_sensitivity as study

        digest = study.freeze_protocol()
        payload = study.PROTOCOL_PATH.read_bytes()
        self.assertEqual(hashlib.sha256(payload).hexdigest(), digest)
        self.assertEqual(study.PROTOCOL_SHA_PATH.read_text().split()[0], digest)
        self.assertEqual(payload, study.canonical_protocol_bytes())
        document = study.protocol_document()
        self.assertEqual(document["frozen"]["checkpoint_sha256"], FROZEN_SHA)
        self.assertEqual(document["frozen"]["temperature"], 0.24038200410185356)
        self.assertEqual(document["frozen"]["threshold"], 0.5)
        self.assertIn("test", document["population"]["excluded_splits"])
        self.assertTrue(document["frozen"]["final_test_unopened"])
        self.assertEqual(study.require_registered_protocol(), digest)

    def test_draw_uses_only_eligible_cells_and_is_deterministic(self) -> None:
        import shortcut_sensitivity as study

        study.freeze_protocol()
        root = Path("/tmp/phase52-eligible")
        membership = {"train": [], "calibration": [], "validation": [], "test": []}

        def add(split: str, label: str, brand: str, name: str) -> None:
            membership[split].append(str(root / label / brand / name))

        for brand in study.BRANDS:
            for index in range(6):
                add("train", "Label_0_Watches", brand, f"a{index}.jpg")
                add("validation", "Label_1_Watches", brand, f"f{index}.jpg")
        secret = str(root / "Label_0_Watches" / study.BRANDS[0] / "secret-test.jpg")
        membership["test"].append(secret)
        opened: list[str] = []

        def opener(path: Path):
            text = str(path)
            if text == secret or "secret-test" in text:
                raise AssertionError("final-test path was opened")
            opened.append(text)
            return object()

        first = [row["path"] for row in study.select_sample(membership, opener=opener) if row.get("path")]
        second = [row["path"] for row in study.select_sample(membership, opener=opener) if row.get("path")]
        self.assertEqual(first, second)
        self.assertEqual(len(first), 4 * 5 * 2)
        self.assertNotIn(secret, first)
        self.assertTrue(all("Label_0_Watches" in path or "Label_1_Watches" in path for path in first))
        self.assertNotIn(secret, opened)

    def test_eligible_split_that_contains_a_test_path_fails_closed(self) -> None:
        import shortcut_sensitivity as study

        shared = "/tmp/phase52-leak/Label_0_Watches/Patek Philippe/same.jpg"
        membership = {"train": [shared], "calibration": [], "validation": [], "test": [shared]}
        with self.assertRaises(RuntimeError):
            study.eligible_records(membership)

    def test_variants_follow_the_registered_geometry_rules(self) -> None:
        import shortcut_sensitivity as study

        buffer = io.BytesIO()
        Image.new("RGB", (80, 40), (10, 20, 30)).save(buffer, format="PNG")
        buffer.seek(0)
        image = Image.open(buffer).convert("RGB")
        jpeg = study.apply_variant(image, "jpeg_q30")
        self.assertEqual(jpeg.size, image.size)
        self.assertTrue(study.pixels_changed(image, jpeg))
        stretched = study.apply_variant(image, "stretch_512")
        self.assertEqual(stretched.size, (512, 512))
        small = study.apply_variant(image, "long_side_256")
        self.assertEqual(max(small.size), 256)
        square = study.apply_variant(Image.new("RGB", (32, 32), (1, 2, 3)), "pad_square_black")
        self.assertEqual(square.size, (32, 32))
        bordered = study.apply_variant(image, "pad_border_32_imagenet_mean")
        self.assertEqual(bordered.size, (80 + 64, 40 + 64))


if __name__ == "__main__":
    unittest.main()
