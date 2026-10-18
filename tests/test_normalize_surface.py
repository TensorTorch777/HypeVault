"""Phase 69 surface normalization. Synthetic images only; no dataset decode."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "ml_rtx5080"))

import normalize_surface as ns


class NormalizeImageTests(unittest.TestCase):
    def test_both_folder_shapes_end_at_one_square_size(self) -> None:
        generated = Image.new("RGB", (512, 512), (10, 20, 30))
        photo = Image.new("RGB", (2100, 1400), (200, 100, 50))
        self.assertEqual(ns.normalize_image(generated).size, (ns.SIZE, ns.SIZE))
        self.assertEqual(ns.normalize_image(photo).size, (ns.SIZE, ns.SIZE))

    def test_short_side_below_target_is_excluded_not_upscaled(self) -> None:
        self.assertIsNone(ns.normalize_image(Image.new("RGB", (1000, ns.SIZE - 1))))

    def test_crop_is_centered_on_the_long_axis(self) -> None:
        image = Image.new("RGB", (900, 600), (0, 0, 0))
        image.paste((255, 255, 255), (150, 0, 750, 600))
        out = np.asarray(ns.normalize_image(image)).astype(int)
        self.assertGreater(out.min(), 240)

    def test_exif_orientation_is_applied(self) -> None:
        image = Image.new("RGB", (800, 500))
        exif = image.getexif()
        exif[0x0112] = 6
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "rotated.jpg"
            image.save(path, exif=exif)
            with Image.open(path) as reopened:
                transposed = ns.ImageOps.exif_transpose(reopened)
                self.assertEqual(transposed.size, (500, 800))
                self.assertEqual(ns.normalize_image(reopened).size, (ns.SIZE, ns.SIZE))

    def test_alpha_is_flattened_on_white(self) -> None:
        clear = Image.new("RGBA", (600, 600), (0, 0, 0, 0))
        out = np.asarray(ns.normalize_image(clear))
        self.assertEqual(out.min(), 255)


class EncodingTests(unittest.TestCase):
    def test_saved_file_has_no_metadata_and_a_baseline_table(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "a" / "b.jpg"
            ns.save_jpeg(ns.normalize_image(Image.new("RGB", (700, 700), (5, 6, 7))), path)
            record = ns._inspect(str(path))
        self.assertEqual(record["size"], [ns.SIZE, ns.SIZE])
        self.assertEqual(record["format"], "JPEG")
        self.assertFalse(record["progressive"])
        self.assertFalse(record["exif"])
        self.assertFalse(record["icc"])

    def test_output_path_keeps_folder_label_and_brand(self) -> None:
        source = REPO / "Label_1_Watches" / "Richard Mille" / "abc.png"
        self.assertEqual(
            ns.output_path(source, 1),
            ns.DATA_ROOT / "Label_1_Watches" / "Richard Mille" / "abc.jpg",
        )
        with self.assertRaises(ValueError):
            ns.output_path(source, 0)


class AuditTests(unittest.TestCase):
    def test_roc_auc_handles_ties_and_order(self) -> None:
        labels = np.array([0, 0, 1, 1])
        self.assertEqual(ns.roc_auc(np.array([1.0, 2.0, 3.0, 4.0]), labels), 1.0)
        self.assertEqual(ns.roc_auc(np.array([4.0, 3.0, 2.0, 1.0]), labels), 0.0)
        self.assertEqual(ns.roc_auc(np.array([1.0, 1.0, 1.0, 1.0]), labels), 0.5)

    def test_remap_marks_excluded_rows_and_keeps_roles(self) -> None:
        rows = [
            {"path": "/s/a.jpg", "role": "train", "label": 0},
            {"path": "/s/b.jpg", "role": "validation", "label": 1},
        ]
        mapping = {
            "/s/a.jpg": {"excluded": True},
            "/s/b.jpg": {"excluded": False, "path": "/n/b.jpg"},
        }
        out = ns._remap(rows, mapping)
        self.assertEqual(out[0]["role"], ns.EXCLUDED_ROLE)
        self.assertEqual(out[0]["source_path"], "/s/a.jpg")
        self.assertEqual(out[1], {"path": "/n/b.jpg", "role": "validation", "label": 1, "source_path": "/s/b.jpg"})


if __name__ == "__main__":
    unittest.main()
