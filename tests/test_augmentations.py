"""Forensic augmentation shapes, normalization, and legacy isolation."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ml_rtx5080"))

import augmentations as aug
from manifest import IMAGENET_MEAN, IMAGENET_STD


def _content_box(image: Image.Image, fill: tuple[int, int, int]) -> tuple[int, int, int, int]:
    pixels = image.load()
    width, height = image.size
    min_x, min_y, max_x, max_y = width, height, 0, 0
    for y in range(height):
        for x in range(width):
            if pixels[x, y] == fill:
                continue
            min_x = min(min_x, x)
            min_y = min(min_y, y)
            max_x = max(max_x, x + 1)
            max_y = max(max_y, y + 1)
    return min_x, min_y, max_x, max_y


def _photo(width: int = 640, height: int = 480) -> Image.Image:
    image = Image.new("RGB", (width, height))
    pixels = image.load()
    for y in range(height):
        for x in range(width):
            pixels[x, y] = ((x * 3) % 256, (y * 5) % 256, (x + y) % 256)
    return image


class AugmentationTests(unittest.TestCase):
    def test_dinov2_train_and_eval_are_504(self) -> None:
        image = _photo()
        train_out = aug.build_transforms("dinov2", train=True)(image)
        eval_out = aug.build_transforms("dinov2", train=False)(image)
        self.assertEqual(tuple(train_out.shape), (3, 504, 504))
        self.assertEqual(tuple(eval_out.shape), (3, 504, 504))

    def test_dinov3_train_and_eval_are_512(self) -> None:
        image = _photo()
        train_out = aug.build_transforms("dinov3", train=True)(image)
        eval_out = aug.build_transforms("dinov3", train=False)(image)
        self.assertEqual(tuple(train_out.shape), (3, 512, 512))
        self.assertEqual(tuple(eval_out.shape), (3, 512, 512))
        self.assertNotEqual(train_out.shape[-1], 504)
        self.assertNotEqual(eval_out.shape[-1], 504)

    def test_eval_is_deterministic_and_train_is_stochastic(self) -> None:
        image = _photo()
        eval_tf = aug.build_transforms("dinov2", train=False)
        first = eval_tf(image)
        second = eval_tf(image)
        self.assertTrue(torch.equal(first, second))

        train_tf = aug.build_transforms("dinov2", train=True)
        samples = [train_tf(image) for _ in range(8)]
        unique = {sample.sum().item() for sample in samples}
        self.assertGreater(len(unique), 1)
        self.assertTrue(all(torch.isfinite(sample).all() for sample in samples))
        self.assertTrue(torch.isfinite(first).all())

    def test_eval_preserves_aspect_ratio(self) -> None:
        wide = Image.new("RGB", (300, 100), (255, 0, 0))
        tall = Image.new("RGB", (80, 240), (0, 0, 255))
        for family, side in (("dinov2", 504), ("dinov3", 512)):
            padded = aug.ResizePadToSquare(side)(wide)
            self.assertEqual(padded.size, (side, side))
            box = _content_box(padded, aug.imagenet_mean_fill())
            content_w = box[2] - box[0]
            content_h = box[3] - box[1]
            self.assertAlmostEqual(content_w / content_h, 3.0, delta=0.03)
            self.assertLess(content_h, side - 10)
            self.assertGreater(content_w, side - 2)
            tensor = aug.build_transforms(family, train=False)(wide)
            self.assertEqual(tuple(tensor.shape), (3, side, side))
            tall_pad = aug.ResizePadToSquare(side)(tall)
            tall_box = _content_box(tall_pad, aug.imagenet_mean_fill())
            tall_w = tall_box[2] - tall_box[0]
            tall_h = tall_box[3] - tall_box[1]
            self.assertAlmostEqual(tall_w / tall_h, 80 / 240, delta=0.03)
            self.assertLess(tall_w, side - 10)

    def test_rotation_fill_is_not_black(self) -> None:
        pipeline = aug.build_transforms("dinov2", train=True, profile=aug.CONSERVATIVE)
        rotation = next(
            transform
            for transform in aug.iter_transforms(pipeline)
            if isinstance(transform, transforms.RandomRotation)
        )
        fill = aug.imagenet_mean_fill()
        self.assertEqual(tuple(rotation.fill), fill)
        image = Image.new("RGB", (180, 180), (250, 250, 250))
        saw_fill = False
        for _ in range(12):
            rotated = rotation(image)
            corners = [
                rotated.getpixel((0, 0)),
                rotated.getpixel((179, 0)),
                rotated.getpixel((0, 179)),
                rotated.getpixel((179, 179)),
            ]
            self.assertNotIn((0, 0, 0), corners)
            if fill in corners:
                saw_fill = True
        self.assertTrue(saw_fill)

    def test_normalization_matches_imagenet(self) -> None:
        image = Image.new("RGB", (64, 64), (128, 64, 32))
        tensor = aug.build_transforms("dinov3", train=False)(image)
        expected = []
        for channel, pixel in enumerate((128, 64, 32)):
            value = (pixel / 255.0 - IMAGENET_MEAN[channel]) / IMAGENET_STD[channel]
            expected.append(value)
        for channel in range(3):
            self.assertTrue(torch.allclose(tensor[channel], torch.full_like(tensor[channel], expected[channel]), atol=1e-5))

    def test_jpeg_keeps_rgb_structure(self) -> None:
        image = _photo(120, 90)
        encoded = aug.JPEGCompression((60, 95))(image)
        self.assertEqual(encoded.mode, "RGB")
        self.assertEqual(encoded.size, image.size)
        extrema = encoded.getextrema()
        self.assertTrue(all(0 <= low <= high <= 255 for low, high in extrema))

    def test_production_profile_has_no_grayscale_and_legacy_is_marked(self) -> None:
        production = aug.build_transforms("dinov2", train=True, profile=aug.CONSERVATIVE)
        legacy = aug.build_transforms("dinov2", train=True, profile=aug.CURRENT_LEGACY)
        self.assertFalse(aug.contains_random_grayscale(production))
        self.assertTrue(aug.contains_random_grayscale(legacy))
        report = aug.augmentation_report("dinov2", aug.CURRENT_LEGACY, True)
        self.assertTrue(report["legacy"])
        self.assertTrue(any(step.get("marker") == aug.LEGACY_MARKER for step in report["steps"]))
        production_report = aug.augmentation_report("dinov3", aug.CONSERVATIVE, True)
        self.assertFalse(production_report["legacy"])
        grayscale = next(step for step in production_report["steps"] if step["op"] == "random_grayscale")
        self.assertFalse(grayscale["enabled"])
        payload = json.dumps(production_report)
        self.assertIn("504", json.dumps(aug.augmentation_report("dinov2", aug.CONSERVATIVE, False)))
        self.assertIn("512", payload)
        self.assertFalse(production_report["mixup"]["applied_in_image_transform"])
        self.assertEqual(production_report["mixup"]["supported_alpha"], [0.0, 0.05])
        self.assertEqual(production_report["normalization"]["mean"], list(IMAGENET_MEAN))
        self.assertEqual(production_report["normalization"]["std"], list(IMAGENET_STD))

    def test_unknown_profile_fails_clearly(self) -> None:
        with self.assertRaises(ValueError) as raised:
            aug.build_transforms("dinov2", train=True, profile="randaugment")
        self.assertIn("randaugment", str(raised.exception))


if __name__ == "__main__":
    unittest.main()
