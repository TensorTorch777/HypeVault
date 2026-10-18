"""Trainer preprocessing uses the shared conservative pipeline."""

from __future__ import annotations

import inspect
import json
import sys
import tempfile
import unittest
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ml_rtx5080"))

import augmentations as aug
import train as tv
import training_config as tc
from manifest import IMAGENET_MEAN, IMAGENET_STD
from model import build_dino_backbone


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


class PreprocessingTests(unittest.TestCase):
    def test_trainer_delegates_to_the_shared_conservative_profile(self) -> None:
        source = inspect.getsource(tv.build_transforms)
        self.assertNotIn("0.55", source)
        self.assertNotIn("RandomGrayscale", source)
        self.assertNotIn("CenterCrop", source)
        self.assertEqual(tv.DEFAULTS["augmentation_profile"], aug.CONSERVATIVE)
        image = Image.new("RGB", (320, 180), (20, 40, 60))
        train_tf = tv.build_transforms("dinov2", train=True)
        shared = aug.build_transforms("dinov2", train=True, profile=aug.CONSERVATIVE)
        self.assertIsInstance(train_tf.transforms[0], transforms.RandomResizedCrop)
        self.assertEqual(tuple(train_tf.transforms[0].scale), (0.75, 1.0))
        rotation = next(item for item in train_tf.transforms if isinstance(item, transforms.RandomRotation))
        self.assertEqual(tuple(rotation.fill), aug.imagenet_mean_fill())
        self.assertNotEqual(tuple(rotation.fill), (0, 0, 0))
        self.assertFalse(aug.contains_random_grayscale(train_tf))
        self.assertFalse(aug.contains_random_grayscale(shared))
        eval_tf = tv.build_transforms("dinov2", train=False)
        self.assertIsInstance(eval_tf.transforms[0], aug.ResizePadToSquare)
        self.assertNotIn(transforms.CenterCrop, [type(item) for item in eval_tf.transforms])

    def test_eval_is_deterministic_and_train_is_stochastic(self) -> None:
        image = Image.new("RGB", (300, 120), (255, 128, 0))
        eval_tf = tv.build_transforms("dinov2", train=False)
        first = eval_tf(image)
        second = tv.build_transforms("dinov2", train=False)(image)
        self.assertTrue(torch.equal(first, second))
        train_tf = tv.build_transforms("dinov3", train=True)
        samples = [train_tf(image) for _ in range(6)]
        self.assertGreater(len({sample.sum().item() for sample in samples}), 1)

    def test_eval_sizes_and_aspect_ratio(self) -> None:
        wide = Image.new("RGB", (300, 100), (255, 0, 0))
        tall = Image.new("RGB", (80, 240), (0, 0, 255))
        fill = aug.imagenet_mean_fill()
        for family, side in (("dinov2", 504), ("dinov3", 512)):
            tensor = tv.build_transforms(family, train=False)(wide)
            self.assertEqual(tuple(tensor.shape), (3, side, side))
            padded_wide = aug.ResizePadToSquare(side)(wide)
            padded_tall = aug.ResizePadToSquare(side)(tall)
            wide_box = _content_box(padded_wide, fill)
            tall_box = _content_box(padded_tall, fill)
            self.assertGreater(wide_box[2] - wide_box[0], wide_box[3] - wide_box[1])
            self.assertGreater(tall_box[3] - tall_box[1], tall_box[2] - tall_box[0])
            self.assertEqual(padded_wide.size, (side, side))
            self.assertEqual(padded_tall.size, (side, side))

    def test_normalization_of_a_mean_colored_square(self) -> None:
        fill = aug.imagenet_mean_fill()
        image = Image.new("RGB", (64, 64), fill)
        tensor = tv.build_transforms("dinov2", train=False)(image)
        self.assertEqual(tuple(tensor.shape), (3, 504, 504))
        for channel, pixel in enumerate(fill):
            expected = (pixel / 255.0 - IMAGENET_MEAN[channel]) / IMAGENET_STD[channel]
            self.assertTrue(
                torch.allclose(tensor[channel], torch.full_like(tensor[channel], expected), atol=1e-4)
            )
            self.assertLess(abs(expected), 0.02)
        norm = next(item for item in tv.build_transforms("dinov3", train=False).transforms if isinstance(item, transforms.Normalize))
        self.assertEqual(tuple(norm.mean), tuple(IMAGENET_MEAN))
        self.assertEqual(tuple(norm.std), tuple(IMAGENET_STD))

    def test_dinov3_rejects_504(self) -> None:
        with self.assertRaises(ValueError):
            build_dino_backbone("dinov3", image_size=504, pretrained=False)

    def test_preprocessing_record_is_written_into_experiment_config(self) -> None:
        record = aug.preprocessing_record("dinov3", aug.CONSERVATIVE, "cls_patch_attention")
        self.assertEqual(record["input_size"], 512)
        self.assertEqual(record["training_transform"], "conservative_train_v1")
        self.assertEqual(record["evaluation_transform"], "resize_pad_square_eval_v1")
        self.assertEqual(record["normalization_mean"], [float(channel) for channel in IMAGENET_MEAN])
        root = Path(__file__).resolve().parents[1] / "ml_rtx5080" / "experiments"
        root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=root) as tmp:
            path = Path(tmp) / "config.json"
            tc.write_experiment_config(
                path,
                {
                    "mixup_alpha": 0.0,
                    "label_smoothing": 0.02,
                    **record,
                },
            )
            payload = json.loads(path.read_text())
        self.assertEqual(payload["model_family"], "dinov3")
        self.assertEqual(payload["model_architecture"], "cls_patch_attention")
        self.assertEqual(payload["augmentation_profile"], "conservative")
        self.assertEqual(payload["evaluation_transform"], "resize_pad_square_eval_v1")


if __name__ == "__main__":
    unittest.main()
