"""Shape contract for the shared DINOv2 / DINOv3 backbone. No pretrained weights."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ml_rtx5080"))

import model as backbone


class BackboneInterfaceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.dinov2 = backbone.build_dino_backbone("dinov2", image_size=504, pretrained=False)
        cls.dinov3 = backbone.build_dino_backbone("dinov3", image_size=512, pretrained=False)
        cls.dinov2.eval()
        cls.dinov3.eval()

    def test_construction_and_dimensions(self) -> None:
        self.assertEqual(self.dinov2.backbone_name, backbone.DINOV2_TIMM_NAME)
        self.assertEqual(self.dinov3.backbone_name, backbone.DINOV3_TIMM_NAME)
        self.assertEqual(self.dinov2.get_embed_dim(), 768)
        self.assertEqual(self.dinov3.get_embed_dim(), 768)
        self.assertEqual(self.dinov2.get_patch_size(), 14)
        self.assertEqual(self.dinov3.get_patch_size(), 16)
        self.assertEqual(self.dinov2.num_register_tokens, 0)
        self.assertEqual(self.dinov3.num_register_tokens, 4)

    def test_dinov2_504_cls_and_patches(self) -> None:
        images = torch.zeros(2, 3, 504, 504)
        with torch.inference_mode():
            features = self.dinov2.forward_features(images)
            cls = self.dinov2.get_cls_token(features)
            patches = self.dinov2.get_patch_tokens(features)
        self.assertEqual(tuple(features.shape), (2, 1297, 768))
        self.assertEqual(tuple(cls.shape), (2, 768))
        self.assertEqual(tuple(patches.shape), (2, 1296, 768))
        self.assertTrue(torch.equal(patches, features[:, 1:]))

    def test_dinov3_512_excludes_register_tokens(self) -> None:
        images = torch.zeros(2, 3, 512, 512)
        with torch.inference_mode():
            features = self.dinov3.forward_features(images)
            cls = self.dinov3.get_cls_token(features)
            patches = self.dinov3.get_patch_tokens(features)
        self.assertEqual(tuple(features.shape), (2, 1029, 768))
        self.assertEqual(tuple(cls.shape), (2, 768))
        self.assertEqual(tuple(patches.shape), (2, 1024, 768))
        registers = features[:, 1:5]
        self.assertEqual(tuple(registers.shape), (2, 4, 768))
        self.assertEqual(features.shape[1], 1 + 4 + patches.shape[1])
        self.assertTrue(torch.equal(patches, features[:, 5:]))
        self.assertFalse(torch.equal(patches[:, :4], registers))

    def test_dinov3_rejects_504(self) -> None:
        with self.assertRaises(ValueError) as raised:
            backbone.build_dino_backbone("dinov3", image_size=504, pretrained=False)
        self.assertIn("504", str(raised.exception))

    def test_configured_backbone_rejects_a_mismatched_spatial_size(self) -> None:
        images = torch.zeros(1, 3, 256, 256)
        with self.assertRaises(ValueError):
            self.dinov3.forward_features(images)


if __name__ == "__main__":
    unittest.main()
