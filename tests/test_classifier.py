"""CLS-only and CLS+patch heads. Random backbone init; no weight download."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ml_rtx5080"))

import classifier as clf


def _finite(tensor: torch.Tensor) -> bool:
    return bool(torch.isfinite(tensor).all())


class ClassifierHeadTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.models = {
            ("dinov2", clf.CLS_ONLY): clf.build_authenticity_classifier(
                "dinov2", clf.CLS_ONLY, image_size=504, pretrained=False
            ),
            ("dinov2", clf.CLS_PATCH_ATTENTION): clf.build_authenticity_classifier(
                "dinov2", clf.CLS_PATCH_ATTENTION, image_size=504, pretrained=False
            ),
            ("dinov3", clf.CLS_ONLY): clf.build_authenticity_classifier(
                "dinov3", clf.CLS_ONLY, image_size=512, pretrained=False
            ),
            ("dinov3", clf.CLS_PATCH_ATTENTION): clf.build_authenticity_classifier(
                "dinov3", clf.CLS_PATCH_ATTENTION, image_size=512, pretrained=False
            ),
        }

    def _images(self, family: str) -> torch.Tensor:
        side = 504 if family == "dinov2" else 512
        return torch.randn(2, 3, side, side)

    def test_dinov2_cls_only_forward(self) -> None:
        model = self.models[("dinov2", clf.CLS_ONLY)].eval()
        with torch.inference_mode():
            logits = model(self._images("dinov2"))
        self.assertEqual(tuple(logits.shape), (2,))
        self.assertTrue(_finite(logits))
        self.assertIsNone(model.attention_pool)

    def test_dinov2_cls_patch_forward(self) -> None:
        model = self.models[("dinov2", clf.CLS_PATCH_ATTENTION)].eval()
        with torch.inference_mode():
            logits, weights = model(self._images("dinov2"), return_attention=True)
        self.assertEqual(tuple(logits.shape), (2,))
        self.assertEqual(tuple(weights.shape), (2, 1296))
        self.assertTrue(_finite(logits))
        self.assertTrue(torch.allclose(weights.sum(dim=-1), torch.ones(2), atol=1e-5))

    def test_dinov3_cls_only_forward(self) -> None:
        model = self.models[("dinov3", clf.CLS_ONLY)].eval()
        with torch.inference_mode():
            logits = model(self._images("dinov3"))
        self.assertEqual(tuple(logits.shape), (2,))
        self.assertTrue(_finite(logits))

    def test_dinov3_cls_patch_excludes_registers(self) -> None:
        model = self.models[("dinov3", clf.CLS_PATCH_ATTENTION)].eval()
        seen: dict[str, int] = {}
        original = model.backbone.get_patch_tokens

        def wrapped(features: torch.Tensor) -> torch.Tensor:
            tokens = original(features)
            seen["tokens"] = int(tokens.shape[1])
            seen["excluded"] = int(features.shape[1] - tokens.shape[1])
            return tokens

        model.backbone.get_patch_tokens = wrapped  # type: ignore[method-assign]
        try:
            with torch.inference_mode():
                logits, weights = model(self._images("dinov3"), return_attention=True)
        finally:
            model.backbone.get_patch_tokens = original  # type: ignore[method-assign]
        self.assertEqual(tuple(logits.shape), (2,))
        self.assertEqual(seen["tokens"], 1024)
        self.assertEqual(seen["excluded"], 5)
        self.assertEqual(tuple(weights.shape), (2, 1024))
        self.assertEqual(model.backbone.num_register_tokens, 4)
        self.assertTrue(torch.allclose(weights.sum(dim=-1), torch.ones(2), atol=1e-5))
        self.assertTrue(_finite(logits))
        self.assertTrue(_finite(weights))

    def test_backprop_and_no_nonfinite_outputs(self) -> None:
        model = self.models[("dinov2", clf.CLS_PATCH_ATTENTION)].train()
        model.zero_grad(set_to_none=True)
        logits = model(self._images("dinov2"))
        self.assertTrue(_finite(logits))
        logits.sum().backward()
        grads = [param.grad for param in model.attention_pool.parameters() if param.grad is not None]
        self.assertTrue(grads)
        self.assertTrue(all(_finite(grad) for grad in grads))
        trunk_grads = [param.grad for param in model.trunk.parameters() if param.grad is not None]
        self.assertTrue(trunk_grads)

    def test_architecture_selection(self) -> None:
        cls_only = clf.build_authenticity_classifier("dinov2", "cls_only", pretrained=False)
        pooled = clf.build_authenticity_classifier("dinov3", "cls_patch_attention", pretrained=False)
        self.assertEqual(cls_only.classifier_arch, clf.CLS_ONLY)
        self.assertIsNone(cls_only.fusion)
        self.assertEqual(pooled.classifier_arch, clf.CLS_PATCH_ATTENTION)
        self.assertIsInstance(pooled.attention_pool, clf.PatchAttentionPool)
        self.assertEqual(cls_only.trunk.drop.p, clf.DEFAULT_DROPOUT)
        self.assertEqual(pooled.trunk.drop.p, clf.DEFAULT_DROPOUT)
        self.assertEqual(cls_only.trunk.fc1.out_features, 512)
        self.assertEqual(pooled.trunk.fc1.out_features, 512)
        self.assertEqual(cls_only.trunk.fc2.out_features, 1)
        self.assertEqual(pooled.trunk.fc2.out_features, 1)

    def test_invalid_architecture_name(self) -> None:
        with self.assertRaises(ValueError) as raised:
            clf.build_authenticity_classifier("dinov2", "transformer_block", pretrained=False)
        message = str(raised.exception)
        self.assertIn("transformer_block", message)
        self.assertIn("cls_only", message)
        self.assertIn("cls_patch_attention", message)


if __name__ == "__main__":
    unittest.main()
