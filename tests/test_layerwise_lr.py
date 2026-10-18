"""Layer-wise learning rates follow block depth for both DINO families."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ml_rtx5080"))

import classifier as clf
import layerwise_lr as ll


def _ids(groups: list[dict]) -> list[int]:
    return [id(parameter) for group in groups for parameter in group["params"]]


class LayerwiseLRTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.models = {
            ("dinov2", clf.CLS_ONLY): clf.build_authenticity_classifier(
                "dinov2", clf.CLS_ONLY, pretrained=False
            ),
            ("dinov2", clf.CLS_PATCH_ATTENTION): clf.build_authenticity_classifier(
                "dinov2", clf.CLS_PATCH_ATTENTION, pretrained=False
            ),
            ("dinov3", clf.CLS_ONLY): clf.build_authenticity_classifier(
                "dinov3", clf.CLS_ONLY, pretrained=False
            ),
            ("dinov3", clf.CLS_PATCH_ATTENTION): clf.build_authenticity_classifier(
                "dinov3", clf.CLS_PATCH_ATTENTION, pretrained=False
            ),
        }

    def test_both_families_and_both_heads_cover_every_parameter_once(self) -> None:
        for (family, arch), model in self.models.items():
            with self.subTest(family=family, arch=arch):
                groups = ll.build_layerwise_param_groups(model)
                names = [group["name"] for group in groups]
                self.assertEqual(names[0], "embed")
                self.assertEqual(names[-1], "head")
                self.assertIn("block_00", names)
                self.assertIn("final_norm", names)
                trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
                found = _ids(groups)
                self.assertEqual(len(found), len(set(found)))
                self.assertEqual(set(found), {id(parameter) for parameter in trainable})
                self.assertEqual(len(found), len(trainable))

    def test_earlier_blocks_have_lower_lr_and_head_is_highest(self) -> None:
        model = self.models[("dinov3", clf.CLS_PATCH_ATTENTION)]
        groups = {group["name"]: group for group in ll.build_layerwise_param_groups(model)}
        n_blocks = len(ll.transformer_blocks(model))
        for index in range(1, n_blocks):
            earlier = groups[f"block_{index - 1:02d}"]["lr"]
            later = groups[f"block_{index:02d}"]["lr"]
            self.assertLess(earlier, later)
        self.assertLess(groups["embed"]["lr"], groups["block_00"]["lr"])
        self.assertAlmostEqual(groups["final_norm"]["lr"], groups[f"block_{n_blocks - 1:02d}"]["lr"])
        backbone_max = max(group["lr"] for name, group in groups.items() if name != "head")
        self.assertGreater(groups["head"]["lr"], backbone_max)
        self.assertAlmostEqual(groups["head"]["lr"], ll.DEFAULT_HEAD_LR)
        self.assertAlmostEqual(groups[f"block_{n_blocks - 1:02d}"]["lr"], ll.DEFAULT_BACKBONE_LR)

    def test_decay_one_makes_backbone_rates_equal(self) -> None:
        model = self.models[("dinov2", clf.CLS_ONLY)]
        groups = ll.build_layerwise_param_groups(model, layer_decay=1.0, base_backbone_lr=1e-5, head_lr=1e-4)
        backbone = [group for group in groups if group["name"] != "head"]
        self.assertTrue(all(group["lr"] == 1e-5 for group in backbone))
        self.assertEqual(groups[-1]["lr"], 1e-4)

    def test_decay_ratio_matches_the_formula(self) -> None:
        model = self.models[("dinov2", clf.CLS_ONLY)]
        decay = 0.5
        base = 2e-5
        groups = {group["name"]: group for group in ll.build_layerwise_param_groups(
            model, layer_decay=decay, base_backbone_lr=base, head_lr=3e-4
        )}
        n_blocks = len(ll.transformer_blocks(model))
        for index in range(n_blocks):
            expected = base * (decay ** (n_blocks - 1 - index))
            self.assertAlmostEqual(groups[f"block_{index:02d}"]["lr"], expected)
        self.assertAlmostEqual(groups["embed"]["lr"], base * (decay ** n_blocks))
        self.assertAlmostEqual(groups["block_01"]["lr"] / groups["block_00"]["lr"], 1.0 / decay)

    def test_weight_decay_is_stored_on_every_group(self) -> None:
        model = self.models[("dinov3", clf.CLS_ONLY)]
        groups = ll.build_layerwise_param_groups(model, weight_decay=0.05)
        self.assertTrue(groups)
        self.assertTrue(all(group["weight_decay"] == 0.05 for group in groups))

    def test_frozen_parameters_are_omitted(self) -> None:
        model = self.models[("dinov2", clf.CLS_PATCH_ATTENTION)]
        frozen = list(model.backbone.encoder.blocks[3].parameters())
        for parameter in frozen:
            parameter.requires_grad = False
        try:
            groups = ll.build_layerwise_param_groups(model)
            found = set(_ids(groups))
            self.assertTrue(all(id(parameter) not in found for parameter in frozen))
            trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
            self.assertEqual(found, {id(parameter) for parameter in trainable})
            names = [group["name"] for group in groups]
            self.assertNotIn("block_03", names)
            self.assertIn("block_02", names)
            self.assertIn("block_04", names)
        finally:
            for parameter in frozen:
                parameter.requires_grad = True

    def test_groups_are_deterministic(self) -> None:
        model = self.models[("dinov3", clf.CLS_ONLY)]
        first = ll.build_layerwise_param_groups(model)
        second = ll.build_layerwise_param_groups(model)
        self.assertEqual([group["name"] for group in first], [group["name"] for group in second])
        self.assertEqual([group["lr"] for group in first], [group["lr"] for group in second])
        self.assertEqual(_ids(first), _ids(second))

    def test_patch_attention_head_contains_the_pool_and_fusion(self) -> None:
        cls_only = ll.build_layerwise_param_groups(self.models[("dinov2", clf.CLS_ONLY)])
        pooled = ll.build_layerwise_param_groups(self.models[("dinov2", clf.CLS_PATCH_ATTENTION)])
        cls_head = sum(parameter.numel() for parameter in cls_only[-1]["params"])
        pool_head = sum(parameter.numel() for parameter in pooled[-1]["params"])
        self.assertGreater(pool_head, cls_head)
        self.assertEqual(cls_only[-1]["name"], "head")
        self.assertEqual(pooled[-1]["name"], "head")


if __name__ == "__main__":
    unittest.main()
