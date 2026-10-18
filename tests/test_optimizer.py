"""Optimizer groups, decay split, and gradient-accumulation scaling."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ml_rtx5080"))

import classifier as clf
import layerwise_lr as ll
import train as tv

_OPT_CFG = {
    "backbone_lr": 1e-5,
    "head_lr": 1e-4,
    "layer_decay": 0.75,
    "weight_decay": 0.05,
}


def _group_index(groups: list[dict]) -> dict[int, dict]:
    found: dict[int, dict] = {}
    for group in groups:
        for parameter in group["params"]:
            if id(parameter) in found:
                raise AssertionError(f"parameter repeated in {group['name']}")
            found[id(parameter)] = group
    return found


def _sgd_snapshots(divisors: list[int]) -> list[torch.Tensor]:
    generator = torch.Generator().manual_seed(1)
    batches = [
        (
            torch.randn(2, 3, generator=generator),
            torch.randint(0, 2, (2,), generator=generator).float(),
        )
        for _ in range(len(divisors))
    ]
    model = nn.Linear(3, 1, bias=False)
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[0.2, -0.4, 0.1]]))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    snapshots: list[torch.Tensor] = []
    num_batches = len(divisors)
    for batch_idx, (features, labels) in enumerate(batches):
        logits = model(features).squeeze(-1)
        loss = F.binary_cross_entropy_with_logits(logits, labels) / divisors[batch_idx]
        loss.backward()
        if tv.optimizer_step_due(batch_idx, num_batches, 2):
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            snapshots.append(model.weight.detach().clone())
    return snapshots


class OptimizerWiringTests(unittest.TestCase):
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

    def test_trainer_wires_layerwise_groups_for_both_families_and_heads(self) -> None:
        for (family, arch), model in self.models.items():
            with self.subTest(family=family, arch=arch):
                optimizer, groups, table = tv.build_optimizer(model, _OPT_CFG)
                self.assertIsInstance(optimizer, torch.optim.AdamW)
                names = [group["name"] for group in groups]
                self.assertIn("block_00", names)
                self.assertIn("block_11", names)
                self.assertIn("head", names)
                self.assertIn("head_no_decay", names)
                self.assertIn("block_00", table)
                self.assertIn("block_11", table)
                by_name = {group["name"]: group for group in groups}
                self.assertLess(by_name["block_00"]["lr"], by_name["block_11"]["lr"])
                self.assertAlmostEqual(by_name["block_11"]["lr"], 1e-5)
                self.assertAlmostEqual(by_name["head"]["lr"], 1e-4)
                self.assertAlmostEqual(by_name["head_no_decay"]["lr"], by_name["head"]["lr"])
                optimizer_ids = {
                    id(parameter)
                    for group in optimizer.param_groups
                    for parameter in group["params"]
                }
                group_ids = {id(parameter) for group in groups for parameter in group["params"]}
                trainable = {
                    id(parameter) for parameter in model.parameters() if parameter.requires_grad
                }
                self.assertEqual(optimizer_ids, group_ids)
                self.assertEqual(optimizer_ids, trainable)
                learning_rates = {group["lr"] for group in optimizer.param_groups}
                self.assertGreater(len(learning_rates), 2)

    def test_decay_split_covers_every_parameter_once(self) -> None:
        token_names = {"cls_token", "pos_embed", "reg_token"}
        for (family, arch), model in self.models.items():
            with self.subTest(family=family, arch=arch):
                groups = ll.build_layerwise_param_groups(
                    model,
                    base_backbone_lr=1e-5,
                    head_lr=1e-4,
                    layer_decay=0.75,
                    weight_decay=0.05,
                    separate_no_decay=True,
                )
                located = _group_index(groups)
                named = dict(model.named_parameters())
                norm_ids = ll.normalization_parameter_ids(model)
                trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
                self.assertEqual(set(located), {id(parameter) for parameter in trainable})
                self.assertEqual(len(located), len(trainable))
                unexpected: list[str] = []
                for name, parameter in named.items():
                    if not parameter.requires_grad:
                        self.assertNotIn(id(parameter), located)
                        continue
                    group = located[id(parameter)]
                    if parameter.ndim <= 1 or name.endswith(".bias") or id(parameter) in norm_ids:
                        self.assertEqual(group["weight_decay"], 0.0)
                        self.assertTrue(group["name"].endswith("_no_decay"))
                    elif name.endswith(".weight") or name.split(".")[-1] in token_names:
                        self.assertEqual(group["weight_decay"], 0.05)
                        self.assertFalse(group["name"].endswith("_no_decay"))
                    else:
                        unexpected.append(name)
                self.assertEqual(unexpected, [])
                by_name = {group["name"]: group for group in groups}
                self.assertIn("final_norm_no_decay", by_name)
                self.assertNotIn("final_norm", by_name)
                self.assertAlmostEqual(by_name["final_norm_no_decay"]["lr"], by_name["block_11"]["lr"])
                self.assertAlmostEqual(by_name["block_11"]["lr"], by_name["block_11_no_decay"]["lr"])
                self.assertEqual(by_name["head"]["weight_decay"], 0.05)
                head_names = [name for name in named if name.startswith("trunk.")]
                self.assertTrue(any(name.endswith("fc1.weight") for name in head_names))
                self.assertEqual(located[id(named["trunk.fc1.weight"])]["weight_decay"], 0.05)
                self.assertEqual(located[id(named["trunk.fc1.bias"])]["weight_decay"], 0.0)
                self.assertEqual(located[id(named["trunk.norm.weight"])]["weight_decay"], 0.0)
                self.assertEqual(located[id(named["trunk.norm.bias"])]["weight_decay"], 0.0)
                proj_weight = next(name for name in named if name.endswith("patch_embed.proj.weight"))
                proj_bias = next(name for name in named if name.endswith("patch_embed.proj.bias"))
                self.assertEqual(located[id(named[proj_weight])]["weight_decay"], 0.05)
                self.assertEqual(located[id(named[proj_bias])]["weight_decay"], 0.0)
                cls_token = next(name for name in named if name.endswith("cls_token"))
                self.assertEqual(located[id(named[cls_token])]["weight_decay"], 0.05)
                layer_scale = next(
                    name for name in named if name.endswith("ls1.gamma") or name.endswith("gamma_1")
                )
                self.assertEqual(named[layer_scale].ndim, 1)
                self.assertEqual(located[id(named[layer_scale])]["weight_decay"], 0.0)
                if arch == clf.CLS_ONLY:
                    self.assertFalse(any(name.startswith("attention_pool.") for name in named))
                    self.assertFalse(any(name.startswith("fusion.") for name in named))
                else:
                    self.assertEqual(located[id(named["attention_pool.score.weight"])]["weight_decay"], 0.05)
                    self.assertEqual(located[id(named["attention_pool.score.bias"])]["weight_decay"], 0.0)
                    self.assertEqual(located[id(named["attention_pool.norm.weight"])]["weight_decay"], 0.0)
                    self.assertEqual(located[id(named["fusion.weight"])]["weight_decay"], 0.05)
                    self.assertEqual(located[id(named["fusion.bias"])]["weight_decay"], 0.0)


class GradientScalingTests(unittest.TestCase):
    def test_five_batches_scale_full_windows_and_leave_the_tail_unscaled(self) -> None:
        divisors = [tv.microbatch_loss_divisor(index, 5, 2) for index in range(5)]
        self.assertEqual(divisors, [2, 2, 2, 2, 1])
        correct = _sgd_snapshots(divisors)
        under_scaled = _sgd_snapshots([2, 2, 2, 2, 2])
        self.assertEqual(len(correct), 3)
        self.assertTrue(torch.allclose(correct[0], under_scaled[0]))
        self.assertTrue(torch.allclose(correct[1], under_scaled[1]))
        self.assertFalse(torch.allclose(correct[2], under_scaled[2]))

        generator = torch.Generator().manual_seed(1)
        batches = [
            (
                torch.randn(2, 3, generator=generator),
                torch.randint(0, 2, (2,), generator=generator).float(),
            )
            for _ in range(5)
        ]
        model = nn.Linear(3, 1, bias=False)
        with torch.no_grad():
            model.weight.copy_(correct[1])
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        features, labels = batches[4]
        loss = F.binary_cross_entropy_with_logits(model(features).squeeze(-1), labels)
        loss.backward()
        optimizer.step()
        self.assertTrue(torch.allclose(model.weight.detach(), correct[2]))

        doubled = nn.Linear(3, 1, bias=False)
        with torch.no_grad():
            doubled.weight.copy_(correct[1])
        optimizer = torch.optim.SGD(doubled.parameters(), lr=0.1)
        loss = F.binary_cross_entropy_with_logits(doubled(features).squeeze(-1), labels) * 2
        loss.backward()
        optimizer.step()
        self.assertFalse(torch.allclose(doubled.weight.detach(), correct[2]))

    def test_flush_rescale_matches_the_batches_actually_accumulated(self) -> None:
        torch.manual_seed(0)
        model = nn.Linear(3, 1, bias=False)
        features = torch.randn(2, 3)
        labels = torch.tensor([0.0, 1.0])
        full = nn.Linear(3, 1, bias=False)
        with torch.no_grad():
            full.weight.copy_(model.weight)
        F.binary_cross_entropy_with_logits(full(features).squeeze(-1), labels).backward()
        expected = full.weight.grad.detach().clone()

        F.binary_cross_entropy_with_logits(model(features).squeeze(-1), labels).backward()
        model.weight.grad.div_(2)
        tv.rescale_accumulated_gradients(model, planned_divisor=2, actual_count=1)
        self.assertTrue(torch.allclose(model.weight.grad, expected))
        unchanged = model.weight.grad.detach().clone()
        tv.rescale_accumulated_gradients(model, planned_divisor=1, actual_count=1)
        self.assertTrue(torch.equal(model.weight.grad, unchanged))

    def test_optimizer_and_scheduler_steps_match_for_even_and_partial_windows(self) -> None:
        for num_batches, accumulate in ((4, 2), (5, 2)):
            with self.subTest(num_batches=num_batches, accumulate=accumulate):
                expected = tv.optimizer_updates_per_epoch(num_batches, accumulate)
                model = nn.Linear(2, 1)
                optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
                scheduler = tv.cosine_schedule_with_warmup(
                    optimizer, warmup_steps=0, total_steps=max(expected, 1)
                )
                scaler = GradScaler("cuda", enabled=False)
                optimizer_steps = 0
                real_step = optimizer.step

                def counting_step(*args, **kwargs):
                    nonlocal optimizer_steps
                    optimizer_steps += 1
                    return real_step(*args, **kwargs)

                optimizer.step = counting_step  # type: ignore[method-assign]
                start = scheduler.last_epoch
                for batch_idx in range(num_batches):
                    loss = model(torch.randn(2, 2)).sum()
                    scaler.scale(loss).backward()
                    if tv.optimizer_step_due(batch_idx, num_batches, accumulate):
                        tv.apply_accumulated_step(model, optimizer, scaler, scheduler, 1.0)
                self.assertEqual(optimizer_steps, expected)
                self.assertEqual(scheduler.last_epoch - start, optimizer_steps)
                self.assertNotEqual(optimizer_steps, num_batches)


if __name__ == "__main__":
    unittest.main()
