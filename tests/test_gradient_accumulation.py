"""Gradient-accumulation windows must include a trailing partial update."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ml_rtx5080"))

import train as tv


def _step_batches(num_batches: int, accumulate_grad: int) -> list[int]:
    """1-based batches that close an optimizer update, matching train_epoch."""
    stepped: list[int] = []
    pending = False
    for batch_idx in range(num_batches):
        pending = True
        if tv.optimizer_step_due(batch_idx, num_batches, accumulate_grad):
            stepped.append(batch_idx + 1)
            pending = False
    if pending:
        stepped.append(num_batches)
    return stepped


class GradientAccumulationTests(unittest.TestCase):
    def test_five_batches_accumulate_two_steps_tail(self) -> None:
        self.assertEqual(_step_batches(5, 2), [2, 4, 5])
        self.assertEqual(tv.optimizer_updates_per_epoch(5, 2), 3)
        # Floor division drops the tail update. The schedule must not use it.
        self.assertEqual(5 // 2, 2)
        self.assertNotEqual(tv.optimizer_updates_per_epoch(5, 2), 5 // 2)

    def test_aligned_window_does_not_double_step(self) -> None:
        self.assertEqual(_step_batches(4, 2), [2, 4])
        self.assertEqual(tv.optimizer_updates_per_epoch(4, 2), 2)

    def test_short_epoch_still_steps_once(self) -> None:
        self.assertEqual(_step_batches(1, 2), [1])
        self.assertEqual(tv.optimizer_updates_per_epoch(1, 2), 1)

    def test_tail_batch_calls_optimizer_step(self) -> None:
        torch.manual_seed(0)
        model = nn.Linear(4, 1)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.5)
        calls: list[int] = []
        real_step = optimizer.step

        def recording_step(*args, **kwargs):
            calls.append(len(calls) + 1)
            return real_step(*args, **kwargs)

        optimizer.step = recording_step  # type: ignore[method-assign]
        num_batches = 5
        accumulate_grad = 2
        pending = False
        before = model.weight.detach().clone()
        for batch_idx in range(num_batches):
            loss = model(torch.randn(2, 4)).sum() / accumulate_grad
            loss.backward()
            pending = True
            if tv.optimizer_step_due(batch_idx, num_batches, accumulate_grad):
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                pending = False
        if pending:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        self.assertEqual(calls, [1, 2, 3])
        self.assertFalse(torch.equal(before, model.weight.detach()))

    def test_scheduler_steps_match_optimizer_updates(self) -> None:
        num_batches = 5
        accumulate_grad = 2
        updates = tv.optimizer_updates_per_epoch(num_batches, accumulate_grad)
        parameter = nn.Parameter(torch.zeros(1))
        optimizer = torch.optim.SGD([parameter], lr=1.0)
        scheduler = tv.cosine_schedule_with_warmup(
            optimizer,
            warmup_steps=updates,
            total_steps=updates * 2,
        )
        scheduler_steps = 0
        pending = False
        for batch_idx in range(num_batches):
            pending = True
            if tv.optimizer_step_due(batch_idx, num_batches, accumulate_grad):
                optimizer.step()
                scheduler.step()
                scheduler_steps += 1
                pending = False
        if pending:
            optimizer.step()
            scheduler.step()
            scheduler_steps += 1
        self.assertEqual(scheduler_steps, 3)
        self.assertEqual(updates, 3)
        # LambdaLR steps once inside its constructor, then once per optimizer update.
        self.assertEqual(scheduler.last_epoch, updates)


if __name__ == "__main__":
    unittest.main()
