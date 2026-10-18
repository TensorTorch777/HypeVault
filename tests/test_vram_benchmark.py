"""VRAM benchmark bookkeeping. These tests do not touch a GPU."""

from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ml_rtx5080"))

import benchmark_vram as bench


def _fit(batch_size: int, allocated: int, reserved: int) -> dict:
    return {
        "model_family": "dinov2",
        "model_architecture": "cls_only",
        "input_size": 504,
        "batch_size": batch_size,
        "status": "FIT",
        "peak_allocated_bytes": allocated,
        "peak_reserved_bytes": reserved,
        "step_time_seconds": 0.2,
        "images_per_sec": batch_size / 0.2,
        "error": None,
    }


class VRAMBenchmarkTests(unittest.TestCase):
    def test_effective_batch_and_accumulation_target_32(self) -> None:
        self.assertEqual(bench.effective_batch(4, 8), 32)
        schedule = bench.accumulation_for_target(8)
        self.assertEqual(schedule["accumulate_grad"], 4)
        self.assertEqual(schedule["effective_batch"], 32)
        six = bench.accumulation_for_target(6)
        self.assertEqual(six["accumulate_grad"], 5)
        self.assertEqual(six["effective_batch"], 30)
        ten = bench.accumulation_for_target(10)
        self.assertEqual(ten["accumulate_grad"], 3)
        self.assertEqual(ten["effective_batch"], 30)
        with self.assertRaises(ValueError):
            bench.effective_batch(0, 2)

    def test_recommendation_keeps_a_vram_margin(self) -> None:
        total = 1000
        free = 1000
        chosen = bench.recommend_microbatch(
            [_fit(2, 700, 750), _fit(4, 860, 900), _fit(8, 990, 995)],
            total_vram_bytes=total,
            baseline_free_bytes=free,
        )
        self.assertEqual(chosen["batch_size"], 2)
        self.assertEqual(chosen["headroom_rule"], 0.15)
        fallback = bench.recommend_microbatch(
            [_fit(4, 860, 900), _fit(8, 990, 995)],
            total_vram_bytes=total,
            baseline_free_bytes=free,
        )
        self.assertEqual(fallback["batch_size"], 4)
        self.assertEqual(fallback["headroom_rule"], 0.10)
        self.assertIsNone(
            bench.recommend_microbatch([_fit(8, 990, 995)], total_vram_bytes=total, baseline_free_bytes=free)
        )

    def test_global_recommendation_uses_the_tightest_safe_batch(self) -> None:
        total = 1000
        free = 1000
        wide = [_fit(8, 700, 750)]
        tight = [_fit(4, 700, 750), _fit(8, 990, 995)]
        cases = {
            ("dinov2", "cls_only"): wide,
            ("dinov2", "cls_patch_attention"): wide,
            ("dinov3", "cls_only"): tight,
            ("dinov3", "cls_patch_attention"): wide,
        }
        recommendation = bench.recommend_training_config(
            cases, total_vram_bytes=total, baseline_free_bytes=free
        )
        self.assertEqual(recommendation["microbatch"], 4)
        self.assertEqual(recommendation["accumulate_grad"], 8)
        self.assertEqual(recommendation["effective_batch"], 32)
        cases[("dinov3", "cls_only")] = [_fit(8, 990, 995)]
        blocked = bench.recommend_training_config(
            cases, total_vram_bytes=total, baseline_free_bytes=free
        )
        self.assertIsNone(blocked["microbatch"])

    def test_oom_and_malformed_results(self) -> None:
        oom = {
            "batch_size": 16,
            "status": "OOM",
            "error": "CUDA out of memory",
        }
        self.assertEqual(bench.validate_measurement(oom)["status"], "OOM")
        self.assertTrue(bench.is_cuda_oom(RuntimeError("CUDA out of memory. Tried to allocate 2 GiB")))
        self.assertFalse(bench.is_cuda_oom(ValueError("CUDA out of memory")))
        with self.assertRaises(ValueError):
            bench.validate_measurement({"batch_size": 16, "status": "OOM"})
        with self.assertRaises(ValueError):
            bench.validate_measurement({**_fit(2, 10, 12), "status": "MAYBE"})
        broken = _fit(2, 10, 12)
        del broken["peak_allocated_bytes"]
        with self.assertRaises(ValueError):
            bench.validate_measurement(broken)

    def test_report_serialization_stays_inside_experiments(self) -> None:
        payload = {
            "timestamp": "2026-10-04T19:00:00+05:30",
            "recommendation": {"microbatch": 4, "accumulate_grad": 8, "effective_batch": 32},
        }
        row = bench.measurement_row(_fit(4, 700, 750), payload["recommendation"])
        root = Path(__file__).resolve().parents[1] / "ml_rtx5080" / "experiments"
        root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=root) as tmp:
            json_path, csv_path = bench.write_benchmark_report(Path(tmp), payload, [row])
            saved = json.loads(json_path.read_text())
            with csv_path.open() as handle:
                table = list(csv.DictReader(handle))
        self.assertEqual(saved["recommendation"]["effective_batch"], 32)
        self.assertEqual(table[0]["status"], "FIT")
        self.assertEqual(table[0]["recommended_microbatch"], "4")
        self.assertEqual(table[0]["recommended_effective_batch"], "32")
        forbidden = Path(__file__).resolve().parents[1] / "ml_rtx5080" / "checkpoints"
        with self.assertRaises(ValueError):
            bench.write_benchmark_report(forbidden, payload, [row])

    def test_catalog_sample_is_deterministic_and_read_only(self) -> None:
        records = []
        for index in range(20):
            records.append(type("Record", (), {"label": index // 10, "image_path": f"img-{index}"})())
        first = bench.select_representative_records(records, 4)
        second = bench.select_representative_records(records, 4)
        self.assertEqual([item.image_path for item in first], [item.image_path for item in second])
        self.assertEqual(sorted({item.label for item in first}), [0, 1])
        self.assertEqual(len(records), 20)


if __name__ == "__main__":
    unittest.main()
