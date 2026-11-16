"""The public benchmark summary names the live model and reads its evidence from the frozen result files."""

from __future__ import annotations

import asyncio
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "backend"))

from inference.benchmark_routes import benchmark_summary  # noqa: E402


class BenchmarkSummaryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.live = asyncio.run(benchmark_summary())["live_model"]

    def test_live_model_is_the_phase76_hybrid(self) -> None:
        self.assertEqual(self.live["id"], "phase76_fusion")
        self.assertEqual(self.live["name"], "DINOv3 + Perception Encoder")
        self.assertEqual(self.live["phase"], 76)
        self.assertTrue(self.live["benchmark"]["met"])

    def test_benchmark_v6_counts_come_from_the_report(self) -> None:
        groups = {g["id"]: g["systems"] for g in self.live["benchmark"]["groups"]}
        self.assertEqual(groups["chatgpt_v6"]["phase76_head"], {"k": 30, "n": 30})
        self.assertEqual(groups["gemini_v6"]["phase76_head"], {"k": 17, "n": 17})
        self.assertEqual(groups["chatgpt_v6"]["dinov2_live"], {"k": 4, "n": 30})
        self.assertEqual(groups["real_ebay_v6"]["phase76_head"], {"k": 2, "n": 102})

    def test_phase78_only_holds_and_known_ai_cut(self) -> None:
        hold = self.live["edited_copy_hold"]
        self.assertIn("only holds", hold["role"])
        self.assertEqual(hold["rows"][1], {"edit": "Screenshot-style", "phase76": "37/60", "phase78": "60/60"})
        self.assertEqual(self.live["known_ai"]["cut"], 0.65)
        self.assertEqual(self.live["known_ai"]["real_matched"], 0)


if __name__ == "__main__":
    unittest.main()
