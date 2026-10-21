"""Phase 71-73 research code: dial-text matching, frozen rules and pre-registered protocol hashes."""

from __future__ import annotations

import hashlib
import json
import re
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ML = REPO / "ml_rtx5080"
sys.path.insert(0, str(ML))

import watch_forensics as wf  # noqa: E402


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class WatchForensicsTests(unittest.TestCase):
    def test_words_strip_accents_and_case(self) -> None:
        self.assertEqual(wf.words("A. Lange & Söhne"), ["LANGE", "SOHNE"])

    def test_brand_text_evidence(self) -> None:
        reading = {"brand": "Patek Philippe", "rapidocr": [["PATEK PHILIPPE", 0.99], ["GENEVE", 0.98]],
                   "gemma": {"text": [], "dial_brand": "NONE", "garbled": False}}
        self.assertEqual(wf.text_evidence(reading)["status"], "CONSISTENT")
        reading["rapidocr"] = [["PATILUE", 0.9]]
        self.assertEqual(wf.text_evidence(reading)["status"], "GARBLED")
        reading["rapidocr"] = []
        self.assertEqual(wf.text_evidence(reading)["status"], "NO_TEXT")

    def test_frozen_rules_are_complete(self) -> None:
        v1 = json.loads((ML / "experiments/watch_forensics_v1/fusion_rule.json").read_text())
        self.assertLess(v1["thresholds"]["t_low"], v1["thresholds"]["T71"])
        v2 = json.loads((ML / "experiments/watch_forensics_v2/fusion_rule.json").read_text())
        self.assertLess(v2["thresholds"]["t_review"], v2["thresholds"]["t_fake"])
        self.assertEqual(len(v2["combiner"]["coef"]), len(v2["combiner"]["features"]))
        probe = REPO / v2["probe"]["path"]
        self.assertEqual(_sha(probe), v2["probe"]["sha256"])

    def test_scorers_pin_the_frozen_files(self) -> None:
        # Each one-time scorer refuses to run if its protocol or rule changed after freezing.
        for scorer, pins in (
            (ML / "benchmark_v2_fresh/score.py", {"PROTOCOL_SHA256": "benchmark_v2_fresh/protocol.json",
                                                  "RULE_SHA256": "experiments/watch_forensics_v1/fusion_rule.json"}),
            (ML / "benchmark_v3_fresh/score.py", {"PROTOCOL_SHA256": "benchmark_v3_fresh/protocol.json",
                                                  "RULE_V1_SHA256": "experiments/watch_forensics_v1/fusion_rule.json",
                                                  "RULE_V2_SHA256": "experiments/watch_forensics_v2/fusion_rule.json"}),
        ):
            source = scorer.read_text()
            for constant, path in pins.items():
                pinned = re.search(rf'^{constant} = "([0-9a-f]{{64}})"', source, re.M).group(1)
                self.assertEqual(pinned, _sha(ML / path), f"{scorer.name}: {constant}")

    def test_final_results_are_recorded(self) -> None:
        for bench in ("benchmark_v2_fresh", "benchmark_v3_fresh"):
            runs = sorted((ML / bench / "results").glob("*/metrics.json"))
            self.assertEqual(len(runs), 1, f"{bench} must be scored exactly once")
        v3 = json.loads(next((ML / "benchmark_v3_fresh/results").glob("*/metrics.json")).read_text())
        self.assertTrue(v3["analysis"]["success"]["met"])


if __name__ == "__main__":
    unittest.main()
