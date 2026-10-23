"""Phase 74 protocol pieces: frozen hashes, scene plan, and the decision rule. No model or image decode."""

from __future__ import annotations

import hashlib
import json
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ML = REPO / "ml_rtx5080"
V4 = ML / "benchmark_v4_fresh"
sys.path.insert(0, str(ML))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def scorer_constant(name: str) -> str:
    for line in (V4 / "score.py").read_text().splitlines():
        if line.startswith(f"{name} = "):
            return line.split('"')[1]
    raise AssertionError(name)


class FrozenArtifactTests(unittest.TestCase):
    def test_scorer_pins_the_protocol(self) -> None:
        self.assertEqual(sha256(V4 / "protocol.json"), scorer_constant("PROTOCOL_SHA256"))

    def test_scorer_pins_the_rule_and_the_rule_pins_its_head(self) -> None:
        rule_path = ML / "experiments" / "kandinsky_v4" / "fusion_rule.json"
        if not rule_path.exists():
            self.skipTest("Phase 74 rule is a local research artifact")
        self.assertEqual(sha256(rule_path), scorer_constant("RULE_V4_SHA256"))
        rule = json.loads(rule_path.read_text())
        self.assertTrue(rule["frozen_before_any_v4_scoring"])
        self.assertLess(rule["thresholds"]["t_review"], rule["thresholds"]["t_fake"])
        head = REPO / rule["model"]["path"]
        if head.exists():
            self.assertEqual(sha256(head), rule["model"]["sha256"])

    def test_protocol_success_criterion_is_80_percent(self) -> None:
        protocol = json.loads((V4 / "protocol.json").read_text())
        self.assertIn("at least 80.0% of kandinsky22_v4", protocol["success_criterion"])
        self.assertIn("at most 3.0% of real_ebay_v4", protocol["success_criterion"])


class ScenePlanTests(unittest.TestCase):
    def setUp(self) -> None:
        sys.path.insert(0, str(ML / "benchmark_v2_fresh"))
        sys.path.insert(0, str(V4))
        import importlib.util

        spec = importlib.util.spec_from_file_location("build_v4", V4 / "build.py")
        self.build = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(self.build)
        except ModuleNotFoundError as exc:  # CI installs no diffusion or scraping extras
            self.skipTest(f"benchmark builder needs {exc.name}")
        self.models = {b: ["Model A", "Model B"] for b in self.build.BRANDS}

    def test_test_set_is_half_new_scenes(self) -> None:
        jobs = self.build.plan("kandinsky22_v4", 60, self.models, 76_000, new_scenes=True)
        self.assertEqual(len(jobs), 300)
        for brand in self.build.BRANDS:
            mine = [j for j in jobs if j["brand"] == brand]
            self.assertEqual(sum(j["new_scene"] for j in mine), 30)

    def test_training_set_never_uses_new_scenes(self) -> None:
        jobs = self.build.plan("kandinsky22_train", 100, self.models, 78_000, new_scenes=False)
        self.assertFalse(any(j["new_scene"] for j in jobs))
        self.assertFalse({j["scene"] for j in jobs} & set(self.build.NEW_SCENES))

    def test_seed_ranges_do_not_overlap(self) -> None:
        test = {j["seed"] for j in self.build.plan("kandinsky22_v4", 60, self.models, 76_000, True)}
        train = {j["seed"] for j in self.build.plan("kandinsky22_train", 200, self.models, 78_000, False)}
        self.assertFalse(test & train)


if __name__ == "__main__":
    unittest.main()
