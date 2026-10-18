"""Phase 57 protocol pieces. No diffusion pipeline, checkpoint, or dataset decode."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "ml_rtx5080"))

import unseen_generator as ug


class PromptTests(unittest.TestCase):
    def test_every_brand_gets_the_same_number_of_prompts(self) -> None:
        rows = ug.prompts()
        self.assertEqual(len(rows), 5 * 5 * len(ug.SCENES))
        counts = {brand: sum(row["brand"] == brand for row in rows) for brand in ug.MODEL_LINES}
        self.assertEqual(set(counts.values()), {5 * len(ug.SCENES)})

    def test_prompt_names_its_brand(self) -> None:
        for row in ug.prompts():
            self.assertIn(row["brand"], row["prompt"])

    def test_protocol_counts_match_the_prompt_list(self) -> None:
        document = ug.protocol_document()
        self.assertEqual(document["images_per_generator"], len(ug.prompts()) * ug.PER_PROMPT)
        self.assertEqual(document["frozen_checkpoint_sha256"], ug.FROZEN_SHA256)


class IntervalTests(unittest.TestCase):
    def test_clopper_pearson_edges(self) -> None:
        self.assertEqual(ug.clopper_pearson(0, 10)[0], 0.0)
        self.assertEqual(ug.clopper_pearson(10, 10)[1], 1.0)
        low, high = ug.clopper_pearson(5, 10)
        self.assertLess(low, 0.5)
        self.assertGreater(high, 0.5)


class ControlSampleTests(unittest.TestCase):
    def test_sample_is_seeded_and_capped_per_brand(self) -> None:
        rows = [{"brand": brand, "path": f"/{brand}/{i:04d}.jpg"} for brand in ug.MODEL_LINES for i in range(150)]
        first = ug._sample_real(rows)
        self.assertEqual(first, ug._sample_real(list(reversed(rows))))
        self.assertEqual(len(first), 5 * ug.CONTROL_PER_BRAND)


if __name__ == "__main__":
    unittest.main()
