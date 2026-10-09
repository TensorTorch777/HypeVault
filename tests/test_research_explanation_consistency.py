"""Regression checks for attaching a research explanation to a verify result."""

from __future__ import annotations

import subprocess
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
TEST_FILE = REPO / "tests" / "researchExplanationConsistency.test.ts"


class ResearchExplanationConsistencyTests(unittest.TestCase):
    def test_model_decision_and_unavailable_cases(self) -> None:
        result = subprocess.run(
            ["node", "--experimental-strip-types", "--test", str(TEST_FILE)],
            cwd=REPO,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("a matching explanation is attached", result.stdout)
        self.assertIn("a model mismatch discards the explanation payload", result.stdout)
        self.assertIn("a decision mismatch discards the explanation payload", result.stdout)
        self.assertIn("an unavailable explanation keeps its own reason", result.stdout)


if __name__ == "__main__":
    unittest.main()
