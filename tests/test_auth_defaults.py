"""Sign-in defaults that must stay safe when a deployment sets no environment variables."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "backend"))

from config import Settings  # noqa: E402


class AuthDefaultsTests(unittest.TestCase):
    def test_password_login_is_off_unless_enabled(self) -> None:
        # Password registration accepts any email domain and the seller role, so it must be opt-in.
        self.assertIs(Settings.model_fields["password_login_enabled"].default, False)

    def test_sso_is_limited_to_the_institution_domain_by_default(self) -> None:
        self.assertEqual(Settings.model_fields["sso_allowed_domains"].default, "somaiya.edu")


if __name__ == "__main__":
    unittest.main()
