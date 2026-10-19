"""Single sign-on: only institution Google Workspace accounts, roles decided by the server."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "backend"))

from auth.sso import SsoRefused, can_sell, check_domain, role_for  # noqa: E402
from database import UserRole  # noqa: E402

ALLOWED = frozenset({"somaiya.edu"})
ADMINS = frozenset({"n.vadgaonkar@somaiya.edu"})


class SsoTests(unittest.TestCase):
    def test_institution_workspace_account_is_accepted(self) -> None:
        self.assertEqual(check_domain("Student@Somaiya.edu", "somaiya.edu", ALLOWED), "somaiya.edu")

    def test_other_domains_and_unmanaged_accounts_are_refused(self) -> None:
        with self.assertRaises(SsoRefused):
            check_domain("someone@gmail.com", None, ALLOWED)
        with self.assertRaises(SsoRefused):
            check_domain("someone@hypevaultapp.com", "hypevaultapp.com", ALLOWED)
        # right-looking address, but not managed by the somaiya.edu Workspace
        with self.assertRaises(SsoRefused):
            check_domain("someone@somaiya.edu", None, ALLOWED)

    def test_admin_comes_only_from_the_server_list(self) -> None:
        self.assertEqual(role_for("n.vadgaonkar@somaiya.edu", None, ADMINS), UserRole.admin)
        self.assertEqual(role_for("N.Vadgaonkar@somaiya.edu", UserRole.buyer, ADMINS), UserRole.admin)
        self.assertEqual(role_for("student@somaiya.edu", None, ADMINS), UserRole.seller)
        self.assertEqual(role_for("student@somaiya.edu", UserRole.buyer, ADMINS), UserRole.seller)

    def test_admins_can_sell_and_admin_is_not_self_assignable(self) -> None:
        from auth.models import PUBLIC_REGISTRATION_ROLES

        self.assertTrue(can_sell(UserRole.admin))
        self.assertTrue(can_sell("seller"))
        self.assertFalse(can_sell(UserRole.buyer))
        self.assertNotIn(UserRole.admin, PUBLIC_REGISTRATION_ROLES)

    def test_google_login_enforces_the_domain_before_creating_a_user(self) -> None:
        routes = (REPO / "backend" / "auth" / "routes.py").read_text()
        self.assertLess(routes.index("check_domain(email"), routes.index("result = await db.execute(select(User).where(User.email == email))"))
        self.assertNotIn("Google sign-in is available for buyers only", routes)


if __name__ == "__main__":
    unittest.main()
