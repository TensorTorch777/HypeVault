"""Single sign-on rules: which Google accounts may sign in, and with which role.

By default only Google Workspace accounts from `sso_allowed_domains` (somaiya.edu) are accepted.
The email domain and Google's `hd` (hosted domain) claim must both match, so a personal Gmail
account that merely uses a lookalike address is refused. `*` in the list accepts any verified
Google account. Roles come from the server: emails in `admin_emails` are admins, everyone else
is a seller (who can also browse).
"""

from __future__ import annotations

from database import UserRole


class SsoRefused(Exception):
    """The identity is valid Google, but not one this deployment accepts."""


def check_domain(email: str, hosted_domain: str | None, allowed: frozenset[str]) -> str:
    """Return the matched domain or raise SsoRefused."""
    email = email.strip().casefold()
    if "@" not in email:
        raise SsoRefused("not an email address")
    domain = email.rsplit("@", 1)[1]
    if "*" in allowed:
        return domain
    hd = (hosted_domain or "").strip().casefold()
    if domain not in allowed:
        raise SsoRefused(f"{domain} is not an allowed domain")
    if hd != domain:
        raise SsoRefused("the account is not managed by the institution's Google Workspace")
    return domain


def role_for(email: str, admins: frozenset[str]) -> UserRole:
    """Role after an SSO sign-in. Admin comes only from the server list; an account dropped from it becomes a seller."""
    if email.strip().casefold() in admins:
        return UserRole.admin
    return UserRole.seller


def can_sell(role: UserRole | str | None) -> bool:
    value = getattr(role, "value", role)
    return value in {UserRole.seller.value, UserRole.admin.value}
