"""Admin role for single sign-on accounts listed in ADMIN_EMAILS."""

from __future__ import annotations

from alembic import op

revision = "20261017_0004"
down_revision = "20261017_0003"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("ALTER TYPE userrole ADD VALUE IF NOT EXISTS 'admin'")


def downgrade() -> None:
    # Postgres cannot drop an enum value; demote admins instead.
    op.execute("UPDATE users SET role = 'seller' WHERE role = 'admin'")
