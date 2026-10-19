"""Listing screening outcome: code, seller-facing reason and evidence details."""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "20261017_0003"
down_revision = "20261016_0002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("listings", sa.Column("screening_code", sa.String(length=48), nullable=True))
    op.add_column("listings", sa.Column("screening_reason", sa.Text(), nullable=True))
    op.add_column("listings", sa.Column("screening_details", postgresql.JSONB(), nullable=True))


def downgrade() -> None:
    op.drop_column("listings", "screening_details")
    op.drop_column("listings", "screening_reason")
    op.drop_column("listings", "screening_code")
