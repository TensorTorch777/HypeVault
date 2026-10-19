"""Market listings: observed market prices with provenance, kept as history."""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "20261016_0002"
down_revision = "20260402_0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "market_listings",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("source", sa.String(length=32), nullable=False),
        sa.Column("external_id", sa.String(length=128), nullable=False),
        sa.Column("brand", sa.String(length=128), nullable=False),
        sa.Column("model", sa.String(length=256), nullable=True),
        sa.Column("reference", sa.String(length=128), nullable=True),
        sa.Column("title", sa.String(length=512), nullable=False),
        sa.Column("price", sa.Float(), nullable=False),
        sa.Column("currency", sa.String(length=8), nullable=False),
        sa.Column("condition", sa.String(length=128), nullable=True),
        sa.Column("year", sa.String(length=16), nullable=True),
        sa.Column("seller_ref", sa.String(length=32), nullable=True),
        sa.Column("seller_feedback_pct", sa.Float(), nullable=True),
        sa.Column("item_url", sa.Text(), nullable=True),
        sa.Column("image_url", sa.Text(), nullable=True),
        sa.Column("observed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("observed_on", sa.Date(), nullable=False),
        sa.Column("provenance", postgresql.JSONB(), nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.Column("imported_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("source", "external_id", "observed_on", name="uq_market_listing_observation"),
    )
    op.create_index(op.f("ix_market_listings_source"), "market_listings", ["source"], unique=False)
    op.create_index(op.f("ix_market_listings_brand"), "market_listings", ["brand"], unique=False)
    op.create_index("ix_market_listings_brand_reference", "market_listings", ["brand", "reference"], unique=False)


def downgrade() -> None:
    op.drop_index("ix_market_listings_brand_reference", table_name="market_listings")
    op.drop_index(op.f("ix_market_listings_brand"), table_name="market_listings")
    op.drop_index(op.f("ix_market_listings_source"), table_name="market_listings")
    op.drop_table("market_listings")
