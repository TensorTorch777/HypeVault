"""Initial schema: users, listings, price_snapshots."""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "20260402_0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "users",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("email", sa.String(length=320), nullable=False),
        sa.Column("password_hash", sa.String(length=255), nullable=False),
        sa.Column(
            "role",
            sa.Enum("buyer", "seller", name="userrole"),
            nullable=False,
        ),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_users_email"), "users", ["email"], unique=True)

    op.create_table(
        "listings",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("seller_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("product_name", sa.String(length=512), nullable=False),
        sa.Column(
            "category",
            sa.Enum("sneaker", "watch", name="listingcategory"),
            nullable=False,
        ),
        sa.Column("brand", sa.String(length=256), nullable=True),
        sa.Column("condition", sa.String(length=128), nullable=True),
        sa.Column("size", sa.String(length=64), nullable=True),
        sa.Column("s3_url", sa.Text(), nullable=True),
        sa.Column("verdict", sa.String(length=32), nullable=True),
        sa.Column("confidence", sa.Float(), nullable=True),
        sa.Column(
            "status",
            sa.Enum("pending", "live", "rejected", name="listingstatus"),
            nullable=False,
        ),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.ForeignKeyConstraint(["seller_id"], ["users.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_listings_seller_id"), "listings", ["seller_id"], unique=False)

    op.create_table(
        "price_snapshots",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("listing_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("platform", sa.String(length=64), nullable=False),
        sa.Column("price", sa.Float(), nullable=False),
        sa.Column("currency", sa.String(length=8), nullable=False),
        sa.Column("delivery_estimate", sa.String(length=128), nullable=True),
        sa.Column("seller_rating", sa.String(length=64), nullable=True),
        sa.Column("scraped_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.ForeignKeyConstraint(["listing_id"], ["listings.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_price_snapshots_listing_id"), "price_snapshots", ["listing_id"], unique=False)


def downgrade() -> None:
    op.drop_index(op.f("ix_price_snapshots_listing_id"), table_name="price_snapshots")
    op.drop_table("price_snapshots")
    op.drop_index(op.f("ix_listings_seller_id"), table_name="listings")
    op.drop_table("listings")
    op.drop_index(op.f("ix_users_email"), table_name="users")
    op.drop_table("users")
    op.execute("DROP TYPE IF EXISTS listingstatus")
    op.execute("DROP TYPE IF EXISTS listingcategory")
    op.execute("DROP TYPE IF EXISTS userrole")
