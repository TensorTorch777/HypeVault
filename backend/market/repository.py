"""Database access for market observations."""

from __future__ import annotations

import math
from datetime import date, timedelta
from typing import Iterable

from sqlalchemy import Select, and_, func, or_, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from database import MarketListing
from market.types import MarketItem

_STOPWORDS = {"watch", "the", "and", "with", "for", "men", "mens", "women", "box", "papers"}


async def save_observations(session: AsyncSession, items: Iterable[MarketItem]) -> int:
    """Insert observations, keeping at most one per item per day. Returns rows written."""
    rows = [item.as_row() for item in items]
    if not rows:
        return 0
    stmt = insert(MarketListing).values(rows).on_conflict_do_nothing(constraint="uq_market_listing_observation")
    result = await session.execute(stmt)
    await session.commit()
    return int(result.rowcount or 0)


def search_terms(query: str) -> list[str]:
    tokens = [t.strip(" ,.;:()\"'").casefold() for t in query.split()]
    return [t for t in tokens if len(t) >= 2 and t not in _STOPWORDS][:6]


def _match(source: str, brand: str | None, query: str) -> list:
    clauses = [MarketListing.source == source]
    if brand:
        clauses.append(MarketListing.brand == brand)
    terms = search_terms(query)
    for term in terms:
        like = f"%{term}%"
        clauses.append(
            or_(
                MarketListing.title.ilike(like),
                MarketListing.reference.ilike(like),
                MarketListing.model.ilike(like),
                MarketListing.brand.ilike(like),
            )
        )
    return clauses


SORTS = ("near_median", "lowest", "highest")


async def find_listings(
    session: AsyncSession,
    source: str,
    query: str,
    brand: str | None,
    *,
    limit: int = 12,
    since: date | None = None,
    sort: str = "lowest",
) -> tuple[list[MarketListing], dict | None]:
    """Matching rows plus count/min/median/max over every match.

    sort: "lowest" (cheapest first), "highest", or "near_median" (closest to the median
    asking price first, which shows what the typical listing looks like).
    """
    clauses = _match(source, brand, query)
    if since is not None:
        clauses.append(MarketListing.observed_on >= since)
    where = and_(*clauses)
    stats_q = select(
        func.count(),
        func.min(MarketListing.price),
        func.percentile_cont(0.5).within_group(MarketListing.price),
        func.max(MarketListing.price),
    ).where(where, MarketListing.currency == "USD")
    count, lo, median, hi = (await session.execute(stats_q)).one()
    stats = None if not count else {"count": int(count), "min": lo, "median": float(median), "max": hi}
    if sort == "highest":
        order = MarketListing.price.desc()
    elif sort == "near_median" and stats is not None:
        order = func.abs(MarketListing.price - stats["median"]).asc()
    else:
        order = MarketListing.price.asc()
    rows_q: Select = select(MarketListing).where(where).order_by(order).limit(limit)
    rows = list((await session.execute(rows_q)).scalars())
    return rows, stats


async def price_histogram(
    session: AsyncSession, source: str, query: str, brand: str | None, stats: dict | None, *, bins: int = 28
) -> list[dict]:
    """Counts of matching listings in equal-width bins of log(price), from min to max."""
    if not stats or stats["count"] < 2 or stats["max"] <= stats["min"]:
        return []
    lo, hi = math.log(stats["min"]), math.log(stats["max"]) + 1e-9
    bucket = func.width_bucket(func.ln(MarketListing.price), lo, hi, bins)
    q = (
        select(bucket.label("b"), func.count())
        .where(and_(*_match(source, brand, query)), MarketListing.currency == "USD", MarketListing.price > 0)
        .group_by("b")
    )
    counts = {int(b): int(n) for b, n in (await session.execute(q)).all()}
    step = (hi - lo) / bins
    return [
        {"from": round(math.exp(lo + i * step), 2), "to": round(math.exp(lo + (i + 1) * step), 2), "count": counts.get(i + 1, 0)}
        for i in range(bins)
    ]


async def daily_medians(
    session: AsyncSession, source: str, query: str, brand: str | None, *, days: int = 365
) -> list[dict]:
    since = date.today() - timedelta(days=days)
    clauses = _match(source, brand, query) + [MarketListing.observed_on >= since, MarketListing.currency == "USD"]
    q = (
        select(
            MarketListing.observed_on,
            func.percentile_cont(0.5).within_group(MarketListing.price),
            func.count(),
        )
        .where(and_(*clauses))
        .group_by(MarketListing.observed_on)
        .order_by(MarketListing.observed_on)
    )
    return [{"date": d.isoformat(), "median": float(m), "n": int(n)} for d, m, n in (await session.execute(q)).all()]


async def source_summary(session: AsyncSession) -> dict:
    q = select(
        MarketListing.source,
        func.count(),
        func.min(MarketListing.observed_on),
        func.max(MarketListing.observed_on),
        func.max(MarketListing.observed_at),
    ).group_by(MarketListing.source)
    out = {}
    for source, n, first, last, last_seen in (await session.execute(q)).all():
        out[source] = {
            "rows": int(n),
            "first_observed_on": first.isoformat() if first else None,
            "last_observed_on": last.isoformat() if last else None,
            "last_observed_at": last_seen.isoformat() if last_seen else None,
        }
    return out


def listing_dict(row: MarketListing) -> dict:
    return {
        "source": row.source,
        "title": row.title,
        "brand": row.brand,
        "model": row.model,
        "reference": row.reference,
        "price": row.price,
        "currency": row.currency,
        "condition": row.condition,
        "year": row.year,
        "seller_feedback_pct": row.seller_feedback_pct,
        "item_url": row.item_url,
        "image_url": row.image_url,
        "observed_on": row.observed_on.isoformat(),
        "observed_at": row.observed_at.isoformat() if row.observed_at else None,
    }
