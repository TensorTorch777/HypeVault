"""Shared market types and helpers. No network or database access here."""

from __future__ import annotations

import hashlib
import hmac
import re
from dataclasses import asdict, dataclass, field
from datetime import date, datetime

from inference.scope_gate import evaluate_declared_brand

SOURCE_EBAY = "ebay"
SOURCE_CHRONO24_DATASET = "chrono24_dataset"

_PRICE_RE = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?")


@dataclass(frozen=True)
class MarketItem:
    source: str
    external_id: str
    brand: str
    title: str
    price: float
    currency: str
    observed_on: date
    model: str | None = None
    reference: str | None = None
    condition: str | None = None
    year: str | None = None
    seller_ref: str | None = None
    seller_feedback_pct: float | None = None
    item_url: str | None = None
    image_url: str | None = None
    observed_at: datetime | None = None
    provenance: dict = field(default_factory=dict)

    def as_row(self) -> dict:
        return asdict(self)


def canonical_brand(raw: str | None) -> str | None:
    """Map a marketplace brand string onto one of the five supported brands, or None."""
    scope = evaluate_declared_brand(raw)
    return scope.canonical_brand if scope.scope_status == "SUPPORTED" else None


def parse_price(raw: str | float | int | None) -> float | None:
    """'$71,968' -> 71968.0. Ranges keep the lower bound. Returns None for missing or non-positive values."""
    if raw is None:
        return None
    if isinstance(raw, (int, float)):
        return float(raw) if raw > 0 else None
    match = _PRICE_RE.search(str(raw).replace("–", "-"))
    if not match:
        return None
    try:
        value = float(match.group(0).replace(",", ""))
    except ValueError:
        return None
    return value if value > 0 else None


_ACCESSORY_RE = re.compile(
    r"\b(?:dial|bezel|crown|hands|strap|bracelet|band|buckle|clasp|movement|case ?back|links?)\s+(?:for|only)\b"
    r"|\b(?:box only|papers only|empty box|for parts|parts only|replica|homage watch)\b",
    re.IGNORECASE,
)


def is_accessory(title: str) -> bool:
    """True for listings that sell a part, accessory or copy rather than a complete watch."""
    return bool(_ACCESSORY_RE.search(title or ""))


# Pocket watches, clocks and pendants share brand names and references with wristwatches,
# but they are a different market: a 1900 Taschenuhr is not a comparable for a Nautilus.
# ("chain" is deliberately absent: chain-link bracelets are wristwatches.)
_NOT_WRIST_RE = re.compile(
    r"\b(?:poc?ket\s?watch\w*|poc?ket|taschenuhr\w*|frackuhr|gousset|savonn?ette|l[eé]pine?|hunter\s?case|pendant|pendentif"
    r"|pendulette|clock\w*|horloge|tischuhr|wanduhr|uhrkette|marine\s?chronometer|deck\s?watch|bolso|brooch)\b",
    re.IGNORECASE,
)


def is_not_wristwatch(title: str) -> bool:
    """True for pocket watches, clocks, pendants and deck watches."""
    return bool(_NOT_WRIST_RE.search(title or ""))


def excluded_listing(title: str) -> bool:
    """Listings kept out of the wristwatch market: accessories, copies and non-wrist pieces."""
    return is_accessory(title) or is_not_wristwatch(title)


def seller_pseudonym(username: str | None, salt: str) -> str | None:
    """One-way, salted seller id. The username itself is never stored."""
    if not username:
        return None
    digest = hmac.new(salt.encode("utf-8"), username.strip().casefold().encode("utf-8"), hashlib.sha256)
    return digest.hexdigest()[:16]


def price_stats(prices: list[float]) -> dict | None:
    """Count, min, median and max of a price list (None when empty)."""
    values = sorted(p for p in prices if p is not None and p > 0)
    if not values:
        return None
    n = len(values)
    mid = n // 2
    median = values[mid] if n % 2 else (values[mid - 1] + values[mid]) / 2
    return {"count": n, "min": values[0], "median": median, "max": values[-1]}
