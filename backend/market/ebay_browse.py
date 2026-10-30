"""eBay Browse API client (official API; eBay's user agreement does not allow HTML scraping).

Uses the client-credentials OAuth flow, which needs only an application key pair from
developer.ebay.com. Searches the Wristwatches category (31387) and returns normalised items.
"""

from __future__ import annotations

import base64
import logging
import time
from datetime import datetime, timezone
from typing import Awaitable, Callable

import httpx

from market.types import SOURCE_EBAY, MarketItem, canonical_brand, excluded_listing, parse_price, seller_pseudonym

_log = logging.getLogger(__name__)

WRISTWATCHES_CATEGORY_ID = "31387"
OAUTH_SCOPE = "https://api.ebay.com/oauth/api_scope"
_HOSTS = {"production": "https://api.ebay.com", "sandbox": "https://api.sandbox.ebay.com"}

TokenGet = Callable[[], Awaitable[str | None]]
TokenSet = Callable[[str, int], Awaitable[None]]


class EbayNotConfigured(RuntimeError):
    """Raised when no application key pair is configured."""


class EbayApiError(RuntimeError):
    def __init__(self, status: int, message: str) -> None:
        super().__init__(f"eBay API {status}: {message}")
        self.status = status


class EbayBrowseClient:
    def __init__(
        self,
        client_id: str,
        client_secret: str,
        *,
        environment: str = "production",
        marketplace_id: str = "EBAY_US",
        pseudonym_salt: str = "change-me",
        http: httpx.AsyncClient | None = None,
        token_get: TokenGet | None = None,
        token_set: TokenSet | None = None,
    ) -> None:
        if not client_id or not client_secret:
            raise EbayNotConfigured("EBAY_CLIENT_ID and EBAY_CLIENT_SECRET are not set")
        if environment not in _HOSTS:
            raise ValueError(f"unknown eBay environment {environment!r}")
        self._client_id = client_id
        self._client_secret = client_secret
        self.environment = environment
        self.marketplace_id = marketplace_id
        self._salt = pseudonym_salt
        self._base = _HOSTS[environment]
        self._http = http or httpx.AsyncClient(timeout=20.0)
        self._owns_http = http is None
        self._token_get = token_get
        self._token_set = token_set
        self._token: tuple[str, float] | None = None

    async def aclose(self) -> None:
        if self._owns_http:
            await self._http.aclose()

    async def app_token(self) -> str:
        now = time.time()
        if self._token and self._token[1] > now + 60:
            return self._token[0]
        if self._token_get:
            cached = await self._token_get()
            if cached:
                self._token = (cached, now + 300)
                return cached
        basic = base64.b64encode(f"{self._client_id}:{self._client_secret}".encode()).decode()
        resp = await self._http.post(
            f"{self._base}/identity/v1/oauth2/token",
            headers={"Authorization": f"Basic {basic}", "Content-Type": "application/x-www-form-urlencoded"},
            data={"grant_type": "client_credentials", "scope": OAUTH_SCOPE},
        )
        if resp.status_code != 200:
            raise EbayApiError(resp.status_code, "token request was refused")
        body = resp.json()
        token, ttl = body["access_token"], int(body.get("expires_in", 7200))
        self._token = (token, now + ttl)
        if self._token_set:
            await self._token_set(token, max(60, ttl - 120))
        return token

    async def search(
        self,
        query: str,
        *,
        brand: str | None = None,
        limit: int = 50,
        offset: int = 0,
        min_price: float | None = None,
    ) -> tuple[list[MarketItem], int]:
        """Return (items, total matches reported by eBay). Items outside the five brands are dropped."""
        token = await self.app_token()
        params: dict[str, str | int] = {
            "q": query,
            "category_ids": WRISTWATCHES_CATEGORY_ID,
            "limit": max(1, min(limit, 200)),
            "offset": max(0, offset),
            "filter": "buyingOptions:{FIXED_PRICE}",
        }
        if min_price:
            params["filter"] = f"buyingOptions:{{FIXED_PRICE}},price:[{min_price:.0f}..],priceCurrency:USD"
        if brand:
            params["aspect_filter"] = f"categoryId:{WRISTWATCHES_CATEGORY_ID},Brand:{{{brand}}}"
        resp = await self._http.get(
            f"{self._base}/buy/browse/v1/item_summary/search",
            params=params,
            headers={"Authorization": f"Bearer {token}", "X-EBAY-C-MARKETPLACE-ID": self.marketplace_id},
        )
        if resp.status_code == 401:
            self._token = None
            raise EbayApiError(401, "token was rejected")
        if resp.status_code != 200:
            raise EbayApiError(resp.status_code, "search request failed")
        body = resp.json()
        observed_at = datetime.now(timezone.utc)
        items = [
            item
            for raw in body.get("itemSummaries") or []
            if (item := self._to_item(raw, brand, observed_at)) is not None
        ]
        return items, int(body.get("total") or 0)

    async def search_by_image(self, image_b64: str, *, limit: int = 50) -> list[MarketItem]:
        """eBay's own visual search over live Wristwatches listings, best match first.

        Same filters as `search`: listings outside the five brands, parts and copies are dropped.
        """
        token = await self.app_token()
        resp = await self._http.post(
            f"{self._base}/buy/browse/v1/item_summary/search_by_image",
            params={"category_ids": WRISTWATCHES_CATEGORY_ID, "limit": max(1, min(limit, 200))},
            headers={"Authorization": f"Bearer {token}", "X-EBAY-C-MARKETPLACE-ID": self.marketplace_id},
            json={"image": image_b64},
        )
        if resp.status_code == 401:
            self._token = None
            raise EbayApiError(401, "token was rejected")
        if resp.status_code != 200:
            raise EbayApiError(resp.status_code, "image search request failed")
        observed_at = datetime.now(timezone.utc)
        return [
            item
            for raw in resp.json().get("itemSummaries") or []
            if (item := self._to_item(raw, None, observed_at)) is not None
        ]

    def _to_item(self, raw: dict, requested_brand: str | None, observed_at: datetime) -> MarketItem | None:
        price_obj = raw.get("price") or {}
        price = parse_price(price_obj.get("value"))
        title = (raw.get("title") or "").strip()
        requested = canonical_brand(requested_brand)
        named = _brand_from_title(title)
        # Trust the title over the request: a title naming a different (or no supported) brand is dropped
        # unless the title names no brand at all and the search was already brand-filtered.
        if named and requested and named != requested:
            return None
        brand = named or (requested if requested and not _names_other_brand(title) else None)
        if price is None or not title or brand is None or not raw.get("itemId") or excluded_listing(title):
            return None
        seller = raw.get("seller") or {}
        feedback = seller.get("feedbackPercentage")
        return MarketItem(
            source=SOURCE_EBAY,
            external_id=str(raw["itemId"])[:128],
            brand=brand,
            title=title[:512],
            price=price,
            currency=(price_obj.get("currency") or "USD")[:8],
            observed_on=observed_at.date(),
            observed_at=observed_at,
            condition=(raw.get("condition") or None),
            seller_ref=seller_pseudonym(seller.get("username"), self._salt),
            seller_feedback_pct=float(feedback) if feedback not in (None, "") else None,
            item_url=raw.get("itemWebUrl"),
            image_url=(raw.get("image") or {}).get("imageUrl"),
            provenance={
                "api": "buy/browse/v1/item_summary/search",
                "environment": self.environment,
                "marketplace": self.marketplace_id,
                "category_id": WRISTWATCHES_CATEGORY_ID,
                "image_rights": "seller-owned listing image; evaluation use only, not redistributed",
            },
        )


_OTHER_BRANDS = ("rolex", "omega", "cartier", "breitling", "tag heuer", "hublot", "iwc", "panerai", "tudor",
                 "jaeger", "longines", "seiko", "grand seiko", "zenith", "breguet", "blancpain", "chopard")


def _names_other_brand(title: str) -> bool:
    lowered = title.casefold()
    return any(name in lowered for name in _OTHER_BRANDS)


def _brand_from_title(title: str) -> str | None:
    lowered = title.casefold()
    for name, key in (
        ("A. Lange & Söhne", "lange"),
        ("Audemars Piguet", "audemars"),
        ("Patek Philippe", "patek"),
        ("Richard Mille", "richard mille"),
        ("Vacheron Constantin", "vacheron"),
    ):
        if key in lowered:
            return name
    return None
