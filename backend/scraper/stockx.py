"""StockX Playwright scraper (search results, top 3)."""

from __future__ import annotations

import asyncio
import logging
import random
from pathlib import Path

from playwright.async_api import TimeoutError as PlaywrightTimeoutError
from playwright.async_api import async_playwright

_log = logging.getLogger(__name__)

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:123.0) Gecko/20100101 Firefox/123.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:122.0) Gecko/20100101 Firefox/122.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Edge/121.0.0.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Safari/605.1.15",
    "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:122.0) Gecko/20100101 Firefox/122.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36 Edg/119.0.0.0",
    "Mozilla/5.0 (iPhone; CPU iPhone OS 17_2 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Mobile/15E148 Safari/604.1",
]


def _parse_price(raw: str) -> float | None:
    cleaned = (raw or "").replace(",", "")
    tokens = cleaned.split()
    for token in tokens:
        tok = token.replace("$", "").strip()
        if "-" in tok or "–" in tok:
            tok = tok.replace("–", "-").split("-")[0]
        try:
            value = float(tok)
            if value > 0:
                return value
        except ValueError:
            continue
    return None


async def scrape_stockx(product_name: str) -> list[dict]:
    """Return up to 3 search rows; empty list on failure."""
    results: list[dict] = []
    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            try:
                context = await browser.new_context(user_agent=random.choice(USER_AGENTS))
                page = await context.new_page()
                await asyncio.sleep(random.uniform(1.0, 3.0))
                url = f"https://stockx.com/search?s={product_name}"
                await page.goto(url, wait_until="domcontentloaded", timeout=60000)
                try:
                    await page.wait_for_selector('[data-testid="product-tile"]', timeout=20000)
                except PlaywrightTimeoutError:
                    await page.screenshot(path=str(Path("/tmp/stockx_fail.png")), full_page=True)

                tiles = await page.query_selector_all('[data-testid="product-tile"]')
                if not tiles:
                    tiles = await page.query_selector_all('[data-component="Tile"]')

                for tile in tiles[:3]:
                    try:
                        await asyncio.sleep(random.uniform(0.2, 0.6))
                        name_el = await tile.query_selector("p")
                        text = await name_el.inner_text() if name_el else ""
                        price_el = await tile.query_selector(
                            '[data-testid="product-tile-price"], span[class*="amount"]'
                        )
                        price_txt = await price_el.inner_text() if price_el else ""
                        price_val = _parse_price(price_txt)
                        results.append(
                            {
                                "product_name": (text or product_name).strip(),
                                "lowest_ask": price_val,
                                "estimated_delivery": "3–7 days",
                                "seller_rating": "N/A",
                            }
                        )
                    except Exception:
                        continue
                return results
            finally:
                await browser.close()
    except Exception as exc:
        _log.exception("scrape_stockx_failed: %s", exc)
        return []
