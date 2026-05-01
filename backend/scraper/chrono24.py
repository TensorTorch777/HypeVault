"""Chrono24 Playwright scraper (search results, top 3)."""

from __future__ import annotations

import asyncio
import logging
import random
import re
from pathlib import Path

from playwright.async_api import TimeoutError as PlaywrightTimeoutError
from playwright.async_api import async_playwright

from config import settings
from scraper.stockx import USER_AGENTS

_log = logging.getLogger(__name__)


def _parse_price(raw: str) -> float | None:
    cleaned = (raw or "").replace(",", "")
    for token in cleaned.split():
        tok = token.strip().replace("$", "").replace("€", "").replace("£", "")
        if "-" in tok or "–" in tok:
            tok = tok.replace("–", "-").split("-")[0]
        try:
            value = float(tok)
            if value > 0:
                return value
        except ValueError:
            continue
    return None


def _parse_seller_rating(raw: str) -> str:
    text = raw or ""
    pct = re.search(r"(\d{2,3}(?:\.\d+)?)\s*%", text)
    if pct:
        return f"{pct.group(1)}%"
    star = re.search(r"(\d(?:\.\d+)?)\s*/\s*5", text)
    if star:
        return star.group(1)
    star_simple = re.search(r"(\d(?:\.\d+)?)\s*stars?", text, flags=re.IGNORECASE)
    if star_simple:
        return star_simple.group(1)
    return "N/A"


async def scrape_chrono24(product_name: str) -> list[dict]:
    results: list[dict] = []
    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            try:
                ctx = await browser.new_context(user_agent=random.choice(USER_AGENTS))
                page = await ctx.new_page()
                await asyncio.sleep(random.uniform(1.0, 3.0))
                q = product_name.replace(" ", "+")
                base = settings.chrono24_search_base.rstrip("/")
                await page.goto(
                    f"{base}/search/index.htm?query={q}",
                    wait_until="domcontentloaded",
                    timeout=60000,
                )
                try:
                    await page.wait_for_selector('[data-name="result-item"], article.watch-item', timeout=20000)
                except PlaywrightTimeoutError:
                    await page.screenshot(path=str(Path("/tmp/chrono24_fail.png")), full_page=True)

                items = await page.query_selector_all('[data-name="result-item"]')
                if not items:
                    items = await page.query_selector_all("article.watch-item")

                for item in items[:3]:
                    try:
                        await asyncio.sleep(random.uniform(0.2, 0.6))
                        title_el = await item.query_selector("a, .text-bold")
                        title = await title_el.inner_text() if title_el else product_name
                        price_el = await item.query_selector(".price, [itemprop='price']")
                        price_txt = await price_el.inner_text() if price_el else ""
                        price_val = _parse_price(price_txt)
                        snippet = await item.inner_text()
                        results.append(
                            {
                                "product_name": title.strip(),
                                "lowest_ask": price_val,
                                "estimated_delivery": "5–10 days",
                                "seller_rating": _parse_seller_rating(snippet),
                            }
                        )
                    except Exception:
                        continue
                return results
            finally:
                await browser.close()
    except Exception as exc:
        _log.exception("scrape_chrono24_failed: %s", exc)
        return []
