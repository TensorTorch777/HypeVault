"""eBay Playwright scraper (search results, top 3)."""

from __future__ import annotations

import asyncio
import logging
import random
import re
from pathlib import Path
from urllib.parse import quote_plus

from playwright.async_api import TimeoutError as PlaywrightTimeoutError
from playwright.async_api import async_playwright

from scraper.stockx import USER_AGENTS

_log = logging.getLogger(__name__)


def _parse_price(raw: str) -> float | None:
    cleaned = (raw or "").replace(",", "")
    for token in cleaned.split():
        tok = token.strip().replace("$", "")
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
    pct = re.search(r"(\d{2,3}(?:\.\d+)?)\s*%(\s*positive feedback)?", text, flags=re.IGNORECASE)
    if pct:
        return f"{pct.group(1)}%"
    return "N/A"


async def scrape_ebay(product_name: str) -> list[dict]:
    results: list[dict] = []
    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            try:
                ctx = await browser.new_context(user_agent=random.choice(USER_AGENTS))
                page = await ctx.new_page()
                await asyncio.sleep(random.uniform(1.0, 3.0))
                q = quote_plus(product_name)
                await page.goto(
                    f"https://www.ebay.com/sch/i.html?_nkw={q}",
                    wait_until="domcontentloaded",
                    timeout=60000,
                )
                try:
                    await page.wait_for_selector(".s-item__title, [role='listitem']", timeout=20000)
                except PlaywrightTimeoutError:
                    await page.screenshot(path=str(Path("/tmp/ebay_fail.png")), full_page=True)

                rows = await page.query_selector_all("li.s-item")
                if not rows:
                    rows = await page.query_selector_all('[role="listitem"]')

                for row in rows[:5]:
                    try:
                        await asyncio.sleep(random.uniform(0.2, 0.6))
                        title_el = await row.query_selector(".s-item__title, [role=heading]")
                        title = await title_el.inner_text() if title_el else ""
                        if not title or "shop on ebay" in title.lower():
                            continue
                        price_el = await row.query_selector(".s-item__price")
                        price_txt = await price_el.inner_text() if price_el else ""
                        price_val = _parse_price(price_txt)
                        snippet = await row.inner_text()
                        results.append(
                            {
                                "product_name": title.strip(),
                                "lowest_ask": price_val,
                                "estimated_delivery": "4–8 days",
                                "seller_rating": _parse_seller_rating(snippet),
                            }
                        )
                        if len(results) >= 3:
                            break
                    except Exception:
                        continue
                return results
            finally:
                await browser.close()
    except Exception as exc:
        _log.exception("scrape_ebay_failed: %s", exc)
        return []
