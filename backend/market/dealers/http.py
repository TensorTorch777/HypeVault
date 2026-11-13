"""Polite HTTP fetching for dealer crawls: robots.txt, per-host spacing, backoff, and stop on bot challenges."""

from __future__ import annotations

import asyncio
import gzip
import logging
import time
import urllib.robotparser
from urllib.parse import urlsplit

import httpx

_log = logging.getLogger(__name__)

DEFAULT_USER_AGENT = (
    "HypeVaultResearchBot/1.0 (KJSSE final-year project; price comparison research; "
    "+https://github.com/TensorTorch777/HypeVault)"
)

# Markers of anti-bot interstitials. Seeing one ends the crawl for that host: we do not try to get past it.
_CHALLENGE_MARKERS = (
    "cf-challenge",
    "challenge-platform",
    "cf_chl_",
    "px-captcha",
    "perimeterx",
    "captcha-delivery",
    "datadome",
    "pardon our interruption",
    "are you a robot",
    "verify you are human",
    "access denied",
    "request unsuccessful. incapsula",
    "awswafintegration",  # AWS WAF JS challenge (e.g. Watchfinder)
    "awswafcookiedomainlist",
    "token.awswaf.com",
    "challenge-container",
)


class FetchError(RuntimeError):
    """The page could not be fetched (network error, server error after retries)."""


class BlockedError(FetchError):
    """The site refused us (403, persistent 429, or a bot challenge). Stop crawling this host."""


class DisallowedError(FetchError):
    """robots.txt disallows this URL for our user agent."""


def looks_like_challenge(html: str) -> bool:
    head = html[:20000].casefold()
    return any(marker in head for marker in _CHALLENGE_MARKERS) and len(html) < 200_000


class PoliteFetcher:
    """Async GET with robots.txt checks, a minimum delay between requests to the same host, and retries."""

    def __init__(
        self,
        *,
        user_agent: str = DEFAULT_USER_AGENT,
        min_delay: float = 3.0,
        timeout: float = 30.0,
        max_retries: int = 3,
        respect_robots: bool = True,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.user_agent = user_agent
        self.min_delay = min_delay
        self.max_retries = max_retries
        self.respect_robots = respect_robots
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(
            timeout=timeout,
            follow_redirects=True,
            headers={
                "User-Agent": user_agent,
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "en-GB,en;q=0.9",
            },
        )
        self._robots: dict[str, urllib.robotparser.RobotFileParser] = {}
        self._robots_sitemaps: dict[str, list[str]] = {}
        self._last: dict[str, float] = {}
        self._locks: dict[str, asyncio.Lock] = {}
        self._blocked: set[str] = set()
        self.requests = 0

    async def __aenter__(self) -> "PoliteFetcher":
        return self

    async def __aexit__(self, *exc) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    # ── robots.txt ───────────────────────────────────────────────────────────

    async def _load_robots(self, origin: str) -> urllib.robotparser.RobotFileParser:
        if origin in self._robots:
            return self._robots[origin]
        parser = urllib.robotparser.RobotFileParser()
        lines: list[str] = []
        try:
            resp = await self._request(f"{origin}/robots.txt", check_robots=False)
            if resp.status_code == 200:
                lines = resp.text.splitlines()
            elif resp.status_code in (401, 403):
                lines = ["User-agent: *", "Disallow: /"]  # RFC 9309: unreachable robots.txt by auth = disallow all
        except FetchError as exc:
            _log.warning("robots.txt unavailable for %s: %s", origin, exc)
        parser.parse(lines)
        self._robots[origin] = parser
        self._robots_sitemaps[origin] = [
            line.split(":", 1)[1].strip() for line in lines if line.lower().startswith("sitemap:")
        ]
        delay = parser.crawl_delay(self.user_agent)
        if delay and float(delay) > self.min_delay:
            _log.info("%s asks for crawl-delay %ss; using it", origin, delay)
            self._host_delay(origin, float(delay))
        return parser

    def _host_delay(self, origin: str, delay: float) -> None:
        self._last.setdefault(f"delay:{origin}", delay)

    async def allowed(self, url: str) -> bool:
        if not self.respect_robots:
            return True
        origin = _origin(url)
        return (await self._load_robots(origin)).can_fetch(self.user_agent, url)

    async def sitemaps(self, origin: str) -> list[str]:
        await self._load_robots(origin)
        return list(self._robots_sitemaps.get(origin, []))

    # ── fetching ─────────────────────────────────────────────────────────────

    async def _wait_turn(self, origin: str) -> None:
        lock = self._locks.setdefault(origin, asyncio.Lock())
        async with lock:
            delay = self._last.get(f"delay:{origin}", self.min_delay)
            wait = self._last.get(origin, 0.0) + delay - time.monotonic()
            if wait > 0:
                await asyncio.sleep(wait)
            self._last[origin] = time.monotonic()

    async def _request(self, url: str, *, check_robots: bool = True) -> httpx.Response:
        origin = _origin(url)
        if origin in self._blocked:
            raise BlockedError(f"{origin} blocked this crawler earlier in the run")
        if check_robots and not await self.allowed(url):
            raise DisallowedError(f"robots.txt disallows {url}")
        last_exc: Exception | None = None
        for attempt in range(self.max_retries + 1):
            await self._wait_turn(origin)
            try:
                resp = await self._client.get(url)
            except httpx.HTTPError as exc:
                last_exc = exc
                await asyncio.sleep(2 ** attempt)
                continue
            self.requests += 1
            if resp.status_code == 429 or resp.status_code >= 500:
                retry_after = resp.headers.get("Retry-After", "")
                pause = float(retry_after) if retry_after.isdigit() else 5.0 * 2 ** attempt
                if attempt == self.max_retries:
                    if resp.status_code == 429:
                        self._blocked.add(origin)
                        raise BlockedError(f"{origin} kept answering 429 (rate limited)")
                    raise FetchError(f"{url}: HTTP {resp.status_code}")
                _log.info("%s: HTTP %s, waiting %.0fs", url, resp.status_code, pause)
                await asyncio.sleep(min(pause, 120))
                continue
            return resp
        raise FetchError(f"{url}: {last_exc}")

    async def get_text(self, url: str, *, allow_missing: bool = True) -> str | None:
        """Body as text; None for 404/410 when allow_missing. Raises BlockedError on 403 or a challenge page."""
        resp = await self._request(url)
        origin = _origin(url)
        if resp.status_code in (404, 410) and allow_missing:
            return None
        if resp.status_code in (401, 403):
            self._blocked.add(origin)
            raise BlockedError(f"{origin} answered {resp.status_code} for {url}")
        if resp.status_code >= 400:
            raise FetchError(f"{url}: HTTP {resp.status_code}")
        body = resp.content
        if body[:2] == b"\x1f\x8b":  # .xml.gz sitemaps
            body = gzip.decompress(body)
        text = body.decode(resp.encoding or "utf-8", errors="replace")
        if looks_like_challenge(text):
            self._blocked.add(origin)
            raise BlockedError(f"{origin} served a bot challenge for {url}; stopping (not bypassing it)")
        return text


def _origin(url: str) -> str:
    parts = urlsplit(url)
    return f"{parts.scheme}://{parts.netloc}"
