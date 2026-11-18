"""Our own crawlers for pre-owned luxury-watch dealers (The 1916 Company and Chronext).

No third-party scraping service. Each site is crawled politely: robots.txt is honoured, requests are
spaced per host, and a bot challenge stops the crawl instead of being worked around.

- ``http``: the polite fetcher.
- ``extract``: site-independent parsing (sitemaps, JSON-LD, OpenGraph, labelled spec rows, prices).
- ``sites``: one class per dealer: how to find product pages and read them.
- ``crawl``: runs a crawl and writes CSV/JSONL and, optionally, ``market_listings`` rows.
"""
