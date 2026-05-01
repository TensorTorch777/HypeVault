"""Scraper readiness checks used by health and compare routes."""

from __future__ import annotations

from pathlib import Path


def check_scraper_readiness() -> tuple[bool, str]:
    try:
        from playwright.async_api import async_playwright  # noqa: F401
        from playwright.sync_api import sync_playwright
    except Exception:
        return False, "playwright_not_installed"

    try:
        with sync_playwright() as p:
            executable = Path(p.chromium.executable_path)
            if executable.exists():
                return True, "ok"
            return False, "chromium_binary_missing"
    except Exception:
        return False, "playwright_runtime_unavailable"
