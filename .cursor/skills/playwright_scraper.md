# Playwright Scraper Skill

When building scraper code:

- Always use async Playwright
- Rotate user agents from predefined list
- Add random delay 1-3 seconds between actions
- Screenshot on failure for debugging
- Never hardcode selectors — use data attributes or aria
- Always handle StaleElementException
- Return empty list on failure, never raise exception to caller
- Cache results in Redis before returning
