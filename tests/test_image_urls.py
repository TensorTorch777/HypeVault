"""Local-disk image URLs follow the current PUBLIC_API_BASE_URL, not the host stored at upload time."""

from __future__ import annotations

import sys
import unittest
import uuid
from datetime import datetime, timezone
from pathlib import Path
from unittest import mock

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "backend"))

import storage  # noqa: E402
from listings.models import ListingRead  # noqa: E402

NAME = "abc_listings_123_photo.jpg"


class LocalPublicUrlTests(unittest.TestCase):
    def setUp(self) -> None:
        patcher = mock.patch.object(storage.settings, "public_api_base_url", "http://localhost:8000/")
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_stale_tunnel_host_is_replaced(self) -> None:
        stored = f"https://old-tunnel.ngrok-free.dev/static/local/{NAME}"
        self.assertEqual(storage.local_public_url(stored), f"http://localhost:8000/static/local/{NAME}")

    def test_other_and_empty_urls_pass_through(self) -> None:
        other = "https://i.ebayimg.com/images/g/abc/s-l1600.jpg"
        self.assertEqual(storage.local_public_url(other), other)
        self.assertIsNone(storage.local_public_url(None))
        self.assertEqual(storage.local_public_url(""), "")

    def test_listing_response_uses_the_current_host(self) -> None:
        listing = ListingRead(
            id=uuid.uuid4(), seller_id=uuid.uuid4(), product_name="x", category="watch", brand=None,
            condition=None, size=None, s3_url=f"https://old-tunnel.ngrok-free.dev/static/local/{NAME}",
            verdict=None, confidence=None, status="live", created_at=datetime.now(timezone.utc),
        )
        self.assertEqual(listing.s3_url, f"http://localhost:8000/static/local/{NAME}")


if __name__ == "__main__":
    unittest.main()
