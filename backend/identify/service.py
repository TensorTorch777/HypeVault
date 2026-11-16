"""Process-wide catalog and gallery for the identify endpoint, loaded on first use."""

from __future__ import annotations

import asyncio
import base64
import io
import logging
import time
from pathlib import Path

from PIL import Image
from sqlalchemy.ext.asyncio import AsyncSession

from config import settings
from identify import recognize
from identify.catalog import Catalog, load_catalog
from identify.embedder import shared_embedder
from identify.gallery import Gallery

_log = logging.getLogger(__name__)
_REPO = Path(__file__).resolve().parents[2]
CATALOG_TTL_SEC = 3600

_catalog: tuple[Catalog, float] | None = None
_gallery: Gallery | None = None
_gallery_checked = False
_lock = asyncio.Lock()


def gallery_path() -> Path:
    p = Path(settings.identify_gallery_path)
    return p if p.is_absolute() else _REPO / p


async def _resources(session: AsyncSession) -> tuple[Catalog, Gallery | None]:
    global _catalog, _gallery, _gallery_checked
    async with _lock:
        if _catalog is None or time.monotonic() - _catalog[1] > CATALOG_TTL_SEC:
            _catalog = (await load_catalog(session), time.monotonic())
        if not _gallery_checked:
            path = gallery_path()
            if path.exists():
                _gallery = await asyncio.to_thread(Gallery.load, path)
                _log.info("identify_gallery_loaded photos=%d built_on=%s", len(_gallery), _gallery.built_on)
            else:
                _log.warning("identify_gallery_missing path=%s; run scripts/build_watch_gallery.py", path)
            _gallery_checked = True
    return _catalog[0], _gallery


EBAY_PHOTO_SIDE = 800  # eBay's visual search does not need more; keeps the request small


async def ebay_photo_search(image: Image.Image) -> list[dict]:
    """eBay visual search over live Wristwatches listings (five brands only), best match first."""
    from market.service import make_ebay_client

    def encode() -> str:
        im = image.convert("RGB")
        im.thumbnail((EBAY_PHOTO_SIDE, EBAY_PHOTO_SIDE))
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=88)
        return base64.b64encode(buf.getvalue()).decode()

    client = make_ebay_client()
    try:
        items = await client.search_by_image(await asyncio.to_thread(encode), limit=50)
    finally:
        await client.aclose()
    return [
        {"title": i.title, "brand": i.brand, "price": i.price, "currency": i.currency,
         "condition": i.condition, "item_url": i.item_url, "image_url": i.image_url}
        for i in items
    ]


async def identify_watch(session: AsyncSession, image: Image.Image | None, text: str | None) -> dict:
    from inference.content_gate import score_watch
    from inference.dial_text import read_dial_lines
    from market.service import ebay_configured

    catalog, gallery = await _resources(session)
    result = await recognize.identify(
        catalog=catalog, gallery=gallery, image=image, text=text,
        embedder=shared_embedder(), read_lines=read_dial_lines, watch_check=score_watch,
        image_search=ebay_photo_search if ebay_configured() else None,
    )
    result["gallery"] = {"ready": gallery is not None, "photos": len(gallery) if gallery else 0,
                         "built_on": gallery.built_on if gallery else None}
    return result


def reset_for_tests() -> None:
    global _catalog, _gallery, _gallery_checked
    _catalog, _gallery, _gallery_checked = None, None, False
