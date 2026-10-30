"""Combine the clues into ranked name guesses.

Four clues, each optional:
  1. Typed text: what the user wrote ("nautilus", "5711", "royal oak blue").
  2. Dial text: words OCR reads on the photo ("PATEK PHILIPPE", "GENEVE").
  3. Look-alikes: the most similar eBay listing photos in the gallery, and the names in their titles.
  4. eBay photo search: eBay's own visual search over all live listings, and the names in their titles.

Each guess gets points from the clues that support it. `confidence` is the guess's share of
all points, so the three guesses add up to at most 1. It says how strongly the clues agree,
not a measured probability of being right.
"""

from __future__ import annotations

import asyncio
import logging
import math
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field

from PIL import Image

from identify.catalog import Catalog, Name, TextMatch
from identify.gallery import Gallery, Neighbour
from watch_forensics import OTHER_BRAND_WORDS, words  # path set up by identify.catalog

_log = logging.getLogger(__name__)

K = 15
TEMPERATURE = 0.03  # softmax temperature over cosine similarity; close look-alikes count much more
MIN_SIMILARITY = 0.55  # below this the closest look-alike is not similar enough to name a watch alone
TOP = 3
SURE = 0.25  # a top guess with less than this share of the points is reported as "unsure"

# Points per clue. A reference number is the most specific clue, then a model name, then a brand.
TYPED = {"reference": 3.0, "model": 2.0, "brand": 0.75}
DIAL = {"reference": 2.0, "model": 1.25, "brand": 0.75}
LOOKALIKE_MODEL = 1.5
LOOKALIKE_BRAND = 0.5
EBAY_PHOTO_MODEL = 1.5  # brand votes from eBay matches share LOOKALIKE_BRAND with the gallery look-alikes
EBAY_PHOTO_TOP = 20  # eBay matches that vote; further down the list they rarely show the same watch


@dataclass
class Guess:
    brand: str
    model: str | None
    points: float = 0.0
    references: Counter = field(default_factory=Counter)
    evidence: list[str] = field(default_factory=list)


def _weights(neighbours: list[Neighbour]) -> list[float]:
    if not neighbours:
        return []
    top = neighbours[0].similarity
    raw = [math.exp((n.similarity - top) / TEMPERATURE) for n in neighbours]
    total = sum(raw)
    return [r / total for r in raw]


def _rank_weights(n: int) -> list[float]:
    """Best match counts most: 1, 1/2, 1/3 ... normalised to sum to 1."""
    raw = [1.0 / (i + 1) for i in range(n)]
    total = sum(raw) or 1.0
    return [r / total for r in raw]


def rank(
    neighbours: list[Neighbour],
    dial: TextMatch | None = None,
    typed: TextMatch | None = None,
    ebay_photo: list[Name] | None = None,
) -> list[dict]:
    """Ranked guesses, best first, at most TOP."""
    guesses: dict[tuple[str, str | None], Guess] = {}

    def guess(brand: str, model: str | None) -> Guess:
        return guesses.setdefault((brand, model), Guess(brand, model))

    # Look-alikes: weighted votes for each model and each brand.
    model_share: dict[tuple[str, str], float] = defaultdict(float)
    model_count: Counter = Counter()
    brand_share: dict[str, float] = defaultdict(float)
    for n, w in zip(neighbours, _weights(neighbours)):
        brand_share[n.brand] += w
        if n.model:
            model_share[(n.brand, n.model)] += w
            model_count[(n.brand, n.model)] += 1
            if n.reference:
                guess(n.brand, n.model).references[n.reference] += 1
    for (brand, model), share in model_share.items():
        g = guess(brand, model)
        g.points += LOOKALIKE_MODEL * share
        g.evidence.append(f"{model_count[(brand, model)]} of the {len(neighbours)} closest eBay photos show the {brand} {model}")

    # eBay photo search: the names in the titles of eBay's best visual matches, best first.
    photo_names = (ebay_photo or [])[:EBAY_PHOTO_TOP]
    photo_model: dict[tuple[str, str], float] = defaultdict(float)
    photo_count: Counter = Counter()
    for name, w in zip(photo_names, _rank_weights(len(photo_names))):
        brand_share[name.brand] += w
        if name.model:
            photo_model[(name.brand, name.model)] += w
            photo_count[(name.brand, name.model)] += 1
            if name.reference:
                guess(name.brand, name.model).references[name.reference] += 1
    for (brand, model), share in photo_model.items():
        g = guess(brand, model)
        g.points += EBAY_PHOTO_MODEL * share
        g.evidence.append(f"{photo_count[(brand, model)]} of eBay's top {len(photo_names)} photo matches show the {brand} {model}")

    def text_clue(match: TextMatch | None, points: dict, said: str) -> None:
        if match is None:
            return
        for name in match.names:
            g = guess(name.brand, name.model)
            if name.reference:
                g.points += points["reference"]
                g.references[name.reference] += 100  # a written reference beats look-alike votes
                g.evidence.append(f"{said} reference {name.reference}")
            elif name.model:
                g.points += points["model"]
                g.evidence.append(f"{said} “{name.model}”")
        for brand in match.brands:
            if not any(b == brand for b, _ in guesses):
                guess(brand, None)  # a brand-only guess when nothing more specific is known
            for (b, _m), g in guesses.items():
                if b == brand:
                    g.points += points["brand"]
            guesses[next(k for k in guesses if k[0] == brand)].evidence.append(f"{said} {brand}")

    text_clue(dial, DIAL, "The dial reads")
    text_clue(typed, TYPED, "You typed")

    # Brand agreement from the look-alikes lifts every guess of that brand.
    for (brand, _m), g in guesses.items():
        g.points += LOOKALIKE_BRAND * brand_share.get(brand, 0.0)
    if brand_share and not any(m for _b, m in guesses):
        top_brand = max(brand_share, key=brand_share.get)
        g = guess(top_brand, None)
        g.points += LOOKALIKE_BRAND * brand_share[top_brand]
        g.evidence.append(f"The most similar eBay photos are mostly {top_brand}")

    # A model guess makes the brand-only guess of the same brand redundant.
    for brand in {b for b, m in guesses if m}:
        brand_only = guesses.pop((brand, None), None)
        if brand_only:
            for g in (g for (b, m), g in guesses.items() if b == brand and m):
                g.evidence += [e for e in brand_only.evidence if e not in g.evidence]

    # A brand read on the dial or typed by the user rules out guesses of other brands.
    named = set().union(*(m.brands for m in (dial, typed) if m is not None))
    if named:
        guesses = {k: g for k, g in guesses.items() if k[0] in named}

    total = sum(g.points for g in guesses.values()) or 1.0
    ordered = sorted(guesses.values(), key=lambda g: g.points, reverse=True)[:TOP]
    out = []
    for g in ordered:
        reference = None
        if g.references:
            ref, votes = g.references.most_common(1)[0]
            if votes >= 2:  # one look-alike's reference is not enough
                reference = ref
        name = Name(g.brand, g.model, reference)
        out.append({
            "brand": g.brand, "model": g.model, "reference": reference,
            "label": name.label, "query": name.query, "short_query": name.short_query,
            "confidence": round(g.points / total, 3),
            "evidence": g.evidence[:4],
        })
    return out


async def identify(
    *,
    catalog: Catalog,
    gallery: Gallery | None,
    image: Image.Image | None = None,
    text: str | None = None,
    embedder=None,
    read_lines=None,
    watch_check=None,
    image_search=None,
) -> dict:
    """Name guesses from a photo, typed text, or both. See module docstring."""
    started = time.perf_counter()
    text = " ".join((text or "").split())[:200] or None
    typed = catalog.match(text) if text else None
    dial = None
    neighbours: list[Neighbour] = []
    dial_lines: list[str] = []
    ebay_matches: list[dict] | None = None
    if image is not None:
        if watch_check is not None:
            check = await asyncio.to_thread(watch_check, image)
            if not check.is_watch:
                return {"status": "not_a_watch", "looks_like": check.looks_like, "candidates": [],
                        "dial_text": [], "lookalikes": [], "elapsed_ms": _ms(started)}
        # eBay's visual search runs while the dial is read and the gallery is searched.
        searching = asyncio.create_task(_photo_search(image_search, image)) if image_search is not None else None
        if read_lines is not None:
            dial_lines = [t for t, _s in await asyncio.to_thread(read_lines, image)]
            dial = catalog.match(" ".join(dial_lines)) if dial_lines else None
        if gallery is not None and len(gallery) and embedder is not None:
            neighbours = gallery.neighbours(await embedder.embed_one(image), K)
        if searching is not None:
            ebay_matches = await searching

    close = bool(neighbours) and neighbours[0].similarity >= MIN_SIMILARITY
    has_text_clue = any(m and (m.names or m.brands) for m in (dial, typed))
    other = None if has_text_clue else other_brand_word(" ".join(filter(None, [text, *dial_lines])))
    if other:
        # "ROLEX" on the dial or "omega seamaster" typed, and none of the five brands named:
        # look-alikes would only find the nearest of our brands, which would be wrong.
        return {"status": "unsupported_brand", "other_brand": other, "candidates": [], "dial_text": dial_lines[:12],
                "lookalikes": [], "used": {"photo": image is not None, "text": bool(text), "dial_text": False,
                                           "lookalikes": False}, "elapsed_ms": _ms(started)}
    photo_names = [catalog.label_title(m["title"], m["brand"]) for m in ebay_matches or []]
    candidates = rank(neighbours if close else [], dial, typed, photo_names)
    if candidates:
        status = "recognised" if candidates[0]["confidence"] >= SURE else "unsure"
    elif text:
        status = "free_text"  # nothing in our catalog, but the user's words can still be searched
        candidates = [{"brand": None, "model": None, "reference": None, "label": text, "query": text,
                       "short_query": text, "confidence": None, "evidence": ["Searched as you typed it"]}]
    else:
        status = "not_recognised"
    result = {
        "status": status,
        "candidates": candidates,
        "dial_text": dial_lines[:12],
        "lookalikes": [
            {"title": n.title, "similarity": round(n.similarity, 3), "item_url": n.item_url}
            for n in neighbours[:5]
        ],
        "ebay_photo_matches": (ebay_matches or [])[:8],
        "used": {"photo": image is not None, "text": bool(text), "dial_text": bool(has_text_clue and dial),
                 "lookalikes": close, "ebay_photo_search": bool(ebay_matches)},
        "elapsed_ms": _ms(started),
    }
    _log.info("identify status=%s top=%s elapsed_ms=%s", status,
              candidates[0]["label"] if candidates else None, result["elapsed_ms"])
    return result


async def _photo_search(image_search, image: Image.Image) -> list[dict] | None:
    """eBay visual matches (five brands only, best first), or None when the search fails."""
    started = time.perf_counter()
    try:
        matches = await image_search(image)
    except Exception as exc:  # a failed eBay call must not stop identification
        _log.warning("identify_ebay_photo_search_failed: %s", exc)
        return None
    _log.info("identify_ebay_photo_search matches=%d ms=%.0f", len(matches), (time.perf_counter() - started) * 1000)
    return matches


def other_brand_word(text: str) -> str | None:
    """A brand or model word from outside the five brands ("ROLEX", "SEAMASTER"), if any."""
    found = set(words(text))
    return next((w.title() for w in OTHER_BRAND_WORDS if w in found), None)


def _ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000, 1)
