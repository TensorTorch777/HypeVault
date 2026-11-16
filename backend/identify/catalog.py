"""Model names and reference numbers known for the five brands, and matching them in free text.

The catalog is built from the Chrono24 dataset rows (brand, model, reference). It is the
dictionary that turns "5711/1A" or "royal oak offshore" into a canonical model name. Free text
can be an eBay title, words read off a dial, or what the user typed.
"""

from __future__ import annotations

import re
import sys
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

_ML = Path(__file__).resolve().parents[2] / "ml_rtx5080"
if str(_ML) not in sys.path:
    sys.path.insert(0, str(_ML))

from watch_forensics import BRAND_WORDS  # noqa: E402

# Dataset "models" that say nothing about which watch it is.
GENERIC_MODELS = {"vintage", "unknown", "other", "ladies", "lady"}
# A reference key must hold at least this share of its rows for one model to name that model.
REFERENCE_MAJORITY = 0.6

_REF_TOKEN_RE = re.compile(r"\b(?:RM\s?)?[A-Z]{0,2}\d{3,6}[A-Z0-9]*(?:[./-][A-Z0-9]+)*")


def normalize(text: str) -> str:
    """ASCII, lower case, words split at letter/digit boundaries, leading zeros dropped from numbers.

    "RM011" and "RM 011" both become "rm 11"; "Twenty~4" becomes "twenty 4".
    """
    plain = unicodedata.normalize("NFKD", text or "").encode("ascii", "ignore").decode().casefold()
    plain = re.sub(r"(?<=[a-z])(?=\d)|(?<=\d)(?=[a-z])", " ", plain)
    tokens = re.findall(r"[a-z0-9]+", plain)
    return " ".join(t.lstrip("0") or "0" if t.isdigit() else t for t in tokens)


def reference_keys(reference: str) -> list[str]:
    """Lookup keys for a reference: the full alphanumeric form, then its family.

    "5711/1A-010" -> ["57111A010", "5711"]; "26320ST.OO.1220ST.03" -> ["26320STOO1220ST03", "26320"];
    "109.032" -> ["109032"] (Lange writes the family as both groups).
    """
    raw = unicodedata.normalize("NFKD", reference or "").encode("ascii", "ignore").decode().upper().strip()
    full = re.sub(r"[^A-Z0-9]", "", raw)
    if len(full) < 3:
        return []
    keys = [full]
    lange = re.match(r"^(\d{3})\.(\d{3})", raw)
    head = re.split(r"[/\-.\s]", raw, maxsplit=1)[0]
    family = lange.group(1) + lange.group(2) if lange else (re.match(r"^(?:RM)?\d{3,6}", head) or [None])[0]
    if family and family != full:
        keys.append(family)
    return keys


@dataclass(frozen=True)
class Name:
    brand: str
    model: str | None = None
    reference: str | None = None

    @property
    def query(self) -> str:
        """Search text without the brand (the brand is sent as a filter)."""
        return " ".join(p for p in (self.model, self.reference) if p) or self.brand

    @property
    def short_query(self) -> str:
        return self.model or self.brand

    @property
    def label(self) -> str:
        return " ".join(p for p in (self.brand, self.model, self.reference) if p)


@dataclass
class TextMatch:
    brands: set[str] = field(default_factory=set)
    names: list[Name] = field(default_factory=list)  # most specific first


class Catalog:
    def __init__(self, rows: Iterable[tuple[str, str | None, str | None]]) -> None:
        """rows: (brand, model, reference) triples, one per listing (repeats count as votes)."""
        self.models: dict[str, list[tuple[str, str]]] = defaultdict(list)  # brand -> [(normalized, model)]
        ref_votes: dict[tuple[str, str], Counter] = defaultdict(Counter)
        ref_display: dict[tuple[str, str], str] = {}
        seen: set[tuple[str, str]] = set()
        for brand, model, reference in rows:
            model = (model or "").strip() or None
            if model and normalize(model) in GENERIC_MODELS:
                model = None
            if model and (brand, model) not in seen:
                seen.add((brand, model))
                self.models[brand].append((normalize(model), model))
            if reference:
                for i, key in enumerate(reference_keys(reference)):
                    ref_votes[(brand, key)][model] += 1
                    # The family key displays as the family; the full key as the reference itself.
                    ref_display.setdefault((brand, key), reference.strip() if i == 0 else key)
        for brand in self.models:
            self.models[brand].sort(key=lambda m: len(m[0]), reverse=True)  # longest name wins
        self.references: dict[tuple[str, str], tuple[str | None, str]] = {}
        for (brand, key), votes in ref_votes.items():
            named = Counter({m: n for m, n in votes.items() if m})
            model = None
            if named:
                top, n = named.most_common(1)[0]
                if n / sum(named.values()) >= REFERENCE_MAJORITY:
                    model = top
            self.references[(brand, key)] = (model, ref_display[(brand, key)])

    @property
    def brands(self) -> list[str]:
        return sorted(BRAND_WORDS)

    def model_count(self) -> int:
        return sum(len(v) for v in self.models.values())

    def brands_in(self, text: str) -> set[str]:
        norm = f" {normalize(text)} "
        return {b for b, words in BRAND_WORDS.items() if any(f" {w.casefold()} " in norm for w in words)}

    def model_in(self, text: str, brand: str) -> str | None:
        norm = f" {normalize(text)} "
        for key, model in self.models.get(brand, ()):
            if f" {key} " in norm:
                return model
        return None

    def references_in(self, text: str, brands: Iterable[str]) -> list[Name]:
        """Known references written in the text, most specific (full) key first."""
        found: list[Name] = []
        upper = unicodedata.normalize("NFKD", text or "").encode("ascii", "ignore").decode().upper()
        for token in _REF_TOKEN_RE.findall(upper):
            for brand in brands:
                for key in reference_keys(token):  # full key first, then the family
                    hit = self.references.get((brand, key))
                    if hit:
                        model, display = hit
                        name = Name(brand, model, display)
                        if name not in found:
                            found.append(name)
                        break
        return found

    def match(self, text: str, brand_hint: str | None = None) -> TextMatch:
        """Brands, models and references named in the text."""
        out = TextMatch(brands=self.brands_in(text))
        brands = [brand_hint] if brand_hint else (sorted(out.brands) or self.brands)
        for name in self.references_in(text, brands):
            out.names.append(name)
            out.brands.add(name.brand)
        for brand in brands:
            model = self.model_in(text, brand)
            if model:
                # A model name alone is only trusted when no brand is named or this brand is.
                if not out.brands or brand in out.brands or brand_hint:
                    if all(n.model != model or n.brand != brand for n in out.names):
                        out.names.append(Name(brand, model))
                    out.brands.add(brand)
        return out

    def label_title(self, title: str, brand: str) -> Name:
        """Best (brand, model, reference) for a listing title whose brand is already known."""
        model = self.model_in(title, brand)
        refs = self.references_in(title, [brand])
        ref = refs[0] if refs else None
        # The reference may name a more specific model ("Royal Oak" + 26120ST -> "Royal Oak Dual Time").
        if ref is not None and ref.model and (model is None or normalize(ref.model).startswith(normalize(model))):
            model = ref.model
        return Name(brand, model, ref.reference if ref and (ref.model in (None, model)) else None)


async def load_catalog(session) -> Catalog:
    from sqlalchemy import select

    from database import MarketListing
    from market.types import SOURCE_CHRONO24_DATASET

    q = select(MarketListing.brand, MarketListing.model, MarketListing.reference).where(
        MarketListing.source == SOURCE_CHRONO24_DATASET
    )
    return Catalog((await session.execute(q)).all())
