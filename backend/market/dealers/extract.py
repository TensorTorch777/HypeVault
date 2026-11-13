"""Site-independent parsing for dealer pages: sitemaps, JSON-LD, OpenGraph, spec rows, money, references.

Everything here works on plain strings so it can be unit-tested without a network.
"""

from __future__ import annotations

import json
import re
import unicodedata
from html import unescape
from typing import Any, Iterable
from urllib.parse import unquote, urljoin, urlsplit

# ── sitemaps ─────────────────────────────────────────────────────────────────

_LOC_RE = re.compile(r"<loc>\s*(?:<!\[CDATA\[)?\s*(.*?)\s*(?:\]\]>)?\s*</loc>", re.IGNORECASE | re.DOTALL)


def parse_sitemap(xml: str) -> tuple[list[str], list[str]]:
    """Returns (child sitemaps, page URLs). A sitemap index lists sitemaps; a urlset lists pages."""
    locs = [unescape(u).strip() for u in _LOC_RE.findall(xml)]
    if re.search(r"<sitemapindex\b", xml, re.IGNORECASE):
        return locs, []
    return [], locs


# ── HTML → structured bits ───────────────────────────────────────────────────

_LDJSON_RE = re.compile(r"<script[^>]*type=[\"']application/ld\+json[\"'][^>]*>(.*?)</script>", re.IGNORECASE | re.DOTALL)
_META_RE = re.compile(r"<meta\b[^>]*>", re.IGNORECASE)
_ATTR_RE = re.compile(r"([a-zA-Z_:.-]+)\s*=\s*(\"[^\"]*\"|'[^']*')")
_DROP_RE = re.compile(r"<(script|style|noscript|svg|template)\b.*?</\1>", re.IGNORECASE | re.DOTALL)
_BREAK_RE = re.compile(r"<\s*(br|/p|/div|/li|/tr|/td|/th|/dt|/dd|/h[1-6]|/span|/a|/label|/button|/section|/header|/footer|/ul|/ol|/dl|/table)\b[^>]*>", re.IGNORECASE)
_TAG_RE = re.compile(r"<[^>]+>")
_HREF_RE = re.compile(r"""href\s*=\s*["']([^"'#]+)["']""", re.IGNORECASE)


def jsonld_nodes(html: str) -> list[dict]:
    """Every JSON object found in ld+json blocks, with @graph and lists flattened."""
    out: list[dict] = []

    def walk(node: Any) -> None:
        if isinstance(node, list):
            for n in node:
                walk(n)
        elif isinstance(node, dict):
            out.append(node)
            if "@graph" in node:
                walk(node["@graph"])
            if isinstance(node.get("itemListElement"), list):
                for el in node["itemListElement"]:
                    walk(el.get("item", el) if isinstance(el, dict) else el)

    for raw in _LDJSON_RE.findall(html):
        text = raw.strip().removeprefix("<!--").removesuffix("-->").strip()
        try:
            walk(json.loads(text))
        except ValueError:
            try:
                walk(json.loads(unescape(text)))
            except ValueError:
                continue
    return out


def _type_is(node: dict, name: str) -> bool:
    t = node.get("@type")
    return t == name or (isinstance(t, list) and name in t)


def jsonld_product(html: str) -> dict | None:
    """The first schema.org Product (or ProductGroup) on the page."""
    for node in jsonld_nodes(html):
        if _type_is(node, "Product") or _type_is(node, "ProductGroup") or _type_is(node, "IndividualProduct"):
            return node
    return None


def meta_tags(html: str) -> dict[str, str]:
    """property/name/itemprop → content, lower-cased keys. First occurrence wins."""
    out: dict[str, str] = {}
    for tag in _META_RE.findall(html):
        attrs = {k.lower(): unescape(v[1:-1]) for k, v in _ATTR_RE.findall(tag)}
        key = attrs.get("property") or attrs.get("name") or attrs.get("itemprop")
        if key and "content" in attrs:
            out.setdefault(key.lower(), attrs["content"].strip())
    return out


def visible_lines(html: str) -> list[str]:
    """Page text, one block element per line, whitespace collapsed, empties dropped."""
    text = _DROP_RE.sub(" ", html)
    text = _BREAK_RE.sub("\n", text)
    text = unescape(_TAG_RE.sub(" ", text))
    lines = []
    for line in text.splitlines():
        line = " ".join(line.split())
        if line:
            lines.append(line)
    return lines


def links(html: str, base_url: str) -> list[str]:
    seen: dict[str, None] = {}
    for href in _HREF_RE.findall(html):
        url = urljoin(base_url, unescape(href).strip())
        if url.startswith("http"):
            seen.setdefault(url.split("#")[0], None)
    return list(seen)


# ── labelled spec rows ("Year" / "2023", "Box: Yes") ─────────────────────────


def label_values(lines: list[str], labels: dict[str, Iterable[str]], *, max_value_len: int = 80) -> dict[str, str]:
    """Find spec values by label. ``labels`` maps our field → the site's label spellings.

    Matches "Label: value", "Label value" on one line, or the label alone with the value on the next line.
    """
    found: dict[str, str] = {}
    folded = [line.casefold() for line in lines]
    for field, spellings in labels.items():
        for spelling in spellings:
            s = spelling.casefold()
            for i, line in enumerate(folded):
                value = None
                if line == s or line == s + ":":
                    if i + 1 < len(lines):
                        value = lines[i + 1]
                elif line.startswith(s + ":"):
                    value = lines[i][len(s) + 1 :]
                elif line.startswith(s + " ") and len(line) - len(s) < max_value_len and field != "price":
                    rest = lines[i][len(s) + 1 :]
                    # "Box and papers" is a different label, not Box = "and papers".
                    if not rest.casefold().startswith(("and ", "& ", "/", "+ ", "&amp;")):
                        value = rest
                if value:
                    value = value.strip(" :")
                    is_label_echo = value.casefold() == s and field != "box_papers"
                    if value and len(value) <= max_value_len and not is_label_echo:
                        found[field] = value
                        break
            if field in found:
                break
    return found


# ── money, references, years ─────────────────────────────────────────────────

_SYMBOLS = {"$": "USD", "US$": "USD", "£": "GBP", "€": "EUR", "CHF": "CHF", "HK$": "HKD", "¥": "JPY", "₹": "INR", "S$": "SGD", "A$": "AUD"}
_CODES = ("USD", "GBP", "EUR", "CHF", "HKD", "JPY", "INR", "SGD", "AUD", "AED", "CAD")
_SYM_PATTERN = "|".join(re.escape(s) for s in sorted(_SYMBOLS, key=len, reverse=True))
_MONEY_RE = re.compile(
    rf"(?P<pre>{_SYM_PATTERN}|\b(?:{'|'.join(_CODES)}))\s?(?P<num>\d[\d.,'   ]*\d|\d)"
    rf"|(?P<num2>\d[\d.,'   ]*\d|\d)\s?(?P<post>{_SYM_PATTERN}|\b(?:{'|'.join(_CODES)})\b)"
)


def _to_number(raw: str) -> float | None:
    s = re.sub(r"[\s'  ]", "", raw)
    if re.fullmatch(r"\d{1,3}(?:\.\d{3})+(?:,\d{1,2})?", s):  # 12.345,50 (European)
        s = s.replace(".", "").replace(",", ".")
    else:
        s = s.replace(",", "")
    try:
        value = float(s)
    except ValueError:
        return None
    return value if value > 0 else None


def parse_money(text: str | None) -> tuple[float | None, str | None]:
    """'$149,950' → (149950, 'USD'); 'USD 119,740' → (119740, 'USD'); '98.500 €' → (98500, 'EUR')."""
    if not text:
        return None, None
    match = _MONEY_RE.search(text)
    if not match:
        return None, None
    marker = match.group("pre") or match.group("post")
    number = match.group("num") or match.group("num2")
    return _to_number(number), _SYMBOLS.get(marker, marker if marker in _CODES else None)


def first_money(lines: list[str], *, after: int = 0, prefer: Iterable[str] = (), window: int = 60) -> tuple[float | None, str | None]:
    """First price in the page text. Lines containing a ``prefer`` phrase (e.g. 'Current Price') win."""
    prefer = [p.casefold() for p in prefer]
    for i, line in enumerate(lines):
        if prefer and any(p in line.casefold() for p in prefer):
            for candidate in (line, lines[i + 1] if i + 1 < len(lines) else ""):
                amount, cur = parse_money(candidate)
                if amount:
                    return amount, cur
    for line in lines[after : after + window]:
        low = line.casefold()
        if any(w in low for w in ("was ", "rrp", "save ", "saving", "monthly", "/mo", "per month", "deposit", "from ")):
            continue
        amount, cur = parse_money(line)
        if amount and amount >= 100:
            return amount, cur
    return None, None


_YEAR_RE = re.compile(r"\b(19[2-9]\d|20[0-4]\d)\b")
_REF_IN_TEXT_RE = re.compile(r"\b(\d{3,5}[A-Z]{0,4}(?:[/-][0-9A-Z]{1,6}){0,3})\b")


def year_of(text: str | None) -> str | None:
    if not text:
        return None
    match = _YEAR_RE.search(text)
    if not match:
        return None
    return f"c. {match.group(1)}" if re.search(r"approx|circa|\bc\.", text, re.IGNORECASE) else match.group(1)


def reference_from_title(title: str) -> str | None:
    """'Patek Philippe Nautilus 5711/1A-010' → '5711/1A-010'. Avoids picking a year or case size."""
    for match in _REF_IN_TEXT_RE.finditer(title.upper()):
        ref = match.group(1)
        if _YEAR_RE.fullmatch(ref) or re.fullmatch(r"\d{2,3}MM", ref):
            continue
        return ref
    return None


def normalize_reference(ref: str | None) -> str | None:
    """Comparable form: '5711/1A-010', '5711 1A 010' and '57111A010' → '57111A010'."""
    if not ref:
        return None
    return re.sub(r"[^0-9A-Z]", "", ref.upper()) or None


def yes_no(text: str | None) -> bool | None:
    if not text:
        return None
    low = text.strip().casefold()
    if low.startswith(("yes", "included", "original", "true")):
        return True
    if low.startswith(("no", "not included", "none", "false")):
        return False
    return None


# ── brand slugs and URL helpers ──────────────────────────────────────────────


def slug_candidates(brand: str) -> list[str]:
    """URL slugs a site might use: 'A. Lange & Söhne' → a-lange-sohne, a-lange-and-sohne, a-lange-soehne, ..."""
    def slugify(s: str) -> str:
        s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
        return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")

    variants = {brand, brand.replace("&", "and"), brand.replace("&", "und"), brand.replace("&", ""),
                brand.replace("ö", "oe").replace("Ö", "Oe"), brand.replace("ö", "oe").replace("&", "and")}
    out = []
    for v in variants:
        s = slugify(v)
        if s and s not in out:
            out.append(s)
    if brand.lower().startswith("a. lange"):
        out += [s for s in ("a-lange-sohne", "a-lange-soehne", "a-lange-und-sohne", "lange-sohne", "a-lange") if s not in out]
    return sorted(out, key=lambda s: (len(s), s))


def path_segments(url: str) -> list[str]:
    return [unquote(p) for p in urlsplit(url).path.split("/") if p]


def clean_title(title: str | None, site_names: Iterable[str] = ()) -> str | None:
    """Drop 'Pre-Owned', ' | Site', ' - Buy at ...' decorations."""
    if not title:
        return None
    t = " ".join(unescape(title).split())
    for sep in (" | ", " – ", " - "):
        head, _, tail = t.rpartition(sep)
        if head and any(n.casefold() in tail.casefold() for n in (*site_names, "buy", "watches", "shop")):
            t = head
    t = re.sub(r"^(pre-?owned|used|certified pre-?owned)\s+", "", t, flags=re.IGNORECASE)
    return t.strip() or None
