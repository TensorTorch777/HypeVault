"""Watch identification: name catalog, clue ranking, gallery, identify statuses, search fallback, upload checks."""

from __future__ import annotations

import asyncio
import io
import sys
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path
from unittest import mock

import numpy as np
from fastapi import HTTPException
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from identify import recognize  # noqa: E402
from identify.catalog import Catalog, Name, normalize, reference_keys  # noqa: E402
from identify.gallery import Gallery, Neighbour  # noqa: E402

ROWS = [
    ("Patek Philippe", "Nautilus", "5711/1A-010"),
    ("Patek Philippe", "Nautilus", "5711/1A-011"),
    ("Patek Philippe", "Nautilus", "5712/1A-001"),
    ("Patek Philippe", "Calatrava", "5196P-001"),
    ("Patek Philippe", "Vintage", "2526"),
    ("Audemars Piguet", "Royal Oak", "15500ST.OO.1220ST.01"),
    ("Audemars Piguet", "Royal Oak Dual Time", "26120ST.OO.1220ST.02"),
    ("Audemars Piguet", "Royal Oak Offshore", "26470ST.OO.A027CA.01"),
    ("A. Lange & Söhne", "Lange 1", "191.032"),
    ("A. Lange & Söhne", "Grand Lange 1", "139.032"),
    ("Richard Mille", "RM 011", "RM011-FM"),
]


def catalog() -> Catalog:
    return Catalog(ROWS)


def nb(brand: str, model: str | None, sim: float, ref: str | None = None, i: int = 0) -> Neighbour:
    return Neighbour(i, sim, f"id{i}", brand, model, ref, f"{brand} {model or ''}", None)


@dataclass
class Check:
    is_watch: bool
    looks_like: str | None = None


class FakeEmbedder:
    def __init__(self, vec):
        self.vec = np.asarray(vec, dtype=np.float32)

    async def embed_one(self, image):
        return self.vec


def tiny_gallery() -> Gallery:
    vecs = np.eye(4, dtype=np.float32)
    meta = {
        "external_id": ["a", "b", "c", "d"],
        "brand": ["Patek Philippe", "Patek Philippe", "Audemars Piguet", "Audemars Piguet"],
        "model": ["Nautilus", "Nautilus", "Royal Oak", ""],
        "reference": ["5711/1A-010", "", "", ""],
        "title": ["t1", "t2", "t3", "t4"],
        "item_url": ["u1", "", "", ""],
    }
    return Gallery(vecs, meta, pool="cls", built_on="2026-10-23")


class CatalogTests(unittest.TestCase):
    def test_normalize(self):
        self.assertEqual(normalize("RM011"), normalize("RM 011"))
        self.assertEqual(normalize("Twenty~4"), "twenty 4")
        self.assertEqual(normalize("A. Lange & Söhne"), "a lange sohne")

    def test_reference_keys(self):
        self.assertEqual(reference_keys("5711/1A-010"), ["57111A010", "5711"])
        self.assertEqual(reference_keys("26320ST.OO.1220ST.03"), ["26320STOO1220ST03", "26320"])
        self.assertEqual(reference_keys("109.032"), ["109032"])
        self.assertEqual(reference_keys("x"), [])

    def test_generic_dataset_models_are_not_names(self):
        self.assertNotIn("Vintage", [m for _k, m in catalog().models["Patek Philippe"]])

    def test_reference_family_names_the_model(self):
        m = catalog().match("5711")
        self.assertEqual(m.names, [Name("Patek Philippe", "Nautilus", "5711")])
        self.assertEqual(m.brands, {"Patek Philippe"})

    def test_full_reference_is_preferred_and_listed_once(self):
        m = catalog().match("patek 5711/1A-010 blue")
        self.assertEqual(m.names, [Name("Patek Philippe", "Nautilus", "5711/1A-010")])

    def test_longest_model_name_wins(self):
        c = catalog()
        self.assertEqual(c.model_in("A. Lange & Söhne Grand Lange 1 Moonphase", "A. Lange & Söhne"), "Grand Lange 1")
        self.assertEqual(c.model_in("Royal Oak Offshore Diver", "Audemars Piguet"), "Royal Oak Offshore")

    def test_title_reference_makes_the_model_more_specific(self):
        name = catalog().label_title("Audemars Piguet Royal Oak 26120ST White Dial", "Audemars Piguet")
        self.assertEqual(name.model, "Royal Oak Dual Time")

    def test_title_with_spaced_rm_number(self):
        self.assertEqual(catalog().label_title("Richard Mille RM011 Felipe Massa", "Richard Mille").model, "RM 011")

    def test_brand_words_alone(self):
        m = catalog().match("AUDEMARS PIGUET AUTOMATIC")
        self.assertEqual(m.brands, {"Audemars Piguet"})
        self.assertEqual(m.names, [])

    def test_name_queries(self):
        n = Name("Patek Philippe", "Nautilus", "5711")
        self.assertEqual((n.query, n.short_query, n.label), ("Nautilus 5711", "Nautilus", "Patek Philippe Nautilus 5711"))
        self.assertEqual(Name("Richard Mille").query, "Richard Mille")


class RankTests(unittest.TestCase):
    def test_lookalike_majority_wins(self):
        out = recognize.rank([nb("Patek Philippe", "Nautilus", 0.92, i=0), nb("Patek Philippe", "Nautilus", 0.91, i=1),
                              nb("Patek Philippe", "Calatrava", 0.90, i=2)])
        self.assertEqual((out[0]["brand"], out[0]["model"]), ("Patek Philippe", "Nautilus"))
        self.assertLessEqual(sum(c["confidence"] for c in out), 1.0001)
        self.assertIn("2 of the 3 closest eBay photos", out[0]["evidence"][0])

    def test_dial_text_outweighs_a_close_lookalike_split(self):
        lookalikes = [nb("Audemars Piguet", "Royal Oak", 0.90), nb("Patek Philippe", "Nautilus", 0.90, i=1)]
        out = recognize.rank(lookalikes, dial=catalog().match("PATEK PHILIPPE GENEVE"))
        self.assertEqual(out[0]["model"], "Nautilus")

    def test_brand_on_the_dial_drops_other_brands(self):
        lookalikes = [nb("Vacheron Constantin", "Overseas", 0.93), nb("Patek Philippe", "Nautilus", 0.90, i=1)]
        out = recognize.rank(lookalikes, dial=catalog().match("PATEK PHILIPPE"))
        self.assertEqual({c["brand"] for c in out}, {"Patek Philippe"})

    def test_typed_reference_beats_lookalikes_and_sets_the_reference(self):
        lookalikes = [nb("Audemars Piguet", "Royal Oak", 0.95, i=i) for i in range(5)]
        out = recognize.rank(lookalikes, typed=catalog().match("5711"))
        self.assertEqual(out[0]["label"], "Patek Philippe Nautilus 5711")
        self.assertEqual(out[0]["query"], "Nautilus 5711")

    def test_single_lookalike_reference_is_not_trusted(self):
        out = recognize.rank([nb("Patek Philippe", "Nautilus", 0.9, ref="5711/1A-010"), nb("Patek Philippe", "Nautilus", 0.9, i=1)])
        self.assertIsNone(out[0]["reference"])
        out = recognize.rank([nb("Patek Philippe", "Nautilus", 0.9, ref="5711/1A-010", i=i) for i in range(2)])
        self.assertEqual(out[0]["reference"], "5711/1A-010")

    def test_brand_only_guess_when_no_model_is_known(self):
        out = recognize.rank([nb("Richard Mille", None, 0.9), nb("Richard Mille", None, 0.88, i=1)])
        self.assertEqual([(c["brand"], c["model"]) for c in out], [("Richard Mille", None)])
        self.assertEqual(out[0]["query"], "Richard Mille")

    def test_at_most_three_guesses(self):
        models = ["Nautilus", "Calatrava", "Aquanaut", "Gondolo"]
        out = recognize.rank([nb("Patek Philippe", m, 0.9 - i * 0.01, i=i) for i, m in enumerate(models)])
        self.assertEqual(len(out), 3)

    def test_nothing_in_nothing_out(self):
        self.assertEqual(recognize.rank([]), [])

    def test_ebay_photo_matches_vote_best_first(self):
        photo = [Name("Vacheron Constantin", "Royal Eagle", "49145"), Name("Vacheron Constantin", "Royal Eagle", "49145"),
                 Name("Patek Philippe", "Nautilus")]
        out = recognize.rank([], ebay_photo=photo)
        self.assertEqual(out[0]["label"], "Vacheron Constantin Royal Eagle 49145")
        self.assertIn("2 of eBay's top 3 photo matches show", out[0]["evidence"][0])

    def test_ebay_photo_matches_can_outvote_a_split_gallery(self):
        gallery = [nb("Patek Philippe", "Nautilus", 0.90), nb("Patek Philippe", "Aquanaut", 0.90, i=1)]
        photo = [Name("Patek Philippe", "Aquanaut")] * 3
        self.assertEqual(recognize.rank(gallery, ebay_photo=photo)[0]["model"], "Aquanaut")

    def test_brand_only_ebay_matches_give_a_brand_guess(self):
        out = recognize.rank([], ebay_photo=[Name("Richard Mille"), Name("Richard Mille")])
        self.assertEqual([(c["brand"], c["model"]) for c in out], [("Richard Mille", None)])


class GalleryTests(unittest.TestCase):
    def test_neighbours_and_exclusion(self):
        g = tiny_gallery()
        top = g.neighbours(np.array([0.9, 0.1, 0, 0], dtype=np.float32), k=2)
        self.assertEqual([n.external_id for n in top], ["a", "b"])
        self.assertNotEqual(g.neighbours(np.array([1, 0, 0, 0], dtype=np.float32), k=1, exclude={0})[0].external_id, "a")
        self.assertIsNone(g.neighbours(np.array([0, 0, 0, 1], dtype=np.float32), k=1)[0].model)

    def test_save_load_roundtrip(self):
        g = tiny_gallery()
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "g.npz"
            g.save(path)
            back = Gallery.load(path)
        self.assertEqual(len(back), 4)
        self.assertEqual((back.pool, back.built_on), ("cls", "2026-10-23"))
        self.assertEqual(back.meta["model"], g.meta["model"])
        self.assertIsNone(back.neighbours(np.array([0, 1, 0, 0], dtype=np.float32), k=1)[0].item_url)  # "" -> None

    def test_misaligned_labels_are_rejected(self):
        meta = {k: v[:2] for k, v in tiny_gallery().meta.items()}
        with self.assertRaises(ValueError):
            Gallery(np.eye(4, dtype=np.float32), meta, pool="cls")


class IdentifyTests(unittest.TestCase):
    def run_identify(self, **kw):
        kw.setdefault("catalog", catalog())
        kw.setdefault("gallery", tiny_gallery())
        return asyncio.run(recognize.identify(**kw))

    def photo(self):
        return Image.new("RGB", (300, 300), "white")

    def test_photo_lookalikes_and_dial(self):
        r = self.run_identify(image=self.photo(), embedder=FakeEmbedder([0.7, 0.7, 0.1, 0]),
                              read_lines=lambda im: [("PATEK PHILIPPE", 0.9)], watch_check=lambda im: Check(True))
        self.assertEqual(r["status"], "recognised")
        self.assertEqual(r["candidates"][0]["model"], "Nautilus")
        self.assertTrue(r["used"]["dial_text"] and r["used"]["lookalikes"])
        self.assertEqual(r["dial_text"], ["PATEK PHILIPPE"])

    def test_not_a_watch(self):
        r = self.run_identify(image=self.photo(), embedder=FakeEmbedder([1, 0, 0, 0]),
                              watch_check=lambda im: Check(False, "a person"))
        self.assertEqual((r["status"], r["looks_like"], r["candidates"]), ("not_a_watch", "a person", []))

    def test_other_brand_on_the_dial(self):
        r = self.run_identify(image=self.photo(), embedder=FakeEmbedder([1, 0, 0, 0]),
                              read_lines=lambda im: [("ROLEX", 0.95), ("OYSTER PERPETUAL", 0.9)])
        self.assertEqual((r["status"], r["other_brand"]), ("unsupported_brand", "Rolex"))

    def test_other_brand_typed(self):
        r = self.run_identify(text="omega seamaster 300")
        self.assertEqual(r["status"], "unsupported_brand")

    def test_unknown_text_is_searched_as_typed(self):
        r = self.run_identify(text="  blue   dress watch ")
        self.assertEqual(r["status"], "free_text")
        self.assertEqual(r["candidates"][0]["query"], "blue dress watch")

    def test_far_lookalikes_alone_do_not_name_a_watch(self):
        r = self.run_identify(image=self.photo(), embedder=FakeEmbedder([0.3, 0.3, 0.3, 0.3]))
        self.assertEqual(r["status"], "not_recognised")
        self.assertFalse(r["used"]["lookalikes"])

    def test_split_clues_are_unsure(self):
        with mock.patch.object(recognize, "SURE", 0.9):
            r = self.run_identify(image=self.photo(), embedder=FakeEmbedder([0.7, 0, 0.7, 0]))
        self.assertEqual(r["status"], "unsure")

    def test_ebay_photo_search_is_used_and_returned(self):
        async def search(image):
            return [{"title": "Patek Philippe Calatrava 5196P", "brand": "Patek Philippe", "price": 30000.0,
                     "currency": "USD", "condition": None, "item_url": "u", "image_url": "i"}] * 3

        r = self.run_identify(gallery=None, image=self.photo(), embedder=FakeEmbedder([1, 0, 0, 0]), image_search=search)
        self.assertEqual(r["candidates"][0]["model"], "Calatrava")
        self.assertTrue(r["used"]["ebay_photo_search"])
        self.assertEqual(len(r["ebay_photo_matches"]), 3)

    def test_failed_ebay_photo_search_does_not_stop_identification(self):
        async def search(image):
            raise RuntimeError("eBay down")

        r = self.run_identify(image=self.photo(), embedder=FakeEmbedder([0.7, 0.7, 0.1, 0]), image_search=search)
        self.assertEqual(r["candidates"][0]["model"], "Nautilus")
        self.assertFalse(r["used"]["ebay_photo_search"])
        self.assertEqual(r["ebay_photo_matches"], [])

    def test_no_gallery_still_uses_text(self):
        r = self.run_identify(gallery=None, image=self.photo(), embedder=FakeEmbedder([1, 0, 0, 0]),
                              read_lines=lambda im: [("AUDEMARS PIGUET", 0.9), ("ROYAL OAK", 0.8)])
        self.assertEqual(r["candidates"][0]["label"], "Audemars Piguet Royal Oak")


class SearchFallbackTests(unittest.TestCase):
    def run_search(self, hits: dict[str, bool], fallback: str | None):
        from market import service

        calls = []

        async def ebay(session, redis, query, brand):
            calls.append(("ebay", query))
            return {"status": "ok", "items": [{"title": query}] if hits.get(query) else [], "stats": None}

        async def c24(session, query, brand, sort="lowest", limit=12):
            calls.append(("c24", query))
            return {"status": "no_match", "items": [], "stats": None}

        with mock.patch.object(service, "_ebay_section", ebay), mock.patch.object(service, "_chrono24_section", c24):
            out = asyncio.run(service.search_market(None, None, "Nautilus 5711", "Patek Philippe", fallback_query=fallback))
        return out, calls

    def test_shorter_name_is_tried_when_nothing_matches(self):
        out, calls = self.run_search({"Nautilus": True}, "Nautilus")
        self.assertEqual(out["tried"], ["Nautilus 5711", "Nautilus"])
        self.assertEqual(out["query_used"], "Nautilus")
        self.assertEqual(out["found"], {"ebay": True, "chrono24_dataset": False})

    def test_no_retry_when_the_full_name_matches(self):
        out, calls = self.run_search({"Nautilus 5711": True}, "Nautilus")
        self.assertEqual(out["tried"], ["Nautilus 5711"])
        self.assertEqual(len(calls), 2)

    def test_not_found_anywhere(self):
        out, _ = self.run_search({}, "Nautilus")
        self.assertEqual(out["found"], {"ebay": False, "chrono24_dataset": False})

    def test_ebay_not_connected_is_unknown_not_missing(self):
        from market.service import found_in

        self.assertEqual(found_in({"status": "not_configured", "items": []}, {"items": []}),
                         {"ebay": None, "chrono24_dataset": False})


class UploadCheckTests(unittest.TestCase):
    def encode(self, size, fmt="JPEG"):
        buf = io.BytesIO()
        Image.new("RGB", size, "gray").save(buf, fmt)
        return buf.getvalue()

    def test_checks(self):
        from market.routes import _read_photo

        self.assertEqual(_read_photo("image/jpeg", self.encode((640, 480))).size, (640, 480))
        for ct, raw in (("image/gif", self.encode((640, 480))), ("image/png", b""), ("image/png", b"not an image"),
                        ("image/jpeg", self.encode((120, 640)))):
            with self.assertRaises(HTTPException) as ctx:
                _read_photo(ct, raw)
            self.assertEqual(ctx.exception.status_code, 400)


if __name__ == "__main__":
    unittest.main()
