"""benchmark_v1_real_vs_generated: statistics, normalisation, exclusions and the frontier rule."""

from __future__ import annotations

import io
import json
import sys
import unittest
from pathlib import Path

from PIL import Image

BENCH = Path(__file__).resolve().parents[1] / "ml_rtx5080" / "benchmark_v1_real_vs_generated"
sys.path.insert(0, str(BENCH))

import bench_stats as st  # noqa: E402
import common  # noqa: E402


def _jpeg(w: int, h: int, colour=(120, 30, 200), fmt="PNG") -> bytes:
    img = Image.new("RGB", (w, h), colour)
    for x in range(0, w, 7):
        img.putpixel((x, h // 2), (255, 255, 255))
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


class StatsTests(unittest.TestCase):
    def test_clopper_pearson_edges(self):
        self.assertEqual(st.clopper_pearson(0, 10)[0], 0.0)
        self.assertEqual(st.clopper_pearson(10, 10)[1], 1.0)
        self.assertAlmostEqual(st.one_sided_upper(0, 300), 0.00994, places=4)

    def test_mcnemar_and_power(self):
        self.assertEqual(st.mcnemar_exact(0, 0), 1.0)
        self.assertLess(st.mcnemar_exact(30, 10), 0.01)
        self.assertEqual(st.required_discordant(), 49)

    def test_binary_metrics_keeps_review_in_coverage(self):
        labels = ["GENERATED", "GENERATED", "GENERATED", "REAL_PHOTO", "REAL_PHOTO"]
        decisions = ["FAKE", "AUTHENTIC", "REVIEW", "AUTHENTIC", "FAKE"]
        m = st.binary_metrics(labels, decisions)
        self.assertEqual(m["false_authentic"]["k"], 1)
        self.assertEqual(m["false_authentic"]["n"], 3)
        self.assertEqual(m["review"]["generated"], 1)
        self.assertAlmostEqual(m["coverage"], 0.8)
        self.assertAlmostEqual(m["accuracy"], 0.5)

    def test_cluster_bootstrap_is_deterministic(self):
        a, b = [1, 1, 0, 1, 1, 0], [0, 1, 0, 0, 1, 0]
        clusters = ["u1", "u1", "u2", "u3", "u3", "u4"]
        self.assertEqual(st.cluster_bootstrap_diff(a, b, clusters), st.cluster_bootstrap_diff(a, b, clusters))
        point, lo, hi = st.cluster_bootstrap_diff(a, b, clusters)
        self.assertAlmostEqual(point, 2 / 6)
        self.assertLessEqual(lo, point)
        self.assertGreaterEqual(hi, point)


class NormalisationTests(unittest.TestCase):
    def test_every_class_ends_as_the_same_jpeg_geometry(self):
        for w, h, fmt in ((1600, 1200, "JPEG"), (768, 768, "PNG"), (1024, 2000, "WEBP")):
            jpeg, _orig, size = common.normalise(_jpeg(w, h, fmt=fmt))
            with Image.open(io.BytesIO(jpeg)) as im:
                self.assertEqual((im.format, im.size), ("JPEG", (768, 768)))
                self.assertFalse(im.info.get("exif"))
            self.assertEqual(size, (w, h))

    def test_admit_excludes_small_and_duplicate_images(self):
        dedup = common.BenchmarkDedup()
        with self._tmp() as out:
            kw = dict(label="REAL_PHOTO", group="real_ebay", brand="Patek Philippe", out_dir=out, training=None, dedup=dedup, meta={})
            small = common.admit(_jpeg(700, 900), sample_id="a", unit_id="u1", **kw)
            first = common.admit(_jpeg(1000, 1000), sample_id="b", unit_id="u2", **kw)
            again = common.admit(_jpeg(1000, 1000), sample_id="c", unit_id="u3", **kw)
        self.assertEqual(small.excluded, "source_short_side_below_768")
        self.assertIsNone(first.excluded)
        self.assertEqual(again.excluded, "benchmark_duplicate_sha256")

    def test_training_overlap_uses_stored_hashes(self):
        raw = _jpeg(1000, 1000, colour=(10, 200, 30))
        _jpeg_bytes, original, _ = common.normalise(raw)
        dh = common.dhash_image(original)
        index = common.TrainingIndex({common.sha256_bytes(raw)}, [])
        self.assertEqual(index.overlap(common.sha256_bytes(raw), dh), "training_sha256_match")
        near = common.TrainingIndex(set(), [dh ^ 0b11])
        self.assertEqual(near.overlap("x", dh), "training_dhash_near_duplicate")
        self.assertIsNone(common.TrainingIndex(set(), [dh ^ 0b111]).overlap("x", dh))

    def _tmp(self):
        import tempfile

        class _D:
            def __enter__(s):
                s.t = tempfile.TemporaryDirectory()
                return Path(s.t.name)

            def __exit__(s, *a):
                s.t.cleanup()

        return _D()


class ProtocolTests(unittest.TestCase):
    def test_protocol_freezes_models_and_rule(self):
        p = json.loads((BENCH / "protocol.json").read_text())
        self.assertEqual(p["status"], "PRE_REGISTERED")
        self.assertEqual(p["models"]["dinov3_experimental"]["checkpoint_sha256"],
                         "5a38c93fd442b03653c65d2a5ecc9c2687ef152f7c5c020763e4ce1fd9c7d28f")
        self.assertEqual(p["sample_size"]["required_discordant"], st.required_discordant())
        self.assertIn("not shown to perform better", p["frontier_rule"]["ENGINEERING_SELECTION"])


if __name__ == "__main__":
    unittest.main()
