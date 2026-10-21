"""Phase 73 fix 3: better dial reading. Centre crop at double resolution, and a second literal reader.

Readers (none of them is told the declared brand):
- rapid_full: RapidOCR on the whole image (Phase 72 reading, reused where it exists)
- rapid_crop: RapidOCR on the central 60% of the image, upscaled to 1024 px (the dial is usually there)
- doctr_full / doctr_crop: docTR mobile models (db_mobilenet_v3_large + crnn_mobilenet_v3_small) on the same two views

    .venv/bin/python ml_rtx5080/watch_forensics_v2.py read --set bench_v1|bench_v2|val_label0|val_label1
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

_PKG = Path(__file__).resolve().parent
if str(_PKG) not in sys.path:
    sys.path.insert(0, str(_PKG))

import watch_forensics as wf  # noqa: E402

OUT = _PKG / "experiments" / "watch_forensics_v2"
READINGS = OUT / "readings"
V2_BENCH = _PKG / "benchmark_v2_fresh"
CROP_FRACTION = 0.6
CROP_SIDE = 1024
MIN_CONFIDENCE = 0.6
READERS = ("rapid_full", "rapid_crop", "doctr_full", "doctr_crop")


def v2_rows() -> list[dict]:
    import csv

    run = sorted((V2_BENCH / "results").iterdir())[0]
    rapid = {r["sample_id"]: json.loads(r["rapidocr"]) for r in csv.DictReader((run / "predictions.csv").open())}
    rows = []
    for line in (V2_BENCH / "data" / "manifest.jsonl").read_text().splitlines():
        s = json.loads(line)
        if s.get("excluded") is None:
            rows.append({"id": s["sample_id"], "path": s["path"], "brand": s["brand"], "group": s["group"],
                         "label": int(s["label"] == "GENERATED"), "rapid_full": rapid[s["sample_id"]]})
    return rows


def phase72_rows(set_name: str) -> list[dict]:
    return [{**{k: r[k] for k in ("id", "path", "brand", "group", "label")}, "rapid_full": r["rapidocr"]}
            for r in wf.load_readings(set_name)]


SETS = {"bench_v1": lambda: phase72_rows("bench_v1"), "bench_v2": v2_rows,
        "val_label0": lambda: phase72_rows("val_label0"), "val_label1": lambda: phase72_rows("val_label1")}


def views(path: str) -> tuple[np.ndarray, np.ndarray]:
    img = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    w, h = img.size
    cw, ch = int(w * CROP_FRACTION), int(h * CROP_FRACTION)
    crop = img.crop(((w - cw) // 2, (h - ch) // 2, (w + cw) // 2, (h + ch) // 2)).resize((CROP_SIDE, CROP_SIDE), Image.LANCZOS)
    return np.asarray(img), np.asarray(crop)


def doctr_lines(predictor, image: np.ndarray) -> list[list]:
    page = predictor([image]).pages[0]
    out = []
    for block in page.blocks:
        for line in block.lines:
            out.append([" ".join(w.value for w in line.words), float(min(w.confidence for w in line.words))])
    return out


def read(set_name: str) -> None:
    from doctr.models import ocr_predictor
    from rapidocr import RapidOCR

    READINGS.mkdir(parents=True, exist_ok=True)
    out = READINGS / f"{set_name}.jsonl"
    done = {json.loads(l)["id"] for l in out.read_text().splitlines()} if out.exists() else set()
    rows = [r for r in SETS[set_name]() if r["id"] not in done]
    rapid = RapidOCR()
    doctr = ocr_predictor(det_arch="db_mobilenet_v3_large", reco_arch="crnn_mobilenet_v3_small", pretrained=True).cuda()
    started = time.time()
    with out.open("a") as fh:
        for i, row in enumerate(rows, 1):
            full, crop = views(row["path"])
            res = rapid(crop)
            row["rapid_crop"] = [[t, float(s)] for t, s in zip(res.txts or (), res.scores or ())]
            row["doctr_full"] = doctr_lines(doctr, full)
            row["doctr_crop"] = doctr_lines(doctr, crop)
            fh.write(json.dumps(row) + "\n")
            fh.flush()
            if i % 200 == 0:
                print(f"{set_name}: {i}/{len(rows)} ({(time.time() - started) / i:.2f} s/image)", flush=True)
    print(f"{set_name}: {len(done) + len(rows)} readings", flush=True)


def brand_hits(reading: dict) -> dict[str, bool]:
    """For each reader: did it read a word containing one of the declared brand's words?"""
    own = wf.BRAND_WORDS[reading["brand"]]
    hits = {}
    for reader in READERS:
        words = {w for t, s in reading.get(reader, []) if s >= MIN_CONFIDENCE for w in wf.words(t)}
        hits[reader] = any(o in w for w in words for o in own)
    return hits


def load_readings(set_name: str) -> list[dict]:
    return [json.loads(l) for l in (READINGS / f"{set_name}.jsonl").read_text().splitlines()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=("read",))
    parser.add_argument("--set", choices=sorted(SETS), required=True)
    args = parser.parse_args()
    read(args.set)


if __name__ == "__main__":
    main()
