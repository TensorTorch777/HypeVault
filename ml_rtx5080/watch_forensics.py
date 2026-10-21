"""Phase 72: dial-text forensics. Does the text on the dial match the declared brand?

Generated watch images often carry garbled lettering or another brand's name. Two readers look at
each image: RapidOCR (PP-OCRv5, literal; it does not correct spelling) and Gemma 4 12B through a
local Ollama server (reads curved and small text, names the brand it sees, and says whether the
lettering is garbled). Neither reader is told the declared brand. The comparison with the declared
brand happens afterwards, in `text_evidence`.

    .venv/bin/python ml_rtx5080/watch_forensics.py read --set bench_v1     # cache readings (resumable)
    .venv/bin/python ml_rtx5080/watch_forensics.py read --set val_label0
    .venv/bin/python ml_rtx5080/watch_forensics.py dev                     # evidence + fusion on development data
"""

from __future__ import annotations

import argparse
import base64
import difflib
import io
import json
import re
import sys
import time
import unicodedata
from pathlib import Path

_PKG = Path(__file__).resolve().parent
if str(_PKG) not in sys.path:
    sys.path.insert(0, str(_PKG))

REPO = _PKG.parent
OUT = _PKG / "experiments" / "watch_forensics_v1"
READINGS = OUT / "readings"
MANIFEST = _PKG / "experiments" / "dataset_audit" / "split_manifest_v2.json"
BENCH = _PKG / "benchmark_v1_real_vs_generated"
OLLAMA = "http://127.0.0.1:11435/api/chat"
GEMMA = "hf.co/google/gemma-4-12B-it-qat-q4_0-gguf"
READ_SIDE = 768
VAL_PER_BRAND = 200

PROMPT = (
    "You are inspecting a photo for a watch resale marketplace. "
    "1) Transcribe every piece of text visible on the watch dial, case, box or papers EXACTLY as written, "
    "letter by letter. Do not correct spelling and do not guess words you cannot read. "
    "2) Name the watch brand written on the dial, or NONE if no brand name is readable. "
    "3) Say whether any lettering looks garbled: malformed letters or nonsense words of the kind AI image "
    "generators produce. "
    'Reply only with JSON: {"text": ["..."], "dial_brand": "...", "garbled": true|false}'
)

# Words that identify each supported brand on a dial. GENEVE/GENEVA is shared, so it is neutral.
BRAND_WORDS = {
    "A. Lange & Söhne": ("LANGE", "SOHNE", "GLASHUTTE"),
    "Audemars Piguet": ("AUDEMARS", "PIGUET"),
    "Patek Philippe": ("PATEK", "PHILIPPE"),
    "Richard Mille": ("RICHARD", "MILLE"),
    "Vacheron Constantin": ("VACHERON", "CONSTANTIN"),
}
OTHER_BRAND_WORDS = (
    "ROLEX", "OMEGA", "TISSOT", "CARTIER", "BREITLING", "HUBLOT", "SEIKO", "LONGINES", "TUDOR", "PANERAI",
    "ZENITH", "CHOPARD", "BREGUET", "BLANCPAIN", "LECOULTRE", "CITIZEN", "CASIO", "GUCCI", "MONTBLANC",
    "BULOVA", "HAMILTON", "ORIS", "RADO", "FOSSIL", "HEUER", "IWC", "MOVADO", "PIAGET", "BVLGARI", "BULGARI",
    "SEAMASTER", "SPEEDMASTER", "SUBMARINER", "DAYTONA", "DATEJUST", "CARRERA", "NAVITIMER",
)
DIAL_WORDS = {
    "GENEVE", "GENEVA", "SWISS", "MADE", "AUTOMATIC", "CHRONOMETER", "CHRONOGRAPH", "OFFICIALLY", "CERTIFIED",
    "WATER", "RESISTANT", "PERPETUAL", "CALENDAR", "TOURBILLON", "ROYAL", "OAK", "OFFSHORE", "NAUTILUS",
    "AQUANAUT", "CALATRAVA", "OVERSEAS", "PATRIMONY", "SAXONIA", "DATOGRAPH", "ZEITWERK", "DATE", "QUARTZ",
}
OCR_MIN_CONFIDENCE = 0.6
NEAR_MISS = (0.6, 0.95)  # difflib ratio band: close to a brand word but not equal


# ---------- sets ----------

def bench_rows() -> list[dict]:
    rows = []
    for line in (BENCH / "data" / "manifest.jsonl").read_text().splitlines():
        s = json.loads(line)
        if s.get("excluded") is None:
            rows.append({"id": s["sample_id"], "path": str(BENCH / s["path"]), "brand": s["brand"],
                         "group": s["group"], "label": int(s["label"] == "GENERATED")})
    return rows


def val_rows(label: int) -> list[dict]:
    rows, counts = [], {}
    for s in json.loads(MANIFEST.read_text())["samples"]:
        if s["split"] != "validation" or s["label"] != label or counts.get(s["brand"], 0) >= VAL_PER_BRAND:
            continue
        counts[s["brand"]] = counts.get(s["brand"], 0) + 1
        rows.append({"id": s["sample_id"], "path": s["path"], "brand": s["brand"], "group": f"label{label}", "label": label})
    return rows


SETS = {"bench_v1": bench_rows, "val_label0": lambda: val_rows(0), "val_label1": lambda: val_rows(1)}


# ---------- readers ----------

def _jpeg_b64(path: str) -> str:
    from PIL import Image, ImageOps

    img = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    img.thumbnail((READ_SIDE, READ_SIDE), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=92)
    return base64.b64encode(buf.getvalue()).decode()


def read_gemma(path: str) -> dict:
    import httpx

    body = {"model": GEMMA, "stream": False, "format": "json", "think": False,
            "options": {"temperature": 0, "num_predict": 200},
            "messages": [{"role": "user", "content": PROMPT, "images": [_jpeg_b64(path)]}]}
    reply = httpx.post(OLLAMA, json=body, timeout=300).json()["message"]["content"]
    try:
        parsed = json.loads(reply)
    except json.JSONDecodeError:
        return {"raw": reply, "parse_error": True}
    return {"text": [str(t) for t in parsed.get("text") or []], "dial_brand": str(parsed.get("dial_brand") or "NONE"),
            "garbled": bool(parsed.get("garbled"))}


def read(set_name: str) -> None:
    from rapidocr import RapidOCR

    READINGS.mkdir(parents=True, exist_ok=True)
    out = READINGS / f"{set_name}.jsonl"
    done = {json.loads(l)["id"] for l in out.read_text().splitlines()} if out.exists() else set()
    rows = [r for r in SETS[set_name]() if r["id"] not in done]
    ocr = RapidOCR()
    started = time.time()
    with out.open("a") as fh:
        for i, row in enumerate(rows, 1):
            res = ocr(row["path"])
            rapid = [[t, round(float(s), 3)] for t, s in zip(res.txts or (), res.scores or ())]
            fh.write(json.dumps({**row, "rapidocr": rapid, "gemma": read_gemma(row["path"])}) + "\n")
            fh.flush()
            if i % 50 == 0:
                print(f"{set_name}: {i}/{len(rows)} ({(time.time() - started) / i:.2f} s/image)", flush=True)
    print(f"{set_name}: {len(done) + len(rows)} readings", flush=True)


# ---------- evidence ----------

def words(text: str) -> list[str]:
    plain = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode().upper()
    return re.findall(r"[A-Z]{2,}", plain)


def text_evidence(reading: dict) -> dict:
    """WRONG_BRAND, GARBLED, CONSISTENT or NO_TEXT, from both readers against the declared brand."""
    own = set(BRAND_WORDS[reading["brand"]])
    others = set(OTHER_BRAND_WORDS) | {w for b, ws in BRAND_WORDS.items() if b != reading["brand"] for w in ws}
    rapid = [w for t, s in reading["rapidocr"] if s >= OCR_MIN_CONFIDENCE for w in words(t)]
    gemma = reading.get("gemma", {})
    gemma_words = [w for t in gemma.get("text", []) for w in words(t)]
    dial_brand = words(gemma.get("dial_brand", ""))
    flags = []
    if (set(rapid) | set(dial_brand)) & others and not (set(rapid) | set(dial_brand)) & own:
        flags.append("WRONG_BRAND")
    near = [w for w in rapid if len(w) >= 4 and w not in own | others | DIAL_WORDS
            and any(NEAR_MISS[0] <= difflib.SequenceMatcher(None, w, b).ratio() < NEAR_MISS[1] for b in own)]
    if near:
        flags.append("GARBLED")
    consistent = bool((set(rapid) | set(gemma_words) | set(dial_brand)) & own)
    if flags:
        status = flags[0]
    elif consistent:
        status = "CONSISTENT"
    else:
        status = "NO_TEXT"
    return {"status": status, "flags": flags, "near_miss": near, "gemma_garbled": bool(gemma.get("garbled")),
            "gemma_dial_brand": gemma.get("dial_brand")}


def load_readings(set_name: str) -> list[dict]:
    return [json.loads(l) for l in (READINGS / f"{set_name}.jsonl").read_text().splitlines()]


def dev() -> None:
    import collections

    table = collections.defaultdict(collections.Counter)
    for set_name in SETS:
        if not (READINGS / f"{set_name}.jsonl").exists():
            continue
        for r in load_readings(set_name):
            ev = text_evidence(r)
            table[r["group"]][ev["status"]] += 1
            table[r["group"]]["gemma_garbled" if ev["gemma_garbled"] else "gemma_clean"] += 1
    for group, counts in table.items():
        print(group, dict(counts))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=("read", "dev"))
    parser.add_argument("--set", choices=sorted(SETS))
    args = parser.parse_args()
    if args.command == "read":
        read(args.set)
    else:
        dev()


if __name__ == "__main__":
    main()
