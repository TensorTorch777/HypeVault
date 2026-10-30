"""Score benchmark_v6_fresh once with every system named in protocol.json.

Systems: the live DINOv2 listing check (0.88 rule), the frozen Phase 75 head and the frozen Phase 76
head (DINOv3 + Perception Encoder), all on the same normalised 512 px images written by build.py.
The C2PA AI label is read from the ORIGINAL downloads at ingest and reported per app.

    .venv/bin/python ml_rtx5080/benchmark_v6_fresh/score.py
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
ML = HERE.parent
REPO = ML.parent
for p in (str(ML / "benchmark_v1_real_vs_generated"), str(ML), str(REPO / "backend")):
    if p not in sys.path:
        sys.path.insert(0, p)

from common import read_manifest  # noqa: E402
from unseen_generator import clopper_pearson  # noqa: E402

PROTOCOL = HERE / "protocol.json"
MANIFEST = HERE / "data" / "manifest.jsonl"
RESULTS = HERE / "results"
SYSTEMS = ("dinov2_live", "phase75_head", "phase76_head")
APPS = ("chatgpt_v6", "gemini_v6")  # primary
REPORTED = APPS + ("gemini_edu_v6",)  # Gemini on the Edu account (Pro): reported apart, not in the criterion


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tail(k: int, n: int, p: float) -> float:
    return sum(math.comb(n, i) * p**i * (1 - p) ** (n - i) for i in range(k, n + 1))


def lower_one_sided(k: int, n: int, alpha: float = 0.05) -> float:
    if k == 0:
        return 0.0
    lo, hi = 0.0, k / n
    for _ in range(60):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if _tail(k, n, mid) < alpha else (lo, mid)
    return lo


def rate(k: int, n: int) -> dict:
    lo, hi = clopper_pearson(k, n) if n else (None, None)
    return {"k": k, "n": n, "rate": round(k / n, 4) if n else None, "ci95": [round(lo, 4), round(hi, 4)] if n else None,
            "lower95_one_sided": round(lower_one_sided(k, n), 4) if n else None}


def mcnemar_one_sided(rows: list[dict], better: str, worse: str) -> dict:
    """Exact one-sided McNemar: is `better` caught on more paired images than `worse`?"""
    b = sum(r[better] and not r[worse] for r in rows)
    c = sum(r[worse] and not r[better] for r in rows)
    return {"only_" + better: b, "only_" + worse: c, "p_one_sided": round(_tail(b, b + c, 0.5), 5) if b + c else 1.0}


def main() -> None:
    import numpy as np
    from PIL import Image

    import frontier_v5 as f5
    import kandinsky_v4 as k4
    import phase76
    from config import settings
    from inference.local_torch import classify_image_torch
    from inference.triton_client import preprocess_chw
    from inference.verdict import apply_min_authentic_confidence
    from kandinsky_v4_design import view

    if RESULTS.exists() and any(RESULTS.iterdir()):
        raise SystemExit("benchmark_v6_fresh has already been scored; it runs once")
    protocol = json.loads(PROTOCOL.read_text())
    if sha256(phase76.RULE) != protocol["frozen_system"]["rule_sha256"]:
        raise SystemExit("Phase 76 rule changed after the protocol was written")
    head76 = phase76.FrozenFusionHead()  # checks the head's SHA-256
    head75 = f5.FrozenHead()
    p75 = head75.rule

    samples = [s for s in read_manifest(MANIFEST) if s.excluded is None]
    for s in samples:
        s.path = str(Path(s.path) if Path(s.path).is_absolute() else HERE / s.path)
    groups = defaultdict(int)
    for s in samples:
        groups[s.group] += 1
    print("samples", dict(groups), flush=True)

    images = [k4.stored(s.path) for s in samples]
    s76 = head76.scores_images(images)
    feats: dict[str, list] = defaultdict(list)
    for start in range(0, len(images), k4.BATCH):
        chunk = images[start:start + k4.BATCH]
        for d in (head75.pre.features(chunk), head75.ft.features(chunk)):
            for k, v in d.items():
                feats[k].append(v)
    s75 = head75.model.decision_function(view({k: np.concatenate(v) for k, v in feats.items()}, p75["features"]))

    async def dinov2(pil):
        verdict, conf = await classify_image_torch(preprocess_chw(np.array(pil)))
        return apply_min_authentic_confidence(verdict, conf, settings.inference_min_authentic_confidence)

    rows = []
    for i, s in enumerate(samples):
        with Image.open(s.path) as im:
            d2_verdict, d2_conf = asyncio.run(dinov2(im.convert("RGB")))
        rows.append({
            "id": s.sample_id, "group": s.group, "brand": s.brand, "scene_type": (s.meta or {}).get("scene_type"),
            "c2pa_ai_label": (s.meta or {}).get("c2pa_ai_label"),
            "web_search_sources_shown": bool((s.meta or {}).get("web_search_sources_shown")),
            "dinov2_live": d2_verdict == "FAKE", "dinov2_confidence": d2_conf,
            "phase75_head": bool(s75[i] >= p75["t_fake"]), "phase75_score": round(float(s75[i]), 4),
            "phase76_head": bool(s76[i] >= head76.rule["t_fake"]), "phase76_score": round(float(s76[i]), 4),
        })

    def summarise(sub: list[dict]) -> dict:
        return {sys_: rate(sum(r[sys_] for r in sub), len(sub)) for sys_ in SYSTEMS}

    report: dict = {
        "benchmark": "benchmark_v6_fresh", "phase": 76,
        "scored_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "protocol_sha256_at_scoring": sha256(PROTOCOL), "phase76_head_sha256": head76.rule["model"]["sha256"],
        "phase75_head_sha256": p75["model"]["sha256"],
        "note": "For generated sets the rate is the share caught (called FAKE); for real_ebay_v6 it is the share wrongly called FAKE.",
        "groups": {},
    }
    for group in sorted({r["group"] for r in rows}):
        sub = [r for r in rows if r["group"] == group]
        g = {"all": summarise(sub), "by_brand": {b: summarise([r for r in sub if r["brand"] == b]) for b in sorted({r["brand"] for r in sub})}}
        if group in REPORTED:
            g["by_scene_type"] = {t: summarise([r for r in sub if r["scene_type"] == t]) for t in sorted({r["scene_type"] for r in sub})}
            g["c2pa_ai_label_on_original"] = rate(sum(bool(r["c2pa_ai_label"]) for r in sub), len(sub))
            g["phase76_vs_dinov2_live"] = mcnemar_one_sided(sub, "phase76_head", "dinov2_live")
            g["phase76_vs_phase75"] = mcnemar_one_sided(sub, "phase76_head", "phase75_head")
            if any(r["web_search_sources_shown"] for r in sub):
                g["all_without_web_search"] = summarise([r for r in sub if not r["web_search_sources_shown"]])
        report["groups"][group] = g

    both = [r for r in rows if r["group"] in ("gemini_v6", "gemini_edu_v6")]
    report["exploratory_gemini_both_accounts"] = {"all": summarise(both),
                                                  "phase76_vs_dinov2_live": mcnemar_one_sided(both, "phase76_head", "dinov2_live"),
                                                  "phase76_vs_phase75": mcnemar_one_sided(both, "phase76_head", "phase75_head")}
    res = {g: report["groups"][g]["all"]["phase76_head"] for g in report["groups"]}
    report["success_criterion"] = {
        "rule": protocol["success_criterion"],
        "met": bool(all(res.get(a, {}).get("rate") is not None and res[a]["rate"] >= 0.80 for a in APPS)
                    and res.get("real_ebay_v6", {}).get("rate") is not None and res["real_ebay_v6"]["rate"] <= 0.03),
    }
    report["secondary"] = {a: {"beats_dinov2_live": report["groups"][a]["phase76_vs_dinov2_live"]["p_one_sided"] < 0.05,
                               "beats_phase75": report["groups"][a]["phase76_vs_phase75"]["p_one_sided"] < 0.05}
                           for a in REPORTED if a in report["groups"]}
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "scores.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (RESULTS / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({g: {k: v["rate"] for k, v in report["groups"][g]["all"].items()} for g in report["groups"]}, indent=1))
    print("success criterion met:", report["success_criterion"]["met"])
    print("secondary:", report["secondary"])


if __name__ == "__main__":
    main()
