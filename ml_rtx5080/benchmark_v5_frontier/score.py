"""Score benchmark_v5_frontier once with every system named in protocol.json.

Systems: the live DINOv2 listing check (0.88 rule), DINOv3 epoch_018 alone (P(fake) >= 0.5), the Phase 74
head (score >= its t_fake) and the frozen Phase 75 head (score >= its t_fake). All read the same normalised
512 px images written by build.py. The motivating image (testtest.jpeg) is scored too and reported apart.

    .venv/bin/python ml_rtx5080/benchmark_v5_frontier/score.py
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
P74_RULE = ML / "experiments" / "kandinsky_v4" / "fusion_rule.json"
P75_RULE = ML / "experiments" / "frontier_v5" / "frozen_rule.json"
MOTIVATING = REPO / "testtest.jpeg"
SYSTEMS = ("dinov2_live", "dinov3_alone", "phase74_head", "phase75_head")
CHATGPT_SCENES = {"S1": "wrist", "S2": "surface", "S3": "box_papers", "S4": "closeup", "S5": "hand"}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _binom_tail(k: int, n: int, p: float) -> float:
    return sum(math.comb(n, i) * p**i * (1 - p) ** (n - i) for i in range(k, n + 1))


def lower_one_sided(k: int, n: int, alpha: float = 0.05) -> float:
    """One-sided Clopper-Pearson lower bound: the smallest rate still consistent with k of n."""
    if k == 0:
        return 0.0
    lo, hi = 0.0, k / n
    for _ in range(60):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if _binom_tail(k, n, mid) < alpha else (lo, mid)
    return lo


def rate(k: int, n: int) -> dict:
    lo, hi = clopper_pearson(k, n) if n else (None, None)
    return {"k": k, "n": n, "rate": round(k / n, 4) if n else None, "ci95": [round(lo, 4), round(hi, 4)] if n else None,
            "lower95_one_sided": round(lower_one_sided(k, n), 4) if n else None}


def scene_of(s) -> str:
    meta = s.meta or {}
    if meta.get("scene_type"):
        return meta["scene_type"]
    return CHATGPT_SCENES.get(meta.get("name", "").rsplit("_", 1)[-1], "unknown")


def main() -> None:
    import joblib
    import numpy as np
    from PIL import Image

    import frontier_v5 as f5
    import kandinsky_v4 as k4
    from config import settings
    from inference.live_screening import _dinov3_probability, verdict_from_probability
    from inference.local_torch import classify_image_torch
    from inference.triton_client import preprocess_chw
    from inference.verdict import apply_min_authentic_confidence
    from kandinsky_v4_design import view

    if RESULTS.exists() and any(RESULTS.iterdir()):
        raise SystemExit("benchmark_v5_frontier has already been scored; it runs once")
    p74, p75 = json.loads(P74_RULE.read_text()), json.loads(P75_RULE.read_text())
    if sha256(REPO / p74["model"]["path"]) != p74["model"]["sha256"]:
        raise SystemExit("Phase 74 head changed after it was frozen")
    head75 = f5.FrozenHead()  # checks the Phase 75 head against its recorded SHA-256
    protocol_sha = sha256(PROTOCOL)

    samples = [s for s in read_manifest(MANIFEST) if s.excluded is None]
    for s in samples:
        s.path = str(Path(s.path) if Path(s.path).is_absolute() else HERE / s.path)
    groups = defaultdict(int)
    for s in samples:
        groups[s.group] += 1
    print("samples", dict(groups), flush=True)

    # Features once (DINOv3 pretrained + fine-tuned) on the stored normalised images, for both heads.
    images_all = [(s.sample_id, s.path) for s in samples] + [("motivating_testtest", None)]
    feats: dict[str, list] = defaultdict(list)
    for start in range(0, len(images_all), k4.BATCH):
        chunk = images_all[start:start + k4.BATCH]
        ims = [k4.stored(p) if p else k4.jpeg90(k4.square(MOTIVATING, 512)) for _sid, p in chunk]
        for d in (head75.pre.features(ims), head75.ft.features(ims)):
            for k, v in d.items():
                feats[k].append(v)
    feats = {k: np.concatenate(v) for k, v in feats.items()}
    s74 = joblib.load(REPO / p74["model"]["path"]).decision_function(view(feats, p74["features"]))
    s75 = head75.model.decision_function(view(feats, p75["features"]))
    ids = [sid for sid, _ in images_all]

    async def dinov2(pil):
        verdict, conf = await classify_image_torch(preprocess_chw(np.array(pil)))
        return apply_min_authentic_confidence(verdict, conf, settings.inference_min_authentic_confidence)

    rows = []
    for i, (sid, path) in enumerate(images_all):
        if path:
            with Image.open(path) as im:
                pil = im.convert("RGB")
        else:
            pil = k4.jpeg90(k4.square(MOTIVATING, 512))
        d2_verdict, d2_conf = asyncio.run(dinov2(pil))
        p_fake = _dinov3_probability(pil)
        d3_verdict, _ = verdict_from_probability(p_fake)
        rows.append({
            "id": sid,
            "dinov2_live": d2_verdict == "FAKE", "dinov2_confidence": d2_conf,
            "dinov3_alone": d3_verdict == "FAKE", "dinov3_p_fake": round(p_fake, 6),
            "phase74_head": bool(s74[i] >= p74["thresholds"]["t_fake"]), "phase74_score": round(float(s74[i]), 4),
            "phase75_head": bool(s75[i] >= p75["t_fake"]), "phase75_score": round(float(s75[i]), 4),
        })
        if i % 50 == 0:
            print(f"  scored {i + 1}/{len(images_all)}", flush=True)
    by_id = {r["id"]: r for r in rows}

    def summarise(subset) -> dict:
        return {sys_: rate(sum(by_id[s.sample_id][sys_] for s in subset), len(subset)) for sys_ in SYSTEMS}

    report: dict = {
        "benchmark": "benchmark_v5_frontier", "phase": 75,
        "scored_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "protocol_sha256_at_scoring": protocol_sha, "phase75_rule_sha256": sha256(P75_RULE),
        "phase75_head_sha256": p75["model"]["sha256"], "phase74_head_sha256": p74["model"]["sha256"],
        "systems": {"dinov2_live": "LEGACY_DINOV2 local torch, FAKE or authentic confidence < 0.88",
                    "dinov3_alone": "DINOv3 epoch_018, calibrated P(fake) >= 0.5",
                    "phase74_head": f"Phase 74 head, score >= {p74['thresholds']['t_fake']:.3f}",
                    "phase75_head": f"Phase 75 head ({p75['selected_candidate']}), score >= {p75['t_fake']:.3f}"},
        "note": "For generated sets the rate is the share caught (called FAKE); for real_ebay_v5 it is the share wrongly called FAKE.",
        "groups": {},
    }
    for group in sorted({s.group for s in samples}):
        sub = [s for s in samples if s.group == group]
        g = {"all": summarise(sub), "by_brand": {}, "by_scene_type": {}}
        for brand in sorted({s.brand for s in sub}):
            g["by_brand"][brand] = summarise([s for s in sub if s.brand == brand])
        if group != "real_ebay_v5":
            for scene in sorted({scene_of(s) for s in sub}):
                g["by_scene_type"][scene] = summarise([s for s in sub if scene_of(s) == scene])
        if group == "gemini_v5":
            num = lambda s: (s.meta or {}).get("prompt_number") or 0  # noqa: E731
            g["prompts_1_36"] = summarise([s for s in sub if num(s) <= 36])
            g["prompts_37_75"] = summarise([s for s in sub if num(s) >= 37])
            g["confident_names_only"] = summarise([s for s in sub if (s.meta or {}).get("name_confidence") != "low"])
        report["groups"][group] = g

    p75_res = {g: report["groups"][g]["all"]["phase75_head"] for g in report["groups"]}
    report["success_criterion"] = {
        "rule": "Phase 75 head caught >= 80% on chatgpt_v5 AND on gemini_v5, and real_ebay_v5 FAKE <= 3%",
        "chatgpt_v5_caught": p75_res.get("chatgpt_v5"), "gemini_v5_caught": p75_res.get("gemini_v5"),
        "real_ebay_v5_fake": p75_res.get("real_ebay_v5"),
        "met": bool(p75_res["chatgpt_v5"]["rate"] >= 0.80 and p75_res["gemini_v5"]["rate"] >= 0.80
                    and p75_res["real_ebay_v5"]["rate"] <= 0.03),
    }
    report["motivating_case_testtest_jpeg"] = by_id["motivating_testtest"]
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "scores.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (RESULTS / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({g: {k: v["rate"] for k, v in report["groups"][g]["all"].items()} for g in report["groups"]}, indent=1))
    print("success criterion met:", report["success_criterion"]["met"])
    print("motivating case:", report["motivating_case_testtest_jpeg"])


if __name__ == "__main__":
    main()
