"""Score benchmark_v2 once with every system named in protocol.json, including the frozen fusion rule.

Needs Triton serving dinov2_vitb14_live and dinov3_authenticity_candidate (as for benchmark_v1).

    HYPEVAULT_DEPLOYMENT_MODE=research .venv/bin/python ml_rtx5080/benchmark_v2_fresh/score.py
"""

from __future__ import annotations

import asyncio
import csv
import hashlib
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
ML = HERE.parent
V1 = ML / "benchmark_v1_real_vs_generated"
for p in (str(V1), str(ML)):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np  # noqa: E402

from bench_stats import mcnemar_exact  # noqa: E402
from common import read_manifest  # noqa: E402
import run_benchmark as v1  # noqa: E402
from unseen_generator import clopper_pearson  # noqa: E402

PROTOCOL = HERE / "protocol.json"
PROTOCOL_SHA256 = "bd1a44877dbac56344b59072fc9f203f2e583de97d093ba82caa6951886ed167"
RULE = ML / "experiments" / "watch_forensics_v1" / "fusion_rule.json"
RULE_SHA256 = "9f37f3286169ed22180dbe82ae0ae09e7b1160dbca0a94cf14994e8a5cda6d5b"
PROBE = ML / "experiments" / "generator_agnostic_v1" / "dinov3_vitb16_linear.joblib"
MANIFEST = HERE / "data" / "manifest.jsonl"
RESULTS = HERE / "results"
SYSTEMS = ("dinov3_alone", "dinov2_alone", "phase71_probe", "watch_forensics")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def brand_text(brand: str, rapid: list) -> bool:
    import watch_forensics as wf

    words = {w for t, s in rapid if s >= 0.6 for w in wf.words(t)}
    return any(o in w for w in words for o in wf.BRAND_WORDS[brand])


def fuse(rule: dict, dinov3: str, p71: float, has_brand_text: bool) -> str:
    t = rule["thresholds"]
    if dinov3 == "FAKE" or p71 >= t["T71"] or (not has_brand_text and p71 >= t["t_low"]):
        return "FAKE"
    return "REVIEW" if dinov3 == "REVIEW" else "AUTHENTIC"


def rate(k: int, n: int) -> dict:
    lo, hi = clopper_pearson(k, n)
    return {"k": k, "n": n, "rate": k / n if n else None, "ci95": [lo, hi]}


def analyse(rows: list[dict]) -> dict:
    out = {}
    groups = sorted({r["group"] for r in rows if r["label"] == "GENERATED"})
    for system in SYSTEMS:
        block = {}
        real = [r[f"{system}_decision"] for r in rows if r["label"] == "REAL_PHOTO"]
        block["real_rejected"] = rate(real.count("FAKE"), len(real))
        block["real_review"] = real.count("REVIEW")
        for group in groups + ["pooled_generated"]:
            gen = [r[f"{system}_decision"] for r in rows if r["label"] == "GENERATED" and (group == "pooled_generated" or r["group"] == group)]
            block[group] = {**rate(gen.count("FAKE"), len(gen)), "review": gen.count("REVIEW")}
        out[system] = block
    gen = [r for r in rows if r["label"] == "GENERATED"]
    binary = [r for r in gen if r["watch_forensics_decision"] != "REVIEW" and r["dinov3_alone_decision"] != "REVIEW"]
    only_fusion = sum(r["watch_forensics_decision"] == "FAKE" and r["dinov3_alone_decision"] != "FAKE" for r in binary)
    only_dinov3 = sum(r["dinov3_alone_decision"] == "FAKE" and r["watch_forensics_decision"] != "FAKE" for r in binary)
    out["paired_generated"] = {"n_binary": len(binary), "fusion_only_caught": only_fusion, "dinov3_only_caught": only_dinov3,
                               "mcnemar_p": mcnemar_exact(only_fusion, only_dinov3)}
    fusion, dinov3 = out["watch_forensics"], out["dinov3_alone"]
    out["success"] = {
        "more_caught_and_significant": fusion["pooled_generated"]["k"] > dinov3["pooled_generated"]["k"] and out["paired_generated"]["mcnemar_p"] < 0.05,
        "real_rejection_within_3_points": fusion["real_rejected"]["rate"] - dinov3["real_rejected"]["rate"] <= 0.03,
    }
    out["success"]["met"] = all(out["success"].values())
    return out


async def score_models(samples) -> dict:
    from inference.checkpoint_identity import verify_frozen_checkpoint

    v1.verify_checkpoints(v1.load_protocol())
    for m in v1.MODELS:
        ready = await v1.model_readiness(m)
        if not ready.get("ready"):
            raise SystemExit(f"{m} is not served: {ready}")
    digest = verify_frozen_checkpoint()
    out = {}
    for i, s in enumerate(samples, 1):
        out[s.sample_id] = await v1.score(s, digest)
        if i % 100 == 0:
            print(f"models: {i}/{len(samples)}", flush=True)
    return out


def score_probe(samples) -> dict:
    import joblib

    import generator_agnostic as ga

    probe = joblib.load(PROBE)
    embed = ga.Embedders()
    out = {}
    for start in range(0, len(samples), ga.BATCH):
        chunk = samples[start:start + ga.BATCH]
        feats = embed([ga.clean_view(ga.base_image(s.path)) for s in chunk])["dinov3_vitb16_linear"]
        out.update(zip((s.sample_id for s in chunk), probe.predict_proba(feats)[:, 1]))
    return out


def read_text(samples) -> dict:
    from rapidocr import RapidOCR

    ocr = RapidOCR()
    out = {}
    for s in samples:
        res = ocr(s.path)
        out[s.sample_id] = [[t, float(c)] for t, c in zip(res.txts or (), res.scores or ())]
    return out


def main() -> None:
    if sha256(PROTOCOL) != PROTOCOL_SHA256 or sha256(RULE) != RULE_SHA256:
        raise SystemExit("protocol.json or fusion_rule.json changed after being frozen; stopping")
    if RESULTS.exists() and any(RESULTS.iterdir()):
        raise SystemExit("benchmark_v2 has already been scored; it runs once")
    rule = json.loads(RULE.read_text())
    samples = [s for s in read_manifest(MANIFEST) if s.excluded is None]
    print("samples", Counter(s.group for s in samples), flush=True)
    models = asyncio.run(score_models(samples))
    probe = score_probe(samples)
    text = read_text(samples)
    rows = []
    for s in samples:
        m = models[s.sample_id]
        has_text = brand_text(s.brand, text[s.sample_id])
        p71 = float(probe[s.sample_id])
        rows.append({
            "sample_id": s.sample_id, "label": s.label, "group": s.group, "brand": s.brand,
            "dinov3_alone_decision": m["dinov3_experimental_decision"], "dinov3_score_fake": m["dinov3_experimental_score_fake"],
            "dinov2_alone_decision": m["dinov2_legacy_decision"], "dinov2_score_fake": m["dinov2_legacy_score_fake"],
            "p71": p71, "phase71_probe_decision": "FAKE" if p71 >= rule["thresholds"]["T71"] else "AUTHENTIC",
            "brand_text": has_text, "rapidocr": json.dumps(text[s.sample_id]),
            "watch_forensics_decision": fuse(rule, m["dinov3_experimental_decision"], p71, has_text),
        })
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = RESULTS / run_id
    out.mkdir(parents=True)
    result = {"run_id": run_id, "protocol_sha256": PROTOCOL_SHA256, "fusion_rule_sha256": RULE_SHA256,
              "counts": dict(Counter(f"{s.label}|{s.group}" for s in samples)), "analysis": analyse(rows)}
    (out / "metrics.json").write_text(json.dumps(result, indent=2, default=float) + "\n")
    with open(out / "predictions.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(result["analysis"], indent=1, default=float))


if __name__ == "__main__":
    main()
