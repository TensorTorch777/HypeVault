"""Score benchmark_v3 once with every system named in protocol.json.

Needs Triton serving dinov2_vitb14_live and dinov3_authenticity_candidate.

    HYPEVAULT_DEPLOYMENT_MODE=research .venv/bin/python ml_rtx5080/benchmark_v3_fresh/score.py
"""

from __future__ import annotations

import asyncio
import csv
import hashlib
import json
import math
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
ML = HERE.parent
for p in (str(ML / "benchmark_v1_real_vs_generated"), str(ML / "benchmark_v2_fresh"), str(ML)):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np  # noqa: E402

from bench_stats import mcnemar_exact  # noqa: E402
from common import read_manifest  # noqa: E402
from unseen_generator import clopper_pearson  # noqa: E402

PROTOCOL = HERE / "protocol.json"
PROTOCOL_SHA256 = "a2b3cc801a900c503326f7bbe00504b97f29d2de3315eb36a99519b9c53b75ca"
RULE_V1 = ML / "experiments" / "watch_forensics_v1" / "fusion_rule.json"
RULE_V1_SHA256 = "9f37f3286169ed22180dbe82ae0ae09e7b1160dbca0a94cf14994e8a5cda6d5b"
RULE_V2 = ML / "experiments" / "watch_forensics_v2" / "fusion_rule.json"
RULE_V2_SHA256 = "2bb10caffd342f51d8931743cf3501361c1fb5fc94eb65b2dce7ef4dec83066e"
PROBE_V1 = ML / "experiments" / "generator_agnostic_v1" / "dinov3_vitb16_linear.joblib"
MANIFEST = HERE / "data" / "manifest.jsonl"
RESULTS = HERE / "results"
PRIMARY = "stable_cascade_lite"
SYSTEMS = ("dinov3_alone", "dinov2_alone", "watch_forensics_v1", "watch_forensics_v2")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def logit(p: float) -> float:
    p = min(max(float(p), 1e-6), 1 - 1e-6)
    return math.log(p / (1 - p))


def combiner_score(rule: dict, x: list[float]) -> float:
    c = rule["combiner"]
    z = sum(w * (v - m) / s for w, v, m, s in zip(c["coef"], x, c["scaler_mean"], c["scaler_scale"])) + c["intercept"]
    return 1 / (1 + math.exp(-z))


def rate(k: int, n: int) -> dict:
    lo, hi = clopper_pearson(k, n)
    return {"k": k, "n": n, "rate": k / n if n else None, "ci95": [lo, hi]}


def analyse(rows: list[dict]) -> dict:
    out = {}
    groups = sorted({r["group"] for r in rows if r["label"] == "GENERATED"})
    for system in SYSTEMS:
        real = [r[f"{system}_decision"] for r in rows if r["label"] == "REAL_PHOTO"]
        block = {"real_rejected": rate(real.count("FAKE"), len(real)), "real_review": real.count("REVIEW")}
        for group in groups:
            gen = [r[f"{system}_decision"] for r in rows if r["group"] == group]
            block[group] = {**rate(gen.count("FAKE"), len(gen)), "review": gen.count("REVIEW")}
        out[system] = block
    gen = [r for r in rows if r["group"] == PRIMARY]
    binary = [r for r in gen if r["watch_forensics_v2_decision"] != "REVIEW" and r["dinov3_alone_decision"] != "REVIEW"]
    b = sum(r["watch_forensics_v2_decision"] == "FAKE" and r["dinov3_alone_decision"] != "FAKE" for r in binary)
    c = sum(r["dinov3_alone_decision"] == "FAKE" and r["watch_forensics_v2_decision"] != "FAKE" for r in binary)
    out["paired_primary"] = {"n_binary": len(binary), "v2_only_caught": b, "dinov3_only_caught": c, "mcnemar_p": mcnemar_exact(b, c)}
    v2, d3 = out["watch_forensics_v2"], out["dinov3_alone"]
    out["success"] = {
        "more_caught_and_significant": v2[PRIMARY]["k"] > d3[PRIMARY]["k"] and out["paired_primary"]["mcnemar_p"] < 0.05,
        "real_rejection_within_3_points": v2["real_rejected"]["rate"] - d3["real_rejected"]["rate"] <= 0.03,
    }
    out["success"]["met"] = all(out["success"].values())
    return out


def probe_scores(samples) -> tuple[dict, dict]:
    """Phase 71 probe (for fusion v1) and the Phase 73 five-decoder probe (for v2), same features."""
    import joblib

    import generator_agnostic as ga
    from generator_agnostic_v2 import DinoOnly

    rule = json.loads(RULE_V2.read_text())
    probe_v2_path = ML.parent / rule["probe"]["path"]
    if sha256(probe_v2_path) != rule["probe"]["sha256"]:
        raise SystemExit("Phase 73 probe changed after the rule was frozen")
    p1, p2 = joblib.load(PROBE_V1), joblib.load(probe_v2_path)
    embed = DinoOnly()
    out1, out2 = {}, {}
    for start in range(0, len(samples), ga.BATCH):
        chunk = samples[start:start + ga.BATCH]
        feats = embed([ga.clean_view(ga.base_image(s.path)) for s in chunk])["dinov3_vitb16_linear"]
        ids = [s.sample_id for s in chunk]
        out1.update(zip(ids, p1.predict_proba(feats)[:, 1]))
        out2.update(zip(ids, p2.predict_proba(feats)[:, 1]))
    return out1, out2


def readings(samples) -> dict:
    from doctr.models import ocr_predictor
    from rapidocr import RapidOCR

    import watch_forensics_v2 as wf2

    rapid = RapidOCR()
    doctr = ocr_predictor(det_arch="db_mobilenet_v3_large", reco_arch="crnn_mobilenet_v3_small", pretrained=True).cuda()
    out = {}
    for s in samples:
        full, crop = wf2.views(s.path)
        r_full, r_crop = rapid(full), rapid(crop)
        out[s.sample_id] = {
            "brand": s.brand,
            "rapid_full": [[t, float(c)] for t, c in zip(r_full.txts or (), r_full.scores or ())],
            "rapid_crop": [[t, float(c)] for t, c in zip(r_crop.txts or (), r_crop.scores or ())],
            "doctr_full": wf2.doctr_lines(doctr, full),
            "doctr_crop": wf2.doctr_lines(doctr, crop),
        }
    return out


def main() -> None:
    import score as v2score  # benchmark_v2's scorer: Triton models and fusion A

    import watch_forensics_v2 as wf2

    if sha256(PROTOCOL) != PROTOCOL_SHA256 or sha256(RULE_V1) != RULE_V1_SHA256 or sha256(RULE_V2) != RULE_V2_SHA256:
        raise SystemExit("protocol or a frozen rule changed; stopping")
    if RESULTS.exists() and any(RESULTS.iterdir()):
        raise SystemExit("benchmark_v3 has already been scored; it runs once")
    rule_v1, rule_v2 = json.loads(RULE_V1.read_text()), json.loads(RULE_V2.read_text())
    samples = [s for s in read_manifest(MANIFEST) if s.excluded is None]
    print("samples", Counter(s.group for s in samples), flush=True)
    models = asyncio.run(v2score.score_models(samples))
    p71, p73 = probe_scores(samples)
    text = readings(samples)
    rows = []
    t = rule_v2["thresholds"]
    for s in samples:
        m = models[s.sample_id]
        d3, d3_score = m["dinov3_experimental_decision"], m["dinov3_experimental_score_fake"]
        hits = wf2.brand_hits(text[s.sample_id])
        n_hits = sum(hits.values())
        x = [logit(d3_score), float(d3 == "REVIEW"), logit(p73[s.sample_id]), float(n_hits > 0), n_hits / len(hits)]
        score = combiner_score(rule_v2, x)
        v2_decision = "FAKE" if score >= t["t_fake"] else ("REVIEW" if score >= t["t_review"] or d3 == "REVIEW" else "AUTHENTIC")
        rows.append({
            "sample_id": s.sample_id, "label": s.label, "group": s.group, "brand": s.brand,
            "dinov3_alone_decision": d3, "dinov3_score_fake": d3_score, "dinov2_alone_decision": m["dinov2_legacy_decision"],
            "p71": float(p71[s.sample_id]), "p73": float(p73[s.sample_id]), "brand_hits": json.dumps(hits),
            "watch_forensics_v1_decision": v2score.fuse(rule_v1, d3, float(p71[s.sample_id]), hits["rapid_full"]),
            "combiner_score": score, "watch_forensics_v2_decision": v2_decision,
        })
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = RESULTS / run_id
    out.mkdir(parents=True)
    result = {"run_id": run_id, "protocol_sha256": PROTOCOL_SHA256, "fusion_rule_v1_sha256": RULE_V1_SHA256,
              "fusion_rule_v2_sha256": RULE_V2_SHA256, "counts": dict(Counter(f"{s.label}|{s.group}" for s in samples)),
              "analysis": analyse(rows)}
    (out / "metrics.json").write_text(json.dumps(result, indent=2, default=float) + "\n")
    with open(out / "predictions.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(result["analysis"], indent=1, default=float))


if __name__ == "__main__":
    main()
