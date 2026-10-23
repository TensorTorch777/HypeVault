"""Score benchmark_v4 once with every system named in protocol.json.

DINOv3 runs in process from the hash-checked FP32 checkpoint (same logits as the Triton path; on
benchmark_v3 the two differ by at most 0.0002 in P(fake) and agree on every decision).

    HYPEVAULT_DEPLOYMENT_MODE=research .venv/bin/python ml_rtx5080/benchmark_v4_fresh/score.py
"""

from __future__ import annotations

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
REPO = ML.parent
for p in (str(ML / "benchmark_v1_real_vs_generated"), str(ML), str(REPO / "backend")):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np  # noqa: E402

from bench_stats import mcnemar_exact  # noqa: E402
from common import read_manifest  # noqa: E402
from unseen_generator import clopper_pearson  # noqa: E402

PROTOCOL = HERE / "protocol.json"
PROTOCOL_SHA256 = "a313d3403d22c566a3f9f7c0fab3471ce9b9508b27b88e87c04dd01c5a2b2a48"
RULE_V2 = ML / "experiments" / "watch_forensics_v2" / "fusion_rule.json"
RULE_V2_SHA256 = "2bb10caffd342f51d8931743cf3501361c1fb5fc94eb65b2dce7ef4dec83066e"
RULE_V4 = ML / "experiments" / "kandinsky_v4" / "fusion_rule.json"
RULE_V4_SHA256 = "d1201673aa2588d32b03d463c88d7dcccb20288bb8211ad6cea02a488044083f"
MANIFEST = HERE / "data" / "manifest.jsonl"
RESULTS = HERE / "results"
PRIMARY = "kandinsky22_v4"
SYSTEMS = ("dinov3_alone", "watch_forensics_v2", "dinov3_kandinsky_v4")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def logit(p: float) -> float:
    p = min(max(float(p), 1e-6), 1 - 1e-6)
    return math.log(p / (1 - p))


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
        for scenes, new in (("T1-T8", False), ("T9-T16", True)):
            gen = [r[f"{system}_decision"] for r in rows if r["group"] == PRIMARY and r["new_scene"] == new]
            block[f"{PRIMARY}_{scenes}"] = {**rate(gen.count("FAKE"), len(gen)), "review": gen.count("REVIEW")}
        out[system] = block
    gen = [r for r in rows if r["group"] == PRIMARY]
    binary = [r for r in gen if r["dinov3_kandinsky_v4_decision"] != "REVIEW" and r["dinov3_alone_decision"] != "REVIEW"]
    b = sum(r["dinov3_kandinsky_v4_decision"] == "FAKE" and r["dinov3_alone_decision"] != "FAKE" for r in binary)
    c = sum(r["dinov3_alone_decision"] == "FAKE" and r["dinov3_kandinsky_v4_decision"] != "FAKE" for r in binary)
    out["paired_primary"] = {"n_binary": len(binary), "v4_only_caught": b, "dinov3_only_caught": c, "mcnemar_p": mcnemar_exact(b, c)}
    v4 = out["dinov3_kandinsky_v4"]
    out["success"] = {
        "kandinsky_caught_at_least_80pct": v4[PRIMARY]["rate"] >= 0.80,
        "real_fake_at_most_3pct": v4["real_rejected"]["rate"] <= 0.03,
    }
    out["success"]["met"] = all(out["success"].values())
    return out


def v4_rows(samples) -> list[dict]:
    return [{"id": s.sample_id, "path": s.path, "brand": s.brand, "group": s.group, "label": int(s.label == "GENERATED")}
            for s in samples]


def main() -> None:
    import joblib
    import torch
    from PIL import Image

    import kandinsky_v4 as k4
    from inference.shadow import _policy_batch
    from kandinsky_v4_design import view

    sys.path.insert(0, str(ML / "benchmark_v3_fresh"))
    import importlib.util

    spec = importlib.util.spec_from_file_location("score_v3", ML / "benchmark_v3_fresh" / "score.py")
    v3 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(v3)
    import watch_forensics_v2 as wf2

    if sha256(PROTOCOL) != PROTOCOL_SHA256 or sha256(RULE_V2) != RULE_V2_SHA256 or sha256(RULE_V4) != RULE_V4_SHA256:
        raise SystemExit("protocol or a frozen rule changed; stopping")
    if RESULTS.exists() and any(RESULTS.iterdir()):
        raise SystemExit("benchmark_v4 has already been scored; it runs once")
    rule_v2, rule_v4 = json.loads(RULE_V2.read_text()), json.loads(RULE_V4.read_text())
    model_path = REPO / rule_v4["model"]["path"]
    if sha256(model_path) != rule_v4["model"]["sha256"]:
        raise SystemExit("Phase 74 model changed after the rule was frozen")
    model = joblib.load(model_path)

    samples = [s for s in read_manifest(MANIFEST) if s.excluded is None]
    for s in samples:
        s.path = str(Path(s.path) if Path(s.path).is_absolute() else HERE / s.path)
    print("samples", Counter(s.group for s in samples), flush=True)
    new_scene = {s.sample_id: bool((s.meta or {}).get("new_scene", False)) for s in samples}

    # Phase 74 features (DINOv3 pretrained + fine-tuned, MoVQ distances), written to a v4 feature file.
    pre, ft, movq = k4.Pretrained(), k4.FineTuned(), k4.MoVQ()
    k4.extract_set("bench_v4", v4_rows(samples), pre, ft, movq)
    feats = dict(np.load(k4.FEATURES / "bench_v4.npz"))
    index = {i: n for n, i in enumerate(feats["id"].tolist())}
    v4_scores = model.decision_function(view(feats, rule_v4["features"]))
    del pre, ft, movq
    torch.cuda.empty_cache()

    # DINOv3 alone: the frozen benchmark policy on the in-process logit.
    from policy_contract import EXPECTED_CHECKPOINT_SHA256

    d3 = {}
    for s in samples:
        with Image.open(s.path) as im:
            pil = im.convert("RGB")
        lg = float(feats["d3ft_logit"][index[s.sample_id]])
        result = _policy_batch([pil], torch.tensor([lg], dtype=torch.float32), EXPECTED_CHECKPOINT_SHA256)[0]
        d3[s.sample_id] = (result.decision or "ERROR", 1.0 / (1.0 + math.exp(-lg / 0.2404)))

    # watch_forensics_v2, unchanged: Phase 73 probe, four literal readers, learned combiner.
    _, p73 = v3.probe_scores(samples)
    text = v3.readings(samples)
    t2, t4 = rule_v2["thresholds"], rule_v4["thresholds"]
    rows = []
    for s in samples:
        d3_decision, d3_score = d3[s.sample_id]
        hits = wf2.brand_hits(text[s.sample_id])
        n_hits = sum(hits.values())
        x = [logit(d3_score), float(d3_decision == "REVIEW"), logit(p73[s.sample_id]), float(n_hits > 0), n_hits / len(hits)]
        c2 = v3.combiner_score(rule_v2, x)
        v2_decision = "FAKE" if c2 >= t2["t_fake"] else ("REVIEW" if c2 >= t2["t_review"] or d3_decision == "REVIEW" else "AUTHENTIC")
        s4 = float(v4_scores[index[s.sample_id]])
        v4_decision = "FAKE" if s4 >= t4["t_fake"] else ("REVIEW" if s4 >= t4["t_review"] or d3_decision == "REVIEW" else "AUTHENTIC")
        rows.append({
            "sample_id": s.sample_id, "label": s.label, "group": s.group, "brand": s.brand, "new_scene": new_scene[s.sample_id],
            "dinov3_alone_decision": d3_decision, "dinov3_score_fake": d3_score, "p73": float(p73[s.sample_id]),
            "brand_hits": json.dumps(hits), "combiner_v2_score": c2, "watch_forensics_v2_decision": v2_decision,
            "v4_score": s4, "movq_l1_768": float(feats["movq_l1_768"][index[s.sample_id]]),
            "dinov3_kandinsky_v4_decision": v4_decision,
        })
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = RESULTS / run_id
    out.mkdir(parents=True)
    result = {"run_id": run_id, "protocol_sha256": PROTOCOL_SHA256, "fusion_rule_v2_sha256": RULE_V2_SHA256,
              "fusion_rule_v4_sha256": RULE_V4_SHA256, "counts": dict(Counter(f"{s.label}|{s.group}" for s in samples)),
              "analysis": analyse(rows)}
    (out / "metrics.json").write_text(json.dumps(result, indent=2, default=float) + "\n")
    with open(out / "predictions.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(result["analysis"], indent=1, default=float))


if __name__ == "__main__":
    main()
