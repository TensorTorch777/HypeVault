"""Phase 73 fixes 1, 2 and 5: learned combiner, thresholds from real photos only, and a REVIEW tier.

Development data only (benchmark_v1, benchmark_v2, Label_0 validation). Writes the frozen rule to
experiments/watch_forensics_v2/fusion_rule.json before benchmark_v3 is scored.

    .venv/bin/python ml_rtx5080/watch_forensics_v2_design.py
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

_PKG = Path(__file__).resolve().parent
if str(_PKG) not in sys.path:
    sys.path.insert(0, str(_PKG))

import watch_forensics_v2 as wf2  # noqa: E402

OUT = _PKG / "experiments" / "watch_forensics_v2"
GA1 = _PKG / "experiments" / "generator_agnostic_v1"
GA2 = _PKG / "experiments" / "generator_agnostic_v2"
PROBE = GA2 / "dinov3_vitb16_linear_5dec.joblib"
V1_PRED = _PKG / "benchmark_v1_real_vs_generated" / "results" / "20261020T110244Z" / "predictions.csv"
V2_PRED = next((_PKG / "benchmark_v2_fresh" / "results").iterdir()) / "predictions.csv"
FAKE_BUDGET = 0.01    # fix 1: at most 1% of real photos called FAKE
REVIEW_BUDGET = 0.05  # fix 2: at most 5% more real photos sent to REVIEW by the score
FEATURES = ("dinov3_logit", "dinov3_review", "probe_logit", "brand_any", "brand_readers")


def logit(p: float) -> float:
    p = min(max(float(p), 1e-6), 1 - 1e-6)
    return math.log(p / (1 - p))


def features(dinov3_score: float, dinov3_decision: str, probe: float, hits: dict) -> list[float]:
    n = sum(hits.values())
    return [logit(dinov3_score), float(dinov3_decision == "REVIEW"), logit(probe), float(n > 0), n / len(hits)]


def probe_scores() -> dict[str, float]:
    import joblib

    model = joblib.load(PROBE)
    out = {}
    for folder, names in ((GA1, ("bench_real_ebay", "bench_heldout_kandinsky22", "bench_sdxl_base10", "val_real")),
                          (GA2, ("bench2_real_ebay_v2", "bench2_kandinsky22_fresh", "bench2_amused512"))):
        for name in names:
            z = np.load(folder / "features" / f"{name}__real_clean.npz")
            out.update(zip(z["ids"].tolist(), model.predict_proba(z["dinov3_vitb16_linear"])[:, 1].tolist()))
    return out


def dev_rows() -> list[dict]:
    probe = probe_scores()
    models = {}
    for path, d3, s3 in ((V1_PRED, "dinov3_experimental_decision", "dinov3_experimental_score_fake"),
                         (V2_PRED, "dinov3_alone_decision", "dinov3_score_fake")):
        for r in csv.DictReader(path.open()):
            models[r["sample_id"]] = (r[d3], float(r[s3]))
    for r in csv.DictReader((OUT / "label0_val_models.csv").open()):
        models[r["sample_id"]] = (r["dinov3_decision"], float(r["dinov3_score_fake"]))
    v2_extra = {r["sample_id"]: (float(r["p71"]), r["brand_text"] == "True") for r in csv.DictReader(V2_PRED.open())}
    rows = []
    for set_name in ("bench_v1", "bench_v2", "val_label0"):
        for r in wf2.load_readings(set_name):
            decision, score = models[r["id"]]
            hits = wf2.brand_hits(r)
            rows.append({"id": r["id"], "set": set_name, "group": r["group"], "label": r["label"], "dinov3": decision,
                         "x": features(score, decision, probe[r["id"]], hits), "probe": probe[r["id"]], "hits": hits})
    return rows


def fit_combiner(x: np.ndarray, y: np.ndarray):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, class_weight="balanced", max_iter=5000)).fit(x, y)


def decide(score: float, dinov3: str, t_fake: float, t_review: float) -> str:
    if score >= t_fake:
        return "FAKE"
    if score >= t_review or dinov3 == "REVIEW":
        return "REVIEW"
    return "AUTHENTIC"


def main() -> None:
    from sklearn.model_selection import StratifiedKFold

    rows = dev_rows()
    bench = [r for r in rows if r["set"] != "val_label0"]
    label0 = [r for r in rows if r["set"] == "val_label0"]
    x = np.array([r["x"] for r in bench])
    y = np.array([r["label"] for r in bench])
    # Out-of-fold combiner scores on the benchmark rows (5 folds, stratified by group).
    oof = np.zeros(len(bench))
    for train, test in StratifiedKFold(5, shuffle=True, random_state=73).split(x, [r["group"] for r in bench]):
        oof[test] = fit_combiner(x[train], y[train]).predict_proba(x[test])[:, 1]
    final = fit_combiner(x, y)
    label0_scores = final.predict_proba(np.array([r["x"] for r in label0]))[:, 1]
    # Thresholds from real photos only: 600 eBay (out-of-fold) + 1,000 Label_0 validation.
    real = np.concatenate([oof[y == 0], label0_scores])
    grid = np.unique(np.round(real, 6))[::-1]
    t_fake = min(t for t in grid if (real >= t).mean() <= FAKE_BUDGET)
    t_review = min(t for t in grid if t <= t_fake and ((real >= t) & (real < t_fake)).mean() <= REVIEW_BUDGET)
    for r, s in zip(bench, oof):
        r["score"] = s
    for r, s in zip(label0, label0_scores):
        r["score"] = s
    table = defaultdict(Counter)
    for r in rows:
        table[(r["group"], "dinov3_alone")][r["dinov3"] if r["dinov3"] != "ERROR" else "ERROR"] += 1
        table[(r["group"], "watch_forensics_v2")][decide(r["score"], r["dinov3"], t_fake, t_review)] += 1
    clf = final.named_steps["logisticregression"]
    scaler = final.named_steps["standardscaler"]
    rule = {
        "rule": "watch_forensics_v2",
        "phase": 73,
        "frozen_before_any_v3_scoring": True,
        "inputs": {
            "dinov3_logit": "logit of fine-tuned DINOv3 calibrated P(fake) (T = 0.2404), benchmark serving path",
            "dinov3_review": "1 if the DINOv3 benchmark policy returned REVIEW",
            "probe_logit": "logit of the Phase 73 five-decoder probe P(fake)",
            "brand_any": "1 if any of the four literal readers read the declared brand's word",
            "brand_readers": "fraction of the four readers (rapid_full, rapid_crop, doctr_full, doctr_crop) that read it",
        },
        "probe": {"path": str(PROBE.relative_to(_PKG.parent)), "sha256": hashlib.sha256(PROBE.read_bytes()).hexdigest()},
        "readers": {"rapidocr": "3.10.0, PP-OCRv6 det", "doctr": "1.1.0, db_mobilenet_v3_large + crnn_mobilenet_v3_small",
                    "crop": f"central {wf2.CROP_FRACTION:.0%} upscaled to {wf2.CROP_SIDE}px", "min_confidence": wf2.MIN_CONFIDENCE},
        "combiner": {"features": FEATURES, "scaler_mean": scaler.mean_.tolist(), "scaler_scale": scaler.scale_.tolist(),
                     "coef": clf.coef_[0].tolist(), "intercept": float(clf.intercept_[0]),
                     "trained_on": "benchmark_v1 + benchmark_v2 (600 real, 1,014 generated)"},
        "thresholds": {"t_fake": float(t_fake), "t_review": float(t_review),
                       "source": "real photos only: 600 eBay (out-of-fold) + 1,000 Label_0 validation; at most 1% FAKE and at most 5% more REVIEW"},
        "decision": "FAKE if score >= t_fake; REVIEW if score >= t_review or DINOv3 REVIEW; else AUTHENTIC",
        "development_results": {f"{g}|{s}": dict(c) for (g, s), c in sorted(table.items())},
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "fusion_rule.json").write_text(json.dumps(rule, indent=2) + "\n")
    print(json.dumps({"t_fake": t_fake, "t_review": t_review, "coef": dict(zip(FEATURES, clf.coef_[0].round(3).tolist()))}))
    for k, v in sorted(table.items()):
        print(k, dict(v))
    print("fusion_rule.json sha256", hashlib.sha256((OUT / "fusion_rule.json").read_bytes()).hexdigest())


if __name__ == "__main__":
    main()
