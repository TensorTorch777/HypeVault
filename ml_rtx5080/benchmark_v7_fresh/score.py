"""Score benchmark_v7_fresh once: the frozen Phase 76 and Phase 77 heads on the same normalised images.

    .venv/bin/python ml_rtx5080/benchmark_v7_fresh/score.py
"""

from __future__ import annotations

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
for p in (str(ML / "benchmark_v1_real_vs_generated"), str(ML)):
    if p not in sys.path:
        sys.path.insert(0, p)

from common import read_manifest  # noqa: E402
from unseen_generator import clopper_pearson  # noqa: E402

PROTOCOL = HERE / "protocol.json"
LOCK = HERE / "lock.json"
MANIFEST = HERE / "data" / "manifest.jsonl"
RESULTS = HERE / "results"
SYSTEMS = ("phase76", "phase77")
PRIMARY_GEN = "unseen_both"
PRIMARY_REAL = "real_ebay_v7"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tail(k: int, n: int, p: float) -> float:
    return sum(math.comb(n, i) * p**i * (1 - p) ** (n - i) for i in range(k, n + 1))


def rate(k: int, n: int) -> dict:
    lo, hi = clopper_pearson(k, n) if n else (None, None)
    return {"k": k, "n": n, "rate": round(k / n, 4) if n else None, "ci95": [round(lo, 4), round(hi, 4)] if n else None}


def mcnemar(rows: list[dict], better: str, worse: str) -> dict:
    b = sum(r[better] and not r[worse] for r in rows)
    c = sum(r[worse] and not r[better] for r in rows)
    return {"only_" + better: b, "only_" + worse: c, "p_one_sided": round(_tail(b, b + c, 0.5), 6) if b + c else 1.0}


def auc(pos: list[float], neg: list[float]) -> float | None:
    if not pos or not neg:
        return None
    wins = sum((p > q) + 0.5 * (p == q) for p in pos for q in neg)
    return round(wins / (len(pos) * len(neg)), 4)


def main() -> None:
    import kandinsky_v4 as k4
    import phase76
    import phase77

    if RESULTS.exists() and any(RESULTS.iterdir()):
        raise SystemExit("benchmark_v7_fresh has already been scored; it runs once")
    protocol = json.loads(PROTOCOL.read_text())
    lock = json.loads(LOCK.read_text())
    if sha256(phase76.RULE) != protocol["systems"]["phase76"]["rule_sha256"]:
        raise SystemExit("Phase 76 rule changed after the protocol was written")
    if sha256(phase77.RULE) != lock["phase77_rule_sha256"]:
        raise SystemExit("Phase 77 rule changed after lock.json was written")
    if sha256(PROTOCOL) != lock["protocol_sha256"]:
        raise SystemExit("protocol.json changed after lock.json was written")

    samples = [s for s in read_manifest(MANIFEST) if s.excluded is None]
    groups = defaultdict(int)
    for s in samples:
        groups[s.group] += 1
    print("samples", dict(groups), flush=True)
    images = [k4.stored(s.path if Path(s.path).is_absolute() else HERE / s.path) for s in samples]

    import torch

    scores, cuts = {}, {}
    for name, make in (("phase76", phase76.FrozenFusionHead), ("phase77", phase77.FrozenPhase77Head)):
        head = make()  # each checks its head's SHA-256
        scores[name], cuts[name] = head.scores_images(images), head.rule["t_fake"]
        print(name, "scored", flush=True)
        del head
        torch.cuda.empty_cache()

    rows = []
    for i, s in enumerate(samples):
        r = {"id": s.sample_id, "group": s.group, "label": s.label, "model": (s.meta or {}).get("model"), "brand": s.brand}
        for n in SYSTEMS:
            r[n + "_score"] = round(scores[n][i], 4)
            r[n] = bool(scores[n][i] >= cuts[n])
        rows.append(r)

    def summarise(sub):
        return {n: rate(sum(r[n] for r in sub), len(sub)) for n in SYSTEMS}

    report = {
        "benchmark": "benchmark_v7_fresh", "phase": 77,
        "scored_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "protocol_sha256": sha256(PROTOCOL), "lock": lock, "cuts": cuts,
        "note": "For generated groups the rate is the share caught (called FAKE); for real groups the share wrongly called FAKE.",
        "groups": {},
    }
    for group in sorted({r["group"] for r in rows}):
        sub = [r for r in rows if r["group"] == group]
        g = {"all": summarise(sub)}
        if sub[0]["label"] == "GENERATED":
            g["by_model"] = {m: summarise([r for r in sub if r["model"] == m]) for m in sorted({r["model"] for r in sub})}
            g["phase77_vs_phase76"] = mcnemar(sub, "phase77", "phase76")
        elif group == PRIMARY_REAL:
            g["by_brand"] = {b: summarise([r for r in sub if r["brand"] == b]) for b in sorted({r["brand"] for r in sub})}
        report["groups"][group] = g
    gen = [r for r in rows if r["group"] == PRIMARY_GEN]
    ebay = [r for r in rows if r["group"] == PRIMARY_REAL]
    report["auc_unseen_both_vs_real_ebay_v7"] = {n: auc([r[n + "_score"] for r in gen], [r[n + "_score"] for r in ebay]) for n in SYSTEMS}

    sup = report["groups"].get(PRIMARY_GEN, {}).get("phase77_vs_phase76", {"p_one_sided": 1.0, "only_phase77": 0, "only_phase76": 0})
    e = report["groups"].get(PRIMARY_REAL, {}).get("all")
    gr = report["groups"].get("real_general", {}).get("all")
    superiority = sup["p_one_sided"] < 0.05 and sup["only_phase77"] > sup["only_phase76"]
    no_more = bool(e and gr and e["phase77"]["k"] <= e["phase76"]["k"] + max(1, math.ceil(0.01 * e["phase77"]["n"]))
                   and e["phase77"]["rate"] <= 0.03 and gr["phase77"]["rate"] <= 0.05)
    report["success_criterion"] = {"rule": protocol["success_criterion"], "superiority": superiority,
                                   "no_more_false_alarms": no_more, "phase77_better": bool(superiority and no_more)}
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "scores.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (RESULTS / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({g: {n: v["all"][n]["rate"] for n in SYSTEMS} for g, v in report["groups"].items()}, indent=1))
    print("AUC", report["auc_unseen_both_vs_real_ebay_v7"])
    print("criterion", report["success_criterion"]["superiority"], report["success_criterion"]["no_more_false_alarms"],
          "-> Phase 77 better:", report["success_criterion"]["phase77_better"])


if __name__ == "__main__":
    main()
