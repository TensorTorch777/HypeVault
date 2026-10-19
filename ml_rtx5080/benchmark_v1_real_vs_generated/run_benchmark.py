"""Score both frozen checkpoints on the same benchmark images and apply the pre-registered analysis.

Uses the serving path itself (allow-listed Triton models, the research endpoint's preprocessing and
each model's native decision policy), so the numbers describe what the application would do.

    .venv/bin/python ml_rtx5080/benchmark_v1_real_vs_generated/run_benchmark.py              # confirmatory
    .venv/bin/python ml_rtx5080/benchmark_v1_real_vs_generated/run_benchmark.py --smoke      # smoke manifest
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import math
import platform
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

from bench_stats import binary_metrics, cluster_bootstrap_diff, mcnemar_exact
from common import BRANDS, DATA, HERE, REPO, RESULTS, PROTOCOL_PATH, load_protocol, read_manifest, sha256_file

from inference.model_router import DINOV2_LEGACY, DINOV3_EXPERIMENTAL, infer_allowlisted_model, model_readiness, resolve_model  # noqa: E402
from inference.research_routes import _array_for_model  # noqa: E402
from inference.verdict import apply_min_authentic_confidence, logits_to_verdict  # noqa: E402

MODELS = (DINOV2_LEGACY, DINOV3_EXPERIMENTAL)
PRIMARY_GENERATED = "heldout_kandinsky22"
FORBIDDEN_ROOTS = ("Label_0_Watches", "Label_1_Watches")


def verify_checkpoints(protocol: dict) -> dict:
    out = {}
    for model_id in MODELS:
        spec = protocol["models"][model_id]
        actual = sha256_file(REPO / spec["checkpoint_path"])
        if actual != spec["checkpoint_sha256"]:
            raise SystemExit(f"{model_id} checkpoint hash {actual} does not match the protocol; stopping before inference")
        out[model_id] = actual
    return out


def go_no_go(samples, protocol: dict) -> tuple[str, dict]:
    counts = Counter((s.label, s.group, s.brand) for s in samples)
    minimum = protocol["sample_size"]["REAL_PHOTO"]["per_brand_minimum"]
    real_ok = all(counts[("REAL_PHOTO", "real_ebay", b)] >= minimum for b in BRANDS)
    gen_ok = all(counts[("GENERATED", PRIMARY_GENERATED, b)] >= minimum for b in BRANDS)
    table = {f"{label}|{group}|{brand}": n for (label, group, brand), n in sorted(counts.items())}
    return ("CONFIRMATORY" if real_ok and gen_ok else "SMOKE_TEST"), table


async def score(sample, digest: str) -> dict:
    path = HERE / sample.path
    if any(part in FORBIDDEN_ROOTS for part in path.parts):
        raise SystemExit(f"refusing a training-corpus path: {path}")
    with Image.open(path) as im:
        pil = im.convert("RGB")
    row = {"sample_id": sample.sample_id, "label": sample.label, "group": sample.group,
           "brand": sample.brand, "unit_id": sample.unit_id}
    for model_id in MODELS:
        spec = resolve_model(model_id)
        routed = await infer_allowlisted_model(model_id, _array_for_model(spec, pil))
        logit = routed["logit"]
        if model_id == DINOV2_LEGACY:
            import numpy as np

            raw, conf = logits_to_verdict(np.asarray([logit], dtype=np.float32))
            decision, _ = apply_min_authentic_confidence(raw, conf, 0.88)
            score_fake = 1.0 / (1.0 + math.exp(-logit))
            reasons = ""
        else:
            import torch
            from inference.shadow import _policy_batch

            result = _policy_batch([pil], torch.tensor([logit], dtype=torch.float32), digest)[0]
            decision = result.decision or "ERROR"
            score_fake = 1.0 / (1.0 + math.exp(-logit / spec["temperature"]))
            reasons = ";".join(result.reason_codes or ())
        row[f"{model_id}_logit"] = logit
        row[f"{model_id}_score_fake"] = score_fake
        row[f"{model_id}_decision"] = decision
        row[f"{model_id}_reasons"] = reasons
    return row


def auc_pair(rows, model_id) -> dict:
    labels = [1 if r["label"] == "GENERATED" else 0 for r in rows]
    if len(set(labels)) < 2:
        return {"roc_auc": None, "pr_auc": None}
    from sklearn.metrics import average_precision_score, roc_auc_score

    scores = [r[f"{model_id}_logit"] for r in rows]
    return {"roc_auc": float(roc_auc_score(labels, scores)), "pr_auc": float(average_precision_score(labels, scores))}


def analyse(rows: list[dict], protocol: dict, generated_group: str) -> dict:
    subset = [r for r in rows if r["group"] == "real_ebay" or r["group"].endswith(generated_group)]
    out = {"set": f"REAL_PHOTO + {generated_group}", "n": len(subset), "models": {}}
    for m in MODELS:
        metrics = binary_metrics([r["label"] for r in subset], [r[f"{m}_decision"] for r in subset])
        metrics.update(auc_pair(subset, m))
        out["models"][m] = metrics
    both = [r for r in subset if all(r[f"{m}_decision"] in ("AUTHENTIC", "FAKE") for m in MODELS)]

    def correct(r, m):
        return int((r[f"{m}_decision"] == "FAKE") == (r["label"] == "GENERATED"))

    b = sum(1 for r in both if correct(r, DINOV3_EXPERIMENTAL) and not correct(r, DINOV2_LEGACY))
    c = sum(1 for r in both if correct(r, DINOV2_LEGACY) and not correct(r, DINOV3_EXPERIMENTAL))
    paired = {"n_both_binary": len(both), "dinov3_only_correct": b, "dinov2_only_correct": c,
              "discordant": b + c, "mcnemar_p": mcnemar_exact(b, c)}
    if both:
        point, lo, hi = cluster_bootstrap_diff(
            [correct(r, DINOV3_EXPERIMENTAL) for r in both], [correct(r, DINOV2_LEGACY) for r in both],
            [r["unit_id"] for r in both])
        paired["accuracy_difference_dinov3_minus_dinov2"] = {"point": point, "ci95": [lo, hi]}
    out["paired"] = paired
    slices = defaultdict(dict)
    for brand in BRANDS:
        rb = [r for r in subset if r["brand"] == brand]
        for m in MODELS:
            if rb:
                slices[brand][m] = binary_metrics([r["label"] for r in rb], [r[f"{m}_decision"] for r in rb])
    out["by_brand"] = slices
    return out


def frontier_decision(primary: dict, protocol: dict, mode: str) -> dict:
    if mode != "CONFIRMATORY":
        return {"outcome": "SMOKE_TEST", "frontier_model": None,
                "statement": "Sample minimums were not met. These numbers check the pipeline and may not be reported as results."}
    need = protocol["sample_size"]["required_discordant"]
    p = primary["paired"]
    m2, m3 = primary["models"][DINOV2_LEGACY], primary["models"][DINOV3_EXPERIMENTAL]
    if p["discordant"] < need:
        outcome = "UNDERPOWERED"
    elif p["mcnemar_p"] < 0.05:
        better = DINOV3_EXPERIMENTAL if p["dinov3_only_correct"] > p["dinov2_only_correct"] else DINOV2_LEGACY
        if primary["models"][better]["false_authentic"]["upper95"] <= 0.10:
            return {"outcome": "PERFORMANCE_WINNER", "frontier_model": better,
                    "statement": f"{better} separated real photos from generated images significantly better (exact McNemar p = {p['mcnemar_p']:.4f}, {p['discordant']} discordant images)."}
        outcome = "NO_SAFE_WINNER"
    else:
        outcome = "NO_SIGNIFICANT_DIFFERENCE"
    ok = (m3["false_authentic"]["upper95"] <= m2["false_authentic"]["upper95"] + 0.02 and (m3["coverage"] or 0) >= 0.85)
    if ok:
        return {"outcome": f"{outcome}+ENGINEERING_SELECTION", "frontier_model": DINOV3_EXPERIMENTAL,
                "statement": "DINOv3 is designated the frontier model for engineering reasons (calibrated score, REVIEW state, checkpoint verification). It was not shown to perform better than DINOv2."}
    return {"outcome": outcome, "frontier_model": DINOV2_LEGACY,
            "statement": "No performance winner, and DINOv3 did not meet the engineering-selection conditions. DINOv2 remains the live model."}


def fmt(x, pct=False):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "n/a"
    return f"{100 * x:.1f}%" if pct else (f"{x:.4f}" if isinstance(x, float) else str(x))


def write_report(path: Path, run: dict) -> None:
    lines = [f"# Benchmark v1 — real photos vs generated images ({run['mode']})", "",
             f"Run `{run['run_id']}`, protocol SHA-256 `{run['protocol_sha256'][:16]}…`. {run['frontier']['statement']}", ""]
    for key in ("primary", "secondary"):
        a = run.get(key)
        if not a:
            continue
        lines += [f"## {key.title()}: {a['set']} ({a['n']} images)", "",
                  "| Model | Coverage | Accuracy | Balanced acc. | False-authentic (upper 95%) | False-fake (upper 95%) | ROC-AUC |",
                  "|---|---|---|---|---|---|---|"]
        for m, x in a["models"].items():
            fa, ff = x["false_authentic"], x["false_fake"]
            lines.append(f"| {m} | {fmt(x['coverage'], True)} | {fmt(x['accuracy'], True)} | {fmt(x['balanced_accuracy'], True)} | "
                         f"{fa['k']}/{fa['n']} ({fmt(fa['upper95'], True)}) | {ff['k']}/{ff['n']} ({fmt(ff['upper95'], True)}) | {fmt(x['roc_auc'])} |")
        p = a["paired"]
        lines += ["", f"Paired: {p['n_both_binary']} images with two binary decisions, {p['discordant']} discordant "
                  f"(DINOv3 only correct {p['dinov3_only_correct']}, DINOv2 only correct {p['dinov2_only_correct']}), exact McNemar p = {fmt(p['mcnemar_p'])}.", ""]
    path.write_text("\n".join(lines) + "\n")


async def main_async(args) -> None:
    protocol = load_protocol()
    manifest = DATA / ("smoke_manifest.jsonl" if args.smoke else "manifest.jsonl")
    samples = [s for s in read_manifest(manifest) if s.excluded is None]
    if not samples:
        raise SystemExit(f"no admitted samples in {manifest}")
    mode, counts = go_no_go(samples, protocol)
    if args.smoke:
        mode = "SMOKE_TEST"
    hashes = verify_checkpoints(protocol)
    for m in MODELS:
        ready = await model_readiness(m)
        if not ready.get("ready"):
            raise SystemExit(f"{m} is not served: {ready}")
    from inference.checkpoint_identity import verify_frozen_checkpoint

    digest = verify_frozen_checkpoint()  # the same start-up check the API runs
    rows = []
    for i, s in enumerate(samples, 1):
        rows.append(await score(s, digest))
        if i % 25 == 0 or i == len(samples):
            print(f"scored {i}/{len(samples)}", flush=True)
    primary = analyse(rows, protocol, PRIMARY_GENERATED)
    secondary = analyse(rows, protocol, "sdxl_base10")
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + ("_smoke" if mode == "SMOKE_TEST" else "")
    run = {
        "run_id": run_id, "mode": mode, "protocol_sha256": sha256_file(PROTOCOL_PATH),
        "manifest": str(manifest.relative_to(HERE)), "manifest_sha256": sha256_file(manifest),
        "checkpoint_sha256": hashes, "counts": counts, "host": platform.node(),
        "primary": primary if primary["n"] else None, "secondary": secondary if secondary["n"] else None,
    }
    run["frontier"] = frontier_decision(primary, protocol, mode)
    out = RESULTS / run_id
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(run, indent=2, default=str))
    with open(out / "predictions.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_report(out / "report.md", run)
    print(f"{mode}: {run['frontier']['outcome']} -> {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--smoke", action="store_true", help="use data/smoke_manifest.jsonl; results are never confirmatory")
    asyncio.run(main_async(parser.parse_args()))


if __name__ == "__main__":
    main()
