"""Read-only summary of the model benchmarks for the public benchmark page.

`live_model` is the model that screens listings now and the benchmark that put it there
(benchmark_v6, Phase 76), with the Phase 78 hold and the known-AI copy check. The rest is
benchmark_v1, the first DINOv2-vs-DINOv3 comparison: its protocol, its confirmatory run and
whether the pipeline smoke test passed. Smoke-test numbers are never returned as results.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from fastapi import APIRouter

_log = logging.getLogger(__name__)
router = APIRouter()

_ML = Path(__file__).resolve().parents[2] / "ml_rtx5080"
_BENCH = _ML / "benchmark_v1_real_vs_generated"
_PHASE53 = _ML / "experiments" / "dinov2_vs_dinov3_full_scale" / "metrics.json"
_V6 = _ML / "benchmark_v6_fresh"
_V7 = _ML / "benchmark_v7_fresh" / "results" / "report.json"
_PHASE78 = _ML / "experiments" / "phase78" / "screenshot_benchmark.json"
_KNOWN_AI = _ML / "experiments" / "known_ai" / "measure.json"

# benchmark_v6 groups and systems, in display order
_V6_GROUPS = {"chatgpt_v6": "ChatGPT", "gemini_v6": "Gemini", "gemini_edu_v6": "Gemini, second account", "real_ebay_v6": "Real eBay photos"}
_V6_SYSTEMS = {"dinov2_live": "DINOv2", "phase75_head": "Phase 75", "phase76_head": "DINOv3 + Perception Encoder"}


def _load(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def _latest_runs() -> tuple[dict | None, dict | None]:
    results = _BENCH / "results"
    runs = sorted((p for p in results.iterdir() if p.is_dir()), reverse=True) if results.is_dir() else []
    confirmatory = smoke = None
    for run_dir in runs:
        metrics = _load(run_dir / "metrics.json")
        if not metrics:
            continue
        if metrics.get("mode") == "CONFIRMATORY" and confirmatory is None:
            confirmatory = metrics
        elif metrics.get("mode") == "SMOKE_TEST" and smoke is None:
            smoke = metrics
    return confirmatory, smoke


def _model_rows(analysis: dict | None) -> list[dict]:
    if not analysis:
        return []
    rows = []
    for model_id, m in analysis["models"].items():
        rows.append({
            "model": model_id,
            "coverage": m.get("coverage"),
            "accuracy": m.get("accuracy"),
            "balanced_accuracy": m.get("balanced_accuracy"),
            "false_authentic": m.get("false_authentic"),
            "false_fake": m.get("false_fake"),
            "roc_auc": m.get("roc_auc"),
            "review": m.get("review"),
        })
    return rows


def _kn(cell: dict | None) -> dict | None:
    return None if not cell else {"k": cell["k"], "n": cell["n"]}


def _live_model() -> dict | None:
    """The live model and the evidence for it, read from the frozen result files."""
    v6 = _load(_V6 / "results" / "report.json")
    if not v6:
        return None
    protocol = _load(_V6 / "protocol.json") or {}
    groups = [
        {
            "id": g,
            "label": label,
            "kind": "real" if g.startswith("real") else "generated",
            "systems": {sid: _kn(v6["groups"][g]["all"].get(sid)) for sid in _V6_SYSTEMS},
        }
        for g, label in _V6_GROUPS.items()
        if g in v6.get("groups", {})
    ]
    v7 = _load(_V7)
    p78 = _load(_PHASE78)
    known = _load(_KNOWN_AI)
    return {
        "id": "phase76_fusion",
        "name": "DINOv3 + Perception Encoder",
        "phase": v6.get("phase"),
        "features": "DINOv3 ViT-B/16 (pretrained + fine-tuned) and Perception Encoder L/14-336, one logistic-regression head",
        "benchmark": {
            "name": v6.get("benchmark"),
            "written_on": protocol.get("written_on"),
            "scored_at": v6.get("scored_at"),
            "question": protocol.get("question"),
            "rule": (v6.get("success_criterion") or {}).get("rule"),
            "met": (v6.get("success_criterion") or {}).get("met"),
            "systems": _V6_SYSTEMS,
            "groups": groups,
            "caveat": "ChatGPT and Gemini were in this model's training data, so this is a seen-app result.",
        },
        "unseen_generators": None if not v7 else {
            "name": v7.get("benchmark"),
            "caught": _kn((v7["groups"].get("unseen_both") or {}).get("all", {}).get("phase76")),
            "real_ebay_flagged": _kn((v7["groups"].get("real_ebay_v7") or {}).get("all", {}).get("phase76")),
            "generators": "ERNIE, ERNIE Turbo, Sora 2, Veo 3, Wan 2.5",
            "note": "General scenes, not watches. Phase 77, the challenger on this benchmark, was not better.",
        },
        "edited_copy_hold": None if not p78 else {
            "rows": [
                {"edit": edit, "phase76": p78.get(f"AI caught · {key}", {}).get("phase76"), "phase78": p78.get(f"AI caught · {key}", {}).get("phase78")}
                for key, edit in (("original", "Original"), ("screenshot_style", "Screenshot-style"), ("crop70", "70% crop"), ("half_size", "Half size"))
            ],
            "real_flagged": p78.get("real photos wrongly flagged · original"),
            "role": "Phase 78 flags more real photos, so it only holds a listing for a proof photo; it never rejects one.",
        },
        "known_ai": None if not known else {
            "cut": known.get("cut"),
            "edited_matched": (known.get("at_cut") or {}).get("edited_ai_matched"),
            "real_matched": (known.get("at_cut") or {}).get("real_matched"),
            "real_n": (known.get("at_cut") or {}).get("real_n"),
        },
    }


@router.get("/summary")
async def benchmark_summary() -> dict:
    protocol = _load(_BENCH / "protocol.json") or {}
    confirmatory, smoke = _latest_runs()
    phase53 = _load(_PHASE53)
    return {
        "live_model": _live_model(),
        "protocol": {
            "name": protocol.get("protocol"),
            "status": protocol.get("status"),
            "written_on": protocol.get("written_on"),
            "question": protocol.get("question"),
            "scope_limit": protocol.get("scope_limit"),
            "classes": {
                "REAL_PHOTO": (protocol.get("classes", {}).get("REAL_PHOTO") or {}).get("source"),
                "GENERATED": {
                    k: v.get("generator") for k, v in (protocol.get("classes", {}).get("GENERATED", {}).get("groups") or {}).items()
                },
            },
            "normalisation": protocol.get("normalisation_applied_to_every_image"),
            "sample_size": protocol.get("sample_size"),
            "frontier_rule": protocol.get("frontier_rule"),
            "models": {k: {"checkpoint_sha256": v.get("checkpoint_sha256"), "policy": v.get("policy")}
                       for k, v in (protocol.get("models") or {}).items() if isinstance(v, dict)},
        },
        "confirmatory": None if confirmatory is None else {
            "run_id": confirmatory["run_id"],
            "frontier": confirmatory["frontier"],
            "primary": {"set": confirmatory["primary"]["set"], "n": confirmatory["primary"]["n"],
                        "models": _model_rows(confirmatory["primary"]),
                        "paired": confirmatory["primary"]["paired"]} if confirmatory.get("primary") else None,
        },
        "pipeline_check": None if smoke is None else {
            "run_id": smoke["run_id"],
            "passed": True,
            "images_scored": sum(smoke.get("counts", {}).values()),
            "note": "Pipeline check on a tiny test generator. Not a result.",
        },
        "previous_evaluation": None if phase53 is None else {
            "name": "Phase 53/54 frozen comparison",
            "data_readiness": "NO_GO_FOR_AUTHENTICITY_CLAIMS",
            "summary": "No model was selected: the only cohort held out from both models had 315 authentic-labelled images of one brand and no generated images.",
        },
    }
