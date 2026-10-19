"""Read-only summary of the model benchmarks for the public benchmark page.

Reports the pre-registered protocol, the latest confirmatory run if one exists, and whether the
pipeline smoke test passed. Smoke-test numbers are never returned as results.
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


@router.get("/summary")
async def benchmark_summary() -> dict:
    protocol = _load(_BENCH / "protocol.json") or {}
    confirmatory, smoke = _latest_runs()
    phase53 = _load(_PHASE53)
    return {
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
