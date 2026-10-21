"""DINOv3 / DINOv2 benchmark-policy decisions on the 1,000 Label_0 validation photos (Phase 73 fix 1).

These real photos set the Phase 73 thresholds. Same serving path and policies as benchmark_v1.

    HYPEVAULT_DEPLOYMENT_MODE=research .venv/bin/python ml_rtx5080/experiments/watch_forensics_v2/score_label0_val.py
"""

import asyncio
import csv
import math
import sys
from pathlib import Path

ML = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ML / "benchmark_v1_real_vs_generated"), str(ML)]

import run_benchmark as v1  # noqa: E402
import watch_forensics as wf  # noqa: E402
from PIL import Image  # noqa: E402

OUT = Path(__file__).resolve().parent / "label0_val_models.csv"


async def main() -> None:
    import torch
    from inference.checkpoint_identity import verify_frozen_checkpoint
    from inference.shadow import _policy_batch

    digest = verify_frozen_checkpoint()
    rows = []
    for row in wf.val_rows(0):
        with Image.open(row["path"]) as im:
            pil = im.convert("RGB")
        out = {"sample_id": row["id"], "brand": row["brand"]}
        for model_id in v1.MODELS:
            spec = v1.resolve_model(model_id)
            logit = (await v1.infer_allowlisted_model(model_id, v1._array_for_model(spec, pil)))["logit"]
            if model_id == v1.DINOV3_EXPERIMENTAL:
                result = _policy_batch([pil], torch.tensor([logit], dtype=torch.float32), digest)[0]
                out["dinov3_decision"], out["dinov3_score_fake"] = result.decision, 1 / (1 + math.exp(-logit / spec["temperature"]))
            else:
                import numpy as np

                raw, conf = v1.logits_to_verdict(np.asarray([logit], dtype=np.float32))
                out["dinov2_decision"] = v1.apply_min_authentic_confidence(raw, conf, 0.88)[0]
        rows.append(out)
    with OUT.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(len(rows), "rows ->", OUT)


asyncio.run(main())
