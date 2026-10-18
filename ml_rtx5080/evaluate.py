"""Evaluate a saved authenticity checkpoint without training it.

Example:
    python -m ml_rtx5080.evaluate \\
        --checkpoint ml_rtx5080/experiments/dinov2_cls_only/best_model.pt \\
        --split_manifest ml_rtx5080/experiments/dinov2_cls_only/split_manifest.json \\
        --output_dir ml_rtx5080/experiments/dinov2_cls_only \\
        --model dinov2 \\
        --head cls_only \\
        --threshold 0.50

Brand evaluation is optional and diagnostic. It does not change the checkpoint.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_PKG = Path(__file__).resolve().parent
if str(_PKG) not in sys.path:
    sys.path.insert(0, str(_PKG))

from evaluation import DEFAULT_THRESHOLD, run_checkpoint_evaluation


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Read-only authenticity evaluation")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split_manifest", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--model", dest="family", choices=("dinov2", "dinov3"), required=True)
    parser.add_argument("--head", dest="classifier_arch", required=True)
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--experiment_name", type=str, default=None)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument(
        "--brand_eval",
        type=int,
        choices=(0, 1),
        default=0,
        help="1 writes a diagnostic per-brand report. It does not change checkpoint selection.",
    )
    parser.add_argument("--minimum_brand_samples", type=int, default=20)
    parser.add_argument(
        "--max_false_authentic_rate",
        type=float,
        default=0.02,
        help="Diagnostic FAR flag only. Does not retune the checkpoint or the threshold.",
    )
    args = parser.parse_args(argv)
    metrics = run_checkpoint_evaluation(
        checkpoint=args.checkpoint,
        split_manifest=args.split_manifest,
        output_dir=args.output_dir,
        family=args.family,
        classifier_arch=args.classifier_arch,
        split=args.split,
        threshold=args.threshold,
        batch_size=args.batch_size,
        device=args.device,
        experiment_name=args.experiment_name,
        num_workers=args.num_workers,
        brand_eval=bool(args.brand_eval),
        minimum_brand_samples=args.minimum_brand_samples,
        max_false_authentic_rate=args.max_false_authentic_rate,
    )
    print(
        f"Evaluated {metrics['sample_count']} {args.split} images at threshold {metrics['threshold']:.2f}. "
        f"false_authentic_rate={metrics['false_authentic_rate']} "
        f"f1={metrics['f1']}"
    )
    print(f"Wrote {Path(args.output_dir) / 'evaluation'}")


if __name__ == "__main__":
    main()
