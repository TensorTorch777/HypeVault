"""Fit post-hoc temperature scaling on the calibration split only.

Example:
    python -m ml_rtx5080.calibrate \\
        --checkpoint ml_rtx5080/experiments/dinov2_cls_only/best_model.pt \\
        --split_manifest ml_rtx5080/experiments/dinov2_cls_only/split_manifest.json \\
        --experiment_dir ml_rtx5080/experiments/dinov2_cls_only \\
        --model dinov2 \\
        --head cls_only \\
        --bins 10

The normal path refuses the checkpoint-selection validation split and the final
test split. --debug_nonproduction_split is not a production calibration.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_PKG = Path(__file__).resolve().parent
if str(_PKG) not in sys.path:
    sys.path.insert(0, str(_PKG))

from calibration import DEFAULT_BINS, DEFAULT_FAR_LIMIT, run_calibration
from dataset import DEFAULT_CALIBRATION_FRACTION


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Post-hoc calibration on a held-out calibration split")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split_manifest", type=Path, required=True)
    parser.add_argument("--experiment_dir", type=Path, required=True)
    parser.add_argument("--model", dest="family", choices=("dinov2", "dinov3"), required=True)
    parser.add_argument("--head", dest="classifier_arch", required=True)
    parser.add_argument("--bins", type=int, default=DEFAULT_BINS)
    parser.add_argument("--calibration_fraction", type=float, default=DEFAULT_CALIBRATION_FRACTION)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_false_authentic_rate", type=float, default=DEFAULT_FAR_LIMIT)
    parser.add_argument("--minimum_brand_samples", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument(
        "--debug_nonproduction_split",
        choices=("validation", "test"),
        default=None,
        help="Development only. Marks the run non-production and does not select a threshold.",
    )
    args = parser.parse_args(argv)
    summary = run_calibration(
        checkpoint=args.checkpoint,
        split_manifest=args.split_manifest,
        experiment_dir=args.experiment_dir,
        family=args.family,
        classifier_arch=args.classifier_arch,
        bins=args.bins,
        calibration_fraction=args.calibration_fraction,
        seed=args.seed,
        max_false_authentic_rate=args.max_false_authentic_rate,
        minimum_brand_samples=args.minimum_brand_samples,
        batch_size=args.batch_size,
        device=args.device,
        num_workers=args.num_workers,
        debug_nonproduction_split=args.debug_nonproduction_split,
    )
    comparison = summary["comparison"]
    print(
        f"Temperature={summary['temperature']:.6f} "
        f"NLL {comparison['nll_absolute_improvement']:+.6f} "
        f"non_production={summary['non_production']}"
    )
    print("production_threshold=null")
    print(f"Wrote {Path(args.experiment_dir) / 'calibration'}")


if __name__ == "__main__":
    main()
