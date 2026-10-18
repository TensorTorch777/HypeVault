"""Fit an OOD reference on the training split and score an explicit OOD corpus.

Example:
    python -m ml_rtx5080.ood_eval \\
        --checkpoint ml_rtx5080/experiments/dinov2_cls_only/best_model.pt \\
        --split_manifest ml_rtx5080/experiments/dinov2_cls_only/split_manifest.json \\
        --output_dir ml_rtx5080/experiments/dinov2_cls_only \\
        --model dinov2 \\
        --head cls_only \\
        --id_split train \\
        --ood_root /data/ood

The reference split must be train. Omitting --ood_root does not invent a corpus
and does not set a production OOD threshold. --threshold records an explicit
cutoff only; it is not searched.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_PKG = Path(__file__).resolve().parent
if str(_PKG) not in sys.path:
    sys.path.insert(0, str(_PKG))

from ood import NO_CORPUS_STATUS, REFERENCE_SPLIT, run_ood_evaluation


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Inference-only feature-distance OOD evaluation")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split_manifest", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--model", dest="family", choices=("dinov2", "dinov3"), required=True)
    parser.add_argument("--head", dest="classifier_arch", required=True)
    parser.add_argument("--id_split", default=REFERENCE_SPLIT)
    parser.add_argument("--ood_root", type=Path, default=None)
    parser.add_argument("--ood_dev_root", type=Path, default=None)
    parser.add_argument("--id_dev_split", choices=("validation", "calibration"), default=None)
    parser.add_argument("--report_final", type=int, choices=(0, 1), default=0)
    parser.add_argument("--method", choices=("mahalanobis", "cosine", "both"), default="both")
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--num_workers", type=int, default=0)
    args = parser.parse_args(argv)
    summary = run_ood_evaluation(
        checkpoint=args.checkpoint,
        split_manifest=args.split_manifest,
        output_dir=args.output_dir,
        family=args.family,
        classifier_arch=args.classifier_arch,
        id_split=args.id_split,
        ood_root=args.ood_root,
        ood_dev_root=args.ood_dev_root,
        id_dev_split=args.id_dev_split,
        report_final=bool(args.report_final),
        method=args.method,
        threshold=args.threshold,
        batch_size=args.batch_size,
        device=args.device,
        num_workers=args.num_workers,
    )
    print(summary["validation_status"])
    print(f"production_ood_threshold={summary['production_ood_threshold']}")
    print(f"Wrote {Path(args.output_dir) / 'ood'}")
    if summary["validation_status"] == NO_CORPUS_STATUS:
        return


if __name__ == "__main__":
    main()
