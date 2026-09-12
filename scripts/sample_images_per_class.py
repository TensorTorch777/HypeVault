#!/usr/bin/env python3
"""Copy N random images from each top-level Label_* dataset folder into a flat output folder per class."""

from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}


def collect_images(root: Path) -> list[Path]:
    out: list[Path] = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES:
            out.append(p)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Sample images per Label_* class folder")
    ap.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="HypeVault repo root (contains Label_0_Watches, Label_1_Watches)",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output directory (default: <repo>/dataset_sample_20_per_class)",
    )
    ap.add_argument("--count", type=int, default=20, help="Images per class (default 20)")
    ap.add_argument("--seed", type=int, default=42, help="RNG seed for reproducible sampling")
    args = ap.parse_args()

    repo = args.repo_root.resolve()
    out_root = (args.out or repo / "dataset_sample_20_per_class").resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    class_dirs = sorted(d for d in repo.iterdir() if d.is_dir() and d.name.startswith("Label_"))
    if not class_dirs:
        raise SystemExit(f"No Label_* folders under {repo}")

    random.seed(args.seed)
    for class_dir in class_dirs:
        imgs = collect_images(class_dir)
        if not imgs:
            print(f"skip (no images): {class_dir.name}")
            continue
        k = min(args.count, len(imgs))
        picked = random.sample(imgs, k=k)
        dest_dir = out_root / class_dir.name
        dest_dir.mkdir(parents=True, exist_ok=True)
        for src in picked:
            dest = dest_dir / src.name
            if dest.exists():
                dest = dest_dir / f"{src.parent.name}__{src.name}"
            shutil.copy2(src, dest)
        print(f"{class_dir.name}: copied {k} images -> {dest_dir}")

    print(f"Done. Output root: {out_root}")


if __name__ == "__main__":
    main()
