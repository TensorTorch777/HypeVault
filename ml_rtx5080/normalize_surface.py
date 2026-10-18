"""Phase 56: give both folder classes the same surface before cross-brand training.

Phase 48 found that every Label_1 file is a 512×512 baseline JPEG at IJG quality
95 with no EXIF, while Label_0 files vary in size, format, and encoder. The eval
transform also pads non-square images, which only Label_0 files are. This
module rewrites every Phase 45 non-test row to one encoding so that size,
padding, container, and metadata stop separating the folders.

Every image is decoded, EXIF-transposed, flattened onto white if it has alpha,
center-cropped to a square, downscaled with Lanczos to 448×448, and written as a
baseline 4:2:0 JPEG at quality 90 with no EXIF or ICC profile. Both classes are
downscaled; nothing is upscaled. A row whose short side is below 448 is
excluded rather than enlarged. Phase 45 roles, duplicate quarantine, and the
locked final-test exclusion are kept as they are. Folder names remain folder
labels, not verified ground truth.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

_PKG = Path(__file__).resolve().parent
_REPO = _PKG.parent
if str(_PKG) not in sys.path:
    sys.path.insert(0, str(_PKG))

from cross_brand import FOLDS

PROTOCOL_ID = "normalized_cross_brand_v1"
SOURCE_MANIFEST = _PKG / "experiments" / "v3_cross_brand" / "fold_manifest.json"
OUT = _PKG / "experiments" / PROTOCOL_ID
DATA_ROOT = _REPO / "data_normalized_v1"
SIZE = 448
JPEG_QUALITY = 90
JPEG_SUBSAMPLING = 2  # 4:2:0
EXCLUDED_ROLE = "excluded_short_side"


def protocol_document() -> dict:
    return {
        "protocol_id": PROTOCOL_ID,
        "question": (
            "Does the Phase 45 leave-one-brand-out result survive when both folder classes share one "
            "size, encoder, and metadata state?"
        ),
        "claim_scope": "real photograph versus AI-generated image. Not authentic versus counterfeit watch.",
        "source_manifest": str(SOURCE_MANIFEST.relative_to(_REPO)),
        "normalization": {
            "steps": [
                "decode",
                "EXIF transpose",
                "alpha composited on white; other modes converted to RGB",
                "center crop to the short side",
                f"Lanczos downscale to {SIZE}x{SIZE}",
                f"baseline JPEG quality {JPEG_QUALITY}, 4:2:0, no EXIF, no ICC",
            ],
            "size": SIZE,
            "jpeg_quality": JPEG_QUALITY,
            "exclusion": f"short side below {SIZE} after EXIF transpose; never upscaled",
        },
        "unchanged_from_phase45": [
            "role assignment per fold and for the random-split control",
            "duplicate and leakage quarantine",
            "locked final test excluded",
            "DINOv3 cls_patch_attention, 512 input, hyperparameters, checkpoint selection, calibration, threshold 0.50",
            "generalization bands",
        ],
        "not_run": ["CLS-only ablation", "DINOv2"],
        "residual_risks": [
            "Every Label_1 file is scaled by the same factor (512 to 448); Label_0 factors vary.",
            "Prompt style and scene content can still differ between folders.",
            "Folder labels are not verified ground truth.",
        ],
    }


def output_path(source: Path, label: int) -> Path:
    folder = source.parent.parent.name
    expected = f"Label_{label}_Watches"
    if folder != expected:
        raise ValueError(f"{source} is not under {expected}")
    return DATA_ROOT / folder / source.parent.name / (source.stem + ".jpg")


def normalize_image(image: Image.Image, size: int = SIZE) -> Image.Image | None:
    """Return the square RGB image, or None when the short side is below ``size``."""
    image = ImageOps.exif_transpose(image)
    if image.mode in ("RGBA", "LA", "PA") or (image.mode == "P" and "transparency" in image.info):
        rgba = image.convert("RGBA")
        canvas = Image.new("RGB", rgba.size, (255, 255, 255))
        canvas.paste(rgba, mask=rgba.getchannel("A"))
        image = canvas
    else:
        image = image.convert("RGB")
    width, height = image.size
    side = min(width, height)
    if side < size:
        return None
    left = (width - side) // 2
    top = (height - side) // 2
    square = image.crop((left, top, left + side, top + side))
    if side != size:
        square = square.resize((size, size), Image.Resampling.LANCZOS)
    return square


def save_jpeg(image: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(
        path,
        format="JPEG",
        quality=JPEG_QUALITY,
        subsampling=JPEG_SUBSAMPLING,
        optimize=False,
        progressive=False,
    )


def _work(item: tuple[str, int]) -> dict:
    source, label = Path(item[0]), int(item[1])
    destination = output_path(source, label)
    with Image.open(source) as image:
        width, height = image.size
        source_format = image.format
        square = normalize_image(image)
    if square is None:
        return {"source": str(source), "excluded": True, "source_size": [width, height], "format": source_format}
    save_jpeg(square, destination)
    return {"source": str(source), "excluded": False, "path": str(destination), "format": source_format}


def _inspect(path: str) -> dict:
    with Image.open(path) as image:
        tables = getattr(image, "quantization", None) or {}
        digest = hashlib.sha256(json.dumps({str(k): list(v) for k, v in sorted(tables.items())}).encode()).hexdigest()
        return {
            "size": list(image.size),
            "mode": image.mode,
            "format": image.format,
            "progressive": bool(image.info.get("progressive") or image.info.get("progression")),
            "exif": bool(image.getexif()),
            "icc": "icc_profile" in image.info,
            "quantization_sha256": digest,
            "bytes": Path(path).stat().st_size,
        }


def roc_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Mann-Whitney AUC with average ranks for ties."""
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=float)
    sorted_scores = scores[order]
    start = 0
    while start < len(scores):
        stop = start
        while stop + 1 < len(scores) and sorted_scores[stop + 1] == sorted_scores[start]:
            stop += 1
        ranks[order[start : stop + 1]] = (start + stop) / 2.0 + 1.0
        start = stop + 1
    positives = labels == 1
    n_pos = int(positives.sum())
    n_neg = len(labels) - n_pos
    return float((ranks[positives].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def surface_audit(records: list[dict], labels: np.ndarray) -> dict:
    """Describe what still separates the folders once the encoding is shared."""
    per_class = {}
    for label in (0, 1):
        chosen = [row for row, value in zip(records, labels) if value == label]
        per_class[str(label)] = {
            "count": len(chosen),
            "sizes": dict(Counter(f"{w}x{h}" for w, h in (row["size"] for row in chosen))),
            "modes": dict(Counter(row["mode"] for row in chosen)),
            "formats": dict(Counter(row["format"] for row in chosen)),
            "progressive": sum(row["progressive"] for row in chosen),
            "exif": sum(row["exif"] for row in chosen),
            "icc": sum(row["icc"] for row in chosen),
            "quantization_tables": len({row["quantization_sha256"] for row in chosen}),
            "median_bytes": float(np.median([row["bytes"] for row in chosen])),
        }
    file_bytes = np.array([row["bytes"] for row in records], dtype=float)
    auc = roc_auc(file_bytes, labels)
    return {
        "per_class": per_class,
        "identical_container": all(
            set(per_class["0"][key]) == set(per_class["1"][key]) for key in ("sizes", "modes", "formats")
        )
        and per_class["0"]["progressive"] == per_class["1"]["progressive"] == 0
        and per_class["0"]["exif"] == per_class["1"]["exif"] == 0
        and per_class["0"]["icc"] == per_class["1"]["icc"] == 0
        and per_class["0"]["quantization_tables"] == per_class["1"]["quantization_tables"] == 1,
        "file_size_roc_auc_for_label_1": auc,
        "file_size_separability": max(auc, 1.0 - auc),
        "file_size_note": (
            "JPEG size at a fixed quality tracks high-frequency detail. It is a pixel-content property, "
            "not a container field, and it is reported as the remaining one-number baseline."
        ),
    }


def _remap(rows: list[dict], mapping: dict[str, dict]) -> list[dict]:
    remapped = []
    for row in rows:
        result = mapping[row["path"]]
        updated = dict(row)
        updated["source_path"] = row["path"]
        if result["excluded"]:
            updated["role"] = EXCLUDED_ROLE
        else:
            updated["path"] = result["path"]
        remapped.append(updated)
    return remapped


def _write_json(path: Path, payload: dict) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(payload, indent=2, ensure_ascii=False) + "\n").encode()
    path.write_bytes(data)
    digest = hashlib.sha256(data).hexdigest()
    path.with_suffix(".sha256").write_text(f"{digest}  {path.name}\n")
    return digest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()

    protocol_sha = _write_json(OUT / "protocol_v1.json", protocol_document())
    source = json.loads(SOURCE_MANIFEST.read_text())
    if not source.get("locked_final_test_excluded"):
        raise RuntimeError("source manifest does not exclude the locked final test")
    items = {(row["path"], int(row["label"])) for row in source["random_split_control"]["rows"]}
    for directory, _brand in FOLDS:
        items.update((row["path"], int(row["label"])) for row in source["folds"][directory]["rows"])
    items = sorted(items)
    destinations = Counter(str(output_path(Path(path), label)) for path, label in items)
    collisions = [path for path, count in destinations.items() if count > 1]
    if collisions:
        raise RuntimeError(f"{len(collisions)} sources map to one output, e.g. {collisions[0]}")

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(_work, items, chunksize=64))
    mapping = {row["source"]: row for row in results}
    excluded = [row for row in results if row["excluded"]]
    excluded_labels = Counter(label for (path, label) in items if mapping[path]["excluded"])
    print(f"normalized {len(results) - len(excluded)} images, excluded {len(excluded)}", flush=True)

    kept = [(mapping[path]["path"], label) for path, label in items if not mapping[path]["excluded"]]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        inspected = list(pool.map(_inspect, [path for path, _label in kept], chunksize=64))
    labels = np.array([label for _path, label in kept])
    audit = surface_audit(inspected, labels)
    audit["excluded_short_side"] = {"total": len(excluded), "by_label": {str(k): v for k, v in excluded_labels.items()}}
    audit["source_formats"] = {
        str(label): dict(Counter(mapping[path]["format"] for path, value in items if value == label))
        for label in (0, 1)
    }
    audit["protocol_sha256"] = protocol_sha
    _write_json(OUT / "surface_audit.json", audit)

    manifest = {
        "protocol_id": PROTOCOL_ID,
        "protocol_sha256": protocol_sha,
        "source_manifest": str(SOURCE_MANIFEST.relative_to(_REPO)),
        "source_manifest_sha256": hashlib.sha256(SOURCE_MANIFEST.read_bytes()).hexdigest(),
        "locked_final_test_excluded": True,
        "folds": {
            directory: {**source["folds"][directory], "rows": _remap(source["folds"][directory]["rows"], mapping)}
            for directory, _brand in FOLDS
        },
        "random_split_control": {
            **source["random_split_control"],
            "rows": _remap(source["random_split_control"]["rows"], mapping),
        },
    }
    manifest_sha = _write_json(OUT / "fold_manifest.json", manifest)
    print(json.dumps({"manifest_sha256": manifest_sha, **{k: audit[k] for k in audit if k != "per_class"}}, indent=2))


if __name__ == "__main__":
    main()
