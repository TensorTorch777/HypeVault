"""Watch-image loading and leakage checks.

Brand is read from the image parent directory. Product, marketplace, and
seller/source ids are not on disk, so those fields stay UNKNOWN. The
product-disjoint split still groups by that parent directory, which is the
historical split key.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from manifest import dataset_hash_from_samples

UNKNOWN = "UNKNOWN"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
LABEL_DIRS = {
    0: ["Label_0_Watches"],
    1: ["Label_1_Watches"],
}
SPLIT_NAMES = ("train", "validation", "test")
CALIBRATION_SPLIT = "calibration"
CHECKED_SPLITS = ("train", "calibration", "validation", "test")
DEFAULT_CALIBRATION_FRACTION = 0.10


@dataclass(frozen=True)
class SampleRecord:
    image_path: Path
    label: int
    brand: str
    product_id: str
    marketplace: str
    source: str
    split_group: str


def product_group_key(path: Path) -> str:
    """Parent directory used by the existing product-disjoint split."""
    return str(path.parent.resolve())


def brand_from_image_path(path: Path) -> str:
    """Brand stored by load_records: the image's parent directory name.

    Label folders are not brands. An empty parent name uses the catalog's
    UNKNOWN token. This does not read pixels, filenames, or predictions.
    """
    name = Path(path).parent.name
    label_folders = {dirname for names in LABEL_DIRS.values() for dirname in names}
    if not name or name in label_folders:
        return UNKNOWN
    return name


def load_records(root: Path) -> list[SampleRecord]:
    """Deterministic catalog order: label folder, brand, then filename."""
    records: list[SampleRecord] = []
    for label, dirs in LABEL_DIRS.items():
        for dirname in dirs:
            folder = root / dirname
            if not folder.exists():
                raise FileNotFoundError(f"Dataset folder not found: {folder}")
            for brand_dir in sorted(p for p in folder.iterdir() if p.is_dir()):
                for img_path in sorted(brand_dir.iterdir()):
                    if img_path.suffix.lower() not in IMAGE_SUFFIXES:
                        continue
                    records.append(
                        SampleRecord(
                            image_path=img_path,
                            label=label,
                            brand=brand_dir.name,
                            product_id=UNKNOWN,
                            marketplace=UNKNOWN,
                            source=UNKNOWN,
                            split_group=product_group_key(img_path),
                        )
                    )
    return records


def load_hypevault_samples(root: Path) -> list[tuple[Path, int]]:
    return [(record.image_path, record.label) for record in load_records(root)]


def _groups_in_first_seen_order(samples: list[tuple[Path, int]]) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = {}
    for idx, (path, _) in enumerate(samples):
        groups.setdefault(product_group_key(path), []).append(idx)
    return groups


def split_indices_by_product(
    samples: list[tuple[Path, int]],
    val_split: float,
    seed: int,
) -> tuple[list[int], list[int]]:
    """Assign whole parent directories to train or val; stratify by class."""
    groups = _groups_in_first_seen_order(samples)
    if not groups:
        return [], []

    labels_by_key = {key: samples[idxs[0]][1] for key, idxs in groups.items()}
    keys_by_label: dict[int, list[str]] = {}
    for key in groups:
        keys_by_label.setdefault(labels_by_key[key], []).append(key)

    rng = np.random.default_rng(seed)
    val_key_set: set[str] = set()
    for label in sorted(keys_by_label):
        keys = list(keys_by_label[label])
        rng.shuffle(keys)
        if val_split <= 0 or len(keys) <= 1:
            n_val_groups = 0
        else:
            n_val_groups = int(round(len(keys) * val_split))
            n_val_groups = max(1, min(len(keys) - 1, n_val_groups))
        val_key_set.update(keys[:n_val_groups])

    train_idx: list[int] = []
    val_idx: list[int] = []
    for key, idxs in groups.items():
        if key in val_key_set:
            val_idx.extend(idxs)
        else:
            train_idx.extend(idxs)
    return train_idx, val_idx


def _holdout_group_assignment(
    samples: list[tuple[Path, int]],
    val_fraction: float,
    test_fraction: float,
    seed: int,
) -> dict[str, str]:
    groups = _groups_in_first_seen_order(samples)
    labels_by_key = {key: samples[idxs[0]][1] for key, idxs in groups.items()}
    keys_by_label: dict[int, list[str]] = {}
    for key in groups:
        keys_by_label.setdefault(labels_by_key[key], []).append(key)

    rng = np.random.default_rng(seed)
    assignment: dict[str, str] = {}
    for label in sorted(keys_by_label):
        keys = list(keys_by_label[label])
        rng.shuffle(keys)
        n = len(keys)
        if n <= 2 or test_fraction <= 0:
            n_test = 0
        else:
            n_test = int(round(n * test_fraction))
            n_test = max(1, min(n - 2, n_test))
        remaining = n - n_test
        if val_fraction <= 0 or remaining <= 1:
            n_val = 0
        else:
            n_val = int(round(n * val_fraction))
            n_val = max(1, min(remaining - 1, n_val))
        for key in keys[:n_test]:
            assignment[key] = "test"
        for key in keys[n_test : n_test + n_val]:
            assignment[key] = "validation"
        for key in keys[n_test + n_val :]:
            assignment[key] = "train"
    return assignment


def _source_cluster_key(records: list[SampleRecord]) -> str:
    known = {record.source for record in records if record.source != UNKNOWN}
    if len(known) > 1:
        raise ValueError(f"Split group {records[0].split_group} mixes sources {sorted(known)}")
    if len(known) == 1:
        return f"source:{next(iter(known))}"
    return f"group:{records[0].split_group}"


def _source_disjoint_assignment(
    records: list[SampleRecord],
    val_fraction: float,
    seed: int,
) -> dict[str, str]:
    grouped: dict[str, list[SampleRecord]] = {}
    for record in records:
        grouped.setdefault(record.split_group, []).append(record)

    clusters: dict[str, list[str]] = {}
    cluster_label: dict[str, int] = {}
    for group, rows in grouped.items():
        cluster = _source_cluster_key(rows)
        clusters.setdefault(cluster, []).append(group)
        cluster_label.setdefault(cluster, rows[0].label)

    keys_by_label: dict[int, list[str]] = {}
    for cluster, label in cluster_label.items():
        keys_by_label.setdefault(label, []).append(cluster)

    rng = np.random.default_rng(seed)
    val_clusters: set[str] = set()
    for label in sorted(keys_by_label):
        keys = list(keys_by_label[label])
        rng.shuffle(keys)
        if val_fraction <= 0 or len(keys) <= 1:
            n_val = 0
        else:
            n_val = int(round(len(keys) * val_fraction))
            n_val = max(1, min(len(keys) - 1, n_val))
        val_clusters.update(keys[:n_val])

    assignment: dict[str, str] = {}
    for cluster, groups in clusters.items():
        split_name = "validation" if cluster in val_clusters else "train"
        for group in groups:
            assignment[group] = split_name
    return assignment


def split_records(
    records: list[SampleRecord],
    strategy: str,
    seed: int,
    val_fraction: float = 0.1,
    test_fraction: float = 0.1,
) -> tuple[dict[str, list[SampleRecord]], dict[str, str]]:
    samples = [(record.image_path, record.label) for record in records]
    meta = {
        "strategy": strategy,
        "source_disjoint_status": "not_requested",
        "source_disjoint_reason": "",
    }
    if strategy == "product_disjoint":
        train_idx, val_idx = split_indices_by_product(samples, val_fraction, seed)
        splits = {
            "train": [records[i] for i in train_idx],
            "validation": [records[i] for i in val_idx],
            "test": [],
        }
    elif strategy == "product_and_source_disjoint":
        known_sources = {record.source for record in records if record.source != UNKNOWN}
        if not known_sources:
            train_idx, val_idx = split_indices_by_product(samples, val_fraction, seed)
            splits = {
                "train": [records[i] for i in train_idx],
                "validation": [records[i] for i in val_idx],
                "test": [],
            }
            meta["source_disjoint_status"] = "unavailable"
            meta["source_disjoint_reason"] = (
                "source metadata is UNKNOWN; used the parent-directory split and "
                "did not claim source disjointness"
            )
        else:
            assignment = _source_disjoint_assignment(records, val_fraction, seed)
            splits = {name: [] for name in SPLIT_NAMES}
            for record in records:
                splits[assignment[record.split_group]].append(record)
            meta["source_disjoint_status"] = "enforced"
    elif strategy == "final_holdout":
        assignment = _holdout_group_assignment(samples, val_fraction, test_fraction, seed)
        splits = {name: [] for name in SPLIT_NAMES}
        for record in records:
            splits[assignment[record.split_group]].append(record)
    else:
        raise ValueError(
            "split strategy must be product_disjoint, product_and_source_disjoint, or final_holdout"
        )
    return splits, meta


def carve_calibration_records(
    records: list[SampleRecord],
    fraction: float,
    seed: int,
) -> tuple[list[SampleRecord], list[SampleRecord]]:
    """Move whole parent directories from the training side into calibration.

    The directory is the same split_group the product-disjoint split uses.
    product_id is UNKNOWN in this catalog, so this is not a SKU split and it
    does not claim marketplace disjointness. Images from one directory stay
    together. Validation and test records must not be passed in. At least one
    directory per class stays in the weight-training set when that class has
    two or more directories.
    """
    portion = float(fraction)
    if not 0.0 <= portion < 1.0:
        raise ValueError(f"calibration_fraction must be in [0, 1), got {fraction}")
    if not records:
        return [], []
    grouped: dict[str, list[SampleRecord]] = {}
    for record in records:
        grouped.setdefault(record.split_group, []).append(record)
    labels_by_key: dict[str, int] = {}
    for key, rows in grouped.items():
        labels = {row.label for row in rows}
        if len(labels) != 1:
            raise ValueError(f"Split group {key} mixes labels {sorted(labels)}")
        labels_by_key[key] = next(iter(labels))
    keys_by_label: dict[int, list[str]] = {}
    for key in sorted(grouped):
        keys_by_label.setdefault(labels_by_key[key], []).append(key)

    rng = np.random.default_rng(int(seed))
    calibration_keys: set[str] = set()
    for label in sorted(keys_by_label):
        keys = list(keys_by_label[label])
        rng.shuffle(keys)
        count = len(keys)
        if portion == 0.0 or count <= 1:
            chosen = 0
        else:
            chosen = int(round(count * portion))
            # A 10% request on only a few brand folders rounds to zero groups.
            # One whole folder is the smallest slice that stays directory-disjoint.
            if chosen == 0:
                chosen = 1
            chosen = min(chosen, count - 1)
        calibration_keys.update(keys[:chosen])

    weight_train = [record for record in records if record.split_group not in calibration_keys]
    calibration = [record for record in records if record.split_group in calibration_keys]
    return weight_train, calibration


def membership_hash(paths: list[str]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(path.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def calibration_split_record(
    records: list[SampleRecord],
    *,
    seed: int,
    requested_fraction: float,
    parent_sample_count: int,
    parent_group_count: int,
    dataset_hash: str,
) -> dict:
    """Describe the calibration partition. It is not a weight-training set."""
    paths = [str(record.image_path.resolve()) for record in records]
    authentic = sum(1 for record in records if record.label == 0)
    fake = sum(1 for record in records if record.label == 1)
    total = len(records)
    groups = {record.split_group for record in records}
    known_products = {record.product_id for record in records if record.product_id != UNKNOWN}
    known_sources = {record.source for record in records if record.source != UNKNOWN}
    return {
        "split": CALIBRATION_SPLIT,
        "parent_split": "train",
        "weight_updates_allowed": False,
        "used_for_checkpoint_selection": False,
        "used_for_final_test": False,
        "sample_count": total,
        "authentic_count": authentic,
        "fake_count": fake,
        "class_ratio_fake": None if total == 0 else fake / total,
        "product_group_count": len(groups),
        "product_id_count": len(known_products),
        "product_id_status": "present" if known_products else "unavailable",
        "brand_counts": dict(sorted(Counter(record.brand for record in records).items())),
        "source_counts": dict(sorted(Counter(record.source for record in records).items())),
        "source_status": "present" if known_sources else "unavailable",
        "marketplace_status": (
            "present"
            if any(record.marketplace != UNKNOWN for record in records)
            else "unavailable"
        ),
        "random_seed": int(seed),
        "requested_fraction": float(requested_fraction),
        "actual_sample_fraction": None if parent_sample_count == 0 else total / parent_sample_count,
        "actual_group_fraction": None if parent_group_count == 0 else len(groups) / parent_group_count,
        "leakage_unit": "parent_directory",
        "leakage_note": (
            "product_id, marketplace, and source are UNKNOWN unless a record says otherwise. "
            "Calibration keeps each parent directory intact. This is the same group as the "
            "product-disjoint split. It is not a SKU split and it is not marketplace-disjoint."
        ),
        "dataset_hash": dataset_hash,
        "membership_hash": membership_hash(paths),
        "membership": sorted(paths),
    }


def verify_no_image_overlap(splits: dict[str, list[SampleRecord]]) -> dict[str, int]:
    seen: dict[str, str] = {}
    for name in CHECKED_SPLITS:
        for record in splits.get(name, []):
            key = str(record.image_path.resolve())
            if key in seen:
                raise RuntimeError(f"Image {key} is in both {seen[key]} and {name}")
            seen[key] = name
    return {"images": len(seen), "overlap": 0}


def verify_no_product_overlap(splits: dict[str, list[SampleRecord]]) -> dict[str, object]:
    """Parent-directory groups must not cross splits.

    product_id is checked only when it is actually known. UNKNOWN is not a
    shared product id.
    """
    group_owner: dict[str, str] = {}
    product_owner: dict[str, str] = {}
    known_products = 0
    for name in CHECKED_SPLITS:
        for record in splits.get(name, []):
            previous = group_owner.get(record.split_group)
            if previous is not None and previous != name:
                raise RuntimeError(
                    f"Split group {record.split_group} is in both {previous} and {name}"
                )
            group_owner[record.split_group] = name
            if record.product_id != UNKNOWN:
                known_products += 1
                previous_product = product_owner.get(record.product_id)
                if previous_product is not None and previous_product != name:
                    raise RuntimeError(
                        f"product_id {record.product_id} is in both {previous_product} and {name}"
                    )
                product_owner[record.product_id] = name
    return {
        "split_groups": len(group_owner),
        "known_product_ids": len(product_owner),
        "product_id_status": "present" if known_products else "unavailable",
        "overlap": 0,
    }


def verify_no_source_leakage(splits: dict[str, list[SampleRecord]]) -> dict[str, object]:
    owners: dict[str, str] = {}
    known = 0
    for name in CHECKED_SPLITS:
        for record in splits.get(name, []):
            if record.source == UNKNOWN:
                continue
            known += 1
            previous = owners.get(record.source)
            if previous is not None and previous != name:
                raise RuntimeError(f"source {record.source} is in both {previous} and {name}")
            owners[record.source] = name
    if known == 0:
        return {
            "status": "unavailable",
            "reason": "source metadata is UNKNOWN",
            "known_sources": 0,
            "overlap": 0,
        }
    return {"status": "ok", "known_sources": len(owners), "overlap": 0}


def verify_product_split(
    train_samples: list[tuple[Path, int]],
    val_samples: list[tuple[Path, int]],
) -> dict[str, int]:
    """Backward-compatible folder and file overlap check used by plotting."""
    train_records = [
        SampleRecord(path, label, path.parent.name, UNKNOWN, UNKNOWN, UNKNOWN, product_group_key(path))
        for path, label in train_samples
    ]
    val_records = [
        SampleRecord(path, label, path.parent.name, UNKNOWN, UNKNOWN, UNKNOWN, product_group_key(path))
        for path, label in val_samples
    ]
    splits = {"train": train_records, "validation": val_records, "test": []}
    verify_no_image_overlap(splits)
    verify_no_product_overlap(splits)
    return {
        "train_products": len({record.split_group for record in train_records}),
        "val_products": len({record.split_group for record in val_records}),
        "train_images": len(train_records),
        "val_images": len(val_records),
        "product_overlap": 0,
        "file_overlap": 0,
    }


def _distribution(records: list[SampleRecord]) -> dict[str, int]:
    authentic = sum(1 for record in records if record.label == 0)
    deepfake = sum(1 for record in records if record.label == 1)
    return {"authentic": authentic, "deepfake": deepfake, "total": len(records)}


def _unique(records: list[SampleRecord], field: str) -> list[str]:
    return sorted({getattr(record, field) for record in records})


def write_split_manifest(
    out_dir: Path,
    records: list[SampleRecord],
    splits: dict[str, list[SampleRecord]],
    *,
    strategy: str,
    seed: int,
    val_fraction: float,
    test_fraction: float,
    split_meta: dict[str, str],
    source_report: dict[str, object],
    product_report: dict[str, object],
) -> dict:
    membership = {
        name: [str(record.image_path.resolve()) for record in splits.get(name, [])]
        for name in CHECKED_SPLITS
    }
    calibration_records = list(splits.get(CALIBRATION_SPLIT, []))
    parent_records = list(splits.get("train", [])) + calibration_records
    data_hash = dataset_hash_from_samples([(record.image_path, record.label) for record in records])
    calibration_fraction = float(split_meta.get("calibration_fraction", DEFAULT_CALIBRATION_FRACTION))
    payload = {
        "strategy": strategy,
        "seed": seed,
        "val_split": val_fraction,
        "test_fraction": test_fraction if strategy == "final_holdout" else None,
        "counts": {name: len(splits.get(name, [])) for name in CHECKED_SPLITS},
        "class_distribution": {name: _distribution(splits.get(name, [])) for name in CHECKED_SPLITS},
        "products": _unique(records, "product_id"),
        "product_groups": _unique(records, "split_group"),
        "brands": _unique(records, "brand"),
        "sources": _unique(records, "source"),
        "marketplaces": _unique(records, "marketplace"),
        "dataset_hash": data_hash,
        "membership": membership,
        "calibration_split": (
            calibration_split_record(
                calibration_records,
                seed=seed,
                requested_fraction=calibration_fraction,
                parent_sample_count=len(parent_records),
                parent_group_count=len({record.split_group for record in parent_records}),
                dataset_hash=data_hash,
            )
            if calibration_records or "calibration_fraction" in split_meta
            else None
        ),
        "product_id_status": product_report["product_id_status"],
        "source_disjoint_status": split_meta.get("source_disjoint_status"),
        "source_disjoint_reason": split_meta.get("source_disjoint_reason", ""),
        "source_check": source_report,
        "train_product_folders": sorted({record.split_group for record in splits.get("train", [])}),
        "val_product_folders": sorted({record.split_group for record in splits.get("validation", [])}),
        "test_product_folders": sorted({record.split_group for record in splits.get("test", [])}),
        "train_images": len(splits.get("train", [])),
        "val_images": len(splits.get("validation", [])),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "split_manifest.json").write_text(json.dumps(payload, indent=2))
    return payload
