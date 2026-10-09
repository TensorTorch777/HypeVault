"""Brand-stratified, duplicate-safe split for the watch catalog.

The on-disk group is a label folder plus a brand folder. product_id,
marketplace, and source are UNKNOWN, so this module does not claim a
product-disjoint split. Exact SHA-256 copies and dHash near-copies stay in
one split. A duplicate group that mixes authentic and fake labels is held
out for manual review.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image

_PKG = Path(__file__).resolve().parent
if str(_PKG) not in sys.path:
    sys.path.insert(0, str(_PKG))

from dataset import UNKNOWN, load_records
from manifest import dataset_hash_from_samples

STRATEGY = "brand_stratified_duplicate_safe"
MANIFEST_VERSION = "v2"
MANIFEST_FILENAME = "split_manifest_v2.json"
SPLIT_NAMES = ("train", "calibration", "validation", "test")
AUTHORITATIVE_MEMBERSHIP_HASH = "78a38ebac1102977ec5598577126ade77fed8edd6563e13dbdee73dba389f678"
AUTHORITATIVE_COUNTS = {
    "train": {"samples": 20999, "authentic": 10499, "fake": 10500},
    "calibration": {"samples": 2998, "authentic": 1498, "fake": 1500},
    "validation": {"samples": 2997, "authentic": 1497, "fake": 1500},
    "test": {"samples": 3006, "authentic": 1506, "fake": 1500},
}
CATALOG_BRANDS = (
    "A. Lange & Söhne",
    "Audemars Piguet",
    "Patek Philippe",
    "Richard Mille",
    "Vacheron Constantin",
)
PHASE15_EXPERIMENT_DIRS = (
    "dinov2_cls_only",
    "dinov2_cls_patch_attention",
    "dinov3_cls_only",
    "dinov3_cls_patch_attention",
)
TARGET_FRACTIONS = {
    "train": 0.70,
    "calibration": 0.10,
    "validation": 0.10,
    "test": 0.10,
}
# dHash is 64 bits from an 8x8 horizontal gradient. A recompressed JPEG of the
# same pixels was distance 0. Random catalog pairs in a 20,000-pair probe had
# no distance below 4. Hamming 1 transitively links 826 authentic images across
# all five brands whose pairwise median distance is 6, so that is not an
# obvious-copy group. The threshold is therefore identical hashes only.
DHASH_HASH_SIZE = 8
DHASH_HAMMING_THRESHOLD = 0
DHASH_METHOD = "dhash"
DHASH_THRESHOLD_REASON = (
    "Hamming 0 keeps identical dHash values together. Hamming 1 was rejected "
    "because its transitive component contains 826 authentic images from all five "
    "brands with pairwise median distance 6, which is not an obvious copy."
)


class SplitLeakageError(RuntimeError):
    def __init__(self, failures: list[str]):
        self.failures = failures
        super().__init__("; ".join(failures))


def assert_v2_destination(path: Path) -> None:
    """Refuse the Phase 15 manifest name and the frozen checkpoint folders."""
    resolved = path.resolve()
    if resolved.name == "split_manifest.json":
        raise ValueError("refusing to write split_manifest.json; the v2 file has its own name")
    forbidden = {"checkpoints", "checkpoints_watches"}
    if forbidden.intersection(resolved.parts):
        raise ValueError(f"refusing to write under a checkpoint directory: {resolved}")


def dhash_image(image: Image.Image, hash_size: int = DHASH_HASH_SIZE) -> int:
    gray = image.convert("L").resize((hash_size + 1, hash_size), Image.Resampling.BOX)
    pixels = list(gray.getdata())
    bits = 0
    width = hash_size + 1
    for row in range(hash_size):
        base = row * width
        for col in range(hash_size):
            bits = (bits << 1) | (1 if pixels[base + col] > pixels[base + col + 1] else 0)
    return bits


def hamming_distance(left: int, right: int) -> int:
    return (int(left) ^ int(right)).bit_count()


def _popcount_u64(values: np.ndarray) -> np.ndarray:
    x = np.array(values, dtype=np.uint64, copy=True)
    x = x - ((x >> np.uint64(1)) & np.uint64(0x5555555555555555))
    x = (x & np.uint64(0x3333333333333333)) + ((x >> np.uint64(2)) & np.uint64(0x3333333333333333))
    x = (x + (x >> np.uint64(4))) & np.uint64(0x0F0F0F0F0F0F0F0F)
    x = x * np.uint64(0x0101010101010101)
    return (x >> np.uint64(56)).astype(np.uint16)


class _UnionFind:
    def __init__(self, size: int):
        self.parent = list(range(size))

    def find(self, index: int) -> int:
        while self.parent[index] != index:
            self.parent[index] = self.parent[self.parent[index]]
            index = self.parent[index]
        return index

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        if left_root < right_root:
            self.parent[right_root] = left_root
        else:
            self.parent[left_root] = right_root


def inventory_records(records, root: Path, progress_every: int = 2000) -> list[dict]:
    """Read each file once. The table stores hashes and sizes, not pixels."""
    root = root.resolve()
    rows: list[dict] = []
    for index, record in enumerate(records):
        path = record.image_path.resolve()
        data = path.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        width = None
        height = None
        perceptual = None
        try:
            with Image.open(BytesIO(data)) as image:
                width, height = image.size
                perceptual = f"{dhash_image(image):016x}"
        except (OSError, ValueError):
            perceptual = None
        relative = path.relative_to(root).as_posix()
        rows.append(
            {
                "sample_id": relative,
                "path": str(path),
                "label": int(record.label),
                "brand": record.brand,
                "current_group": record.split_group,
                "image_width": width,
                "image_height": height,
                "file_size": path.stat().st_size,
                "sha256": digest,
                "perceptual_hash": perceptual,
                "duplicate_group_id": relative,
            }
        )
        if progress_every and (index + 1) % progress_every == 0:
            print(f"inventoried {index + 1}/{len(records)}", flush=True)
    return rows


def assign_duplicate_groups(
    rows: list[dict],
    hamming_threshold: int = DHASH_HAMMING_THRESHOLD,
) -> dict:
    """Union exact SHA-256 copies and dHash pairs within the Hamming threshold."""
    count = len(rows)
    union = _UnionFind(count)
    exact_members: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        exact_members[row["sha256"]].append(index)
    exact_groups = []
    for digest, members in exact_members.items():
        if len(members) < 2:
            continue
        for other in members[1:]:
            union.union(members[0], other)
        exact_groups.append(
            {
                "sha256": digest,
                "size": len(members),
                "sample_ids": [rows[index]["sample_id"] for index in members],
                "paths": [rows[index]["path"] for index in members],
                "labels": sorted({rows[index]["label"] for index in members}),
                "brands": sorted({rows[index]["brand"] for index in members}),
            }
        )
    exact_groups.sort(key=lambda group: (-group["size"], group["sha256"]))

    hash_bits = []
    hash_index = []
    for index, row in enumerate(rows):
        if row["perceptual_hash"]:
            hash_bits.append(int(row["perceptual_hash"], 16))
            hash_index.append(index)
    hashes = np.array(hash_bits, dtype=np.uint64)
    print(f"clustering {len(hash_index)} perceptual hashes", flush=True)
    for position, origin in enumerate(hash_index):
        if position + 1 >= len(hash_index):
            break
        distances = _popcount_u64(hashes[position + 1 :] ^ np.uint64(hashes[position]))
        hits = np.flatnonzero(distances <= hamming_threshold)
        for hit in hits.tolist():
            union.union(origin, hash_index[position + 1 + hit])

    components: dict[int, list[int]] = defaultdict(list)
    for index in range(count):
        components[union.find(index)].append(index)
    near_groups = []
    for members in components.values():
        group_id = min(rows[index]["sample_id"] for index in members)
        labels = sorted({rows[index]["label"] for index in members})
        brands = sorted({rows[index]["brand"] for index in members})
        mixed_labels = len(labels) > 1
        for index in members:
            rows[index]["duplicate_group_id"] = group_id
            rows[index]["manual_review"] = mixed_labels
            rows[index]["manual_review_reason"] = (
                "duplicate group contains both authentic and fake labels" if mixed_labels else None
            )
        if len(members) < 2:
            continue
        near_groups.append(
            {
                "duplicate_group_id": group_id,
                "size": len(members),
                "sample_ids": [rows[index]["sample_id"] for index in members],
                "paths": [rows[index]["path"] for index in members],
                "labels": labels,
                "brands": brands,
                "mixed_labels": mixed_labels,
                "hamming_threshold": hamming_threshold,
            }
        )
    near_groups.sort(key=lambda group: (-group["size"], group["duplicate_group_id"]))
    row_by_id = {row["sample_id"]: row for row in rows}
    exact_counts: dict[tuple[int, str], int] = Counter()
    for group in exact_groups:
        for sample_id in group["sample_ids"]:
            row = row_by_id[sample_id]
            exact_counts[(row["label"], row["brand"])] += 1
    return {
        "exact_duplicate_groups": len(exact_groups),
        "largest_exact_duplicate_group": exact_groups[0]["size"] if exact_groups else 0,
        "exact_groups": exact_groups,
        "exact_image_counts_by_label_brand": [
            {"label": label, "brand": brand, "images": count}
            for (label, brand), count in sorted(exact_counts.items())
        ],
        "near_duplicate_groups": len(near_groups),
        "largest_near_duplicate_group": near_groups[0]["size"] if near_groups else 0,
        "near_groups": near_groups,
        "hamming_threshold": hamming_threshold,
        "perceptual_method": DHASH_METHOD,
        "manual_review_images": sum(1 for row in rows if row["manual_review"]),
        "manual_review_groups": sum(1 for group in near_groups if group["mixed_labels"]),
    }


def _largest_remainder(total: int, fractions: dict[str, float]) -> dict[str, int]:
    names = list(SPLIT_NAMES)
    raw = [total * fractions[name] for name in names]
    counts = [math.floor(value) for value in raw]
    leftover = total - sum(counts)
    order = sorted(range(len(names)), key=lambda index: (-(raw[index] - counts[index]), index))
    for index in order[:leftover]:
        counts[index] += 1
    if total >= len(names):
        for index, count in enumerate(counts):
            if count == 0:
                donor = max(range(len(names)), key=lambda item: (counts[item], -item))
                if counts[donor] > 1:
                    counts[donor] -= 1
                    counts[index] += 1
    return {name: count for name, count in zip(names, counts)}


def assign_brand_stratified_splits(
    rows: list[dict],
    seed: int,
    fractions: dict[str, float] | None = None,
) -> None:
    """Assign whole duplicate groups. Mixed-label groups stay out of every split."""
    fractions = dict(TARGET_FRACTIONS if fractions is None else fractions)
    if abs(sum(fractions[name] for name in SPLIT_NAMES) - 1.0) > 1e-9:
        raise ValueError("split fractions must sum to 1")
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        row["split"] = "manual_review" if row["manual_review"] else None
        grouped[row["duplicate_group_id"]].append(row)

    strata: dict[tuple[str, int], list[list[dict]]] = defaultdict(list)
    for group_id in sorted(grouped):
        members = grouped[group_id]
        if any(member["manual_review"] for member in members):
            continue
        brands = {member["brand"] for member in members}
        labels = {member["label"] for member in members}
        stratum_brand = min(brands)
        stratum_label = min(labels)
        strata[(stratum_brand, stratum_label)].append(members)

    rng = np.random.default_rng(int(seed))
    for key in sorted(strata):
        groups = strata[key]
        order = rng.permutation(len(groups))
        shuffled = [groups[int(index)] for index in order]
        image_total = sum(len(group) for group in shuffled)
        targets = _largest_remainder(image_total, fractions)
        assigned = {name: 0 for name in SPLIT_NAMES}
        for group in shuffled:
            choice = max(
                SPLIT_NAMES,
                key=lambda name: (targets[name] - assigned[name], -SPLIT_NAMES.index(name)),
            )
            for member in group:
                member["split"] = choice
            assigned[choice] += len(group)


def records_from_fixed_manifest(
    records,
    manifest_path: Path,
    *,
    expected_hash: str | None = None,
) -> tuple[dict[str, list], dict]:
    """Use a written membership list. Does not call the old split generator or carve."""
    manifest_path = Path(manifest_path).resolve()
    payload = json.loads(manifest_path.read_text())
    if payload.get("strategy") != STRATEGY:
        raise RuntimeError(
            f"{manifest_path} strategy is {payload.get('strategy')!r}, expected {STRATEGY}"
        )
    if payload.get("product_disjoint_guarantee") is True:
        raise RuntimeError("refusing a manifest that claims product-disjoint evaluation")
    membership = payload.get("membership")
    if not isinstance(membership, dict):
        raise RuntimeError(f"{manifest_path} has no membership object")
    digest = membership_hash({name: list(membership.get(name) or []) for name in SPLIT_NAMES})
    stored = payload.get("membership_hash")
    if digest != stored:
        raise RuntimeError(
            f"{manifest_path} membership hash {digest} does not match stored hash {stored}"
        )
    if expected_hash is not None and digest != expected_hash:
        raise RuntimeError(
            "manifest hash does not match the Phase 16 manifest. Refusing to train."
        )
    by_path = {str(record.image_path.resolve()): record for record in records}
    splits: dict[str, list] = {}
    seen: dict[str, str] = {}
    for name in SPLIT_NAMES:
        paths = membership.get(name)
        if not isinstance(paths, list):
            raise RuntimeError(f"{manifest_path} is missing membership.{name}")
        chosen = []
        for raw_path in paths:
            key = str(Path(raw_path).resolve())
            previous = seen.get(key)
            if previous is not None:
                raise RuntimeError(f"{key} is in both {previous} and {name}")
            record = by_path.get(key)
            if record is None:
                raise RuntimeError(f"{key} is not in the loaded catalog")
            seen[key] = name
            chosen.append(record)
        splits[name] = chosen
    if expected_hash == AUTHORITATIVE_MEMBERSHIP_HASH:
        _require_authoritative_counts(splits)
    return splits, {
        "strategy": STRATEGY,
        "source_disjoint_status": "not_requested",
        "source_disjoint_reason": "fixed manifest membership was used; splits were not regenerated",
        "fixed_split_manifest": str(manifest_path),
        "manifest_membership_hash": digest,
        "calibration_membership_source": "split_manifest_v2",
        "product_disjoint_guarantee": False,
    }


def _require_authoritative_counts(splits: dict[str, list]) -> None:
    for name, expected in AUTHORITATIVE_COUNTS.items():
        rows = splits[name]
        authentic = sum(1 for row in rows if row.label == 0)
        fake = sum(1 for row in rows if row.label == 1)
        if len(rows) != expected["samples"] or authentic != expected["authentic"] or fake != expected["fake"]:
            raise RuntimeError(
                f"{name} counts {(len(rows), authentic, fake)} do not match the Phase 16 manifest"
            )
        for label in (0, 1):
            present = {row.brand for row in rows if row.label == label}
            missing = [brand for brand in CATALOG_BRANDS if brand not in present]
            if missing:
                raise RuntimeError(f"{name} label {label} is missing brands {missing}")


def write_fixed_membership_copy(out_dir: Path, splits: dict[str, list], meta: dict) -> dict:
    """Record the fixed membership in the experiment directory. Does not rewrite the source manifest."""
    if out_dir.name in PHASE15_EXPERIMENT_DIRS or out_dir.name == "dataset_audit":
        raise RuntimeError(f"refusing to write a v2 run into {out_dir}")
    membership = {
        name: [str(record.image_path.resolve()) for record in splits.get(name, [])] for name in SPLIT_NAMES
    }
    digest = membership_hash(membership)
    if digest != meta["manifest_membership_hash"]:
        raise RuntimeError("experiment membership copy does not match the fixed manifest hash")
    payload = {
        "strategy": STRATEGY,
        "source_manifest": meta["fixed_split_manifest"],
        "membership_hash": digest,
        "calibration_membership_source": "split_manifest_v2",
        "calibration_policy": {
            "weight_updates_allowed": False,
            "post_hoc_directory_carve": False,
            "used_for_checkpoint_selection": False,
            "used_for_final_test": False,
        },
        "product_disjoint_guarantee": False,
        "test_set_used_for_selection": False,
        "counts": {name: len(membership[name]) for name in SPLIT_NAMES},
        "membership": membership,
    }
    destination = out_dir / "split_manifest.json"
    if "checkpoints" in destination.parts or "checkpoints_watches" in destination.parts:
        raise RuntimeError(f"refusing to write under a checkpoint directory: {destination}")
    destination.write_text(json.dumps(payload, indent=2) + "\n")
    return payload


def membership_hash(membership: dict[str, list[str]]) -> str:
    digest = hashlib.sha256()
    for name in SPLIT_NAMES:
        digest.update(name.encode("utf-8"))
        digest.update(b"\n")
        for path in sorted(membership.get(name, [])):
            digest.update(path.encode("utf-8"))
            digest.update(b"\n")
    return digest.hexdigest()


def _brand_entropy(counts: dict[str, int]) -> float | None:
    total = sum(counts.values())
    if total <= 0:
        return None
    entropy = 0.0
    for count in counts.values():
        if count <= 0:
            continue
        share = count / total
        entropy -= share * math.log2(share)
    return entropy


def distribution_for_rows(rows: list[dict]) -> dict:
    authentic = sum(1 for row in rows if row["label"] == 0)
    fake = sum(1 for row in rows if row["label"] == 1)
    total = len(rows)
    brand_counts = Counter(row["brand"] for row in rows)
    largest_brand = max(brand_counts.values()) if brand_counts else 0
    largest_class = max(authentic, fake)
    return {
        "total": total,
        "authentic": authentic,
        "fake": fake,
        "fake_ratio": None if total == 0 else fake / total,
        "brands": len(brand_counts),
        "brand_entropy_bits": _brand_entropy(dict(brand_counts)),
        "largest_brand_share": None if total == 0 else largest_brand / total,
        "largest_class_share": None if total == 0 else largest_class / total,
        "brand_counts": dict(sorted(brand_counts.items())),
    }


def build_manifest(
    rows: list[dict],
    *,
    dataset_hash: str,
    seed: int,
    duplicate_summary: dict,
    fractions: dict[str, float] | None = None,
) -> dict:
    fractions = dict(TARGET_FRACTIONS if fractions is None else fractions)
    membership = {
        name: sorted(row["path"] for row in rows if row["split"] == name) for name in SPLIT_NAMES
    }
    brand_label_rows = []
    for split in SPLIT_NAMES:
        selected = [row for row in rows if row["split"] == split]
        brands = sorted({row["brand"] for row in selected})
        for brand in brands:
            authentic = sum(1 for row in selected if row["brand"] == brand and row["label"] == 0)
            fake = sum(1 for row in selected if row["brand"] == brand and row["label"] == 1)
            brand_label_rows.append(
                {
                    "split": split,
                    "brand": brand,
                    "authentic": authentic,
                    "fake": fake,
                    "total": authentic + fake,
                }
            )
    samples = [
        {
            "sample_id": row["sample_id"],
            "path": row["path"],
            "label": row["label"],
            "brand": row["brand"],
            "sha256": row["sha256"],
            "perceptual_hash": row["perceptual_hash"],
            "duplicate_group_id": row["duplicate_group_id"],
            "split": row["split"],
            "manual_review": bool(row["manual_review"]),
        }
        for row in sorted(rows, key=lambda item: item["sample_id"])
    ]
    known_products = 0
    return {
        "manifest_version": MANIFEST_VERSION,
        "strategy": STRATEGY,
        "product_disjoint_guarantee": False,
        "product_id_status": "unavailable" if known_products == 0 else "present",
        "marketplace_status": "unavailable",
        "source_status": "unavailable",
        "product_metadata_note": (
            "product_id, marketplace, and source are UNKNOWN. The filesystem group is the "
            "brand directory inside a label directory. This manifest does not claim "
            "product-disjoint or marketplace-disjoint evaluation."
        ),
        "seed": int(seed),
        "fractions": fractions,
        "perceptual_hash": {
            "method": DHASH_METHOD,
            "bits": DHASH_HASH_SIZE * DHASH_HASH_SIZE,
            "hamming_threshold": duplicate_summary["hamming_threshold"],
            "threshold_reason": DHASH_THRESHOLD_REASON,
            "resize_limitation": (
                "A 400px resize of one catalog photo had dHash distance 1, so identical-hash "
                "grouping can miss a modest resize. Those pairs are not product identities."
            ),
        },
        "duplicate_group_policy": (
            "Exact SHA-256 copies and dHash pairs at or below the Hamming threshold form one "
            "duplicate group. Every member of a group receives the same split. A group whose "
            "labels include both authentic and fake is manual_review and is in none of the "
            "four splits."
        ),
        "calibration_policy": {
            "weight_updates_allowed": False,
            "used_for_checkpoint_selection": False,
            "used_for_final_test": False,
            "generated_with_the_split": True,
            "post_hoc_directory_carve": False,
        },
        "dataset_hash": dataset_hash,
        "membership_hash": membership_hash(membership),
        "created_at": None,
        "created_at_note": "omitted so the same dataset, seed, and duplicate groups reproduce identical bytes",
        "counts": {name: len(membership[name]) for name in SPLIT_NAMES},
        "manual_review_count": sum(1 for row in rows if row["split"] == "manual_review"),
        "class_counts": {
            name: {
                "authentic": sum(1 for row in rows if row["split"] == name and row["label"] == 0),
                "fake": sum(1 for row in rows if row["split"] == name and row["label"] == 1),
            }
            for name in SPLIT_NAMES
        },
        "distribution": {
            name: distribution_for_rows([row for row in rows if row["split"] == name]) for name in SPLIT_NAMES
        },
        "brand_by_label": brand_label_rows,
        "duplicate_summary": {
            "exact_duplicate_groups": duplicate_summary["exact_duplicate_groups"],
            "largest_exact_duplicate_group": duplicate_summary["largest_exact_duplicate_group"],
            "near_duplicate_groups": duplicate_summary["near_duplicate_groups"],
            "largest_near_duplicate_group": duplicate_summary["largest_near_duplicate_group"],
            "hamming_threshold": duplicate_summary["hamming_threshold"],
            "manual_review_images": duplicate_summary["manual_review_images"],
            "manual_review_groups": duplicate_summary["manual_review_groups"],
        },
        "membership": membership,
        "samples": samples,
    }


def validate_split_manifest(manifest: dict) -> dict:
    """Fail closed on leakage, an empty test, or brand-confounded evaluation."""
    failures: list[str] = []
    limitations: list[str] = []
    samples = manifest.get("samples")
    if not isinstance(samples, list) or not samples:
        raise SplitLeakageError(["manifest has no samples"])
    if manifest.get("product_disjoint_guarantee") is True and manifest.get("product_id_status") != "present":
        failures.append("product_disjoint_guarantee is true while product_id is unavailable")
    policy = manifest.get("calibration_policy") or {}
    if policy.get("weight_updates_allowed") is not False:
        failures.append("calibration weight_updates_allowed must be false")

    owners: dict[str, str] = {}
    hash_owners: dict[str, str] = {}
    group_owners: dict[str, str] = {}
    for row in samples:
        split = row["split"]
        if split == "manual_review":
            continue
        if split not in SPLIT_NAMES:
            failures.append(f"sample {row['sample_id']} has unknown split {split}")
            continue
        previous = owners.get(row["path"])
        if previous is not None and previous != split:
            failures.append(f"sample {row['path']} is in both {previous} and {split}")
        owners[row["path"]] = split
        previous_hash = hash_owners.get(row["sha256"])
        if previous_hash is not None and previous_hash != split:
            failures.append(f"sha256 {row['sha256']} is in both {previous_hash} and {split}")
        hash_owners[row["sha256"]] = split
        previous_group = group_owners.get(row["duplicate_group_id"])
        if previous_group is not None and previous_group != split:
            failures.append(
                f"duplicate group {row['duplicate_group_id']} is in both {previous_group} and {split}"
            )
        group_owners[row["duplicate_group_id"]] = split

    catalog_brands = {row["brand"] for row in samples}
    catalog_labels = {row["label"] for row in samples}
    eligible = [row for row in samples if row["split"] != "manual_review"]
    groups_for_cell: dict[tuple[str, int], set[str]] = defaultdict(set)
    for row in eligible:
        groups_for_cell[(row["brand"], row["label"])].add(row["duplicate_group_id"])

    for name in SPLIT_NAMES:
        selected = [row for row in samples if row["split"] == name]
        if not selected:
            failures.append(f"{name} is empty")
            continue
        labels = {row["label"] for row in selected}
        brands = {row["brand"] for row in selected}
        if name in {"validation", "test"} and len(labels) < 2 and len(catalog_labels) >= 2:
            failures.append(f"{name} contains only one class")
        if name == "validation" and len(brands) < 2 and len(catalog_brands) >= 2:
            failures.append("validation has only one brand")
        for label, class_name in ((0, "authentic"), (1, "fake")):
            class_brands = {row["brand"] for row in selected if row["label"] == label}
            catalog_class_brands = {row["brand"] for row in eligible if row["label"] == label}
            if name in {"validation", "test"} and len(catalog_class_brands) >= 2 and len(class_brands) < 2:
                failures.append(f"{name} {class_name} samples come from fewer than two brands")

    for (brand, label), groups in sorted(groups_for_cell.items()):
        if len(groups) < len(SPLIT_NAMES):
            limitations.append(
                f"{brand} label {label} has {len(groups)} duplicate groups, fewer than four splits"
            )
            continue
        for name in SPLIT_NAMES:
            present = any(row["split"] == name and row["brand"] == brand and row["label"] == label for row in samples)
            if not present:
                failures.append(f"{name} is missing brand {brand} label {label}")

    if failures:
        raise SplitLeakageError(failures)
    return {"ok": True, "limitations": limitations, "splits": list(SPLIT_NAMES)}


def _example_groups(groups: list[dict], limit: int = 12) -> list[dict]:
    examples = []
    for group in groups[:limit]:
        examples.append(
            {
                "size": group["size"],
                "labels": group["labels"],
                "brands": group["brands"],
                "paths": group["paths"][:4],
            }
        )
    return examples


def structure_audit(records, rows: list[dict], duplicate_summary: dict) -> dict:
    brands = sorted({record.brand for record in records})
    groups = sorted({record.split_group for record in records})
    product_ids = sorted({record.product_id for record in records})
    marketplaces = sorted({record.marketplace for record in records})
    sources = sorted({record.source for record in records})
    stems: dict[str, set[str]] = defaultdict(set)
    for record in records:
        stems[record.image_path.stem].add(record.split_group)
    repeated_stems = {stem: sorted(owners) for stem, owners in stems.items() if len(owners) > 1}
    images_per_group = Counter(record.split_group for record in records)
    return {
        "verified_metadata": {
            "label": "top-level directory Label_0_Watches (authentic, 0) or Label_1_Watches (fake, 1)",
            "brand": "immediate parent directory name",
            "filename": "12-character hexadecimal stem plus .jpg; not documented as a SKU",
            "image_bytes": "present",
        },
        "unavailable_metadata": {
            "product_id": product_ids == [UNKNOWN],
            "marketplace": marketplaces == [UNKNOWN],
            "source": sources == [UNKNOWN],
            "physical_watch_identity": "not on disk",
            "capture_time": "not read; no timestamp is treated as identity",
        },
        "current_group_meaning": (
            "load_records sets split_group and product_group_key to the resolved parent directory. "
            "Each such directory is one brand inside one label folder. It is a brand-and-label "
            "bucket of many files, not a product or SKU."
        ),
        "product_disjoint_possible": False,
        "product_disjoint_reason": (
            "No product or SKU field exists. A parent directory holds thousands of images, and "
            "nothing on disk proves which of those images are the same physical watch."
        ),
        "directory_contains_multiple_products": "unknown; each brand directory contains many images and no product key",
        "same_physical_watch_possible": "unknown; filenames and folders do not identify a physical watch",
        "brand_count": len(brands),
        "brands": brands,
        "current_group_count": len(groups),
        "images_per_current_group_min": min(images_per_group.values()) if images_per_group else 0,
        "images_per_current_group_max": max(images_per_group.values()) if images_per_group else 0,
        "sample_count": len(records),
        "filename_stems_shared_across_groups": len(repeated_stems),
        "exact_duplicate_groups": duplicate_summary["exact_duplicate_groups"],
        "largest_exact_duplicate_group": duplicate_summary["largest_exact_duplicate_group"],
        "exact_image_counts_by_label_brand": duplicate_summary["exact_image_counts_by_label_brand"],
        "near_duplicate_groups": duplicate_summary["near_duplicate_groups"],
        "largest_near_duplicate_group": duplicate_summary["largest_near_duplicate_group"],
        "near_duplicate_examples": _example_groups(duplicate_summary["near_groups"]),
        "exact_duplicate_examples": _example_groups(duplicate_summary["exact_groups"]),
        "hamming_threshold": duplicate_summary["hamming_threshold"],
        "threshold_reason": DHASH_THRESHOLD_REASON,
        "perceptual_method": DHASH_METHOD,
        "manual_review_images": duplicate_summary["manual_review_images"],
        "rows_missing_perceptual_hash": sum(1 for row in rows if not row["perceptual_hash"]),
    }


def _write_json(path: Path, payload: dict) -> None:
    assert_v2_destination(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def write_inventory_csv(path: Path, rows: list[dict]) -> None:
    assert_v2_destination(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "sample_id",
        "path",
        "label",
        "brand",
        "current_group",
        "image_width",
        "image_height",
        "file_size",
        "sha256",
        "perceptual_hash",
        "duplicate_group_id",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_brand_balance_csv(path: Path, manifest: dict) -> None:
    assert_v2_destination(path)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["split", "brand", "authentic", "fake", "total"])
        writer.writeheader()
        for row in manifest["brand_by_label"]:
            writer.writerow(row)


def render_audit_markdown(audit: dict, manifest: dict) -> str:
    lines = [
        "# Dataset structure audit",
        "",
        "Verified metadata:",
        "",
        f"- Label: {audit['verified_metadata']['label']}",
        f"- Brand: {audit['verified_metadata']['brand']}",
        f"- Filename: {audit['verified_metadata']['filename']}",
        "",
        "Unavailable metadata:",
        "",
        "- product_id is UNKNOWN on every record.",
        "- marketplace is UNKNOWN on every record.",
        "- source is UNKNOWN on every record.",
        "- No physical-watch or SKU identity is stored.",
        "",
        f"Current group: {audit['current_group_meaning']}",
        "",
        f"True product-disjoint splitting is possible: {str(audit['product_disjoint_possible']).lower()}.",
        audit["product_disjoint_reason"],
        "",
        f"Samples: {audit['sample_count']}. Brands: {audit['brand_count']}. Current groups: {audit['current_group_count']}.",
        (
            f"Images per current group: {audit['images_per_current_group_min']} to "
            f"{audit['images_per_current_group_max']}."
        ),
        "",
        (
            f"Exact duplicate groups: {audit['exact_duplicate_groups']}. "
            f"Largest exact group: {audit['largest_exact_duplicate_group']}."
        ),
        (
            f"Near-duplicate groups ({audit['perceptual_method']}, Hamming <= {audit['hamming_threshold']}): "
            f"{audit['near_duplicate_groups']}. Largest near-duplicate group: {audit['largest_near_duplicate_group']}."
        ),
        audit["threshold_reason"],
        "Near-duplicate groups are not product groups.",
        f"Manual-review images (mixed authentic/fake duplicate group): {audit['manual_review_images']}.",
        "",
        f"Corrected strategy: {manifest['strategy']}. product_disjoint_guarantee: false.",
        f"Membership hash: {manifest['membership_hash']}.",
        "",
        "| Split | Total | Authentic | Fake | Brands |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for name in SPLIT_NAMES:
        distribution = manifest["distribution"][name]
        lines.append(
            f"| {name} | {distribution['total']} | {distribution['authentic']} | "
            f"{distribution['fake']} | {distribution['brands']} |"
        )
    lines.extend(
        [
            "",
            "| Split | Brand | Authentic | Fake | Total |",
            "| --- | --- | ---: | ---: | ---: |",
        ]
    )
    for row in manifest["brand_by_label"]:
        lines.append(
            f"| {row['split']} | {row['brand']} | {row['authentic']} | {row['fake']} | {row['total']} |"
        )
    lines.append("")
    return "\n".join(lines)


def build_split_from_records(records, root: Path, seed: int) -> tuple[list[dict], dict, dict]:
    rows = inventory_records(records, root)
    summary = assign_duplicate_groups(rows)
    assign_brand_stratified_splits(rows, seed)
    dataset_hash = dataset_hash_from_samples([(record.image_path, record.label) for record in records])
    manifest = build_manifest(rows, dataset_hash=dataset_hash, seed=seed, duplicate_summary=summary)
    return rows, summary, manifest


def write_audit_directory(destination: Path, records, root: Path, seed: int) -> dict:
    rows, summary, manifest = build_split_from_records(records, root, seed)
    leakage = validate_split_manifest(manifest)
    audit = structure_audit(records, rows, summary)
    destination.mkdir(parents=True, exist_ok=True)
    write_inventory_csv(destination / "sample_inventory.csv", rows)
    _write_json(
        destination / "exact_duplicates.json",
        {
            "exact_duplicate_groups": summary["exact_duplicate_groups"],
            "largest_exact_duplicate_group": summary["largest_exact_duplicate_group"],
            "counts_by_label_brand": summary["exact_image_counts_by_label_brand"],
            "examples": _example_groups(summary["exact_groups"]),
            "groups": summary["exact_groups"],
        },
    )
    _write_json(
        destination / "near_duplicates.json",
        {
            "method": DHASH_METHOD,
            "bits": DHASH_HASH_SIZE * DHASH_HASH_SIZE,
            "hamming_threshold": summary["hamming_threshold"],
            "near_duplicate_groups": summary["near_duplicate_groups"],
            "largest_near_duplicate_group": summary["largest_near_duplicate_group"],
            "note": "These are near-duplicate image groups, not product groups.",
            "examples": _example_groups(summary["near_groups"]),
            "groups": summary["near_groups"],
        },
    )
    _write_json(destination / "dataset_structure_audit.json", audit)
    (destination / "dataset_structure_audit.md").write_text(render_audit_markdown(audit, manifest))
    _write_json(destination / MANIFEST_FILENAME, manifest)
    write_brand_balance_csv(destination / "brand_balance.csv", manifest)
    _write_json(destination / "distribution_report.json", manifest["distribution"])
    _write_json(destination / "leakage_validation.json", leakage)
    return manifest


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Write the v2 brand-stratified duplicate-safe split")
    parser.add_argument("--data_root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "experiments" / "dataset_audit",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    records = load_records(args.data_root)
    print(f"loaded {len(records)} records", flush=True)
    manifest = write_audit_directory(args.output, records, args.data_root, args.seed)
    print(
        json.dumps(
            {
                "strategy": manifest["strategy"],
                "membership_hash": manifest["membership_hash"],
                "counts": manifest["counts"],
                "product_disjoint_guarantee": manifest["product_disjoint_guarantee"],
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
