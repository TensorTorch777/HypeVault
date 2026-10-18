"""Leave-one-brand-out protocol for DINOv3 authenticity research.

The locked final-test membership is excluded by path. This module does not
open images, fit a temperature, or load a checkpoint.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

PROTOCOL_ID = "v3_cross_brand_v1"
SEED = 45
FROZEN_CHECKPOINT = "ml_rtx5080/experiments/v2_dinov3_cls_patch_attention/epoch_018.pt"
FROZEN_CHECKPOINT_SHA256 = "5a38c93fd442b03653c65d2a5ecc9c2687ef152f7c5c020763e4ce1fd9c7d28f"
INITIALIZATION = (
    "timm pretrained vit_base_patch16_dinov3.lvd1689m. "
    "The frozen epoch_018.pt checkpoint is not loaded."
)
CHECKPOINT_SELECTION_RULE = (
    "Select only on the four training brands' validation split. "
    "Maximum validation PR-AUC subject to false_authentic_rate <= 0.02. "
    "Ties prefer higher F1, then lower false-authentic rate, then lower validation loss, "
    "then the earlier epoch. The held-out brand and the locked final test are not inputs. "
    "The decision threshold stays 0.50 and is not tuned."
)
CALIBRATION_RULE = (
    "Fit one temperature on the four training brands' calibration split only. "
    "Freeze that temperature. Apply it unchanged to in-domain validation and to the held-out brand."
)
WITHIN_FOLD_FRACTIONS = {"train": 0.70 / 0.90, "calibration": 0.10 / 0.90, "validation": 0.10 / 0.90}
RANDOM_FRACTIONS = {"train": 0.70, "calibration": 0.10, "validation": 0.10, "random_test": 0.10}
FOLDS = (
    ("fold_1_lange", "A. Lange & Söhne"),
    ("fold_2_ap", "Audemars Piguet"),
    ("fold_3_patek", "Patek Philippe"),
    ("fold_4_rm", "Richard Mille"),
    ("fold_5_vacheron", "Vacheron Constantin"),
)
METRIC_NAMES = (
    "roc_auc",
    "pr_auc",
    "f1",
    "false_authentic_rate",
    "false_fake_rate",
    "nll",
    "brier",
    "ece",
)
# Pre-specified before any cross-brand score exists. These read the qualitative
# bands. They are not a threshold searched on a held-out brand.
GENERALIZATION_BANDS = {
    "strong": {
        "min_mean_unseen_pr_auc": 0.90,
        "max_mean_unseen_false_authentic_rate": 0.05,
        "max_mean_pr_auc_gap": 0.05,
        "max_fold_unseen_false_authentic_rate": 0.10,
    },
    "moderate": {
        "min_mean_unseen_pr_auc": 0.75,
        "max_mean_unseen_false_authentic_rate": 0.15,
    },
    "weak": {"min_mean_unseen_pr_auc": 0.60},
}


def canonical_json(payload: dict) -> bytes:
    return json.dumps(payload, indent=2, sort_keys=True).encode() + b"\n"


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _largest_remainder(total: int, fractions: dict[str, float], names: tuple[str, ...]) -> dict[str, int]:
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


def assign_whole_groups(
    rows: list[dict],
    *,
    seed: int,
    fractions: dict[str, float],
    split_names: tuple[str, ...],
) -> list[dict]:
    """Give every member of a duplicate group the same split. Mixed-label groups stay out."""
    if abs(sum(fractions[name] for name in split_names) - 1.0) > 1e-9:
        raise ValueError("split fractions must sum to 1")
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["duplicate_group_id"]].append(dict(row))
    strata: dict[tuple[str, int], list[list[dict]]] = defaultdict(list)
    excluded: list[dict] = []
    for group_id in sorted(grouped):
        members = grouped[group_id]
        labels = {int(member["label"]) for member in members}
        if len(labels) != 1:
            for member in members:
                excluded.append({**member, "role": "manual_review"})
            continue
        brands = {member["brand"] for member in members}
        strata[(min(brands), min(labels))].append(members)

    rng = np.random.default_rng(int(seed))
    assigned: list[dict] = []
    for key in sorted(strata):
        groups = strata[key]
        order = rng.permutation(len(groups))
        shuffled = [groups[int(index)] for index in order]
        targets = _largest_remainder(sum(len(group) for group in shuffled), fractions, split_names)
        filled = {name: 0 for name in split_names}
        for group in shuffled:
            choice = max(
                split_names,
                key=lambda name: (
                    int(filled[name] == 0 and targets[name] > 0),
                    targets[name] - filled[name],
                    -split_names.index(name),
                ),
            )
            for member in group:
                assigned.append({**member, "role": choice})
            filled[choice] += len(group)
    return assigned + excluded


def locked_test_paths(samples: list[dict]) -> set[str]:
    return {row["path"] for row in samples if row["split"] == "test"}


def exclude_final_test(samples: list[dict]) -> tuple[list[dict], dict]:
    """Drop the locked test split and every duplicate group that touches it."""
    test_groups = {row["duplicate_group_id"] for row in samples if row["split"] == "test"}
    test_paths = locked_test_paths(samples)
    kept = []
    excluded_test = 0
    excluded_copies = 0
    for row in samples:
        if row["path"] in test_paths or row["split"] == "test":
            excluded_test += 1
            continue
        if row["duplicate_group_id"] in test_groups:
            excluded_copies += 1
            continue
        kept.append(row)
    return kept, {
        "locked_final_test_images_excluded": excluded_test,
        "duplicate_copies_of_final_test_excluded": excluded_copies,
        "eligible_images": len(kept),
    }


def _compact(row: dict, role: str) -> dict:
    return {
        "path": row["path"],
        "label": int(row["label"]),
        "brand": row["brand"],
        "duplicate_group_id": row["duplicate_group_id"],
        "sha256": row["sha256"],
        "role": role,
    }


def build_fold(samples: list[dict], held_out_brand: str, *, seed: int = SEED) -> dict:
    eligible, exclusion = exclude_final_test(samples)
    held = [row for row in eligible if row["brand"] == held_out_brand]
    others = [row for row in eligible if row["brand"] != held_out_brand]
    held_groups = {row["duplicate_group_id"] for row in held}
    quarantine = []
    pool = []
    for row in others:
        if row["duplicate_group_id"] in held_groups:
            quarantine.append(_compact(row, "leakage_quarantine"))
        else:
            pool.append(row)
    assigned = assign_whole_groups(
        pool,
        seed=seed,
        fractions=WITHIN_FOLD_FRACTIONS,
        split_names=("train", "calibration", "validation"),
    )
    grouped_held: dict[str, set[int]] = defaultdict(set)
    for row in held:
        grouped_held[row["duplicate_group_id"]].add(int(row["label"]))
    mixed_held_groups = {group for group, labels in grouped_held.items() if len(labels) > 1}
    rows = []
    for row in held:
        role = "manual_review" if row["duplicate_group_id"] in mixed_held_groups else "cross_brand_test"
        rows.append(_compact(row, role))
    rows.extend(quarantine)
    rows.extend(_compact(row, row["role"]) for row in assigned)
    fold = {
        "held_out_brand": held_out_brand,
        "seed": int(seed),
        "exclusion": exclusion,
        "leakage_quarantine": sum(1 for row in rows if row["role"] == "leakage_quarantine"),
        "rows": rows,
    }
    assert_fold_isolation(fold)
    return fold


def build_random_control(samples: list[dict], *, seed: int = SEED) -> dict:
    eligible, exclusion = exclude_final_test(samples)
    assigned = assign_whole_groups(
        eligible,
        seed=seed,
        fractions=RANDOM_FRACTIONS,
        split_names=("train", "calibration", "validation", "random_test"),
    )
    rows = [_compact(row, row["role"]) for row in assigned]
    control = {
        "held_out_brand": None,
        "seed": int(seed),
        "exclusion": exclusion,
        "rows": rows,
        "note": "Brand-mixed split of images that are not in the locked final test.",
    }
    brands = {row["brand"] for row in rows if row["role"] == "random_test"}
    if len(brands) < 2:
        raise RuntimeError("random control test collapsed to one brand")
    return control


def role_paths(fold: dict, role: str) -> set[str]:
    return {row["path"] for row in fold["rows"] if row["role"] == role}


def assert_fold_isolation(fold: dict) -> None:
    held = fold["held_out_brand"]
    failures = []
    for role in ("train", "calibration", "validation"):
        brands = {row["brand"] for row in fold["rows"] if row["role"] == role}
        if held in brands:
            failures.append(f"{held} appears in {role}")
        if not brands:
            failures.append(f"{role} is empty")
    test_rows = [row for row in fold["rows"] if row["role"] == "cross_brand_test"]
    if not test_rows:
        failures.append("cross-brand test is empty")
    if any(row["brand"] != held for row in test_rows):
        failures.append("cross-brand test contains another brand")
    owners: dict[str, str] = {}
    groups: dict[str, str] = {}
    hashes: dict[str, str] = {}
    for row in fold["rows"]:
        if row["role"] in {"manual_review", "leakage_quarantine"}:
            continue
        previous = owners.get(row["path"])
        if previous is not None and previous != row["role"]:
            failures.append(f"{row['path']} is in both {previous} and {row['role']}")
        owners[row["path"]] = row["role"]
        previous_group = groups.get(row["duplicate_group_id"])
        if previous_group is not None and previous_group != row["role"]:
            failures.append(f"duplicate group {row['duplicate_group_id']} crosses {previous_group} and {row['role']}")
        groups[row["duplicate_group_id"]] = row["role"]
        previous_hash = hashes.get(row["sha256"])
        if previous_hash is not None and previous_hash != row["role"]:
            failures.append(f"sha256 {row['sha256']} crosses splits")
        hashes[row["sha256"]] = row["role"]
    if failures:
        raise RuntimeError("; ".join(failures))


def assert_final_test_excluded(fold: dict, test_paths: set[str]) -> None:
    present = {row["path"] for row in fold["rows"]}
    overlap = present & test_paths
    if overlap:
        raise RuntimeError(f"{len(overlap)} locked final-test paths entered a research split")


def assert_selection_isolation(rows: list[dict], held_out_brand: str) -> list[dict]:
    """Checkpoint selection may see validation rows from the training brands only."""
    selected = [row for row in rows if row["role"] == "validation"]
    if any(row["brand"] == held_out_brand for row in selected):
        raise RuntimeError("checkpoint selection received the held-out brand")
    if any(row["role"] == "cross_brand_test" for row in selected):
        raise RuntimeError("checkpoint selection received the held-out test")
    return selected


def assert_calibration_isolation(rows: list[dict], held_out_brand: str) -> list[dict]:
    selected = [row for row in rows if row["role"] == "calibration"]
    if any(row["brand"] == held_out_brand for row in selected):
        raise RuntimeError("calibration received the held-out brand")
    return selected


def generalization_gap(in_domain: dict, unseen: dict) -> dict:
    gap = {}
    for name in METRIC_NAMES:
        left = in_domain.get(name)
        right = unseen.get(name)
        if left is None or right is None:
            gap[name] = None
            gap[f"{name}_relative_degradation"] = None
            continue
        difference = float(left) - float(right)
        gap[name] = difference
        gap[f"{name}_relative_degradation"] = None if float(left) == 0.0 else difference / float(left)
    return gap


def classify_generalization(fold_rows: list[dict]) -> str:
    """Read the pre-specified bands. One strong brand does not make the result strong."""
    if len(fold_rows) < len(FOLDS):
        raise RuntimeError("generalization is not classified until every fold has an unseen-brand score")
    unseen_pr = [float(row["unseen"]["pr_auc"]) for row in fold_rows]
    unseen_far = [float(row["unseen"]["false_authentic_rate"]) for row in fold_rows]
    gaps = [float(row["gap"]["pr_auc"]) for row in fold_rows]
    mean_pr = sum(unseen_pr) / len(unseen_pr)
    mean_far = sum(unseen_far) / len(unseen_far)
    mean_gap = sum(gaps) / len(gaps)
    strong = GENERALIZATION_BANDS["strong"]
    moderate = GENERALIZATION_BANDS["moderate"]
    weak = GENERALIZATION_BANDS["weak"]
    if (
        mean_pr >= strong["min_mean_unseen_pr_auc"]
        and mean_far <= strong["max_mean_unseen_false_authentic_rate"]
        and mean_gap <= strong["max_mean_pr_auc_gap"]
        and max(unseen_far) <= strong["max_fold_unseen_false_authentic_rate"]
    ):
        return "STRONG"
    if mean_pr >= moderate["min_mean_unseen_pr_auc"] and mean_far <= moderate["max_mean_unseen_false_authentic_rate"]:
        return "MODERATE"
    if mean_pr >= weak["min_mean_unseen_pr_auc"]:
        return "WEAK"
    return "FAILED"


def protocol_document() -> dict:
    return {
        "protocol_id": PROTOCOL_ID,
        "research_question": (
            "Can DINOv3 learn authenticity cues that generalize to a luxury-watch brand "
            "that was completely unseen during training?"
        ),
        "initialization": INITIALIZATION,
        "frozen_checkpoint_not_used_for_initialization": FROZEN_CHECKPOINT,
        "frozen_checkpoint_sha256": FROZEN_CHECKPOINT_SHA256,
        "architecture": {
            "family": "dinov3",
            "backbone": "vit_base_patch16_dinov3.lvd1689m",
            "head": "cls_patch_attention",
            "input_size": 512,
            "training_precision": "bf16",
            "evaluation_precision": "fp32",
        },
        "hyperparameters": {
            "batch_size": 16,
            "accumulate_grad": 2,
            "effective_batch": 32,
            "backbone_lr": 1e-5,
            "head_lr": 1e-4,
            "layer_decay": 0.75,
            "weight_decay": 0.05,
            "warmup_epochs": 2,
            "schedule": "cosine",
            "max_epochs": 30,
            "early_stop_patience": 6,
            "grad_clip": 1.0,
            "label_smoothing": 0.02,
            "mixup_alpha": 0.0,
            "seed": SEED,
        },
        "folds": [{"directory": name, "held_out_brand": brand} for name, brand in FOLDS],
        "within_fold_fractions": WITHIN_FOLD_FRACTIONS,
        "checkpoint_selection_rule": CHECKPOINT_SELECTION_RULE,
        "calibration_rule": CALIBRATION_RULE,
        "validation_threshold": 0.50,
        "production_ood_threshold": None,
        "phase35_ood_thresholds_reused": False,
        "locked_final_test_used": False,
        "ood_images_used_for_training_or_thresholds": False,
        "source_identity": "UNKNOWN on every catalog row, so source-family disjointness is not claimed",
        "duplicate_policy": (
            "Exact SHA-256 copies and the audited dHash groups stay in one role. "
            "A group that touches the locked final test is excluded. "
            "A training-brand image that shares a duplicate group with the held-out brand is quarantined."
        ),
        "generalization_bands": GENERALIZATION_BANDS,
        "mean_generalization_gap_definition": "mean over folds of in-domain PR-AUC minus unseen-brand PR-AUC",
        "secondary_ablation": "cls_only versus cls_patch_attention on the same fold manifests",
        "dinov2_baseline": "deferred; a second backbone family would materially expand this phase",
    }


def build_experiment(samples: list[dict]) -> dict:
    test_paths = locked_test_paths(samples)
    folds = {}
    for directory, brand in FOLDS:
        fold = build_fold(samples, brand, seed=SEED)
        assert_final_test_excluded(fold, test_paths)
        assert_selection_isolation(fold["rows"], brand)
        assert_calibration_isolation(fold["rows"], brand)
        folds[directory] = fold
    control = build_random_control(samples, seed=SEED)
    assert_final_test_excluded(control, test_paths)
    return {"folds": folds, "random_split_control": control}


def summarize_fold(fold: dict) -> dict:
    counts = {}
    for role in ("train", "calibration", "validation", "cross_brand_test", "random_test", "leakage_quarantine", "manual_review"):
        chosen = [row for row in fold["rows"] if row["role"] == role]
        if not chosen and role in {"random_test", "leakage_quarantine", "manual_review"}:
            continue
        counts[role] = {
            "images": len(chosen),
            "authentic": sum(1 for row in chosen if int(row["label"]) == 0),
            "fake": sum(1 for row in chosen if int(row["label"]) == 1),
            "brands": sorted({row["brand"] for row in chosen}),
        }
    return counts
