"""Per-block learning rates for DINOv2 and DINOv3 fine-tuning.

The last transformer block uses ``base_backbone_lr``. Each step toward the
input multiplies that rate by ``layer_decay``. Embeddings sit one step below
the first block. The classifier head uses ``head_lr`` and is not decayed.

Block count comes from the encoder's ``blocks`` module. It is not hardcoded.
"""

from __future__ import annotations

import re

import torch.nn as nn

DEFAULT_BACKBONE_LR = 1e-5
DEFAULT_HEAD_LR = 1e-4
DEFAULT_LAYER_DECAY = 0.75
DEFAULT_WEIGHT_DECAY = 0.05

_BLOCK_RE = re.compile(r"(?:^|\.)blocks\.(\d+)(?:\.|$)")
_HEAD_RE = re.compile(r"(?:^|\.)(?:trunk|attention_pool|fusion|head)(?:\.|$)")
_FINAL_NORM_RE = re.compile(r"(?:^|\.)norm(?:\.|$)")


def encoder_of(model: nn.Module) -> nn.Module:
    if hasattr(model, "backbone") and hasattr(model.backbone, "encoder"):
        return model.backbone.encoder
    if hasattr(model, "encoder"):
        return model.encoder
    if hasattr(model, "blocks"):
        return model
    raise ValueError("Cannot find a transformer encoder with a block list")


def transformer_blocks(model: nn.Module) -> nn.Module:
    blocks = getattr(encoder_of(model), "blocks", None)
    if blocks is None or len(blocks) == 0:
        raise ValueError("Encoder has no transformer blocks")
    return blocks


def backbone_learning_rate(depth_from_output: int, base_backbone_lr: float, layer_decay: float) -> float:
    """``depth_from_output=0`` is the last block and receives ``base_backbone_lr``."""
    if layer_decay <= 0 or layer_decay > 1:
        raise ValueError(f"layer_decay must be in (0, 1], got {layer_decay}")
    if base_backbone_lr < 0:
        raise ValueError(f"base_backbone_lr must be non-negative, got {base_backbone_lr}")
    return float(base_backbone_lr) * (float(layer_decay) ** int(depth_from_output))


def normalization_parameter_ids(model: nn.Module) -> set[int]:
    """Affine parameters owned by a normalization module.

    DINOv2 and DINOv3 both use ``nn.LayerNorm`` (timm's LayerNorm subclasses it)
    for block norms and the final norm. The authenticity trunk and attention
    pool do the same. Biases of those norms are included here and also caught
    by the ``.bias`` name rule.
    """
    norm_types = (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d, nn.GroupNorm)
    found: set[int] = set()
    for module in model.modules():
        if isinstance(module, norm_types):
            for parameter in module.parameters(recurse=False):
                found.add(id(parameter))
    return found


def uses_weight_decay(name: str, parameter: nn.Parameter, norm_ids: set[int]) -> bool:
    """Decay rank-2+ matrices. Skip biases, norms, and LayerScale vectors.

    Measured names, not a blanket ``weight_decay`` on every parameter:

    * DINOv2 blocks expose ``ls1.gamma`` and ``ls2.gamma`` (shape ``[768]``).
    * DINOv3 blocks expose ``gamma_1`` and ``gamma_2`` (shape ``[768]``).
    * Both families use ``nn.LayerNorm`` weight/bias, plus linear biases.

    Those are rank-1. Weight decay would pull a LayerScale gamma toward zero
    and shrink the residual it gates. Token embeddings (``cls_token``,
    ``pos_embed``, ``reg_token``) are rank-3 and stay in the decay group.
    RoPE period buffers are not parameters and never reach this check.
    """
    if parameter.ndim <= 1:
        return False
    if name == "bias" or name.endswith(".bias"):
        return False
    if id(parameter) in norm_ids:
        return False
    return True


def _group_for_parameter(name: str, n_blocks: int) -> tuple[str, int]:
    block_match = _BLOCK_RE.search(name)
    if block_match:
        index = int(block_match.group(1))
        if index >= n_blocks:
            raise ValueError(f"Parameter {name} refers to block {index}, but the encoder has {n_blocks}")
        return f"block_{index:02d}", n_blocks - 1 - index
    if _HEAD_RE.search(name):
        return "head", -1
    if _FINAL_NORM_RE.search(name):
        return "final_norm", 0
    return "embed", n_blocks


def build_layerwise_param_groups(
    model: nn.Module,
    *,
    base_backbone_lr: float = DEFAULT_BACKBONE_LR,
    head_lr: float = DEFAULT_HEAD_LR,
    layer_decay: float = DEFAULT_LAYER_DECAY,
    weight_decay: float = DEFAULT_WEIGHT_DECAY,
    separate_no_decay: bool = False,
) -> list[dict]:
    """One optimizer group per depth. Frozen parameters are omitted.

    ``separate_no_decay=False`` keeps the Phase 7 grouping: every parameter at
    a depth shares one weight decay. The trainer passes ``True``, which splits
    each depth into a decay group and a ``*_no_decay`` group at the same
    learning rate. Biases and normalization affine parameters are the no-decay
    side. Group order and parameter order follow ``named_parameters``.
    """
    n_blocks = len(transformer_blocks(model))
    order = ["embed", *[f"block_{index:02d}" for index in range(n_blocks)], "final_norm", "head"]
    buckets: dict[str, list] = {name: [] for name in order}
    depths = {
        "embed": n_blocks,
        **{f"block_{index:02d}": n_blocks - 1 - index for index in range(n_blocks)},
        "final_norm": 0,
        "head": None,
    }
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        group_name, depth = _group_for_parameter(name, n_blocks)
        if group_name not in buckets:
            raise ValueError(f"No learning-rate bucket for {name}")
        if depths[group_name] != depth and group_name != "head":
            raise ValueError(f"Inconsistent depth for {name}")
        buckets[group_name].append((name, parameter))

    norm_ids = normalization_parameter_ids(model) if separate_no_decay else set()
    groups: list[dict] = []
    for group_name in order:
        named = buckets[group_name]
        if not named:
            continue
        if group_name == "head":
            learning_rate = float(head_lr)
        else:
            learning_rate = backbone_learning_rate(depths[group_name], base_backbone_lr, layer_decay)
        if not separate_no_decay:
            groups.append(
                {
                    "name": group_name,
                    "params": [parameter for _, parameter in named],
                    "lr": learning_rate,
                    "weight_decay": float(weight_decay),
                }
            )
            continue
        decay = [
            parameter
            for name, parameter in named
            if uses_weight_decay(name, parameter, norm_ids)
        ]
        no_decay = [
            parameter
            for name, parameter in named
            if not uses_weight_decay(name, parameter, norm_ids)
        ]
        if decay:
            groups.append(
                {
                    "name": group_name,
                    "params": decay,
                    "lr": learning_rate,
                    "weight_decay": float(weight_decay),
                }
            )
        if no_decay:
            groups.append(
                {
                    "name": f"{group_name}_no_decay",
                    "params": no_decay,
                    "lr": learning_rate,
                    "weight_decay": 0.0,
                }
            )
    return groups


def format_param_group_table(groups: list[dict]) -> str:
    lines = ["group                  params        lr              weight_decay"]
    for group in groups:
        count = sum(parameter.numel() for parameter in group["params"])
        lines.append(
            f"{group['name']:<22} {count:<12}  {group['lr']:.8e}  {group['weight_decay']:.4f}"
        )
    return "\n".join(lines)


def format_decay_summary(groups: list[dict]) -> str:
    """Counts and a few representative groups. Safe to print at startup."""
    decay_count = 0
    no_decay_count = 0
    for group in groups:
        count = sum(parameter.numel() for parameter in group["params"])
        if float(group["weight_decay"]) == 0.0:
            no_decay_count += count
        else:
            decay_count += count
    representative = {
        "embed",
        "embed_no_decay",
        "block_00",
        "block_00_no_decay",
        "block_11",
        "block_11_no_decay",
        "final_norm",
        "final_norm_no_decay",
        "head",
        "head_no_decay",
    }
    lines = [
        f"decay parameters: {decay_count}",
        f"no-decay parameters: {no_decay_count}",
    ]
    for group in groups:
        if group["name"] not in representative:
            continue
        count = sum(parameter.numel() for parameter in group["params"])
        lines.append(
            f"{group['name']}: count={count} lr={group['lr']:.8e} weight_decay={group['weight_decay']:.4f}"
        )
    return "\n".join(lines)
