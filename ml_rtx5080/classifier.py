"""Authenticity heads shared by DINOv2 and DINOv3.

``cls_only`` and ``cls_patch_attention`` use the same trunk:

    LayerNorm -> Linear(512) -> GELU -> Dropout -> Linear(1)

The patch variant adds one attention pool and a concat fusion down to the
trunk width. Neither path applies sigmoid. The logit is P(fake) before calibration.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from attention_pool import PatchAttentionPool
from model import DinoBackbone, build_dino_backbone

CLS_ONLY = "cls_only"
CLS_PATCH_ATTENTION = "cls_patch_attention"
CLASSIFIER_ARCHS = (CLS_ONLY, CLS_PATCH_ATTENTION)
DEFAULT_DROPOUT = 0.15


class ClassifierTrunk(nn.Module):
    """Shared MLP from an embed_dim vector to one raw logit."""

    def __init__(self, embed_dim: int, dropout: float = DEFAULT_DROPOUT) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim)
        self.fc1 = nn.Linear(embed_dim, 512)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(512, 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        hidden = self.drop(self.act(self.fc1(self.norm(features))))
        return self.fc2(hidden).squeeze(-1)


class AuthenticityClassifier(nn.Module):
    """Backbone plus one of the two authenticity heads."""

    def __init__(
        self,
        backbone: DinoBackbone,
        classifier_arch: str,
        dropout: float = DEFAULT_DROPOUT,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.classifier_arch = _validate_classifier_arch(classifier_arch)
        embed_dim = backbone.get_embed_dim()
        self.trunk = ClassifierTrunk(embed_dim, dropout=dropout)
        if self.classifier_arch == CLS_PATCH_ATTENTION:
            self.attention_pool: PatchAttentionPool | None = PatchAttentionPool(embed_dim)
            self.fusion: nn.Linear | None = nn.Linear(embed_dim * 2, embed_dim)
        else:
            self.attention_pool = None
            self.fusion = None

    def forward(
        self,
        images: torch.Tensor,
        return_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor | None]:
        features = self.backbone.forward_features(images)
        cls_token = self.backbone.get_cls_token(features)
        attention_weights: torch.Tensor | None = None
        if self.classifier_arch == CLS_ONLY:
            hidden = cls_token
        else:
            patch_tokens = self.backbone.get_patch_tokens(features)
            assert self.attention_pool is not None and self.fusion is not None
            pooled, attention_weights = self.attention_pool(patch_tokens)
            hidden = self.fusion(torch.cat((cls_token, pooled), dim=-1))
        logits = self.trunk(hidden)
        if return_attention:
            return logits, attention_weights
        return logits


def _validate_classifier_arch(classifier_arch: str) -> str:
    if classifier_arch not in CLASSIFIER_ARCHS:
        allowed = ", ".join(CLASSIFIER_ARCHS)
        raise ValueError(f"Unknown classifier_arch '{classifier_arch}'. Expected one of: {allowed}.")
    return classifier_arch


def build_authenticity_classifier(
    family: str,
    classifier_arch: str,
    *,
    image_size: int | None = None,
    pretrained: bool = False,
    dropout: float = DEFAULT_DROPOUT,
) -> AuthenticityClassifier:
    """Build a backbone and head. ``classifier_arch`` selects the head."""
    arch = _validate_classifier_arch(classifier_arch)
    backbone = build_dino_backbone(family, image_size=image_size, pretrained=pretrained)
    return AuthenticityClassifier(backbone, arch, dropout=dropout)
