"""
DINOv2 + binary head — must match `ml/train.py` `DINOv2Classifier` for checkpoint compatibility.
"""

from __future__ import annotations

import torch
import torch.nn as nn

try:
    import timm

    HAS_TIMM = True
except ImportError:
    HAS_TIMM = False


class DINOv2Classifier(nn.Module):
    """DINOv2 backbone + binary classification head (same as training)."""

    def __init__(self, model_name: str, dropout: float = 0.2):
        super().__init__()
        self.backbone = self._load_backbone(model_name)
        embed_dim = self._get_embed_dim(self.backbone)
        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Dropout(p=dropout),
            nn.Linear(embed_dim, 512),
            nn.GELU(),
            nn.Dropout(p=dropout / 2),
            nn.Linear(512, 1),
        )

    def _load_backbone(self, model_name: str):
        hub_names = {
            "dinov2_vitg14_reg": "dinov2_vitg14_reg",
            "dinov2_vitg14": "dinov2_vitg14",
            "dinov2_vitl14_reg": "dinov2_vitl14_reg",
            "dinov2_vitl14": "dinov2_vitl14",
        }
        if model_name in hub_names:
            try:
                return torch.hub.load(
                    "facebookresearch/dinov2",
                    hub_names[model_name],
                    pretrained=True,
                    force_reload=False,
                )
            except Exception:
                pass

        if HAS_TIMM:
            timm_names = {
                "dinov2_vitg14_reg": "vit_giant_patch14_reg4_dinov2.lvd142m",
                "dinov2_vitg14": "vit_giant_patch14_dinov2.lvd142m",
                "dinov2_vitl14_reg": "vit_large_patch14_reg4_dinov2.lvd142m",
                "dinov2_vitl14": "vit_large_patch14_dinov2.lvd142m",
            }
            timm_name = timm_names.get(model_name, model_name)
            return timm.create_model(timm_name, pretrained=True, num_classes=0)

        raise RuntimeError("Install timm or ensure torch.hub can load facebookresearch/dinov2")

    def _get_embed_dim(self, backbone) -> int:
        if hasattr(backbone, "embed_dim"):
            return int(backbone.embed_dim)
        if hasattr(backbone, "num_features"):
            return int(backbone.num_features)
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 518, 518)
            out = backbone(dummy)
            if isinstance(out, dict):
                return int(out["x_norm_clstoken"].shape[-1])
            return int(out.shape[-1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        if isinstance(features, dict):
            features = features["x_norm_clstoken"]
        return self.head(features).squeeze(1)
