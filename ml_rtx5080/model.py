"""Shared DINOv2 / DINOv3 backbone adapter.

Token layout was measured with timm 1.0.26, pretrained=False, on CPU:

DINOv2 ViT-B/14 (``vit_base_patch14_dinov2.lvd142m``), dynamic 504×504
    VisionTransformer, global_pool='token', num_prefix_tokens=1, no register token
    forward_features: [B, 1297, 768] = 1 CLS + 1296 patches
    forward: [B, 768] CLS

DINOv3 ViT-B/16 (``vit_base_patch16_dinov3.lvd1689m``), dynamic 512×512
    Eva, default global_pool='avg' (this adapter forces 'token')
    num_prefix_tokens=5, reg_token [1, 4, 768], no num_reg_tokens attribute
    forward_features: [B, 1029, 768] = 1 CLS + 4 registers + 1024 patches
    504×504 raises AssertionError inside PatchEmbed because 504 % 16 != 0

Order in both ``_pos_embed`` implementations is CLS, then registers, then patches.
"""

from __future__ import annotations

import torch
import torch.nn as nn

try:
    import timm
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("timm is required to build DINO backbones") from exc


DINOV2_TIMM_NAME = "vit_base_patch14_dinov2.lvd142m"
DINOV3_TIMM_NAME = "vit_base_patch16_dinov3.lvd1689m"
DINOV2_IMAGE_SIZE = 504
DINOV3_IMAGE_SIZE = 512
DINOV2_PATCH_SIZE = 14
DINOV3_PATCH_SIZE = 16
EMBED_DIM = 768


def _register_count(module: nn.Module) -> int:
    register = getattr(module, "reg_token", None)
    if register is None:
        return 0
    return int(register.shape[1])


class DinoBackbone(nn.Module):
    """Thin adapter over a timm DINOv2 or DINOv3 encoder."""

    def __init__(
        self,
        encoder: nn.Module,
        *,
        family: str,
        backbone_name: str,
        image_size: int,
        patch_size: int,
        num_register_tokens: int,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.family = family
        self.backbone_name = backbone_name
        self.image_size = int(image_size)
        self.patch_size = int(patch_size)
        self.num_register_tokens = int(num_register_tokens)
        self.embed_dim = int(getattr(encoder, "embed_dim"))

    def forward_features(self, images: torch.Tensor) -> torch.Tensor:
        """Return token sequence [B, 1 + registers + patches, C]."""
        self._check_images(images)
        features = self.encoder.forward_features(images)
        if features.ndim != 3:
            raise RuntimeError(f"Expected token features [B, N, C], got {tuple(features.shape)}")
        prefix = 1 + self.num_register_tokens
        if features.shape[1] <= prefix or features.shape[-1] != self.embed_dim:
            raise RuntimeError(
                f"Unexpected feature shape {tuple(features.shape)} for "
                f"{self.family} registers={self.num_register_tokens}"
            )
        return features

    def get_cls_token(self, features: torch.Tensor) -> torch.Tensor:
        """CLS token at index 0. Shape [B, C]."""
        self._check_features(features)
        return features[:, 0]

    def get_patch_tokens(self, features: torch.Tensor) -> torch.Tensor:
        """Patch tokens with register tokens removed. Shape [B, patches, C]."""
        self._check_features(features)
        start = 1 + self.num_register_tokens
        return features[:, start:]

    def get_embed_dim(self) -> int:
        return self.embed_dim

    def get_patch_size(self) -> int:
        return self.patch_size

    def _check_images(self, images: torch.Tensor) -> None:
        if images.ndim != 4:
            raise ValueError(f"Expected NCHW images, got {tuple(images.shape)}")
        _, _, height, width = images.shape
        if height != self.image_size or width != self.image_size:
            raise ValueError(
                f"{self.family} backbone is configured for {self.image_size}×{self.image_size}, "
                f"got {height}×{width}"
            )
        if height % self.patch_size != 0 or width % self.patch_size != 0:
            raise ValueError(
                f"Image size {height}×{width} is not divisible by patch size {self.patch_size}"
            )

    def _check_features(self, features: torch.Tensor) -> None:
        if features.ndim != 3:
            raise ValueError(
                "get_cls_token/get_patch_tokens expect forward_features output [B, N, C], "
                f"got {tuple(features.shape)}"
            )


def _validate_resolution(family: str, image_size: int) -> tuple[str, int, int]:
    if family == "dinov2":
        if image_size % DINOV2_PATCH_SIZE != 0:
            raise ValueError(
                f"DINOv2 ViT-B/14 requires a multiple of {DINOV2_PATCH_SIZE}, got {image_size}"
            )
        return DINOV2_TIMM_NAME, DINOV2_PATCH_SIZE, image_size
    if family == "dinov3":
        if image_size == 504:
            raise ValueError("DINOv3 ViT-B/16 cannot use 504×504; use 512×512")
        if image_size % DINOV3_PATCH_SIZE != 0:
            raise ValueError(
                f"DINOv3 ViT-B/16 requires a multiple of {DINOV3_PATCH_SIZE}, got {image_size}"
            )
        return DINOV3_TIMM_NAME, DINOV3_PATCH_SIZE, image_size
    raise ValueError(f"Unsupported backbone family '{family}'. Expected dinov2 or dinov3")


def build_dino_backbone(
    family: str,
    *,
    image_size: int | None = None,
    pretrained: bool = False,
) -> DinoBackbone:
    """Build a ViT-B backbone. ``pretrained=False`` skips the weight download."""
    if image_size is None:
        image_size = DINOV2_IMAGE_SIZE if family == "dinov2" else DINOV3_IMAGE_SIZE
    timm_name, patch_size, image_size = _validate_resolution(family, int(image_size))
    create_kwargs: dict = {"pretrained": pretrained, "num_classes": 0}
    if family == "dinov2":
        # Native timm grid is 518. 504 only works when absolute positions can be resampled.
        create_kwargs["dynamic_img_size"] = True
    else:
        # timm's DINOv3 default global_pool is 'avg', which mixes registers into the vector.
        create_kwargs["global_pool"] = "token"
    encoder = timm.create_model(timm_name, **create_kwargs)
    registers = _register_count(encoder)
    expected_registers = 0 if family == "dinov2" else 4
    if registers != expected_registers:
        raise RuntimeError(
            f"{timm_name} exposed {registers} register tokens, expected {expected_registers}"
        )
    prefix = int(getattr(encoder, "num_prefix_tokens"))
    if prefix != 1 + registers:
        raise RuntimeError(f"{timm_name} num_prefix_tokens={prefix}, expected {1 + registers}")
    return DinoBackbone(
        encoder,
        family=family,
        backbone_name=timm_name,
        image_size=image_size,
        patch_size=patch_size,
        num_register_tokens=registers,
    )
