"""Lightweight patch-token attention pool.

Scores each patch with a LayerNorm and a single linear map to one logit, then
softmaxes over the spatial tokens and takes a weighted sum. That is O(N)
in the number of patches. It does not flatten patches into a wide dense layer
and does not run a transformer over them.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class PatchAttentionPool(nn.Module):
    """Pool [B, N, C] patch tokens to [B, C] with learned spatial attention."""

    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim)
        self.score = nn.Linear(embed_dim, 1)

    def forward(self, patch_tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if patch_tokens.ndim != 3:
            raise ValueError(f"Expected patch tokens [B, N, C], got {tuple(patch_tokens.shape)}")
        if patch_tokens.shape[1] < 1:
            raise ValueError("Attention pooling received no patch tokens")
        scores = self.score(self.norm(patch_tokens)).squeeze(-1)
        weights = torch.softmax(scores, dim=-1)
        pooled = torch.bmm(weights.unsqueeze(1), patch_tokens).squeeze(1)
        return pooled, weights
