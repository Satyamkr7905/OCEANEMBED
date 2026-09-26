"""Multi-head depth cross-attention decoder with Arabian Sea / Bay of Bengal heads."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from track2_model_engine.config import BOB_LON, N_DEPTH
from track2_model_engine.models.layers import MCDropout


class DepthAttentionDecoder(nn.Module):
    """Map a spatial latent (B×512×H×W) onto 15 vertical tokens per column.

    The 512 latent channels are split into ``n_tokens`` memory slots. Fifteen
    learned depth queries attend over those slots (not over the full basin),
    so each water column keeps its local embedding while sharing depth bases.
    """

    def __init__(
        self,
        latent_dim: int = 512,
        n_depths: int = N_DEPTH,
        n_tokens: int = 8,
        n_heads: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if latent_dim % n_tokens != 0:
            raise ValueError("latent_dim must be divisible by n_tokens")
        if latent_dim % n_heads != 0:
            raise ValueError("latent_dim must be divisible by n_heads")
        self.latent_dim = latent_dim
        self.n_depths = n_depths
        self.n_tokens = n_tokens
        self.n_heads = n_heads
        self.token_dim = latent_dim // n_tokens
        self.head_dim = latent_dim // n_heads
        self.depth_query = nn.Parameter(torch.randn(n_depths, latent_dim) * 0.02)
        self.q_proj = nn.Linear(latent_dim, latent_dim)
        self.k_proj = nn.Linear(self.token_dim, latent_dim)
        self.v_proj = nn.Linear(self.token_dim, latent_dim)
        self.out_proj = nn.Linear(latent_dim, latent_dim)
        self.drop = MCDropout(dropout)
        self.norm_q = nn.LayerNorm(latent_dim)
        self.norm_out = nn.LayerNorm(latent_dim)
        self.ff = nn.Sequential(
            nn.Linear(latent_dim, latent_dim * 2),
            nn.GELU(),
            MCDropout(dropout),
            nn.Linear(latent_dim * 2, latent_dim),
        )
        self.head_as = nn.Linear(latent_dim, 4)
        self.head_bob = nn.Linear(latent_dim, 4)
        self.blend_scale = 0.35  # degrees of longitude for a smooth AS/BoB seam

    def _attend(self, latent: Tensor) -> Tensor:
        # latent: B, 512, H, W → tokens B, H, W, N, token_dim
        b, c, h, w = latent.shape
        tokens = latent.view(b, self.n_tokens, self.token_dim, h, w)
        tokens = tokens.permute(0, 3, 4, 1, 2).contiguous()
        q = self.norm_q(self.depth_query).view(1, 1, 1, self.n_depths, c)
        q = q.expand(b, h, w, self.n_depths, c).contiguous()
        q = self.q_proj(q)
        k = self.k_proj(tokens)
        v = self.v_proj(tokens)

        def split_heads(t: Tensor) -> Tensor:
            return t.view(*t.shape[:-1], self.n_heads, self.head_dim)

        qh = split_heads(q)   # B, H, W, Q, heads, hd
        kh = split_heads(k)   # B, H, W, N, heads, hd
        vh = split_heads(v)   # B, H, W, N, heads, hd
        scale = 1.0 / math.sqrt(self.head_dim)
        # Use distinct subscripts: b=batch, i=height, j=width, q=query, n=key, a=head, d=headdim
        attn = torch.einsum("bijqad,bijnad->bijqan", qh, kh) * scale
        attn = torch.softmax(attn, dim=-1)
        attn = self.drop(attn)
        ctx = torch.einsum("bijqan,bijnad->bijqad", attn, vh)
        ctx = ctx.reshape(b, h, w, self.n_depths, c)
        ctx = self.out_proj(ctx)
        ctx = ctx + q
        ctx = self.norm_out(ctx)
        ctx = ctx + self.ff(ctx)
        return ctx  # B, H, W, 15, C

    def _basin_blend(self, features: Tensor, lon: Tensor) -> Tensor:
        """Soft switch between AS and BoB linear heads along 80°E."""
        # features: B, H, W, 15, C
        as_logits = self.head_as(features)
        bob_logits = self.head_bob(features)
        lon = lon.to(dtype=features.dtype, device=features.device).view(1, 1, -1, 1, 1)
        mix = torch.sigmoid((lon - BOB_LON) / self.blend_scale)
        return (1.0 - mix) * as_logits + mix * bob_logits

    def forward(self, latent: Tensor, lon: Tensor) -> Tensor:
        """Return B×(15*4)×H×W packed dual-head tensor."""
        feats = self._attend(latent)
        packed = self._basin_blend(feats, lon)  # B H W 15 4
        packed = packed.permute(0, 3, 4, 1, 2).contiguous()
        b, d, four, h, w = packed.shape
        return packed.view(b, d * four, h, w)
