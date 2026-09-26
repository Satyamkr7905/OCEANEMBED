"""Flagship OceanEmbed: dual-stream encoder + depth-attention T–S decoder."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from track2_model_engine.config import N_CHANNELS, N_DEPTH, N_LON, WINDOW_DAYS, target_longitudes
from track2_model_engine.models.depth_decoder import DepthAttentionDecoder
from track2_model_engine.models.dual_stream_encoder import DualStreamEncoder
from track2_model_engine.models.heads import ModelOutput, split_dual_head
from track2_model_engine.models.layers import set_mc_dropout


class OceanEmbed(nn.Module):
    """Physics-informed satellite-to-subsurface reconstruction network.

    Input:  ``(B, 7, C, 100, 240)``
    Output: residual Δθ, ΔS_p and log-σ at 15 depths, each ``(B, 15, 100, 240)``.
    """

    def __init__(
        self,
        in_channels: int = N_CHANNELS,
        window: int = WINDOW_DAYS,
        stream_a_dim: int = 64,
        stream_b_dim: int = 64,
        latent_dim: int = 512,
        n_depths: int = N_DEPTH,
        n_tokens: int = 8,
        n_heads: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.window = window
        self.in_channels = in_channels
        self.encoder = DualStreamEncoder(
            in_channels=in_channels,
            stream_a_dim=stream_a_dim,
            stream_b_dim=stream_b_dim,
            latent_dim=latent_dim,
            dropout=dropout,
        )
        self.decoder = DepthAttentionDecoder(
            latent_dim=latent_dim,
            n_depths=n_depths,
            n_tokens=n_tokens,
            n_heads=n_heads,
            dropout=dropout,
        )
        lon = torch.from_numpy(target_longitudes()).float()
        self.register_buffer("lon", lon, persistent=False)

    def set_mc_dropout(self, enabled: bool) -> None:
        set_mc_dropout(self, enabled)

    def forward(self, x: Tensor, lon: Tensor | None = None) -> ModelOutput:
        if x.ndim != 5:
            raise ValueError("OceanEmbed expects B×T×C×H×W")
        b, t, c, h, w = x.shape
        if t != self.window or c != self.in_channels:
            raise ValueError(f"Expected T={self.window} C={self.in_channels}, got T={t} C={c}")
        if w != N_LON and lon is None:
            raise ValueError("Provide lon when width != 240")
        x_bcthw = x.permute(0, 2, 1, 3, 4).contiguous()
        latent = self.encoder(x_bcthw)
        packed = self.decoder(latent, lon if lon is not None else self.lon)
        return split_dual_head(packed, n_depths=self.decoder.n_depths)


def build_model(name: str, **kwargs) -> nn.Module:
    """Factory used by the Lightning harness and export scripts."""
    key = name.lower().replace("-", "_")
    if key in {"mlp", "baseline_mlp", "pointwise_mlp"}:
        from track2_model_engine.models.baseline_mlp import PointwiseMLP

        return PointwiseMLP(**kwargs)
    if key in {"unet", "baseline_unet", "unet2d"}:
        from track2_model_engine.models.baseline_unet import UNet2D

        return UNet2D(**kwargs)
    if key in {"oceanembed", "ocean_embed", "flagship"}:
        return OceanEmbed(**kwargs)
    raise ValueError(f"Unknown model '{name}'")
