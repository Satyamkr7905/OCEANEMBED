"""Point-wise MLP baseline (1×1 convolutions over the spatial grid).

Each water column is mapped independently from the flattened 7-day channel
vector onto 15-depth residual T/S (+ log-variance).
"""

from __future__ import annotations

from torch import Tensor, nn

from track2_model_engine.config import N_CHANNELS, N_DEPTH, WINDOW_DAYS
from track2_model_engine.models.layers import MCDropout
from track2_model_engine.models.heads import ModelOutput, split_dual_head


class PointwiseMLP(nn.Module):
    def __init__(
        self,
        in_channels: int = N_CHANNELS,
        window: int = WINDOW_DAYS,
        hidden: int = 256,
        n_depths: int = N_DEPTH,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.window = window
        self.in_channels = in_channels
        feat = in_channels * window
        self.net = nn.Sequential(
            nn.Conv2d(feat, hidden, kernel_size=1),
            nn.GELU(),
            MCDropout(dropout),
            nn.Conv2d(hidden, hidden, kernel_size=1),
            nn.GELU(),
            MCDropout(dropout),
            nn.Conv2d(hidden, n_depths * 4, kernel_size=1),
        )

    def forward(self, x: Tensor) -> ModelOutput:
        # x: B, T, C, H, W  — use the full window (pointwise in space only)
        b, t, c, h, w = x.shape
        if t != self.window or c != self.in_channels:
            raise ValueError(f"Expected T={self.window}, C={self.in_channels}, got {t}, {c}")
        flat = x.reshape(b, t * c, h, w)
        return split_dual_head(self.net(flat))
