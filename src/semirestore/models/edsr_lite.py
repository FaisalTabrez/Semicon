"""Compact EDSR-style learned baseline for joint denoising and 2x SR."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class ResidualBlock(nn.Module):
    def __init__(self, width: int, residual_scale: float) -> None:
        super().__init__()
        self.residual_scale = residual_scale
        self.body = nn.Sequential(
            nn.Conv2d(width, width, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(width, width, 3, padding=1),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs + self.body(inputs) * self.residual_scale


class EDSRLite(nn.Module):
    """Low-resolution residual trunk with a 2x pixel-shuffle residual head."""

    scale: int = 2

    def __init__(
        self,
        *,
        width: int = 64,
        num_blocks: int = 16,
        residual_scale: float = 0.1,
    ) -> None:
        super().__init__()
        if width < 4:
            raise ValueError("EDSRLite width must be at least 4")
        if num_blocks < 1:
            raise ValueError("EDSRLite needs at least one residual block")
        if not 0.0 < residual_scale <= 1.0:
            raise ValueError("residual_scale must be in (0, 1]")

        self.width = width
        self.num_blocks = num_blocks
        self.residual_scale = residual_scale
        self.head = nn.Conv2d(1, width, 3, padding=1)
        self.blocks = nn.Sequential(
            *(ResidualBlock(width, residual_scale) for _ in range(num_blocks))
        )
        self.body_tail = nn.Conv2d(width, width, 3, padding=1)
        self.upsample = nn.Sequential(
            nn.Conv2d(width, width * self.scale * self.scale, 3, padding=1),
            nn.PixelShuffle(self.scale),
            nn.ReLU(inplace=True),
            nn.Conv2d(width, 1, 3, padding=1),
        )

        # Start close to the transparent bicubic baseline without blocking
        # gradients into the learned residual path.
        final = self.upsample[-1]
        nn.init.normal_(final.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(final.bias)

    def model_config(self) -> dict[str, int | float]:
        return {
            "width": self.width,
            "num_blocks": self.num_blocks,
            "residual_scale": self.residual_scale,
        }

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 4 or inputs.shape[1] != 1:
            raise ValueError(
                f"EDSRLite expects NCHW input with one channel; got {tuple(inputs.shape)}"
            )
        features = self.head(inputs)
        residual_features = self.body_tail(self.blocks(features)) + features
        learned_residual = self.upsample(residual_features)
        bicubic = F.interpolate(
            inputs,
            scale_factor=self.scale,
            mode="bicubic",
            align_corners=False,
            antialias=True,
        )
        return bicubic + learned_residual
