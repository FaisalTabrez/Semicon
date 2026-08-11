"""Parameter-free lower-bound baseline."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class BicubicRestorer(nn.Module):
    """Upscale a one-channel NCHW tensor by exactly 2× using bicubic sampling."""

    scale: int = 2

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 4 or inputs.shape[1] != 1:
            raise ValueError(
                f"BicubicRestorer expects NCHW input with one channel; got {tuple(inputs.shape)}"
            )
        return F.interpolate(
            inputs,
            scale_factor=self.scale,
            mode="bicubic",
            align_corners=False,
            antialias=True,
        )
