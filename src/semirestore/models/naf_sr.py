"""Three-scale NAF encoder/decoder with a conservative 2x SR head."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional as F

from .naf_blocks import NAFBlock


def _block_stack(channels: int, count: int, dropout: float) -> nn.Sequential:
    if count < 1:
        raise ValueError("Every NAF stage needs at least one block")
    return nn.Sequential(*(NAFBlock(channels, dropout=dropout) for _ in range(count)))


class NAFSR(nn.Module):
    """Unconditioned primary restoration model operating mostly at LR."""

    scale: int = 2

    def __init__(
        self,
        *,
        width: int = 48,
        encoder_blocks: Sequence[int] = (2, 2, 4),
        middle_blocks: int = 6,
        decoder_blocks: Sequence[int] = (2, 2, 2),
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        encoder_counts = tuple(int(value) for value in encoder_blocks)
        decoder_counts = tuple(int(value) for value in decoder_blocks)
        if width < 4:
            raise ValueError("NAFSR width must be at least 4")
        if not encoder_counts or len(encoder_counts) != len(decoder_counts):
            raise ValueError("encoder_blocks and decoder_blocks need the same non-zero length")
        if middle_blocks < 1:
            raise ValueError("NAFSR needs at least one middle block")

        self.width = width
        self.encoder_counts = encoder_counts
        self.middle_blocks = middle_blocks
        self.decoder_counts = decoder_counts
        self.dropout = dropout
        self.padder_size = 2 ** len(encoder_counts)

        self.intro = nn.Conv2d(1, width, 3, padding=1)
        self.encoders = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        channels = width
        for count in encoder_counts:
            self.encoders.append(_block_stack(channels, count, dropout))
            self.downsamples.append(nn.Conv2d(channels, channels * 2, 2, stride=2))
            channels *= 2

        self.middle = _block_stack(channels, middle_blocks, dropout)
        self.upsamples = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for count in decoder_counts:
            self.upsamples.append(
                nn.Sequential(
                    nn.Conv2d(channels, channels * 2, 1, bias=False),
                    nn.PixelShuffle(2),
                )
            )
            channels //= 2
            self.decoders.append(_block_stack(channels, count, dropout))

        self.sr_head = nn.Sequential(
            nn.Conv2d(width, self.scale * self.scale, 3, padding=1),
            nn.PixelShuffle(self.scale),
        )
        final = self.sr_head[0]
        nn.init.normal_(final.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(final.bias)

    def model_config(self) -> dict[str, object]:
        return {
            "width": self.width,
            "encoder_blocks": list(self.encoder_counts),
            "middle_blocks": self.middle_blocks,
            "decoder_blocks": list(self.decoder_counts),
            "dropout": self.dropout,
        }

    def _pad(self, inputs: torch.Tensor) -> torch.Tensor:
        height, width = inputs.shape[-2:]
        pad_height = (self.padder_size - height % self.padder_size) % self.padder_size
        pad_width = (self.padder_size - width % self.padder_size) % self.padder_size
        return F.pad(inputs, (0, pad_width, 0, pad_height), mode="replicate")

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 4 or inputs.shape[1] != 1:
            raise ValueError(f"NAFSR expects NCHW input with one channel; got {tuple(inputs.shape)}")
        height, width = inputs.shape[-2:]
        features = self.intro(self._pad(inputs))
        skips: list[torch.Tensor] = []
        for encoder, downsample in zip(self.encoders, self.downsamples, strict=True):
            features = encoder(features)
            skips.append(features)
            features = downsample(features)

        features = self.middle(features)
        for upsample, decoder, skip in zip(
            self.upsamples, self.decoders, reversed(skips), strict=True
        ):
            features = decoder(upsample(features) + skip)

        learned_residual = self.sr_head(features)[..., : height * 2, : width * 2]
        bicubic = F.interpolate(
            inputs,
            scale_factor=self.scale,
            mode="bicubic",
            align_corners=False,
            antialias=True,
        )
        return bicubic + learned_residual
