"""Training losses for conservative restoration baselines."""

from __future__ import annotations

import torch
from torch import nn


class CharbonnierLoss(nn.Module):
    """Smooth L1-like fidelity loss: mean(sqrt(error^2 + epsilon^2))."""

    def __init__(self, epsilon: float = 1e-3) -> None:
        super().__init__()
        if epsilon <= 0.0:
            raise ValueError("Charbonnier epsilon must be positive")
        self.epsilon = epsilon

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if prediction.shape != target.shape:
            raise ValueError(
                f"Charbonnier shape mismatch: {tuple(prediction.shape)} vs {tuple(target.shape)}"
            )
        difference = prediction - target
        return torch.sqrt(difference.square() + self.epsilon**2).mean()
