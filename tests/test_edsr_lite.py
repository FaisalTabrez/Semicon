from __future__ import annotations

from pathlib import Path

import pytest
import torch

from semirestore.checkpoints import atomic_torch_save, load_model_checkpoint
from semirestore.losses import CharbonnierLoss
from semirestore.models import EDSRLite


def test_edsr_lite_dynamic_shape_backward_and_budget() -> None:
    model = EDSRLite(width=16, num_blocks=2)
    inputs = torch.randn((2, 1, 12, 15), requires_grad=True)
    target = torch.rand((2, 1, 24, 30))

    prediction = model(inputs)
    loss = CharbonnierLoss()(prediction, target)
    loss.backward()

    assert prediction.shape == target.shape
    assert torch.isfinite(prediction).all()
    assert torch.isfinite(loss)
    assert inputs.grad is not None and torch.isfinite(inputs.grad).all()
    assert sum(parameter.numel() for parameter in model.parameters()) < 12_000_000


def test_charbonnier_validates_inputs() -> None:
    with pytest.raises(ValueError, match="positive"):
        CharbonnierLoss(0.0)
    with pytest.raises(ValueError, match="shape mismatch"):
        CharbonnierLoss()(torch.zeros(1, 1, 2, 2), torch.zeros(1, 1, 4, 4))


def test_self_describing_checkpoint_round_trip(tmp_path: Path) -> None:
    torch.manual_seed(9)
    model = EDSRLite(width=8, num_blocks=1, residual_scale=0.2)
    checkpoint = tmp_path / "model.pt"
    atomic_torch_save(
        {
            "format_version": 1,
            "model_name": "edsr_lite",
            "model_config": model.model_config(),
            "model_state_dict": model.state_dict(),
        },
        checkpoint,
    )

    restored, payload = load_model_checkpoint(checkpoint)
    inputs = torch.randn((1, 1, 7, 9))

    assert payload["model_name"] == "edsr_lite"
    assert torch.equal(model(inputs), restored(inputs))
