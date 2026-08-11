from __future__ import annotations

from pathlib import Path

import torch

from semirestore.checkpoints import atomic_torch_save, load_model_checkpoint
from semirestore.losses import CharbonnierLoss
from semirestore.models import NAFSR
from semirestore.models.naf_blocks import NAFBlock


def test_naf_block_starts_as_identity_and_backpropagates() -> None:
    block = NAFBlock(8)
    inputs = torch.randn((2, 8, 9, 11), requires_grad=True)
    output = block(inputs)

    assert torch.equal(output, inputs)
    output.mean().backward()
    assert inputs.grad is not None and torch.isfinite(inputs.grad).all()


def test_primary_naf_sr_budget_dynamic_shape_and_backward(tmp_path: Path) -> None:
    primary = NAFSR()
    parameter_count = sum(parameter.numel() for parameter in primary.parameters())
    assert parameter_count == 8_974_084
    assert parameter_count < 12_000_000
    assert parameter_count * 4 < 60 * 1024 * 1024
    checkpoint = tmp_path / "primary_model_only.pt"
    atomic_torch_save(
        {
            "format_version": 1,
            "checkpoint_role": "best_inference",
            "model_name": "naf_sr",
            "model_config": primary.model_config(),
            "model_state_dict": primary.state_dict(),
        },
        checkpoint,
    )
    assert checkpoint.stat().st_size < 60 * 1024 * 1024

    model = NAFSR(
        width=8,
        encoder_blocks=[1, 1],
        middle_blocks=1,
        decoder_blocks=[1, 1],
    )
    inputs = torch.randn((1, 1, 9, 11), requires_grad=True)
    target = torch.rand((1, 1, 18, 22))
    prediction = model(inputs)
    loss = CharbonnierLoss()(prediction, target)
    loss.backward()

    assert prediction.shape == target.shape
    assert torch.isfinite(prediction).all()
    assert torch.isfinite(loss)
    assert inputs.grad is not None and torch.isfinite(inputs.grad).all()


def test_naf_sr_self_describing_checkpoint_round_trip(tmp_path: Path) -> None:
    torch.manual_seed(22)
    model = NAFSR(
        width=8,
        encoder_blocks=[1],
        middle_blocks=1,
        decoder_blocks=[1],
    )
    checkpoint = tmp_path / "naf.pt"
    atomic_torch_save(
        {
            "format_version": 1,
            "model_name": "naf_sr",
            "model_config": model.model_config(),
            "model_state_dict": model.state_dict(),
        },
        checkpoint,
    )
    restored, payload = load_model_checkpoint(checkpoint)
    inputs = torch.randn((1, 1, 8, 10))

    assert payload["model_name"] == "naf_sr"
    assert torch.equal(model(inputs), restored(inputs))
