from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from semirestore.data import InputValidationError
from semirestore.inference import resolve_precision, restore_directory
from semirestore.models import BicubicRestorer


class RecordingRestorer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.observed_min: float | None = None
        self.observed_max: float | None = None

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        self.observed_min = float(inputs.min())
        self.observed_max = float(inputs.max())
        return torch.nn.functional.interpolate(inputs, scale_factor=2, mode="nearest")


def test_inference_preserves_raw_input_then_clips_output(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    np.save(input_dir / "sample.npy", np.array([[-0.5, 2.0], [0.25, 0.75]], dtype=np.float32))
    model = RecordingRestorer()

    summary = restore_directory(
        model, input_dir, output_dir, model_name="recording", device="cpu"
    )

    output = np.load(output_dir / "sample.npy", allow_pickle=False)
    assert model.observed_min == -0.5
    assert model.observed_max == 2.0
    assert output.shape == (4, 4)
    assert output.dtype == np.float32
    assert float(output.min()) == 0.0
    assert float(output.max()) == 1.0
    assert summary.input_count == 1


def test_bicubic_model_shape() -> None:
    model = BicubicRestorer()
    output = model(torch.zeros((2, 1, 7, 9), dtype=torch.float32))
    assert output.shape == (2, 1, 14, 18)


def test_mixed_input_shapes_can_share_a_requested_batch(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    np.save(input_dir / "small.npy", np.zeros((3, 4), dtype=np.float32))
    np.save(input_dir / "large.npy", np.zeros((5, 6), dtype=np.float32))

    summary = restore_directory(
        BicubicRestorer(),
        input_dir,
        output_dir,
        model_name="bicubic",
        device="cpu",
        batch_size=2,
    )

    assert summary.input_count == 2
    assert np.load(output_dir / "small.npy", allow_pickle=False).shape == (6, 8)
    assert np.load(output_dir / "large.npy", allow_pickle=False).shape == (10, 12)


def test_output_directory_cannot_be_nested_under_input(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    np.save(input_dir / "sample.npy", np.zeros((2, 2), dtype=np.float32))

    with pytest.raises(InputValidationError, match="must not be"):
        restore_directory(
            BicubicRestorer(),
            input_dir,
            input_dir / "outputs",
            model_name="bicubic",
            device="cpu",
        )


def test_fp16_precision_requires_cuda() -> None:
    with pytest.raises(InputValidationError, match="FP16 inference is only enabled on CUDA"):
        resolve_precision("fp16", torch.device("cpu"))
