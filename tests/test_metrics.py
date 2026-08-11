from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from semirestore.metrics import (
    compute_image_metrics,
    lpips_distance,
    prepare_lpips_input,
    psnr,
    ssim,
)


class FakeLPIPS:
    def __init__(self) -> None:
        self.normalize: bool | None = None

    def __call__(
        self, prediction: torch.Tensor, target: torch.Tensor, *, normalize: bool
    ) -> torch.Tensor:
        self.normalize = normalize
        assert prediction.shape[1] == target.shape[1] == 3
        assert float(prediction.min()) >= -1.0
        assert float(prediction.max()) <= 1.0
        return torch.mean(torch.abs(prediction - target), dim=(1, 2, 3), keepdim=True)


def test_psnr_and_ssim_known_fixtures() -> None:
    target = np.zeros((11, 11), dtype=np.float32)
    prediction = np.full((11, 11), 0.5, dtype=np.float32)

    assert psnr(prediction, target) == pytest.approx(6.020599913, abs=1e-8)
    assert ssim(target, target) == pytest.approx(1.0, abs=1e-7)
    assert math.isinf(psnr(target, target))


def test_lpips_preprocessing_repeats_and_maps_exactly_once() -> None:
    grayscale = torch.tensor([[[[0.0, 0.5, 1.0]]]], dtype=torch.float32)
    prepared = prepare_lpips_input(grayscale)

    assert prepared.shape == (1, 3, 1, 3)
    expected = torch.tensor([-1.0, 0.0, 1.0])
    for channel in range(3):
        assert torch.equal(prepared[0, channel, 0], expected)
    assert torch.equal(grayscale, torch.tensor([[[[0.0, 0.5, 1.0]]]]))


def test_lpips_distance_disables_package_normalization() -> None:
    model = FakeLPIPS()
    prediction = torch.zeros((2, 1, 11, 11))
    target = torch.ones((2, 1, 11, 11))

    distances = lpips_distance(model, prediction, target)

    assert model.normalize is False
    assert distances.tolist() == pytest.approx([2.0, 2.0])


def test_diagnostics_use_raw_prediction_only_for_out_of_range_rate() -> None:
    target = np.full((11, 11), 0.5, dtype=np.float32)
    prediction = target.copy()
    prediction[0, 0] = -0.1
    prediction[0, 1] = 1.1

    metrics = compute_image_metrics(prediction, target, lpips_value=0.25)

    assert metrics.pre_clamp_out_of_range_rate == pytest.approx(2 / 121)
    assert metrics.lpips_alex == 0.25
    assert metrics.mean_intensity_bias >= 0.0
    assert metrics.sobel_l1 >= 0.0


def test_metrics_reject_small_or_mismatched_arrays() -> None:
    with pytest.raises(ValueError, match="at least 11x11"):
        ssim(np.zeros((10, 10)), np.zeros((10, 10)))
    with pytest.raises(ValueError, match="shape mismatch"):
        ssim(np.zeros((11, 11)), np.zeros((12, 12)))
