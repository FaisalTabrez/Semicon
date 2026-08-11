"""Locked image-quality metrics for labeled restoration evidence."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import torch
from torch.nn import functional as F


SSIM_POLICY = {
    "implementation": "skimage.metrics.structural_similarity",
    "data_range": 1.0,
    "win_size": 11,
    "gaussian_weights": True,
    "sigma": 1.5,
    "use_sample_covariance": False,
}

LPIPS_POLICY = {
    "implementation": "lpips.LPIPS",
    "net": "alex",
    "grayscale_to_rgb": "repeat channel exactly once",
    "input_mapping": "clamp [0,1], then x * 2 - 1 exactly once",
    "normalize_argument": False,
}


class LPIPSModel(Protocol):
    def __call__(
        self, prediction: torch.Tensor, target: torch.Tensor, *, normalize: bool
    ) -> torch.Tensor: ...


@dataclass(frozen=True)
class ImageMetrics:
    psnr_db: float
    ssim: float
    lpips_alex: float | None
    sobel_l1: float
    mean_intensity_bias: float
    pre_clamp_out_of_range_rate: float


def _validate_pair(prediction: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    prediction_array = np.asarray(prediction)
    target_array = np.asarray(target)
    if prediction_array.ndim != 2 or target_array.ndim != 2:
        raise ValueError(
            "Metrics require two 2D grayscale arrays; got "
            f"{prediction_array.shape} and {target_array.shape}"
        )
    if prediction_array.shape != target_array.shape:
        raise ValueError(
            f"Prediction/target shape mismatch: {prediction_array.shape} vs {target_array.shape}"
        )
    if not np.isfinite(prediction_array).all() or not np.isfinite(target_array).all():
        raise ValueError("Metrics require finite prediction and target arrays")
    return (
        np.ascontiguousarray(prediction_array.astype(np.float32, copy=False)),
        np.ascontiguousarray(target_array.astype(np.float32, copy=False)),
    )


def clamp_for_scoring(array: np.ndarray) -> np.ndarray:
    """Clamp only at the scoring boundary and return contiguous float32."""

    return np.ascontiguousarray(np.clip(array, 0.0, 1.0).astype(np.float32, copy=False))


def psnr(prediction_01: np.ndarray, target_01: np.ndarray) -> float:
    """Compute PSNR with the fixed unit data range."""

    prediction, target = _validate_pair(prediction_01, target_01)
    error = prediction.astype(np.float64) - target.astype(np.float64)
    mse = float(np.mean(error * error))
    if mse == 0.0:
        return math.inf
    return float(-10.0 * math.log10(mse))


def ssim(prediction_01: np.ndarray, target_01: np.ndarray) -> float:
    """Compute SSIM with the immutable policy declared in :data:`SSIM_POLICY`."""

    from skimage.metrics import structural_similarity

    prediction, target = _validate_pair(prediction_01, target_01)
    if min(prediction.shape) < SSIM_POLICY["win_size"]:
        raise ValueError(
            f"Images must be at least {SSIM_POLICY['win_size']}x{SSIM_POLICY['win_size']} "
            "for the locked SSIM window"
        )
    return float(
        structural_similarity(
            target,
            prediction,
            data_range=1.0,
            win_size=11,
            gaussian_weights=True,
            sigma=1.5,
            use_sample_covariance=False,
        )
    )


def _as_nchw(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim == 2:
        tensor = tensor[None, None]
    elif tensor.ndim == 3:
        tensor = tensor[:, None]
    if tensor.ndim != 4 or tensor.shape[1] != 1:
        raise ValueError(f"Expected grayscale HW, NHW, or N1HW tensor; got {tuple(tensor.shape)}")
    if not torch.isfinite(tensor).all():
        raise ValueError("LPIPS preprocessing requires finite tensors")
    return tensor.to(dtype=torch.float32)


def prepare_lpips_input(images_01: torch.Tensor) -> torch.Tensor:
    """Convert grayscale values to LPIPS RGB ``[-1,1]`` exactly once."""

    grayscale = _as_nchw(images_01).clamp(0.0, 1.0)
    return grayscale.repeat(1, 3, 1, 1).mul(2.0).sub(1.0)


def create_lpips_model(device: torch.device | str, net: str = "alex") -> torch.nn.Module:
    """Create the official LPIPS model lazily so judge inference stays lightweight."""

    try:
        import lpips
    except ImportError as exc:
        raise RuntimeError(
            "LPIPS evaluation requires the 'lpips' package; install requirements.txt"
        ) from exc
    model = lpips.LPIPS(net=net, verbose=False).to(device)
    model.eval()
    return model


def lpips_distance(
    model: LPIPSModel,
    prediction_01: torch.Tensor,
    target_01: torch.Tensor,
) -> torch.Tensor:
    """Score a batch without allowing the LPIPS package to normalize it again."""

    prediction = prepare_lpips_input(prediction_01)
    target = prepare_lpips_input(target_01)
    with torch.inference_mode():
        distances = model(prediction, target, normalize=False)
    return distances.reshape(-1).to(dtype=torch.float32)


def sobel_l1(prediction_01: np.ndarray, target_01: np.ndarray) -> float:
    prediction, target = _validate_pair(prediction_01, target_01)
    pair = torch.from_numpy(np.stack([prediction, target]))[:, None]
    kernel_x = torch.tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
        dtype=torch.float32,
    )[None, None]
    kernel_y = kernel_x.transpose(-1, -2)
    padded = F.pad(pair, (1, 1, 1, 1), mode="reflect")
    grad_x = F.conv2d(padded, kernel_x)
    grad_y = F.conv2d(padded, kernel_y)
    magnitude = torch.sqrt(grad_x.square() + grad_y.square() + 1e-12)
    return float(torch.mean(torch.abs(magnitude[0] - magnitude[1])).item())


def compute_image_metrics(
    prediction_raw: np.ndarray,
    target_raw: np.ndarray,
    *,
    lpips_value: float | None = None,
) -> ImageMetrics:
    """Compute fixed scalar metrics for one raw prediction/target pair."""

    prediction, target = _validate_pair(prediction_raw, target_raw)
    out_of_range = np.logical_or(prediction < 0.0, prediction > 1.0)
    prediction_01 = clamp_for_scoring(prediction)
    target_01 = clamp_for_scoring(target)
    return ImageMetrics(
        psnr_db=psnr(prediction_01, target_01),
        ssim=ssim(prediction_01, target_01),
        lpips_alex=lpips_value,
        sobel_l1=sobel_l1(prediction_01, target_01),
        mean_intensity_bias=float(abs(prediction_01.mean() - target_01.mean())),
        pre_clamp_out_of_range_rate=float(out_of_range.mean()),
    )
