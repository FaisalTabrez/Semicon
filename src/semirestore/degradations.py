"""Paired geometry and training-only fitted synthetic degradations."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .data import InputValidationError, load_npy_image


def apply_d4(image: torch.Tensor, transform: int) -> torch.Tensor:
    """Apply one of the eight square dihedral transforms to a CHW tensor."""

    if image.ndim != 3 or transform not in range(8):
        raise ValueError("D4 expects a CHW tensor and transform index in [0, 7]")
    result = torch.rot90(image, transform % 4, dims=(-2, -1))
    if transform >= 4:
        result = torch.flip(result, dims=(-1,))
    return result.contiguous()


def _range(profile: dict[str, object], name: str) -> tuple[float, float]:
    value = profile.get(name)
    if not isinstance(value, dict):
        raise InputValidationError(f"Degradation profile is missing '{name}'")
    low, high = value.get("low"), value.get("high")
    if not isinstance(low, (int, float)) or not isinstance(high, (int, float)):
        raise InputValidationError(f"Degradation range '{name}' must be numeric")
    low, high = float(low), float(high)
    if not np.isfinite((low, high)).all() or low < 0 or high < low:
        raise InputValidationError(f"Invalid degradation range '{name}'")
    return low, high


def validate_degradation_profile(profile: dict[str, object]) -> dict[str, object]:
    if profile.get("schema_version") != 1:
        raise InputValidationError("Unsupported degradation profile schema")
    for name in ("blur_sigma", "gaussian_noise_std", "speckle_std"):
        _range(profile, name)
    bias = profile.get("additive_bias")
    if not isinstance(bias, dict) or not all(
        isinstance(bias.get(key), (int, float)) for key in ("low", "high")
    ):
        raise InputValidationError("Invalid degradation range 'additive_bias'")
    if float(bias["high"]) < float(bias["low"]):
        raise InputValidationError("Invalid degradation range 'additive_bias'")
    modes = profile.get("downsample_modes")
    if not isinstance(modes, list) or not modes or not set(modes) <= {"area", "bicubic"}:
        raise InputValidationError("downsample_modes must contain area and/or bicubic")
    return profile


def load_degradation_profile(path: str | Path) -> tuple[dict[str, object], str]:
    profile_path = Path(path).expanduser().resolve()
    if not profile_path.is_file():
        raise InputValidationError(f"Degradation profile does not exist: {profile_path}")
    content = profile_path.read_bytes()
    try:
        payload = json.loads(content)
    except json.JSONDecodeError as exc:
        raise InputValidationError(f"Invalid degradation profile JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise InputValidationError("Degradation profile root must be a mapping")
    return validate_degradation_profile(payload), hashlib.sha256(content).hexdigest()


def _uniform(low: float, high: float, *, device: torch.device) -> float:
    if low == high:
        return low
    return float((torch.rand((), device=device) * (high - low) + low).item())


def _gaussian_blur(image: torch.Tensor, sigma: float) -> torch.Tensor:
    if sigma < 0.05:
        return image
    radius = max(1, min(4, int(round(3 * sigma))))
    positions = torch.arange(-radius, radius + 1, device=image.device, dtype=image.dtype)
    kernel_1d = torch.exp(-(positions.square()) / (2 * sigma * sigma))
    kernel_1d /= kernel_1d.sum()
    kernel = torch.outer(kernel_1d, kernel_1d)[None, None]
    padded = F.pad(image[None], (radius, radius, radius, radius), mode="reflect")
    return F.conv2d(padded, kernel)[0]


def synthesize_degraded(
    target: torch.Tensor, profile: dict[str, object]
) -> torch.Tensor:
    """Generate a raw-range LR observation from one CHW ground truth tensor."""

    validate_degradation_profile(profile)
    if target.ndim != 3 or target.shape[0] != 1:
        raise ValueError("Synthetic degradation expects one-channel CHW target")
    if target.shape[-2] % 2 or target.shape[-1] % 2:
        raise ValueError("Synthetic degradation requires even target dimensions")
    device = target.device
    blur = _uniform(*_range(profile, "blur_sigma"), device=device)
    gaussian = _uniform(*_range(profile, "gaussian_noise_std"), device=device)
    speckle = _uniform(*_range(profile, "speckle_std"), device=device)
    bias_profile = profile["additive_bias"]
    assert isinstance(bias_profile, dict)
    bias = _uniform(float(bias_profile["low"]), float(bias_profile["high"]), device=device)
    modes = profile["downsample_modes"]
    assert isinstance(modes, list)
    mode = str(modes[int(torch.randint(len(modes), ()).item())])

    operations = ["blur", "gaussian", "speckle", "downsample"]
    order = torch.randperm(len(operations)).tolist()
    image = target
    for index in order:
        operation = operations[index]
        if operation == "blur":
            image = _gaussian_blur(image, blur)
        elif operation == "gaussian":
            image = image + torch.randn_like(image) * gaussian + bias
        elif operation == "speckle":
            image = image + image * torch.randn_like(image) * speckle
        else:
            size = (target.shape[-2] // 2, target.shape[-1] // 2)
            if mode == "bicubic":
                image = F.interpolate(
                    image[None], size=size, mode=mode, align_corners=False, antialias=True
                )[0]
            else:
                image = F.interpolate(image[None], size=size, mode=mode)[0]
    return image.contiguous()


def _gradient_mean(array: np.ndarray) -> float:
    gy, gx = np.gradient(array.astype(np.float64))
    return float(np.hypot(gx, gy).mean())


def fit_degradation_profile(
    pairs: Iterable[object],
    output_path: str | Path,
    *,
    manifest_sha256: str,
    overwrite: bool = False,
) -> dict[str, object]:
    """Fit conservative noise/blur ranges using manifest-selected training pairs only."""

    output = Path(output_path).expanduser().resolve()
    if output.exists() and not overwrite:
        raise InputValidationError(f"Artifact already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    residual_stds, residual_biases, speckle_stds, blur_proxies = [], [], [], []
    count = 0
    for pair in pairs:
        degraded = load_npy_image(pair.input_path)
        target = load_npy_image(pair.target_path)
        target_tensor = torch.from_numpy(target)[None, None]
        base = F.interpolate(
            target_tensor,
            size=degraded.shape,
            mode="bicubic",
            align_corners=False,
            antialias=True,
        )[0, 0].numpy()
        residual = degraded.astype(np.float64) - base.astype(np.float64)
        residual_stds.append(float(residual.std()))
        residual_biases.append(float(residual.mean()))
        mask = np.abs(base) > 0.10
        ratio = residual[mask] / base[mask] if mask.any() else residual.ravel()
        median = np.median(ratio)
        speckle_stds.append(float(1.4826 * np.median(np.abs(ratio - median))))
        base_gradient = max(_gradient_mean(base), 1e-8)
        gradient_ratio = _gradient_mean(degraded) / base_gradient
        blur_proxies.append(float(np.clip((1.0 - gradient_ratio) * 1.5, 0.0, 1.5)))
        count += 1
    if count < 2:
        raise InputValidationError("At least two training pairs are required to fit degradation")

    def quantile_range(values, *, multiplier=1.0, cap=None):
        low, high = np.quantile(values, (0.10, 0.90)) * multiplier
        low, high = max(0.0, float(low)), max(0.0, float(high))
        if cap is not None:
            low, high = min(low, cap), min(high, cap)
        return {"low": low, "high": max(low, high)}

    bias_low, bias_high = np.quantile(residual_biases, (0.10, 0.90))
    payload: dict[str, object] = {
        "schema_version": 1,
        "fit_split": "train",
        "fit_pair_count": count,
        "manifest_sha256": manifest_sha256,
        "public_test_used": False,
        "blur_sigma": quantile_range(blur_proxies, cap=1.5),
        "gaussian_noise_std": quantile_range(residual_stds, multiplier=0.70, cap=0.20),
        "speckle_std": quantile_range(speckle_stds, multiplier=0.30, cap=0.30),
        "additive_bias": {"low": float(bias_low), "high": float(bias_high)},
        "downsample_modes": ["area", "bicubic"],
        "operation_order": "uniform_random_per_sample",
        "output_clipped": False,
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(rendered, encoding="utf-8")
        temporary.replace(output)
    finally:
        if temporary.exists():
            temporary.unlink()
    return payload
