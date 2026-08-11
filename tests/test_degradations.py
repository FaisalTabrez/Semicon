from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from semirestore.degradations import (
    apply_d4,
    fit_degradation_profile,
    load_degradation_profile,
    synthesize_degraded,
)
from semirestore.training_data import ManifestPair


PROFILE = {
    "schema_version": 1,
    "blur_sigma": {"low": 0.2, "high": 0.8},
    "gaussian_noise_std": {"low": 0.01, "high": 0.03},
    "speckle_std": {"low": 0.01, "high": 0.04},
    "additive_bias": {"low": -0.01, "high": 0.01},
    "downsample_modes": ["area", "bicubic"],
}


def test_paired_d4_preserves_two_x_alignment() -> None:
    degraded = torch.arange(12, dtype=torch.float32).reshape(1, 3, 4)
    target = degraded.repeat_interleave(2, -2).repeat_interleave(2, -1)
    for transform in range(8):
        transformed_input = apply_d4(degraded, transform)
        transformed_target = apply_d4(target, transform)
        expected = transformed_input.repeat_interleave(2, -2).repeat_interleave(2, -1)
        assert torch.equal(transformed_target, expected)


def test_synthetic_degradation_is_seeded_finite_and_raw_range() -> None:
    target = torch.linspace(0, 1, 16 * 20).reshape(1, 16, 20)
    torch.manual_seed(91)
    first = synthesize_degraded(target, PROFILE)
    torch.manual_seed(91)
    second = synthesize_degraded(target, PROFILE)
    assert first.shape == (1, 8, 10)
    assert torch.equal(first, second)
    assert torch.isfinite(first).all()


def test_degradation_fit_uses_training_pairs_and_round_trips(tmp_path: Path) -> None:
    input_path, target_path = tmp_path / "input.npy", tmp_path / "target.npy"
    target = np.linspace(0, 1, 16 * 16, dtype=np.float32).reshape(16, 16)
    base = target.reshape(8, 2, 8, 2).mean((1, 3))
    rng = np.random.default_rng(4)
    pairs = []
    for index in range(3):
        current_input = input_path.with_name(f"input_{index}.npy")
        current_target = target_path.with_name(f"target_{index}.npy")
        np.save(current_input, (base + rng.normal(0, 0.02, base.shape)).astype(np.float32))
        np.save(current_target, target)
        pairs.append(ManifestPair(str(index), current_input, current_target, "train"))
    output = tmp_path / "profile.json"
    payload = fit_degradation_profile(
        pairs, output, manifest_sha256="a" * 64
    )
    loaded, digest = load_degradation_profile(output)
    assert loaded == payload
    assert len(digest) == 64
    assert payload["fit_split"] == "train"
    assert payload["public_test_used"] is False
