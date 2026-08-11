"""Manifest-backed paired NumPy dataset used by training scripts."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import Dataset

from .data import InputValidationError, load_npy_image
from .degradations import apply_d4, synthesize_degraded, validate_degradation_profile


@dataclass(frozen=True)
class ManifestPair:
    stem: str
    input_path: Path
    target_path: Path
    split: str


def _safe_dataset_path(root: Path, relative_text: str, *, label: str, stem: str) -> Path:
    relative = Path(relative_text)
    if relative.is_absolute():
        raise InputValidationError(
            f"Manifest {label} path for '{stem}' must be relative: {relative}"
        )
    resolved = (root / relative).resolve()
    if not resolved.is_relative_to(root):
        raise InputValidationError(
            f"Manifest {label} path for '{stem}' escapes dataset root: {relative}"
        )
    if not resolved.is_file():
        raise InputValidationError(f"Manifest {label} file does not exist for '{stem}': {resolved}")
    return resolved


def read_manifest_pairs(
    manifest_path: str | Path,
    dataset_root: str | Path,
    *,
    split: str,
) -> list[ManifestPair]:
    manifest = Path(manifest_path).expanduser().resolve()
    root = Path(dataset_root).expanduser().resolve()
    if not manifest.is_file():
        raise InputValidationError(f"Manifest does not exist: {manifest}")
    if not root.is_dir():
        raise InputValidationError(f"Dataset root does not exist: {root}")

    pairs: list[ManifestPair] = []
    seen: set[str] = set()
    with manifest.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"stem", "input_relpath", "target_relpath", "split"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise InputValidationError(
                f"Manifest is missing required training columns: {sorted(required)}"
            )
        for row in reader:
            if row["split"].strip() != split:
                continue
            stem = row["stem"].strip()
            if not stem:
                raise InputValidationError("Manifest contains an empty stem")
            if stem in seen:
                raise InputValidationError(f"Manifest contains duplicate stem '{stem}'")
            seen.add(stem)
            pairs.append(
                ManifestPair(
                    stem=stem,
                    input_path=_safe_dataset_path(
                        root, row["input_relpath"], label="input", stem=stem
                    ),
                    target_path=_safe_dataset_path(
                        root, row["target_relpath"], label="target", stem=stem
                    ),
                    split=split,
                )
            )

    if not pairs:
        raise InputValidationError(f"Manifest contains no rows assigned to split '{split}'")
    pairs.sort(key=lambda pair: pair.stem)
    return pairs


class PairedNpyDataset(Dataset[tuple[torch.Tensor, torch.Tensor, str]]):
    def __init__(
        self,
        pairs: list[ManifestPair],
        *,
        d4_augmentation: bool = False,
        synthetic_probability: float = 0.0,
        degradation_profile: dict[str, object] | None = None,
    ) -> None:
        if not pairs:
            raise InputValidationError("PairedNpyDataset requires at least one pair")
        if not 0.0 <= synthetic_probability <= 1.0:
            raise InputValidationError("synthetic_probability must be in [0, 1]")
        if synthetic_probability and degradation_profile is None:
            raise InputValidationError(
                "Synthetic training requires a fitted degradation profile"
            )
        self.pairs = list(pairs)
        self.d4_augmentation = d4_augmentation
        self.synthetic_probability = synthetic_probability
        self.degradation_profile = (
            None
            if degradation_profile is None
            else validate_degradation_profile(degradation_profile)
        )

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        pair = self.pairs[index]
        degraded = load_npy_image(pair.input_path)
        target = load_npy_image(pair.target_path)
        expected = (degraded.shape[0] * 2, degraded.shape[1] * 2)
        if target.shape != expected:
            raise InputValidationError(
                f"Pair '{pair.stem}' has input shape {degraded.shape} and target shape "
                f"{target.shape}; expected {expected}"
            )
        degraded_tensor = torch.from_numpy(degraded[None])
        target_tensor = torch.from_numpy(target[None])
        if self.d4_augmentation:
            transform = int(torch.randint(8, ()).item())
            degraded_tensor = apply_d4(degraded_tensor, transform)
            target_tensor = apply_d4(target_tensor, transform)
        if self.synthetic_probability and float(torch.rand(()).item()) < self.synthetic_probability:
            assert self.degradation_profile is not None
            degraded_tensor = synthesize_degraded(target_tensor, self.degradation_profile)
        return degraded_tensor, target_tensor, pair.stem
