"""Safe discovery and loading for organizer-provided NumPy images."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


class InputValidationError(ValueError):
    """Raised when an input directory or image violates the inference contract."""


@dataclass(frozen=True)
class InputImage:
    """A discovered input and the relative path used for its output."""

    source: Path
    relative_path: Path


def is_ignored_npy_path(path: Path, root: Path) -> bool:
    """Return whether an archive-derived NumPy path is metadata, not a sample."""

    try:
        relative = path.relative_to(root)
    except ValueError:
        relative = path
    return "__MACOSX" in relative.parts or path.name.startswith("._")


def discover_npy_files(root: str | Path) -> list[InputImage]:
    """Return real `.npy` inputs in stable relative-path order.

    AppleDouble files and `__MACOSX` directories are excluded because both are
    present in the public KLA archives.
    """

    root_path = Path(root).expanduser().resolve()
    if not root_path.exists():
        raise InputValidationError(f"Input directory does not exist: {root_path}")
    if not root_path.is_dir():
        raise InputValidationError(f"Input path is not a directory: {root_path}")

    discovered = [
        InputImage(path, path.relative_to(root_path))
        for path in root_path.rglob("*.npy")
        if path.is_file() and not is_ignored_npy_path(path, root_path)
    ]
    discovered.sort(key=lambda item: item.relative_path.as_posix().casefold())

    if not discovered:
        raise InputValidationError(f"No valid .npy inputs found under: {root_path}")
    return discovered


def count_ignored_npy_files(root: str | Path) -> int:
    """Count metadata `.npy` entries excluded by :func:`discover_npy_files`."""

    root_path = Path(root).expanduser().resolve()
    if not root_path.is_dir():
        raise InputValidationError(f"Input path is not a directory: {root_path}")
    return sum(
        1
        for path in root_path.rglob("*.npy")
        if path.is_file() and is_ignored_npy_path(path, root_path)
    )


def load_npy_image(path: str | Path) -> np.ndarray:
    """Load one finite 2D numeric image as contiguous float32 without clipping."""

    image_path = Path(path)
    try:
        array = np.load(image_path, allow_pickle=False)
    except Exception as exc:  # NumPy exposes several format-specific exceptions.
        raise InputValidationError(f"Could not load NumPy image {image_path}: {exc}") from exc

    if not isinstance(array, np.ndarray):
        raise InputValidationError(f"Expected a NumPy array in {image_path}")
    if array.ndim != 2:
        raise InputValidationError(
            f"Expected a 2D grayscale array in {image_path}; received shape {array.shape}"
        )
    if array.size == 0:
        raise InputValidationError(f"Input image is empty: {image_path}")
    if not np.issubdtype(array.dtype, np.number) or np.issubdtype(array.dtype, np.complexfloating):
        raise InputValidationError(
            f"Expected a real numeric array in {image_path}; received dtype {array.dtype}"
        )
    if not np.isfinite(array).all():
        raise InputValidationError(f"Input image contains NaN or infinity: {image_path}")

    # astype(copy=False) deliberately preserves negative and >1 values.
    return np.ascontiguousarray(array.astype(np.float32, copy=False))


def validate_output_directory(path: str | Path, overwrite: bool = False) -> Path:
    """Create or validate an output directory without deleting existing content."""

    output = Path(path).expanduser().resolve()
    if output.exists() and not output.is_dir():
        raise InputValidationError(f"Output path is not a directory: {output}")
    if output.exists() and any(output.iterdir()) and not overwrite:
        raise InputValidationError(
            f"Output directory is not empty: {output}. Use --overwrite to replace matching outputs."
        )
    output.mkdir(parents=True, exist_ok=True)
    return output


def ensure_unique_output_paths(items: Iterable[InputImage]) -> None:
    """Guard against case-insensitive output collisions on Windows."""

    seen: dict[str, Path] = {}
    for item in items:
        key = item.relative_path.as_posix().casefold()
        previous = seen.get(key)
        if previous is not None:
            raise InputValidationError(
                f"Input paths collide in the output directory: {previous} and {item.source}"
            )
        seen[key] = item.source
