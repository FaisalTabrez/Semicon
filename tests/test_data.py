from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from semirestore.data import InputValidationError, discover_npy_files, load_npy_image


def test_discovery_ignores_macos_metadata_and_sorts(tmp_path: Path) -> None:
    np.save(tmp_path / "b.npy", np.zeros((2, 2), dtype=np.float32))
    np.save(tmp_path / "a.npy", np.ones((2, 2), dtype=np.float32))
    metadata = tmp_path / "__MACOSX"
    metadata.mkdir()
    np.save(metadata / "ignored.npy", np.ones((2, 2), dtype=np.float32))
    np.save(tmp_path / "._ignored.npy", np.ones((2, 2), dtype=np.float32))

    files = discover_npy_files(tmp_path)

    assert [item.relative_path.as_posix() for item in files] == ["a.npy", "b.npy"]


def test_loader_preserves_values_outside_unit_range(tmp_path: Path) -> None:
    source = tmp_path / "sample.npy"
    expected = np.array([[-0.25, 0.5], [1.25, 2.0]], dtype=np.float64)
    np.save(source, expected)

    actual = load_npy_image(source)

    assert actual.dtype == np.float32
    assert actual.flags.c_contiguous
    np.testing.assert_allclose(actual, expected.astype(np.float32))


@pytest.mark.parametrize(
    "array, message",
    [
        (np.zeros((1, 2, 3), dtype=np.float32), "2D grayscale"),
        (np.array([[np.nan]], dtype=np.float32), "NaN or infinity"),
        (np.empty((0, 0), dtype=np.float32), "empty"),
        (np.array([[1 + 2j]], dtype=np.complex64), "real numeric"),
    ],
)
def test_loader_rejects_invalid_arrays(tmp_path: Path, array: np.ndarray, message: str) -> None:
    source = tmp_path / "invalid.npy"
    np.save(source, array)

    with pytest.raises(InputValidationError, match=message):
        load_npy_image(source)


def test_discovery_rejects_empty_directory(tmp_path: Path) -> None:
    with pytest.raises(InputValidationError, match="No valid"):
        discover_npy_files(tmp_path)
