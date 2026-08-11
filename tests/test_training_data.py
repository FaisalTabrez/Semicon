from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from semirestore.data import InputValidationError
from semirestore.training_data import PairedNpyDataset, read_manifest_pairs


def test_manifest_dataset_loads_raw_values_and_pair(tmp_path: Path) -> None:
    root = tmp_path / "train"
    input_dir = root / "NoisyLR"
    target_dir = root / "GT"
    input_dir.mkdir(parents=True)
    target_dir.mkdir()
    degraded = np.linspace(-0.2, 1.4, 36, dtype=np.float32).reshape(6, 6)
    target = np.zeros((12, 12), dtype=np.float32)
    np.save(input_dir / "sample.npy", degraded)
    np.save(target_dir / "sample.npy", target)
    manifest = tmp_path / "manifest.csv"
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("stem", "input_relpath", "target_relpath", "split"),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerow(
            {
                "stem": "sample",
                "input_relpath": "NoisyLR/sample.npy",
                "target_relpath": "GT/sample.npy",
                "split": "train",
            }
        )

    pairs = read_manifest_pairs(manifest, root, split="train")
    loaded_input, loaded_target, stem = PairedNpyDataset(pairs)[0]

    assert stem == "sample"
    assert float(loaded_input.min()) == pytest.approx(-0.2)
    assert float(loaded_input.max()) == pytest.approx(1.4)
    assert loaded_input.shape == (1, 6, 6)
    assert loaded_target.shape == (1, 12, 12)


def test_manifest_rejects_path_escape(tmp_path: Path) -> None:
    root = tmp_path / "train"
    root.mkdir()
    outside = tmp_path / "outside.npy"
    np.save(outside, np.zeros((4, 4), dtype=np.float32))
    manifest = tmp_path / "manifest.csv"
    manifest.write_text(
        "stem,input_relpath,target_relpath,split\n"
        "bad,../outside.npy,../outside.npy,train\n",
        encoding="utf-8",
    )

    with pytest.raises(InputValidationError, match="escapes dataset root"):
        read_manifest_pairs(manifest, root, split="train")
