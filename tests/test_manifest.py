from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from semirestore.data import InputValidationError
from semirestore.manifest import build_paired_manifest, sha256_file

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_SCRIPT = PROJECT_ROOT / "scripts" / "build_manifest.py"


def _pair(root: Path, stem: str, low: np.ndarray, high: np.ndarray) -> None:
    input_dir = root / "NoisyLR"
    target_dir = root / "GT"
    input_dir.mkdir(parents=True, exist_ok=True)
    target_dir.mkdir(parents=True, exist_ok=True)
    np.save(input_dir / f"{stem}.npy", low)
    np.save(target_dir / f"{stem}.npy", high)


def test_manifest_is_deterministic_and_records_ranges_hashes_and_metadata(tmp_path: Path) -> None:
    train = tmp_path / "train"
    _pair(
        train,
        "000001",
        np.array([[-0.2, 0.4], [1.3, 0.7]], dtype=np.float32),
        np.linspace(0, 1, 16, dtype=np.float32).reshape(4, 4),
    )
    _pair(
        train,
        "000000",
        np.full((2, 2), 0.25, dtype=np.float32),
        np.full((4, 4), 0.5, dtype=np.float32),
    )
    metadata = train / "NoisyLR" / "__MACOSX"
    metadata.mkdir()
    np.save(metadata / "ignored.npy", np.zeros((2, 2), dtype=np.float32))
    np.save(train / "NoisyLR" / "._ignored.npy", np.zeros((2, 2), dtype=np.float32))
    manifest = tmp_path / "manifest.csv"
    audit = tmp_path / "audit.json"

    result = build_paired_manifest(
        train / "NoisyLR",
        train / "GT",
        manifest,
        audit,
        dataset_root=train,
        expected_pairs=2,
    )
    first_manifest = manifest.read_bytes()
    first_audit = audit.read_bytes()

    with manifest.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    payload = json.loads(audit.read_text(encoding="utf-8"))

    assert result.pair_count == 2
    assert [row["stem"] for row in rows] == ["000000", "000001"]
    assert rows[0]["input_relpath"] == "NoisyLR/000000.npy"
    assert rows[0]["target_relpath"] == "GT/000000.npy"
    assert rows[0]["input_shape"] == "2x2"
    assert rows[0]["target_shape"] == "4x4"
    assert len(rows[0]["sha256_input"]) == 64
    assert rows[0]["sha256_input"] == sha256_file(train / "NoisyLR" / "000000.npy")
    assert payload["pair_count"] == 2
    assert payload["dataset_root"] == "."
    assert payload["input_directory"] == "NoisyLR"
    assert payload["target_directory"] == "GT"
    assert str(tmp_path) not in audit.read_text(encoding="utf-8")
    assert payload["ignored_metadata_files"]["input"] == 2
    assert payload["input"]["shape_counts"] == {"2x2": 2}
    assert payload["target"]["shape_counts"] == {"4x4": 2}
    assert payload["input"]["global_min"] == pytest.approx(-0.2)
    assert payload["input"]["global_max"] == pytest.approx(1.3)
    assert payload["input_clipped_or_normalized"] is False
    assert payload["manifest_sha256"] == result.manifest_sha256

    build_paired_manifest(
        train / "NoisyLR",
        train / "GT",
        manifest,
        audit,
        dataset_root=train,
        expected_pairs=2,
        overwrite=True,
    )
    assert manifest.read_bytes() == first_manifest
    assert audit.read_bytes() == first_audit


def test_manifest_rejects_missing_pair_before_writing(tmp_path: Path) -> None:
    inputs = tmp_path / "NoisyLR"
    targets = tmp_path / "GT"
    inputs.mkdir()
    targets.mkdir()
    np.save(inputs / "orphan.npy", np.zeros((2, 2), dtype=np.float32))
    np.save(targets / "different.npy", np.zeros((4, 4), dtype=np.float32))
    manifest = tmp_path / "manifest.csv"

    with pytest.raises(InputValidationError, match="Pairing failed"):
        build_paired_manifest(inputs, targets, manifest, tmp_path / "audit.json")
    assert not manifest.exists()


def test_manifest_rejects_duplicate_stem(tmp_path: Path) -> None:
    inputs = tmp_path / "NoisyLR"
    targets = tmp_path / "GT"
    (inputs / "nested").mkdir(parents=True)
    targets.mkdir()
    np.save(inputs / "same.npy", np.zeros((2, 2), dtype=np.float32))
    np.save(inputs / "nested" / "same.npy", np.zeros((2, 2), dtype=np.float32))
    np.save(targets / "same.npy", np.zeros((4, 4), dtype=np.float32))

    with pytest.raises(InputValidationError, match="Duplicate input stem"):
        build_paired_manifest(inputs, targets, tmp_path / "manifest.csv", tmp_path / "audit.json")


def test_manifest_rejects_wrong_scale_and_expected_count(tmp_path: Path) -> None:
    train = tmp_path / "train"
    _pair(
        train,
        "bad",
        np.zeros((2, 3), dtype=np.float32),
        np.zeros((4, 5), dtype=np.float32),
    )
    with pytest.raises(InputValidationError, match="expected target shape"):
        build_paired_manifest(
            train / "NoisyLR",
            train / "GT",
            tmp_path / "manifest.csv",
            tmp_path / "audit.json",
        )

    (train / "GT" / "bad.npy").unlink()
    np.save(train / "GT" / "bad.npy", np.zeros((4, 6), dtype=np.float32))
    with pytest.raises(InputValidationError, match="Expected 2 pairs"):
        build_paired_manifest(
            train / "NoisyLR",
            train / "GT",
            tmp_path / "manifest.csv",
            tmp_path / "audit.json",
            expected_pairs=2,
        )


def test_manifest_cli_runs_from_foreign_working_directory(tmp_path: Path) -> None:
    train = tmp_path / "train"
    _pair(
        train,
        "000000",
        np.zeros((2, 2), dtype=np.float32),
        np.zeros((4, 4), dtype=np.float32),
    )
    foreign = tmp_path / "foreign"
    foreign.mkdir()
    manifest = tmp_path / "artifacts" / "manifest.csv"
    audit = tmp_path / "artifacts" / "audit.json"
    command = [
        sys.executable,
        str(MANIFEST_SCRIPT),
        "--input-dir",
        str(train / "NoisyLR"),
        "--target-dir",
        str(train / "GT"),
        "--dataset-root",
        str(train),
        "--manifest",
        str(manifest),
        "--audit",
        str(audit),
        "--expected-pairs",
        "1",
    ]

    completed = subprocess.run(command, cwd=foreign, capture_output=True, text=True)

    assert completed.returncode == 0, completed.stderr
    assert "Validated 1 pair(s)" in completed.stdout
    assert manifest.exists()
    assert audit.exists()
