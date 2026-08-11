"""Deterministic paired-data manifest and audit generation."""

from __future__ import annotations

import csv
import hashlib
import json
import os
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from .data import (
    InputImage,
    InputValidationError,
    count_ignored_npy_files,
    discover_npy_files,
    load_npy_image,
)

MANIFEST_FIELDS = (
    "stem",
    "input_relpath",
    "target_relpath",
    "input_shape",
    "target_shape",
    "input_dtype",
    "target_dtype",
    "input_min",
    "input_max",
    "input_mean",
    "input_std",
    "target_min",
    "target_max",
    "target_mean",
    "target_std",
    "source_group",
    "texture_cluster",
    "split",
    "sha256_input",
    "sha256_target",
)


@dataclass(frozen=True)
class ManifestResult:
    pair_count: int
    manifest_path: Path
    audit_path: Path
    manifest_sha256: str
    input_global_min: float
    input_global_max: float
    target_global_min: float
    target_global_max: float


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _index_by_stem(items: Iterable[InputImage], label: str) -> dict[str, InputImage]:
    indexed: dict[str, InputImage] = {}
    for item in items:
        stem = item.source.stem
        previous = indexed.get(stem)
        if previous is not None:
            raise InputValidationError(
                f"Duplicate {label} stem '{stem}': {previous.source} and {item.source}"
            )
        indexed[stem] = item
    return indexed


def _display_stems(stems: Iterable[str], limit: int = 10) -> str:
    values = sorted(stems)
    shown = ", ".join(values[:limit])
    if len(values) > limit:
        shown += f", ... (+{len(values) - limit} more)"
    return shown


def _shape_text(array: np.ndarray) -> str:
    return "x".join(str(dimension) for dimension in array.shape)


def _number(value: float) -> str:
    return format(float(value), ".10g")


def _relative_to_dataset(path: Path, dataset_root: Path, label: str) -> str:
    try:
        return path.relative_to(dataset_root).as_posix()
    except ValueError as exc:
        raise InputValidationError(
            f"{label} file is outside dataset root {dataset_root}: {path}"
        ) from exc


def _relative_directory(path: Path, dataset_root: Path, label: str) -> str:
    relative = _relative_to_dataset(path, dataset_root, label)
    return relative or "."


def _default_dataset_root(input_root: Path, target_root: Path) -> Path:
    try:
        common = Path(os.path.commonpath([input_root, target_root]))
    except ValueError as exc:
        raise InputValidationError(
            "Input and target directories have no common dataset root; pass --dataset-root"
        ) from exc
    if common in (input_root, target_root):
        # Directories should normally be siblings. Their parent still gives stable,
        # descriptive paths when one name happens to prefix the other.
        common = common.parent
    return common.resolve()


def _artifact_path(path: str | Path, overwrite: bool) -> Path:
    artifact = Path(path).expanduser().resolve()
    if artifact.exists() and not overwrite:
        raise InputValidationError(
            f"Artifact already exists: {artifact}. Use --overwrite to replace it."
        )
    if artifact.exists() and not artifact.is_file():
        raise InputValidationError(f"Artifact path is not a file: {artifact}")
    artifact.parent.mkdir(parents=True, exist_ok=True)
    return artifact


def _atomic_write_text(path: Path, content: str) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(content, encoding="utf-8", newline="")
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _csv_text(rows: list[dict[str, str]]) -> str:
    from io import StringIO

    output = StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=MANIFEST_FIELDS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def _counter_dict(counter: Counter[str]) -> dict[str, int]:
    return {key: counter[key] for key in sorted(counter)}


def build_paired_manifest(
    input_dir: str | Path,
    target_dir: str | Path,
    manifest_path: str | Path,
    audit_path: str | Path,
    *,
    dataset_root: str | Path | None = None,
    expected_pairs: int | None = None,
    overwrite: bool = False,
) -> ManifestResult:
    """Validate aligned 2× pairs and write a deterministic manifest and audit."""

    input_root = Path(input_dir).expanduser().resolve()
    target_root = Path(target_dir).expanduser().resolve()
    if input_root == target_root:
        raise InputValidationError("Input and target directories must be different")

    manifest = _artifact_path(manifest_path, overwrite)
    audit = _artifact_path(audit_path, overwrite)
    if manifest == audit:
        raise InputValidationError("Manifest and audit paths must be different")

    resolved_dataset_root = (
        Path(dataset_root).expanduser().resolve()
        if dataset_root is not None
        else _default_dataset_root(input_root, target_root)
    )
    input_items = discover_npy_files(input_root)
    target_items = discover_npy_files(target_root)
    inputs = _index_by_stem(input_items, "input")
    targets = _index_by_stem(target_items, "target")

    missing_targets = inputs.keys() - targets.keys()
    missing_inputs = targets.keys() - inputs.keys()
    if missing_targets or missing_inputs:
        details: list[str] = []
        if missing_targets:
            details.append(f"missing target(s) for: {_display_stems(missing_targets)}")
        if missing_inputs:
            details.append(f"missing input(s) for: {_display_stems(missing_inputs)}")
        raise InputValidationError("Pairing failed; " + "; ".join(details))

    stems = sorted(inputs)
    if expected_pairs is not None and len(stems) != expected_pairs:
        raise InputValidationError(
            f"Expected {expected_pairs} pairs, but discovered {len(stems)}"
        )

    rows: list[dict[str, str]] = []
    input_shapes: Counter[str] = Counter()
    target_shapes: Counter[str] = Counter()
    input_dtypes: Counter[str] = Counter()
    target_dtypes: Counter[str] = Counter()
    input_min = float("inf")
    input_max = float("-inf")
    target_min = float("inf")
    target_max = float("-inf")
    input_bytes = 0
    target_bytes = 0

    for stem in stems:
        input_item = inputs[stem]
        target_item = targets[stem]
        input_array = load_npy_image(input_item.source)
        target_array = load_npy_image(target_item.source)
        expected_target_shape = (input_array.shape[0] * 2, input_array.shape[1] * 2)
        if target_array.shape != expected_target_shape:
            raise InputValidationError(
                f"Pair '{stem}' has input shape {input_array.shape} and target shape "
                f"{target_array.shape}; expected target shape {expected_target_shape}"
            )

        input_shape = _shape_text(input_array)
        target_shape = _shape_text(target_array)
        input_shapes[input_shape] += 1
        target_shapes[target_shape] += 1
        input_dtypes[str(input_array.dtype)] += 1
        target_dtypes[str(target_array.dtype)] += 1
        pair_input_min = float(input_array.min())
        pair_input_max = float(input_array.max())
        pair_target_min = float(target_array.min())
        pair_target_max = float(target_array.max())
        input_min = min(input_min, pair_input_min)
        input_max = max(input_max, pair_input_max)
        target_min = min(target_min, pair_target_min)
        target_max = max(target_max, pair_target_max)
        input_bytes += input_item.source.stat().st_size
        target_bytes += target_item.source.stat().st_size

        rows.append(
            {
                "stem": stem,
                "input_relpath": _relative_to_dataset(
                    input_item.source, resolved_dataset_root, "Input"
                ),
                "target_relpath": _relative_to_dataset(
                    target_item.source, resolved_dataset_root, "Target"
                ),
                "input_shape": input_shape,
                "target_shape": target_shape,
                "input_dtype": str(input_array.dtype),
                "target_dtype": str(target_array.dtype),
                "input_min": _number(pair_input_min),
                "input_max": _number(pair_input_max),
                "input_mean": _number(float(input_array.mean(dtype=np.float64))),
                "input_std": _number(float(input_array.std(dtype=np.float64))),
                "target_min": _number(pair_target_min),
                "target_max": _number(pair_target_max),
                "target_mean": _number(float(target_array.mean(dtype=np.float64))),
                "target_std": _number(float(target_array.std(dtype=np.float64))),
                "source_group": "",
                "texture_cluster": "",
                "split": "",
                "sha256_input": sha256_file(input_item.source),
                "sha256_target": sha256_file(target_item.source),
            }
        )

    manifest_content = _csv_text(rows)
    manifest_digest = hashlib.sha256(manifest_content.encode("utf-8")).hexdigest()
    audit_payload = {
        "schema_version": 1,
        # Keep committed evidence portable and avoid leaking a contributor's
        # machine-specific absolute path.
        "dataset_root": ".",
        "input_directory": _relative_directory(input_root, resolved_dataset_root, "Input"),
        "target_directory": _relative_directory(target_root, resolved_dataset_root, "Target"),
        "pair_count": len(rows),
        "expected_pairs": expected_pairs,
        "ignored_metadata_files": {
            "input": count_ignored_npy_files(input_root),
            "target": count_ignored_npy_files(target_root),
        },
        "input": {
            "shape_counts": _counter_dict(input_shapes),
            "dtype_counts_after_loading": _counter_dict(input_dtypes),
            "global_min": input_min,
            "global_max": input_max,
            "file_bytes": input_bytes,
        },
        "target": {
            "shape_counts": _counter_dict(target_shapes),
            "dtype_counts_after_loading": _counter_dict(target_dtypes),
            "global_min": target_min,
            "global_max": target_max,
            "file_bytes": target_bytes,
        },
        "manifest_sha256": manifest_digest,
        "input_clipped_or_normalized": False,
    }
    audit_content = json.dumps(audit_payload, indent=2, sort_keys=True) + "\n"
    _atomic_write_text(manifest, manifest_content)
    _atomic_write_text(audit, audit_content)

    return ManifestResult(
        pair_count=len(rows),
        manifest_path=manifest,
        audit_path=audit,
        manifest_sha256=manifest_digest,
        input_global_min=input_min,
        input_global_max=input_max,
        target_global_min=target_min,
        target_global_max=target_max,
    )
