"""Reproducible manifest split helpers.

The provisional hash split is intentionally source-agnostic. It provides a stable
validation-ID holdout before the texture-group OOD split is built later.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
from collections import Counter
from io import StringIO
from pathlib import Path

import numpy as np

from .data import InputValidationError
from .data import load_npy_image


def _sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _artifact_path(path: str | Path, *, overwrite: bool) -> Path:
    artifact = Path(path).expanduser().resolve()
    if artifact.exists() and not overwrite:
        raise InputValidationError(
            f"Artifact already exists: {artifact}. Use --overwrite to replace it."
        )
    if artifact.exists() and not artifact.is_file():
        raise InputValidationError(f"Artifact path is not a file: {artifact}")
    artifact.parent.mkdir(parents=True, exist_ok=True)
    return artifact


def _atomic_write(path: Path, content: str) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(content, encoding="utf-8", newline="")
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def assign_provisional_hash_split(
    manifest_path: str | Path,
    output_manifest_path: str | Path,
    audit_path: str | Path,
    *,
    validation_fraction: float = 0.15,
    seed: int = 2026,
    overwrite: bool = False,
) -> dict[str, object]:
    """Assign an exact-size deterministic ``train``/``val_id`` holdout.

    This is a temporary, explicitly non-OOD split. The ranking is based on a
    SHA-256 digest of ``seed:stem``, so row order cannot change membership.
    """

    if not 0.0 < validation_fraction < 1.0:
        raise InputValidationError("validation_fraction must be strictly between 0 and 1")

    source = Path(manifest_path).expanduser().resolve()
    if not source.is_file():
        raise InputValidationError(f"Manifest does not exist: {source}")

    output = _artifact_path(output_manifest_path, overwrite=overwrite)
    audit = _artifact_path(audit_path, overwrite=overwrite)
    if output == source:
        raise InputValidationError(
            "Output manifest must differ from the source manifest so the checksum audit is preserved"
        )
    if output == audit:
        raise InputValidationError("Output manifest and audit paths must be different")

    source_bytes = source.read_bytes()
    with source.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames
        if not fieldnames or "stem" not in fieldnames or "split" not in fieldnames:
            raise InputValidationError("Manifest must contain 'stem' and 'split' columns")
        rows = list(reader)

    if len(rows) < 2:
        raise InputValidationError("Manifest needs at least two rows to create a holdout")

    stems = [row["stem"].strip() for row in rows]
    if any(not stem for stem in stems):
        raise InputValidationError("Manifest contains an empty stem")
    if len(set(stems)) != len(stems):
        duplicates = sorted(stem for stem, count in Counter(stems).items() if count > 1)
        raise InputValidationError(f"Manifest contains duplicate stem(s): {', '.join(duplicates[:10])}")

    ranked = sorted(
        stems,
        key=lambda stem: (hashlib.sha256(f"{seed}:{stem}".encode()).digest(), stem),
    )
    validation_count = max(1, min(len(rows) - 1, round(len(rows) * validation_fraction)))
    validation_stems = set(ranked[:validation_count])
    for row in rows:
        row["split"] = "val_id" if row["stem"].strip() in validation_stems else "train"

    csv_buffer = StringIO(newline="")
    writer = csv.DictWriter(csv_buffer, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    output_content = csv_buffer.getvalue()
    output_digest = _sha256_bytes(output_content.encode("utf-8"))
    payload: dict[str, object] = {
        "schema_version": 1,
        "strategy": "provisional_sha256_stem_holdout",
        "warning": "This is validation-ID only, not a texture/source OOD split.",
        "seed": seed,
        "requested_validation_fraction": validation_fraction,
        "pair_count": len(rows),
        "split_counts": {"train": len(rows) - validation_count, "val_id": validation_count},
        "source_manifest_sha256": _sha256_bytes(source_bytes),
        "output_manifest_sha256": output_digest,
    }

    _atomic_write(output, output_content)
    _atomic_write(audit, json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


TEXTURE_DESCRIPTOR_NAMES = (
    "mean",
    "std",
    "q10",
    "q90",
    "entropy_32bin",
    "gradient_mean",
    "gradient_std",
    "laplacian_abs_mean",
    "fft_low_fraction",
    "fft_mid_fraction",
    "fft_high_fraction",
)


def texture_descriptor(image: np.ndarray) -> np.ndarray:
    """Return inexpensive deterministic texture statistics for one GT image."""

    array = np.asarray(image, dtype=np.float64)
    if array.ndim != 2 or not np.isfinite(array).all():
        raise InputValidationError("Texture descriptors require a finite 2D image")
    histogram, _ = np.histogram(array, bins=32, range=(0.0, 1.0))
    probabilities = histogram.astype(np.float64)
    probabilities /= max(float(probabilities.sum()), 1.0)
    nonzero = probabilities[probabilities > 0]
    entropy = float(-(nonzero * np.log2(nonzero)).sum())

    gradient_y, gradient_x = np.gradient(array)
    gradient = np.hypot(gradient_x, gradient_y)
    laplacian = (
        -4.0 * array
        + np.roll(array, 1, axis=0)
        + np.roll(array, -1, axis=0)
        + np.roll(array, 1, axis=1)
        + np.roll(array, -1, axis=1)
    )

    centered = array - array.mean()
    power = np.abs(np.fft.rfft2(centered)) ** 2
    frequencies_y = np.fft.fftfreq(array.shape[0])[:, None]
    frequencies_x = np.fft.rfftfreq(array.shape[1])[None, :]
    radius = np.hypot(frequencies_y, frequencies_x)
    total_power = max(float(power.sum()), np.finfo(np.float64).eps)
    low = float(power[radius < 0.12].sum() / total_power)
    middle = float(power[(radius >= 0.12) & (radius < 0.30)].sum() / total_power)
    high = float(power[radius >= 0.30].sum() / total_power)

    return np.asarray(
        [
            array.mean(),
            array.std(),
            np.quantile(array, 0.10),
            np.quantile(array, 0.90),
            entropy,
            gradient.mean(),
            gradient.std(),
            np.abs(laplacian[1:-1, 1:-1]).mean(),
            low,
            middle,
            high,
        ],
        dtype=np.float64,
    )


def _standardize_descriptors(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    center = np.median(values, axis=0)
    q25, q75 = np.quantile(values, (0.25, 0.75), axis=0)
    scale = q75 - q25
    scale[scale < 1e-12] = 1.0
    return (values - center) / scale, center, scale


def _kmeans_plus_plus(
    values: np.ndarray, cluster_count: int, rng: np.random.Generator
) -> np.ndarray:
    centers = [values[int(rng.integers(len(values)))]]
    closest_squared = ((values - centers[0]) ** 2).sum(axis=1)
    for _ in range(1, cluster_count):
        total = float(closest_squared.sum())
        if total <= 0:
            index = next(
                (candidate for candidate in range(len(values)) if not any(
                    np.array_equal(values[candidate], center) for center in centers
                )),
                0,
            )
        else:
            index = int(rng.choice(len(values), p=closest_squared / total))
        centers.append(values[index])
        distance = ((values - values[index]) ** 2).sum(axis=1)
        closest_squared = np.minimum(closest_squared, distance)
    return np.stack(centers)


def _deterministic_kmeans(
    values: np.ndarray,
    *,
    cluster_count: int,
    seed: int,
    restarts: int = 8,
    max_iterations: int = 100,
) -> tuple[np.ndarray, np.ndarray, float]:
    best: tuple[np.ndarray, np.ndarray, float] | None = None
    for restart in range(restarts):
        rng = np.random.default_rng(np.random.SeedSequence([seed, restart]))
        centers = _kmeans_plus_plus(values, cluster_count, rng)
        assignments = np.zeros(len(values), dtype=np.int64)
        for _ in range(max_iterations):
            distances = ((values[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
            next_assignments = distances.argmin(axis=1)
            if np.array_equal(next_assignments, assignments):
                assignments = next_assignments
                break
            assignments = next_assignments
            nearest_distance = distances[np.arange(len(values)), assignments]
            for cluster in range(cluster_count):
                members = values[assignments == cluster]
                if len(members):
                    centers[cluster] = members.mean(axis=0)
                else:
                    farthest = int(np.argmax(nearest_distance))
                    centers[cluster] = values[farthest]
                    assignments[farthest] = cluster
                    nearest_distance[farthest] = -1.0
        inertia = float(((values - centers[assignments]) ** 2).sum())
        candidate = (assignments.copy(), centers.copy(), inertia)
        if best is None or inertia < best[2]:
            best = candidate
    assert best is not None
    return best


def _safe_manifest_target(dataset_root: Path, relative_text: str, stem: str) -> Path:
    relative = Path(relative_text)
    if relative.is_absolute():
        raise InputValidationError(f"Target path for '{stem}' must be relative")
    target = (dataset_root / relative).resolve()
    if not target.is_relative_to(dataset_root) or not target.is_file():
        raise InputValidationError(f"Invalid target path for '{stem}': {relative}")
    return target


def assign_texture_ood_split(
    manifest_path: str | Path,
    dataset_root: str | Path,
    output_manifest_path: str | Path,
    audit_path: str | Path,
    *,
    cluster_count: int = 12,
    ood_fraction: float = 0.15,
    id_fraction: float = 0.15,
    seed: int = 2026,
    overwrite: bool = False,
) -> dict[str, object]:
    """Create train/val_id/val_ood using complete held-out GT texture clusters."""

    if cluster_count < 3:
        raise InputValidationError("cluster_count must be at least 3")
    if ood_fraction <= 0 or id_fraction <= 0 or ood_fraction + id_fraction >= 1:
        raise InputValidationError("OOD and ID fractions must be positive and sum below 1")
    source = Path(manifest_path).expanduser().resolve()
    root = Path(dataset_root).expanduser().resolve()
    if not source.is_file() or not root.is_dir():
        raise InputValidationError("Manifest and dataset root must exist")
    output = _artifact_path(output_manifest_path, overwrite=overwrite)
    audit = _artifact_path(audit_path, overwrite=overwrite)
    if source in {output, audit} or output == audit:
        raise InputValidationError("Source manifest, output manifest, and audit must differ")

    source_bytes = source.read_bytes()
    with source.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames
        required = {"stem", "target_relpath", "texture_cluster", "split"}
        if not fieldnames or not required.issubset(fieldnames):
            raise InputValidationError(f"Manifest is missing required columns: {sorted(required)}")
        rows = sorted(list(reader), key=lambda row: row["stem"].strip())
    if cluster_count >= len(rows):
        raise InputValidationError("cluster_count must be smaller than the pair count")
    stems = [row["stem"].strip() for row in rows]
    if any(not stem for stem in stems) or len(set(stems)) != len(stems):
        raise InputValidationError("Manifest stems must be non-empty and unique")

    descriptors = np.stack(
        [
            texture_descriptor(
                load_npy_image(_safe_manifest_target(root, row["target_relpath"], stem))
            )
            for row, stem in zip(rows, stems, strict=True)
        ]
    )
    standardized, descriptor_center, descriptor_scale = _standardize_descriptors(descriptors)
    assignments, centers, inertia = _deterministic_kmeans(
        standardized, cluster_count=cluster_count, seed=seed
    )

    global_center = standardized.mean(axis=0)
    cluster_distances = np.linalg.norm(centers - global_center, axis=1)
    cluster_sizes = np.bincount(assignments, minlength=cluster_count)
    ranked_clusters = sorted(
        range(cluster_count), key=lambda cluster: (-cluster_distances[cluster], cluster)
    )
    target_ood_count = round(len(rows) * ood_fraction)
    ood_clusters: list[int] = []
    current_count = 0
    for cluster in ranked_clusters:
        candidate_count = current_count + int(cluster_sizes[cluster])
        if ood_clusters and abs(current_count - target_ood_count) <= abs(
            candidate_count - target_ood_count
        ):
            break
        ood_clusters.append(cluster)
        current_count = candidate_count
        if current_count >= target_ood_count:
            break
    ood_cluster_set = set(ood_clusters)

    target_id_count = round(len(rows) * id_fraction)
    remaining_clusters = [
        cluster for cluster in range(cluster_count) if cluster not in ood_cluster_set
    ]
    remaining_count = sum(int(cluster_sizes[cluster]) for cluster in remaining_clusters)
    allocation_rate = target_id_count / remaining_count
    allocations = {
        cluster: min(int(cluster_sizes[cluster]) - 1, int(cluster_sizes[cluster] * allocation_rate))
        for cluster in remaining_clusters
    }
    remainders = sorted(
        remaining_clusters,
        key=lambda cluster: (
            -(cluster_sizes[cluster] * allocation_rate - allocations[cluster]),
            cluster,
        ),
    )
    while sum(allocations.values()) < target_id_count:
        changed = False
        for cluster in remainders:
            if allocations[cluster] < int(cluster_sizes[cluster]) - 1:
                allocations[cluster] += 1
                changed = True
                if sum(allocations.values()) == target_id_count:
                    break
        if not changed:
            raise InputValidationError("Could not allocate validation-ID rows")

    validation_id_stems: set[str] = set()
    for cluster in remaining_clusters:
        cluster_stems = [
            stem
            for stem, assignment in zip(stems, assignments, strict=True)
            if int(assignment) == cluster
        ]
        ranked_stems = sorted(
            cluster_stems,
            key=lambda stem: (hashlib.sha256(f"{seed}:val_id:{stem}".encode()).digest(), stem),
        )
        validation_id_stems.update(ranked_stems[: allocations[cluster]])

    split_counts: Counter[str] = Counter()
    cluster_summary: list[dict[str, object]] = []
    for row, assignment in zip(rows, assignments, strict=True):
        cluster = int(assignment)
        row["texture_cluster"] = f"texture_{cluster:02d}"
        if cluster in ood_cluster_set:
            row["split"] = "val_ood"
        elif row["stem"].strip() in validation_id_stems:
            row["split"] = "val_id"
        else:
            row["split"] = "train"
        split_counts[row["split"]] += 1
    for cluster in range(cluster_count):
        cluster_rows = [row for row in rows if row["texture_cluster"] == f"texture_{cluster:02d}"]
        cluster_summary.append(
            {
                "cluster": f"texture_{cluster:02d}",
                "count": len(cluster_rows),
                "distance_from_global_center": float(cluster_distances[cluster]),
                "split_counts": dict(sorted(Counter(row["split"] for row in cluster_rows).items())),
                "held_out_ood": cluster in ood_cluster_set,
            }
        )

    csv_buffer = StringIO(newline="")
    writer = csv.DictWriter(csv_buffer, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    output_content = csv_buffer.getvalue()
    output_digest = _sha256_bytes(output_content.encode())
    payload: dict[str, object] = {
        "schema_version": 1,
        "strategy": "gt_texture_descriptor_kmeans_complete_ood_clusters",
        "seed": seed,
        "pair_count": len(rows),
        "cluster_count": cluster_count,
        "requested_fractions": {"val_id": id_fraction, "val_ood": ood_fraction},
        "split_counts": dict(sorted(split_counts.items())),
        "ood_clusters": [f"texture_{cluster:02d}" for cluster in sorted(ood_clusters)],
        "descriptor_names": list(TEXTURE_DESCRIPTOR_NAMES),
        "descriptor_robust_center": descriptor_center.tolist(),
        "descriptor_robust_scale": descriptor_scale.tolist(),
        "kmeans_inertia": inertia,
        "cluster_summary": cluster_summary,
        "source_manifest_sha256": _sha256_bytes(source_bytes),
        "output_manifest_sha256": output_digest,
        "fit_data": "organizer labeled training GT only",
        "public_test_used": False,
        "ood_cluster_overlap_with_train_or_val_id": False,
    }
    _atomic_write(output, output_content)
    _atomic_write(audit, json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload
