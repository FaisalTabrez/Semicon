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

from .data import InputValidationError


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
