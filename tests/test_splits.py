from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from semirestore.data import InputValidationError
from semirestore.splits import assign_provisional_hash_split


def _write_manifest(path: Path, count: int = 20) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("stem", "split"), lineterminator="\n")
        writer.writeheader()
        for index in range(count):
            writer.writerow({"stem": f"{index:06d}", "split": ""})


def test_provisional_split_is_exact_and_row_order_independent(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.csv"
    reversed_manifest = tmp_path / "manifest_reversed.csv"
    _write_manifest(manifest)
    rows = manifest.read_text(encoding="utf-8").splitlines()
    reversed_manifest.write_text("\n".join([rows[0], *reversed(rows[1:])]) + "\n")

    memberships: list[set[str]] = []
    for index, source in enumerate((manifest, reversed_manifest)):
        output = tmp_path / f"split_{index}.csv"
        audit = tmp_path / f"audit_{index}.json"
        payload = assign_provisional_hash_split(
            source,
            output,
            audit,
            validation_fraction=0.15,
            seed=2026,
        )
        with output.open("r", encoding="utf-8", newline="") as handle:
            assigned = list(csv.DictReader(handle))
        memberships.append({row["stem"] for row in assigned if row["split"] == "val_id"})
        assert payload["split_counts"] == {"train": 17, "val_id": 3}
        assert json.loads(audit.read_text(encoding="utf-8"))["warning"].startswith("This is")

    assert memberships[0] == memberships[1]


def test_provisional_split_protects_artifacts(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.csv"
    output = tmp_path / "split.csv"
    audit = tmp_path / "audit.json"
    _write_manifest(manifest, count=2)
    output.write_text("owned", encoding="utf-8")

    with pytest.raises(InputValidationError, match="already exists"):
        assign_provisional_hash_split(manifest, output, audit)
