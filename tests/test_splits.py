from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
import numpy as np

from semirestore.data import InputValidationError
from semirestore.splits import assign_provisional_hash_split, assign_texture_ood_split


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


def test_texture_split_holds_out_complete_clusters_and_is_row_order_independent(
    tmp_path: Path,
) -> None:
    root = tmp_path / "dataset"
    targets = root / "GT"
    targets.mkdir(parents=True)
    rows = []
    y, x = np.mgrid[:24, :24]
    rng = np.random.default_rng(8)
    patterns = (
        lambda offset: np.full((24, 24), 0.1 + offset),
        lambda offset: np.clip(x / 23 + offset, 0, 1),
        lambda offset: ((x + y) % 2).astype(np.float32) * (0.8 - offset),
        lambda offset: np.clip(rng.random((24, 24)) * 0.7 + offset, 0, 1),
    )
    for group, pattern in enumerate(patterns):
        for item in range(15):
            stem = f"{group:02d}{item:04d}"
            np.save(targets / f"{stem}.npy", pattern(item / 1000).astype(np.float32))
            rows.append(
                {
                    "stem": stem,
                    "target_relpath": f"GT/{stem}.npy",
                    "texture_cluster": "",
                    "split": "",
                }
            )

    outputs = []
    for index, ordered_rows in enumerate((rows, list(reversed(rows)))):
        manifest = tmp_path / f"source_{index}.csv"
        with manifest.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=rows[0].keys(), lineterminator="\n")
            writer.writeheader()
            writer.writerows(ordered_rows)
        output = tmp_path / f"texture_{index}.csv"
        audit = tmp_path / f"texture_{index}.json"
        payload = assign_texture_ood_split(
            manifest,
            root,
            output,
            audit,
            cluster_count=4,
            seed=2026,
        )
        outputs.append(output.read_text(encoding="utf-8"))
        assigned = list(csv.DictReader(output.open(encoding="utf-8", newline="")))
        ood_clusters = {
            row["texture_cluster"] for row in assigned if row["split"] == "val_ood"
        }
        assert ood_clusters
        assert not any(
            row["texture_cluster"] in ood_clusters and row["split"] != "val_ood"
            for row in assigned
        )
        assert set(payload["split_counts"]) == {"train", "val_id", "val_ood"}
        assert payload["public_test_used"] is False
    assert outputs[0] == outputs[1]
