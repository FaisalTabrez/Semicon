from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_metrics_cli_writes_split_filtered_evidence(tmp_path: Path) -> None:
    input_dir = tmp_path / "NoisyLR"
    target_dir = tmp_path / "GT"
    input_dir.mkdir()
    target_dir.mkdir()
    manifest = tmp_path / "manifest.csv"
    per_image = tmp_path / "per_image.csv"
    summary = tmp_path / "summary.json"

    rng = np.random.default_rng(17)
    for index in range(3):
        degraded = rng.uniform(0.0, 1.0, size=(12, 12)).astype(np.float32)
        target = np.repeat(np.repeat(degraded, 2, axis=0), 2, axis=1)
        np.save(input_dir / f"{index:06d}.npy", degraded)
        np.save(target_dir / f"{index:06d}.npy", target.astype(np.float32))

    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("stem", "split"), lineterminator="\n")
        writer.writeheader()
        writer.writerow({"stem": "000000", "split": "train"})
        writer.writerow({"stem": "000001", "split": "val_id"})
        writer.writerow({"stem": "000002", "split": "val_id"})

    completed = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "evaluate_metrics.py"),
            str(input_dir),
            str(target_dir),
            "--manifest",
            str(manifest),
            "--split",
            "val_id",
            "--no-lpips",
            "--bootstrap-samples",
            "20",
            "--per-image-csv",
            str(per_image),
            "--summary-json",
            str(summary),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert completed.returncode == 0, completed.stderr
    with per_image.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    payload = json.loads(summary.read_text(encoding="utf-8"))
    assert [row["stem"] for row in rows] == ["000001", "000002"]
    assert payload["pair_count"] == 2
    assert payload["split_counts"] == {"val_id": 2}
    assert payload["metric_policy"]["lpips"]["enabled"] is False
    assert "LPIPS=disabled" in completed.stdout
