from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _summary(path: Path, psnr: float, ssim: float, lpips: float, checkpoint: str) -> None:
    path.write_text(
        json.dumps(
            {
                "manifest_sha256": "a" * 64,
                "checkpoint_sha256": checkpoint,
                "aggregates": {
                    "all": {
                        "psnr_db": {"mean": psnr},
                        "ssim": {"mean": ssim},
                        "lpips_alex": {"mean": lpips},
                    }
                },
            }
        ),
        encoding="utf-8",
    )


def test_ablation_cli_applies_predeclared_id_and_ood_gate(tmp_path: Path) -> None:
    paths = [tmp_path / f"{name}.json" for name in ("base_id", "base_ood", "new_id", "new_ood")]
    _summary(paths[0], 28.0, 0.75, 0.30, "b" * 64)
    _summary(paths[1], 26.0, 0.65, 0.40, "b" * 64)
    _summary(paths[2], 27.9, 0.749, 0.29, "c" * 64)
    _summary(paths[3], 26.2, 0.66, 0.37, "c" * 64)
    output = tmp_path / "ablations.csv"
    completed = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "record_ablation.py"),
            "--experiment", "synthetic15",
            "--isolated-change", "synthetic_probability=0.15",
            "--baseline-id", str(paths[0]),
            "--baseline-ood", str(paths[1]),
            "--candidate-id", str(paths[2]),
            "--candidate-ood", str(paths[3]),
            "--output", str(output),
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr
    with output.open(encoding="utf-8", newline="") as handle:
        row = next(csv.DictReader(handle))
    assert row["decision"] == "keep"
    assert row["ood_metric_wins"] == "3"
