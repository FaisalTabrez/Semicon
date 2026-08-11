from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVALUATION_SCRIPT = PROJECT_ROOT / "evaluation.py"


def test_cli_runs_from_foreign_working_directory_and_protects_outputs(tmp_path: Path) -> None:
    input_dir = tmp_path / "inputs"
    output_dir = tmp_path / "outputs"
    report = tmp_path / "report.json"
    foreign_cwd = tmp_path / "elsewhere"
    input_dir.mkdir()
    foreign_cwd.mkdir()
    sample = np.linspace(-0.2, 1.3, 20, dtype=np.float32).reshape(4, 5)
    np.save(input_dir / "000001.npy", sample)

    command = [
        sys.executable,
        str(EVALUATION_SCRIPT),
        str(input_dir),
        str(output_dir),
        "--report-json",
        str(report),
    ]
    first = subprocess.run(command, cwd=foreign_cwd, capture_output=True, text=True)
    assert first.returncode == 0, first.stderr
    assert "Restored 1 image(s)" in first.stdout

    restored = np.load(output_dir / "000001.npy", allow_pickle=False)
    assert restored.shape == (8, 10)
    assert restored.dtype == np.float32
    assert np.isfinite(restored).all()
    assert float(restored.min()) >= 0.0
    assert float(restored.max()) <= 1.0
    assert json.loads(report.read_text(encoding="utf-8"))["input_count"] == 1

    protected = subprocess.run(command, cwd=foreign_cwd, capture_output=True, text=True)
    assert protected.returncode == 1
    assert "not empty" in protected.stderr

    overwritten = subprocess.run(
        [*command, "--overwrite"], cwd=foreign_cwd, capture_output=True, text=True
    )
    assert overwritten.returncode == 0, overwritten.stderr

    report_inside_output = subprocess.run(
        [
            sys.executable,
            str(EVALUATION_SCRIPT),
            str(input_dir),
            str(tmp_path / "other_outputs"),
            "--report-json",
            str(tmp_path / "other_outputs" / "report.json"),
        ],
        cwd=foreign_cwd,
        capture_output=True,
        text=True,
    )
    assert report_inside_output.returncode == 1
    assert "must be outside" in report_inside_output.stderr
