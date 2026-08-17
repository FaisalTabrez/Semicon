from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_required_submission_layout_is_present() -> None:
    assert (PROJECT_ROOT / "run.py").is_file()
    assert (PROJECT_ROOT / "requirements.txt").is_file()
    assert (PROJECT_ROOT / "README.md").is_file()
    assert (PROJECT_ROOT / "models" / "model.pt").is_file()
    assert (PROJECT_ROOT / "models" / "model.sha256").is_file()


def test_run_py_restores_npy_with_required_contract(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    source = np.linspace(-0.1, 1.1, 64, dtype=np.float32).reshape(8, 8)
    np.save(input_dir / "sample.npy", source)

    completed = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "run.py"), str(input_dir), str(output_dir)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert completed.returncode == 0, completed.stderr

    output = np.load(output_dir / "sample.npy", allow_pickle=False)
    assert output.shape == (16, 16)
    assert output.dtype == np.float32
    assert np.isfinite(output).all()
    assert float(output.min()) >= 0.0
    assert float(output.max()) <= 1.0
