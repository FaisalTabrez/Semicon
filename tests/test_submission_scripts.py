from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path

import numpy as np

from semirestore.checkpoints import atomic_torch_save
from semirestore.models import NAFSR


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_generate_and_verify_submission_outputs(tmp_path: Path) -> None:
    inputs = tmp_path / "inputs"
    outputs = tmp_path / "outputs"
    inputs.mkdir()
    np.save(inputs / "alpha.npy", np.zeros((8, 10), dtype=np.float32))
    np.save(inputs / "beta.npy", np.ones((8, 10), dtype=np.float32))
    model = NAFSR(width=8, encoder_blocks=[1], middle_blocks=1, decoder_blocks=[1])
    weights = tmp_path / "model.pt"
    atomic_torch_save(
        {
            "format_version": 1,
            "model_name": "naf_sr",
            "model_config": model.model_config(),
            "model_state_dict": model.state_dict(),
        },
        weights,
    )
    digest = hashlib.sha256(weights.read_bytes()).hexdigest()
    digest_path = tmp_path / "model.sha256"
    digest_path.write_text(f"{digest}  model.pt\n", encoding="utf-8")

    generated = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "generate_test_outputs.py"),
            str(inputs),
            str(outputs),
            "--weights", str(weights),
            "--device", "cpu",
            "--expected-count", "2",
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert generated.returncode == 0, generated.stderr

    verified = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "verify_submission.py"),
            "--input-dir", str(inputs),
            "--output-dir", str(outputs),
            "--weights", str(weights),
            "--sha256", str(digest_path),
            "--expected-count", "2",
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert verified.returncode == 0, verified.stderr
    assert "Submission verification passed" in verified.stdout
