from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from semirestore.checkpoints import atomic_torch_save
from semirestore.models import EDSRLite


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_cpu_benchmark_emits_auditable_shape_and_latency_artifact(tmp_path: Path) -> None:
    model = EDSRLite(width=8, num_blocks=1)
    weights = tmp_path / "edsr.pt"
    atomic_torch_save(
        {
            "format_version": 1,
            "model_name": "edsr_lite",
            "model_config": model.model_config(),
            "model_state_dict": model.state_dict(),
        },
        weights,
    )
    output = tmp_path / "benchmark.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "benchmark.py"),
            "--weights", str(weights),
            "--model", "edsr_lite",
            "--device", "cpu",
            "--batch-size", "1",
            "--height", "4",
            "--width", "5",
            "--warmup", "0",
            "--iterations", "2",
            "--output", str(output),
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert completed.returncode == 0, completed.stderr
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["scope"].startswith("model-only")
    assert payload["input_shape"] == [1, 1, 4, 5]
    assert payload["output_shape"] == [1, 1, 8, 10]
    assert payload["latency_ms"]["median"] > 0
