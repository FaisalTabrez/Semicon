from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from semirestore.checkpoints import atomic_torch_save
from semirestore.models import NAFSR


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_external_protocol_emits_paired_control_evidence(tmp_path: Path) -> None:
    image_root = tmp_path / "images"
    (image_root / "class_1").mkdir(parents=True)
    rng = np.random.default_rng(44)
    for index in range(2):
        array = rng.integers(0, 256, size=(24, 24), dtype=np.uint8)
        Image.fromarray(array, mode="L").save(image_root / "class_1" / f"{index}.jpg")

    model = NAFSR(
        width=4,
        encoder_blocks=(1,),
        middle_blocks=1,
        decoder_blocks=(1,),
        statistics_conditioning=True,
        conditioning_hidden=4,
    )
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
    output = tmp_path / "output"
    completed = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "evaluate_external_carinthia.py"),
            str(image_root),
            "--weights", str(weights),
            "--device", "cpu",
            "--batch-size", "2",
            "--expected-count", "2",
            "--bootstrap-samples", "10",
            "--output-dir", str(output),
            "--no-lpips",
        ],
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert completed.returncode == 0, completed.stderr
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    with (output / "per_image_metrics.csv").open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle))
    assert summary["evidence_label"] == "external-domain controlled-degradation validation"
    assert summary["dataset"]["image_count"] == 2
    assert summary["dataset"]["used_for_training_or_tuning"] is False
    assert summary["aggregates"]["downsample_only"]["naf_sr"]["count"] == 2
    assert summary["paired_comparison"]["profile_high"]["count"] == 2
    assert len(rows) == 12
