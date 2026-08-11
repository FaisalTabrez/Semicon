from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

from semirestore.checkpoints import load_model_checkpoint

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    ("model_config", "expected_model_config"),
    [
        (
            {
                "name": "edsr_lite",
                "width": 8,
                "num_blocks": 1,
                "residual_scale": 0.1,
            },
            {"width": 8, "num_blocks": 1, "residual_scale": 0.1},
        ),
        (
            {
                "name": "naf_sr",
                "width": 8,
                "encoder_blocks": [1],
                "middle_blocks": 1,
                "decoder_blocks": [1],
                "dropout": 0.0,
            },
            {
                "width": 8,
                "encoder_blocks": [1],
                "middle_blocks": 1,
                "decoder_blocks": [1],
                "dropout": 0.0,
            },
        ),
    ],
)
def test_tiny_training_run_writes_reloadable_artifacts(
    tmp_path: Path,
    model_config: dict[str, object],
    expected_model_config: dict[str, object],
) -> None:
    root = tmp_path / "train"
    input_dir = root / "NoisyLR"
    target_dir = root / "GT"
    input_dir.mkdir(parents=True)
    target_dir.mkdir()
    rng = np.random.default_rng(41)
    rows: list[dict[str, str]] = []
    for index in range(6):
        degraded = rng.uniform(-0.1, 1.1, size=(6, 6)).astype(np.float32)
        target = np.repeat(np.repeat(np.clip(degraded, 0, 1), 2, axis=0), 2, axis=1)
        name = f"{index:06d}.npy"
        np.save(input_dir / name, degraded)
        np.save(target_dir / name, target.astype(np.float32))
        rows.append(
            {
                "stem": Path(name).stem,
                "input_relpath": f"NoisyLR/{name}",
                "target_relpath": f"GT/{name}",
                "split": "train" if index < 4 else "val_id",
            }
        )

    manifest = tmp_path / "manifest.csv"
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys(), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    config = tmp_path / "tiny.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "model": model_config,
                "data": {
                    "manifest": "unused.csv",
                    "dataset_root": "unused",
                    "train_split": "train",
                    "val_split": "val_id",
                },
                "training": {
                    "seed": 7,
                    "device": "cpu",
                    "amp": False,
                    "batch_size": 2,
                    "num_workers": 0,
                    "max_steps": 4,
                    "validation_interval": 2,
                    "log_interval": 1,
                    "learning_rate": 0.001,
                    "weight_decay": 0.0,
                    "warmup_steps": 0,
                    "gradient_clip_norm": 1.0,
                    "charbonnier_epsilon": 0.001,
                },
                "output": {"run_dir": "unused-run"},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    run_dir = tmp_path / "run"

    completed = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "train.py"),
            "--config",
            str(config),
            "--manifest",
            str(manifest),
            "--dataset-root",
            str(root),
            "--run-dir",
            str(run_dir),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=90,
    )

    assert completed.returncode == 0, completed.stderr
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    model, payload = load_model_checkpoint(run_dir / "best.pt")
    assert summary["step"] == 4
    assert summary["parameter_count"] > 0
    assert np.isfinite(summary["best_val_psnr_db"])
    assert payload["data"]["input_policy"] == "raw_float32_no_clip"
    assert payload["checkpoint_role"] == "best_inference"
    assert "optimizer_state_dict" not in payload
    assert model.model_config() == expected_model_config
    assert (run_dir / "history.csv").is_file()
    assert (run_dir / "resolved_config.yaml").is_file()
