from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml

from semirestore.checkpoints import atomic_torch_save, load_checkpoint_payload
from semirestore.models import EDSRLite


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_tiny_distillation_run_records_frozen_teacher_provenance(tmp_path: Path) -> None:
    dataset_root = tmp_path / "train"
    input_dir = dataset_root / "NoisyLR"
    target_dir = dataset_root / "GT"
    input_dir.mkdir(parents=True)
    target_dir.mkdir()
    rows: list[dict[str, str]] = []
    rng = np.random.default_rng(29)
    for index in range(4):
        degraded = rng.random((6, 6), dtype=np.float32)
        target = np.repeat(np.repeat(degraded, 2, axis=0), 2, axis=1)
        name = f"{index:06d}.npy"
        np.save(input_dir / name, degraded)
        np.save(target_dir / name, target)
        rows.append(
            {
                "stem": Path(name).stem,
                "input_relpath": f"NoisyLR/{name}",
                "target_relpath": f"GT/{name}",
                "split": "train" if index < 3 else "val_id",
            }
        )

    manifest = tmp_path / "manifest.csv"
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys(), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    teacher = EDSRLite(width=8, num_blocks=1)
    teacher_path = tmp_path / "teacher.pt"
    atomic_torch_save(
        {
            "format_version": 1,
            "model_name": "edsr_lite",
            "model_config": teacher.model_config(),
            "model_state_dict": teacher.state_dict(),
        },
        teacher_path,
    )

    config = tmp_path / "distill.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "model": {
                    "name": "edsr_lite",
                    "width": 8,
                    "num_blocks": 1,
                    "residual_scale": 0.1,
                },
                "data": {
                    "manifest": str(manifest),
                    "dataset_root": str(dataset_root),
                    "train_split": "train",
                    "val_split": "val_id",
                },
                "training": {
                    "seed": 2026,
                    "device": "cpu",
                    "amp": False,
                    "ema_enabled": False,
                    "batch_size": 2,
                    "num_workers": 0,
                    "max_steps": 2,
                    "validation_interval": 1,
                    "log_interval": 1,
                    "learning_rate": 0.001,
                    "weight_decay": 0.0,
                    "warmup_steps": 0,
                    "gradient_clip_norm": 1.0,
                    "charbonnier_epsilon": 0.001,
                },
                "distillation": {
                    "enabled": True,
                    "teacher_checkpoint": str(teacher_path),
                    "teacher_precision": "fp32",
                    "supervised_weight": 0.7,
                    "teacher_weight": 0.3,
                },
                "output": {"run_dir": str(tmp_path / "unused")},
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
            "--config", str(config),
            "--run-dir", str(run_dir),
        ],
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert completed.returncode == 0, completed.stderr

    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    payload = load_checkpoint_payload(run_dir / "best.pt")
    assert summary["distillation"]["enabled"] is True
    assert summary["final_distillation_loss"] is not None
    assert len(summary["distillation"]["teacher_sha256"]) == 64
    assert payload["distillation"] == summary["distillation"]
    assert payload["loss"]["supervised_weight"] == 0.7
    assert payload["loss"]["teacher_weight"] == 0.3
