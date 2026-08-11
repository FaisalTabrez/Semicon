from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml

from semirestore.checkpoints import load_checkpoint_payload

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    root = tmp_path / "data"
    input_dir, target_dir = root / "NoisyLR", root / "GT"
    input_dir.mkdir(parents=True)
    target_dir.mkdir()
    rng = np.random.default_rng(12)
    rows = []
    for index in range(6):
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
                "split": "train" if index < 4 else "val_id",
            }
        )
    manifest = tmp_path / "manifest.csv"
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys(), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    config = tmp_path / "config.yaml"
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
                    "manifest": "unused",
                    "dataset_root": "unused",
                    "train_split": "train",
                    "val_split": "val_id",
                },
                "training": {
                    "seed": 2026,
                    "device": "cpu",
                    "amp": False,
                    "deterministic": True,
                    "ema_enabled": True,
                    "ema_decay": 0.9,
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
                "output": {"run_dir": "unused"},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return root, manifest, config


def _run(
    root: Path,
    manifest: Path,
    config: Path,
    run_dir: Path,
    *extra: str,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
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
            *extra,
        ],
        cwd=run_dir.parent,
        capture_output=True,
        text=True,
        timeout=90,
    )


def test_deterministic_repeat_and_resume(tmp_path: Path) -> None:
    root, manifest, config = _fixture(tmp_path)
    first, second = tmp_path / "first", tmp_path / "second"
    first_result = _run(root, manifest, config, first, "--stop-after-step", "2")
    second_result = _run(root, manifest, config, second, "--stop-after-step", "2")
    assert first_result.returncode == 0, first_result.stderr
    assert second_result.returncode == 0, second_result.stderr

    comparison = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "compare_training_runs.py"),
            str(first),
            str(second),
            "--atol",
            "0",
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert comparison.returncode == 0, comparison.stderr
    assert json.loads(comparison.stdout)["maximum_absolute_parameter_difference"] == 0.0

    resumed = tmp_path / "resumed"
    resume_result = _run(
        root,
        manifest,
        config,
        resumed,
        "--resume",
        str(first / "last.pt"),
        "--stop-after-step",
        "4",
    )
    assert resume_result.returncode == 0, resume_result.stderr
    summary = json.loads((resumed / "summary.json").read_text(encoding="utf-8"))
    payload = load_checkpoint_payload(resumed / "last.pt")
    assert summary["start_step"] == 2
    assert summary["step"] == 4
    assert summary["resumed_from"] == str((first / "last.pt").resolve())
    assert payload["step"] == 4
    assert (resumed / "best.pt").is_file()
