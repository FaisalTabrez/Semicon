from __future__ import annotations

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


def test_selection_prefers_ood_rank_after_six_metric_rank_tie(tmp_path: Path) -> None:
    baseline_id = tmp_path / "baseline_id.json"
    baseline_ood = tmp_path / "baseline_ood.json"
    conditioned_id = tmp_path / "conditioned_id.json"
    conditioned_ood = tmp_path / "conditioned_ood.json"
    _summary(baseline_id, 29.228, 0.77649, 0.27088, "b" * 64)
    _summary(baseline_ood, 25.244, 0.66455, 0.35236, "b" * 64)
    _summary(conditioned_id, 29.226, 0.77657, 0.27144, "c" * 64)
    _summary(conditioned_ood, 25.251, 0.66607, 0.35308, "c" * 64)
    output = tmp_path / "selection.json"

    completed = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "select_final_model.py"),
            "--candidate", "baseline", str(baseline_id), str(baseline_ood),
            "--candidate", "conditioned", str(conditioned_id), str(conditioned_ood),
            "--output", str(output),
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["winner"] == "conditioned"
    assert payload["candidates"][0]["mean_rank"] == 1.5
