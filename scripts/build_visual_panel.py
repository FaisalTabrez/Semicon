#!/usr/bin/env python
"""Render a fixed noisy/bicubic/restored/GT panel for the worst PSNR examples."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from semirestore.checkpoints import load_model_checkpoint  # noqa: E402
from semirestore.data import InputValidationError, load_npy_image  # noqa: E402
from semirestore.inference import resolve_device  # noqa: E402
from semirestore.models import BicubicRestorer  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics-csv", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--target-dir", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--model", choices=("edsr_lite", "naf_sr"), default="naf_sr")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--limit", type=int, default=6)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _normalise(array: np.ndarray) -> np.ndarray:
    return np.clip(array, 0.0, 1.0)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.limit < 1:
            raise InputValidationError("--limit must be at least 1")
        output = args.output.expanduser().resolve()
        if output.exists() and not args.overwrite:
            raise InputValidationError(f"Panel already exists: {output}; use --overwrite")
        with args.metrics_csv.expanduser().resolve().open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        if not rows or "stem" not in rows[0] or "psnr_db" not in rows[0]:
            raise InputValidationError("Metrics CSV must contain stem and psnr_db columns")
        selected = sorted(rows, key=lambda row: float(row["psnr_db"]))[: args.limit]
        device = resolve_device(args.device)
        model, payload = load_model_checkpoint(args.weights)
        if payload["model_name"] != args.model:
            raise InputValidationError(f"Checkpoint model is {payload['model_name']}, not {args.model}")
        model = model.to(device).eval()
        bicubic = BicubicRestorer().to(device).eval()
        examples: list[tuple[str, float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
        with torch.inference_mode():
            for row in selected:
                stem = row["stem"]
                source = args.input_dir.expanduser().resolve() / f"{stem}.npy"
                target = args.target_dir.expanduser().resolve() / f"{stem}.npy"
                if not source.is_file() or not target.is_file():
                    raise InputValidationError(f"Panel source/target missing for stem {stem}")
                noisy = load_npy_image(source)
                reference = load_npy_image(target)
                tensor = torch.from_numpy(noisy)[None, None].to(device)
                restored = model(tensor).float().cpu().numpy()[0, 0]
                upsampled = bicubic(tensor).float().cpu().numpy()[0, 0]
                examples.append((stem, float(row["psnr_db"]), noisy, upsampled, restored, reference))
        try:
            import matplotlib.pyplot as plt
        except ImportError as exc:
            raise RuntimeError("Panel rendering requires matplotlib") from exc
        figure, axes = plt.subplots(len(examples), 4, figsize=(12, 3 * len(examples)), squeeze=False)
        labels = ("Noisy input", "Bicubic 2×", "NAF-SR restore", "Ground truth")
        for row_index, (stem, psnr_db, noisy, upsampled, restored, reference) in enumerate(examples):
            images = (noisy, upsampled, restored, reference)
            for column, (axis, image, label) in enumerate(zip(axes[row_index], images, labels, strict=True)):
                axis.imshow(_normalise(image), cmap="gray", vmin=0.0, vmax=1.0)
                axis.set_axis_off()
                if row_index == 0:
                    axis.set_title(label)
                if column == 0:
                    axis.set_ylabel(f"{stem}\nPSNR {psnr_db:.2f}", rotation=0, ha="right", va="center")
        figure.suptitle("Worst-PSNR restoration review", y=0.995)
        figure.tight_layout()
        output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output, dpi=180, bbox_inches="tight")
        plt.close(figure)
    except (InputValidationError, OSError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(f"Rendered {len(examples)} worst-PSNR examples: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
