#!/usr/bin/env python
"""Generate and validate organizer test outputs with the frozen final checkpoint."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from semirestore.checkpoints import load_model_checkpoint  # noqa: E402
from semirestore.data import InputValidationError, discover_npy_files, load_npy_image  # noqa: E402
from semirestore.inference import restore_directory  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", type=Path, help="Organizer public-test NoisyLR directory")
    parser.add_argument("output_dir", type=Path, help="Destination for restored arrays")
    parser.add_argument(
        "--weights", type=Path, default=PROJECT_ROOT / "models" / "model.pt"
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--expected-count", type=int, default=400)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        inputs = discover_npy_files(args.input_dir)
        if len(inputs) != args.expected_count:
            raise InputValidationError(
                f"Expected {args.expected_count} test inputs; discovered {len(inputs)}"
            )
        model, payload = load_model_checkpoint(args.weights)
        if payload["model_name"] != "naf_sr":
            raise InputValidationError("Final test-output generation requires the NAF-SR checkpoint")
        summary = restore_directory(
            model,
            args.input_dir,
            args.output_dir,
            model_name="naf_sr",
            device=args.device,
            precision="fp32",
            batch_size=args.batch_size,
            overwrite=args.overwrite,
            checkpoint=str(args.weights.expanduser().resolve()),
        )
        for item in inputs:
            output = args.output_dir.expanduser().resolve() / item.relative_path
            array = np.load(output, allow_pickle=False)
            input_shape = load_npy_image(item.source).shape
            expected_shape = (input_shape[0] * 2, input_shape[1] * 2)
            if array.dtype != np.float32 or array.shape != expected_shape:
                raise RuntimeError(f"Invalid output dtype/shape for {output}")
            if not np.isfinite(array).all() or float(array.min()) < 0.0 or float(array.max()) > 1.0:
                raise RuntimeError(f"Invalid output range/values for {output}")
    except (InputValidationError, OSError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(
        f"Generated and validated {summary.input_count} outputs on {summary.device}; "
        f"mean {summary.mean_milliseconds_per_image:.3f} ms/image."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
