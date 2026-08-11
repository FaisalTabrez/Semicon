#!/usr/bin/env python
"""Standalone KLA directory-to-directory restoration entry point."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from semirestore.data import InputValidationError  # noqa: E402
from semirestore.checkpoints import load_model_checkpoint  # noqa: E402
from semirestore.inference import restore_directory, write_report  # noqa: E402
from semirestore.models import BicubicRestorer  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Restore every grayscale .npy image in a directory and write 2× outputs."
    )
    parser.add_argument("input_dir", type=Path, help="Directory containing degraded .npy images")
    parser.add_argument("output_dir", type=Path, help="Directory for restored .npy images")
    parser.add_argument(
        "--model",
        choices=("bicubic", "edsr_lite"),
        default="bicubic",
        help="Restoration model (bicubic is the current lower-bound baseline)",
    )
    parser.add_argument(
        "--weights",
        type=Path,
        help="Self-describing .pt checkpoint required by learned models",
    )
    parser.add_argument(
        "--device", choices=("auto", "cpu", "cuda"), default="auto", help="Inference device"
    )
    parser.add_argument(
        "--precision",
        choices=("auto", "fp32", "bf16"),
        default="auto",
        help="Inference precision; BF16 remains opt-in pending parity tests",
    )
    parser.add_argument("--batch-size", type=int, default=1, help="Images per inference batch")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacement of matching outputs in a non-empty directory; nothing is deleted",
    )
    parser.add_argument("--report-json", type=Path, help="Optional report path outside output_dir")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.report_json:
            report_path = args.report_json.expanduser().resolve()
            output_path = args.output_dir.expanduser().resolve()
            if report_path == output_path or report_path.is_relative_to(output_path):
                raise InputValidationError(
                    f"--report-json must be outside the restored output directory: {report_path}"
                )
        checkpoint_text = None
        if args.model == "bicubic":
            if args.weights is not None:
                raise InputValidationError("--weights cannot be used with the bicubic model")
            model = BicubicRestorer()
        else:
            if args.weights is None:
                raise InputValidationError(f"--weights is required for model {args.model}")
            model, payload = load_model_checkpoint(args.weights)
            if payload["model_name"] != args.model:
                raise InputValidationError(
                    f"Checkpoint model is {payload['model_name']}, not requested {args.model}"
                )
            checkpoint_text = str(args.weights.expanduser().resolve())
        summary = restore_directory(
            model,
            args.input_dir,
            args.output_dir,
            model_name=args.model,
            device=args.device,
            precision=args.precision,
            batch_size=args.batch_size,
            overwrite=args.overwrite,
            checkpoint=checkpoint_text,
        )
        if args.report_json:
            write_report(args.report_json, summary)
    except (InputValidationError, OSError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(
        f"Restored {summary.input_count} image(s) with {summary.model} on {summary.device} "
        f"({summary.precision}) in {summary.elapsed_seconds:.3f}s; "
        f"mean {summary.mean_milliseconds_per_image:.3f} ms/image."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
