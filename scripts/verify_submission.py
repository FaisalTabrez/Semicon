#!/usr/bin/env python
"""Validate final checkpoint integrity and restored public-test outputs."""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from semirestore.checkpoints import load_model_checkpoint  # noqa: E402
from semirestore.data import InputValidationError, discover_npy_files, load_npy_image  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "restored_test_outputs")
    parser.add_argument("--weights", type=Path, default=PROJECT_ROOT / "weights" / "model.pt")
    parser.add_argument("--sha256", type=Path, default=PROJECT_ROOT / "weights" / "model.sha256")
    parser.add_argument("--expected-count", type=int, default=400)
    return parser


def _digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.expected_count < 1:
            raise InputValidationError("--expected-count must be positive")
        inputs = discover_npy_files(args.input_dir)
        outputs = discover_npy_files(args.output_dir)
        if len(inputs) != args.expected_count or len(outputs) != args.expected_count:
            raise InputValidationError(
                f"Expected {args.expected_count} inputs/outputs; got {len(inputs)}/{len(outputs)}"
            )
        expected_paths = {item.relative_path for item in inputs}
        actual_paths = {item.relative_path for item in outputs}
        if actual_paths != expected_paths:
            raise InputValidationError("Output relative paths do not exactly match input relative paths")
        model, payload = load_model_checkpoint(args.weights)
        if payload["model_name"] != "naf_sr" or not any(parameter.numel() for parameter in model.parameters()):
            raise InputValidationError("Final checkpoint is not a valid non-empty NAF-SR model")
        sha_path = args.sha256.expanduser().resolve()
        if not sha_path.is_file():
            raise InputValidationError(f"Missing checkpoint SHA-256 file: {sha_path}")
        expected_digest = sha_path.read_text(encoding="utf-8").strip().split()[0]
        actual_digest = _digest(args.weights.expanduser().resolve())
        if expected_digest != actual_digest:
            raise InputValidationError("Checkpoint SHA-256 does not match weights/model.sha256")
        input_map = {item.relative_path: item for item in inputs}
        for output in outputs:
            array = np.load(output.source, allow_pickle=False)
            source = input_map[output.relative_path]
            source_shape = load_npy_image(source.source).shape
            expected_shape = (source_shape[0] * 2, source_shape[1] * 2)
            if array.dtype != np.float32 or array.shape != expected_shape:
                raise InputValidationError(f"Invalid output dtype/shape: {output.source}")
            if not np.isfinite(array).all() or float(array.min()) < 0.0 or float(array.max()) > 1.0:
                raise InputValidationError(f"Invalid output range/values: {output.source}")
    except (InputValidationError, OSError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(f"Submission verification passed: {len(outputs)} float32 2x outputs and checkpoint hash match.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
