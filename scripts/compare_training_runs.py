#!/usr/bin/env python
"""Compare deterministic training artifacts and fail outside a numeric tolerance."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from semirestore.checkpoints import load_checkpoint_payload  # noqa: E402
from semirestore.data import InputValidationError  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_a", type=Path)
    parser.add_argument("run_b", type=Path)
    parser.add_argument("--atol", type=float, default=0.0)
    parser.add_argument("--report", type=Path)
    return parser


def _checkpoint_path(run: Path) -> Path:
    return run / "best.pt" if run.is_dir() else run


def _compare_state_dicts(
    first: dict[str, object], second: dict[str, object]
) -> tuple[float, str | None]:
    if first.keys() != second.keys():
        raise InputValidationError("Checkpoint tensor keys differ")
    maximum = 0.0
    maximum_key: str | None = None
    for key in first:
        left, right = first[key], second[key]
        if not isinstance(left, torch.Tensor) or not isinstance(right, torch.Tensor):
            raise InputValidationError(f"Non-tensor state entry: {key}")
        if left.shape != right.shape or left.dtype != right.dtype:
            raise InputValidationError(f"Tensor metadata differs: {key}")
        difference = (
            float((left.double() - right.double()).abs().max().item())
            if left.numel()
            else 0.0
        )
        if difference > maximum:
            maximum, maximum_key = difference, key
    return maximum, maximum_key


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.atol < 0:
            raise InputValidationError("--atol cannot be negative")
        first = load_checkpoint_payload(_checkpoint_path(args.run_a))
        second = load_checkpoint_payload(_checkpoint_path(args.run_b))
        for key in ("model_name", "model_config"):
            if first.get(key) != second.get(key):
                raise InputValidationError(f"Checkpoint {key} differs")
        first_data, second_data = first.get("data"), second.get("data")
        if not isinstance(first_data, dict) or not isinstance(second_data, dict):
            raise InputValidationError("Checkpoint data provenance is missing")
        if first_data.get("manifest_sha256") != second_data.get("manifest_sha256"):
            raise InputValidationError("Manifest hashes differ")
        first_state, second_state = first.get("model_state_dict"), second.get(
            "model_state_dict"
        )
        if not isinstance(first_state, dict) or not isinstance(second_state, dict):
            raise InputValidationError("Model state is missing")
        maximum, maximum_key = _compare_state_dicts(first_state, second_state)
        report = {
            "schema_version": 1,
            "run_a": str(args.run_a.resolve()),
            "run_b": str(args.run_b.resolve()),
            "manifest_sha256": first_data["manifest_sha256"],
            "maximum_absolute_parameter_difference": maximum,
            "maximum_difference_tensor": maximum_key,
            "absolute_tolerance": args.atol,
            "passed": maximum <= args.atol,
        }
        rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
        if args.report is not None:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(rendered, encoding="utf-8")
        print(rendered, end="")
        return 0 if report["passed"] else 1
    except (InputValidationError, OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
