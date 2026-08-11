#!/usr/bin/env python
"""Build the deterministic KLA training-pair manifest and audit report."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from semirestore.data import InputValidationError  # noqa: E402
from semirestore.manifest import build_paired_manifest  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Pair NoisyLR/GT arrays by stem and write a validated manifest plus audit."
    )
    parser.add_argument("--input-dir", type=Path, required=True, help="Extracted NoisyLR directory")
    parser.add_argument("--target-dir", type=Path, required=True, help="Extracted GT directory")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        help="Root used for relative paths; defaults to the directories' common parent",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_ROOT / "data" / "splits" / "manifest.csv",
        help="Output CSV path",
    )
    parser.add_argument(
        "--audit",
        type=Path,
        default=PROJECT_ROOT / "reports" / "dataset_audit.json",
        help="Output JSON audit path",
    )
    parser.add_argument(
        "--expected-pairs",
        type=int,
        default=3200,
        help="Fail unless this many pairs are found; use 0 to disable the count gate",
    )
    parser.add_argument("--overwrite", action="store_true", help="Replace existing artifacts")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.expected_pairs < 0:
        print("ERROR: --expected-pairs cannot be negative", file=sys.stderr)
        return 1
    expected_pairs = args.expected_pairs or None
    try:
        result = build_paired_manifest(
            args.input_dir,
            args.target_dir,
            args.manifest,
            args.audit,
            dataset_root=args.dataset_root,
            expected_pairs=expected_pairs,
            overwrite=args.overwrite,
        )
    except (InputValidationError, OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(
        f"Validated {result.pair_count} pair(s). Input range "
        f"[{result.input_global_min:.6g}, {result.input_global_max:.6g}]; target range "
        f"[{result.target_global_min:.6g}, {result.target_global_max:.6g}]."
    )
    print(f"Manifest: {result.manifest_path}")
    print(f"Audit: {result.audit_path}")
    print(f"Manifest SHA-256: {result.manifest_sha256}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
