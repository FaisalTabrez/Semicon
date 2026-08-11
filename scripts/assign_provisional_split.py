#!/usr/bin/env python
"""Create the temporary deterministic validation-ID manifest."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from semirestore.data import InputValidationError  # noqa: E402
from semirestore.splits import assign_provisional_hash_split  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Assign a reproducible provisional train/val_id split. This is not the final "
            "texture-aware OOD split."
        )
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_ROOT / "data" / "splits" / "manifest.csv",
        help="Input paired-data manifest",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "data" / "splits" / "manifest_provisional.csv",
        help="Output manifest with train/val_id assignments",
    )
    parser.add_argument(
        "--audit",
        type=Path,
        default=PROJECT_ROOT / "reports" / "provisional_split_audit.json",
        help="Output split audit JSON",
    )
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        payload = assign_provisional_hash_split(
            args.manifest,
            args.output,
            args.audit,
            validation_fraction=args.validation_fraction,
            seed=args.seed,
            overwrite=args.overwrite,
        )
    except (InputValidationError, OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    counts = payload["split_counts"]
    print(
        f"Assigned provisional split: {counts['train']} train, "
        f"{counts['val_id']} val_id. No samples are labeled OOD."
    )
    print(f"Manifest: {args.output.expanduser().resolve()}")
    print(f"Audit: {args.audit.expanduser().resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
