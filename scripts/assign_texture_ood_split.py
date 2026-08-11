#!/usr/bin/env python
"""Assign deterministic train/validation-ID/pseudo-OOD texture splits."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from semirestore.data import InputValidationError  # noqa: E402
from semirestore.splits import assign_texture_ood_split  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "data" / "splits" / "manifest_texture_ood.csv",
    )
    parser.add_argument(
        "--audit",
        type=Path,
        default=PROJECT_ROOT / "reports" / "texture_split_audit.json",
    )
    parser.add_argument("--clusters", type=int, default=12)
    parser.add_argument("--validation-id-fraction", type=float, default=0.15)
    parser.add_argument("--validation-ood-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        payload = assign_texture_ood_split(
            args.manifest,
            args.dataset_root,
            args.output,
            args.audit,
            cluster_count=args.clusters,
            ood_fraction=args.validation_ood_fraction,
            id_fraction=args.validation_id_fraction,
            seed=args.seed,
            overwrite=args.overwrite,
        )
    except (InputValidationError, OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    counts = payload["split_counts"]
    print(
        f"Assigned texture split: {counts['train']} train, {counts['val_id']} val_id, "
        f"{counts['val_ood']} val_ood across {payload['cluster_count']} clusters."
    )
    print("Held-out OOD clusters:", ", ".join(payload["ood_clusters"]))
    print(f"Manifest: {args.output.resolve()}")
    print(f"Audit: {args.audit.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
