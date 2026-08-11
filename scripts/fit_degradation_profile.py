#!/usr/bin/env python
"""Fit conservative synthetic-degradation ranges from training pairs only."""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from semirestore.data import InputValidationError  # noqa: E402
from semirestore.degradations import fit_degradation_profile  # noqa: E402
from semirestore.training_data import read_manifest_pairs  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "reports" / "degradation_profile.json",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        manifest = args.manifest.expanduser().resolve()
        if args.split != "train":
            raise InputValidationError(
                "Degradation fitting is restricted to the manifest's train split"
            )
        pairs = read_manifest_pairs(manifest, args.dataset_root, split=args.split)
        payload = fit_degradation_profile(
            pairs,
            args.output,
            manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
            overwrite=args.overwrite,
        )
    except (InputValidationError, OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(f"Fitted degradation profile from {payload['fit_pair_count']} training pairs.")
    print(f"Gaussian noise std: {payload['gaussian_noise_std']}")
    print(f"Speckle std: {payload['speckle_std']}")
    print(f"Blur sigma: {payload['blur_sigma']}")
    print(f"Profile: {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
