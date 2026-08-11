#!/usr/bin/env python
"""Record an isolated ID/OOD ablation decision from locked metric summaries."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from io import StringIO
from pathlib import Path


FIELDS = (
    "experiment",
    "isolated_change",
    "baseline_checkpoint_sha256",
    "candidate_checkpoint_sha256",
    "manifest_sha256",
    "id_psnr_delta_db",
    "id_ssim_delta",
    "id_lpips_delta",
    "ood_psnr_delta_db",
    "ood_ssim_delta",
    "ood_lpips_delta",
    "ood_metric_wins",
    "id_psnr_floor_db",
    "id_ssim_floor",
    "decision",
    "reason",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--isolated-change", required=True)
    parser.add_argument("--baseline-id", type=Path, required=True)
    parser.add_argument("--baseline-ood", type=Path, required=True)
    parser.add_argument("--candidate-id", type=Path, required=True)
    parser.add_argument("--candidate-ood", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--id-psnr-tolerance", type=float, default=0.15)
    parser.add_argument("--id-ssim-tolerance", type=float, default=0.002)
    parser.add_argument("--replace", action="store_true")
    return parser


def _load(path: Path) -> dict[str, object]:
    payload = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or "aggregates" not in payload:
        raise ValueError(f"Invalid metric summary: {path}")
    return payload


def _mean(payload: dict[str, object], metric: str) -> float:
    return float(payload["aggregates"]["all"][metric]["mean"])


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        summaries = [
            _load(path)
            for path in (
                args.baseline_id,
                args.baseline_ood,
                args.candidate_id,
                args.candidate_ood,
            )
        ]
        manifest_hashes = {summary.get("manifest_sha256") for summary in summaries}
        if len(manifest_hashes) != 1 or None in manifest_hashes:
            raise ValueError("All ablation summaries must use the same manifest")
        baseline_id, baseline_ood, candidate_id, candidate_ood = summaries
        id_deltas = {
            metric: _mean(candidate_id, metric) - _mean(baseline_id, metric)
            for metric in ("psnr_db", "ssim", "lpips_alex")
        }
        ood_deltas = {
            metric: _mean(candidate_ood, metric) - _mean(baseline_ood, metric)
            for metric in ("psnr_db", "ssim", "lpips_alex")
        }
        id_safe = (
            id_deltas["psnr_db"] >= -args.id_psnr_tolerance
            and id_deltas["ssim"] >= -args.id_ssim_tolerance
        )
        ood_wins = sum(
            (
                ood_deltas["psnr_db"] > 0,
                ood_deltas["ssim"] > 0,
                ood_deltas["lpips_alex"] < 0,
            )
        )
        decision = "keep" if id_safe and ood_wins >= 2 else "reject"
        reason = (
            f"OOD improved on {ood_wins}/3 locked metrics; "
            f"ID PSNR/SSIM deltas were {id_deltas['psnr_db']:+.6f} dB/"
            f"{id_deltas['ssim']:+.6f}."
        )
        row = {
            "experiment": args.experiment,
            "isolated_change": args.isolated_change,
            "baseline_checkpoint_sha256": baseline_id.get("checkpoint_sha256"),
            "candidate_checkpoint_sha256": candidate_id.get("checkpoint_sha256"),
            "manifest_sha256": next(iter(manifest_hashes)),
            "id_psnr_delta_db": id_deltas["psnr_db"],
            "id_ssim_delta": id_deltas["ssim"],
            "id_lpips_delta": id_deltas["lpips_alex"],
            "ood_psnr_delta_db": ood_deltas["psnr_db"],
            "ood_ssim_delta": ood_deltas["ssim"],
            "ood_lpips_delta": ood_deltas["lpips_alex"],
            "ood_metric_wins": ood_wins,
            "id_psnr_floor_db": -args.id_psnr_tolerance,
            "id_ssim_floor": -args.id_ssim_tolerance,
            "decision": decision,
            "reason": reason,
        }
        output = args.output.expanduser().resolve()
        existing = []
        if output.is_file():
            with output.open("r", encoding="utf-8", newline="") as handle:
                existing = list(csv.DictReader(handle))
        duplicate = any(item["experiment"] == args.experiment for item in existing)
        if duplicate and not args.replace:
            raise ValueError("Experiment already exists; use --replace")
        existing = [item for item in existing if item["experiment"] != args.experiment]
        existing.append(row)
        buffer = StringIO(newline="")
        writer = csv.DictWriter(buffer, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(existing)
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
        temporary.write_text(buffer.getvalue(), encoding="utf-8", newline="")
        temporary.replace(output)
    except (KeyError, OSError, TypeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(f"Ablation {args.experiment}: {decision}. {reason}")
    print(f"Table: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
