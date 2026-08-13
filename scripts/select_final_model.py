#!/usr/bin/env python
"""Select a final checkpoint by the predeclared six-metric rank policy.

Candidates are ranked separately on PSNR (higher), SSIM (higher), and LPIPS-Alex
(lower) for validation-ID and pseudo-OOD.  The lowest mean rank wins; ties prefer
the lower pseudo-OOD mean rank, then the lower validation-ID mean rank.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path


METRICS = ("psnr_db", "ssim", "lpips_alex")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidate",
        action="append",
        nargs=3,
        metavar=("NAME", "ID_SUMMARY", "OOD_SUMMARY"),
        required=True,
        help="Repeat for each candidate checkpoint.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _load(path: str) -> dict[str, object]:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise ValueError(f"Metric summary does not exist: {resolved}")
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Metric summary is not valid JSON: {resolved}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Metric summary must contain an object: {resolved}")
    return payload


def _metric(payload: dict[str, object], metric: str) -> float:
    try:
        value = payload["aggregates"]["all"][metric]["mean"]  # type: ignore[index]
        return float(value)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Metric summary lacks a finite {metric} aggregate mean") from exc


def _rank(values: list[tuple[str, float]], *, higher_is_better: bool) -> dict[str, int]:
    """Competition ranks: equal values receive the same rank."""

    ordered = sorted(values, key=lambda item: item[1], reverse=higher_is_better)
    ranks: dict[str, int] = {}
    previous_value: float | None = None
    current_rank = 0
    for position, (name, value) in enumerate(ordered, start=1):
        if previous_value is None or value != previous_value:
            current_rank = position
            previous_value = value
        ranks[name] = current_rank
    return ranks


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        candidates: dict[str, dict[str, object]] = {}
        manifest_hashes: set[str] = set()
        for name, id_path, ood_path in args.candidate:
            if name in candidates:
                raise ValueError(f"Duplicate candidate name: {name}")
            id_summary = _load(id_path)
            ood_summary = _load(ood_path)
            id_manifest = id_summary.get("manifest_sha256")
            ood_manifest = ood_summary.get("manifest_sha256")
            if not isinstance(id_manifest, str) or id_manifest != ood_manifest:
                raise ValueError(f"Candidate {name} has mismatched ID/OOD manifest hashes")
            manifest_hashes.add(id_manifest)
            candidates[name] = {
                "id": id_summary,
                "ood": ood_summary,
                "id_summary": str(Path(id_path).expanduser().resolve()),
                "ood_summary": str(Path(ood_path).expanduser().resolve()),
            }
        if len(manifest_hashes) != 1:
            raise ValueError("Every candidate must use the same manifest")

        ranks: dict[str, dict[str, int]] = {name: {} for name in candidates}
        for split in ("id", "ood"):
            for metric in METRICS:
                score_ranks = _rank(
                    [(name, _metric(item[split], metric)) for name, item in candidates.items()],  # type: ignore[arg-type]
                    higher_is_better=metric != "lpips_alex",
                )
                for name, rank in score_ranks.items():
                    ranks[name][f"{split}_{metric}"] = rank

        table: list[dict[str, object]] = []
        for name, item in candidates.items():
            id_scores = {metric: _metric(item["id"], metric) for metric in METRICS}  # type: ignore[arg-type]
            ood_scores = {metric: _metric(item["ood"], metric) for metric in METRICS}  # type: ignore[arg-type]
            id_mean_rank = sum(ranks[name][f"id_{metric}"] for metric in METRICS) / 3.0
            ood_mean_rank = sum(ranks[name][f"ood_{metric}"] for metric in METRICS) / 3.0
            table.append(
                {
                    "name": name,
                    "checkpoint_sha256": item["id"].get("checkpoint_sha256"),  # type: ignore[index]
                    "id": id_scores,
                    "ood": ood_scores,
                    "ranks": ranks[name],
                    "id_mean_rank": id_mean_rank,
                    "ood_mean_rank": ood_mean_rank,
                    "mean_rank": (id_mean_rank + ood_mean_rank) / 2.0,
                    "id_summary": item["id_summary"],
                    "ood_summary": item["ood_summary"],
                }
            )
        table.sort(key=lambda row: (row["mean_rank"], row["ood_mean_rank"], row["id_mean_rank"], row["name"]))  # type: ignore[index]
        for ordinal, row in enumerate(table, start=1):
            row["selection_order"] = ordinal

        winner = table[0]
        output = args.output.expanduser().resolve()
        if output.exists() and not args.overwrite:
            raise ValueError(f"Selection artifact already exists: {output}; use --overwrite")
        output.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": 1,
            "policy": {
                "metrics": {"psnr_db": "higher", "ssim": "higher", "lpips_alex": "lower"},
                "splits": ["val_id", "val_ood"],
                "aggregation": "mean of six competition ranks",
                "tie_break": "lower val_ood mean rank, then lower val_id mean rank, then name",
            },
            "manifest_sha256": next(iter(manifest_hashes)),
            "winner": winner["name"],
            "winner_checkpoint_sha256": winner["checkpoint_sha256"],
            "candidates": table,
        }
        content = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
        temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
        temporary.write_text(content, encoding="utf-8", newline="")
        temporary.replace(output)
    except (OSError, TypeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(f"Selected {winner['name']} by mean rank {winner['mean_rank']:.3f}.")
    print(f"Selection: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
