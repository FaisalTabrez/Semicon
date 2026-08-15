#!/usr/bin/env python
"""Evaluate a labeled restoration baseline and emit reproducible evidence."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
import time
from collections import Counter
from contextlib import nullcontext
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from semirestore.data import (  # noqa: E402
    InputImage,
    InputValidationError,
    discover_npy_files,
    load_npy_image,
)
from semirestore.checkpoints import load_model_checkpoint  # noqa: E402
from semirestore.inference import resolve_device, resolve_precision  # noqa: E402
from semirestore.metrics import (  # noqa: E402
    LPIPS_POLICY,
    SSIM_POLICY,
    compute_image_metrics,
    create_lpips_model,
    lpips_distance,
)
from semirestore.models import BicubicRestorer  # noqa: E402

CSV_FIELDS = (
    "stem",
    "split",
    "texture_cluster",
    "psnr_db",
    "ssim",
    "lpips_alex",
    "sobel_l1",
    "mean_intensity_bias",
    "pre_clamp_out_of_range_rate",
    "prediction_shape",
    "target_shape",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Score a restoration model on paired arrays.")
    parser.add_argument("input_dir", type=Path, help="Directory containing degraded arrays")
    parser.add_argument("target_dir", type=Path, help="Directory containing aligned GT arrays")
    parser.add_argument(
        "--model", choices=("bicubic", "edsr_lite", "naf_sr"), default="bicubic"
    )
    parser.add_argument("--weights", type=Path, help="Checkpoint required by learned models")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--precision",
        choices=("fp32", "bf16", "fp16"),
        default="fp32",
        help="Inference precision for a measured parity experiment; default is FP32",
    )
    parser.add_argument(
        "--memory-format",
        choices=("contiguous", "channels_last"),
        default="contiguous",
        help="Model/input CUDA memory format; default preserves the packaged path",
    )
    parser.add_argument(
        "--compile-mode",
        choices=("none", "default", "reduce-overhead", "max-autotune"),
        default="none",
        help="Optional torch.compile mode for fixed-runtime parity evaluation",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--manifest",
        type=Path,
        help="Optional manifest supplying split labels; required when --split is used",
    )
    parser.add_argument(
        "--split",
        action="append",
        help="Evaluate only this manifest split; repeat to select multiple splits",
    )
    parser.add_argument(
        "--per-image-csv",
        type=Path,
        default=PROJECT_ROOT / "reports" / "bicubic_validation_metrics.csv",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=PROJECT_ROOT / "reports" / "bicubic_validation_summary.json",
    )
    parser.add_argument(
        "--no-lpips",
        action="store_true",
        help="Development-only fast path; summary records LPIPS as disabled",
    )
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=2026)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _index_by_stem(items: Iterable[InputImage], label: str) -> dict[str, InputImage]:
    indexed: dict[str, InputImage] = {}
    for item in items:
        stem = item.source.stem
        if stem in indexed:
            raise InputValidationError(
                f"Duplicate {label} stem '{stem}': {indexed[stem].source} and {item.source}"
            )
        indexed[stem] = item
    return indexed


def _load_manifest_labels(path: Path) -> tuple[dict[str, dict[str, str]], str]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise InputValidationError(f"Manifest does not exist: {resolved}")
    labels: dict[str, dict[str, str]] = {}
    with resolved.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or not {"stem", "split"}.issubset(reader.fieldnames):
            raise InputValidationError("Manifest must contain 'stem' and 'split' columns")
        for row in reader:
            stem = row["stem"].strip()
            if not stem:
                raise InputValidationError("Manifest contains an empty stem")
            if stem in labels:
                raise InputValidationError(f"Manifest contains duplicate stem '{stem}'")
            labels[stem] = {
                "split": row["split"].strip() or "unassigned",
                "texture_cluster": row.get("texture_cluster", "").strip() or "unassigned",
            }
    digest = hashlib.sha256(resolved.read_bytes()).hexdigest()
    return labels, digest


def _artifact(path: Path, *, overwrite: bool) -> Path:
    resolved = path.expanduser().resolve()
    if resolved.exists() and not overwrite:
        raise InputValidationError(
            f"Evidence artifact already exists: {resolved}. Use --overwrite to replace it."
        )
    if resolved.exists() and not resolved.is_file():
        raise InputValidationError(f"Evidence artifact path is not a file: {resolved}")
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved


def _atomic_text(path: Path, content: str) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(content, encoding="utf-8", newline="")
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _chunks(values: list[str], size: int) -> Iterable[list[str]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


def _number(value: float | None) -> str:
    return "" if value is None else format(float(value), ".10g")


def _finite_summary(
    values: list[float | None], *, bootstrap_samples: int, seed: int
) -> dict[str, object]:
    array = np.asarray([value for value in values if value is not None], dtype=np.float64)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return {
            "mean": None,
            "median": None,
            "bootstrap_95_ci": [None, None],
            "finite_count": 0,
            "non_finite_or_missing_count": len(values),
        }
    rng = np.random.default_rng(seed)
    if bootstrap_samples:
        means = np.empty(bootstrap_samples, dtype=np.float64)
        for index in range(bootstrap_samples):
            means[index] = rng.choice(finite, size=finite.size, replace=True).mean()
        interval = [float(value) for value in np.quantile(means, [0.025, 0.975])]
    else:
        interval = [None, None]
    return {
        "mean": float(finite.mean()),
        "median": float(np.median(finite)),
        "bootstrap_95_ci": interval,
        "finite_count": int(finite.size),
        "non_finite_or_missing_count": len(values) - int(finite.size),
    }


def _aggregate(
    rows: list[dict[str, str]], *, bootstrap_samples: int, seed: int
) -> dict[str, object]:
    fields = (
        "psnr_db",
        "ssim",
        "lpips_alex",
        "sobel_l1",
        "mean_intensity_bias",
        "pre_clamp_out_of_range_rate",
    )
    payload: dict[str, object] = {"count": len(rows)}
    for offset, field in enumerate(fields):
        values = [float(row[field]) if row[field] else None for row in rows]
        payload[field] = _finite_summary(
            values,
            bootstrap_samples=bootstrap_samples,
            seed=seed + offset,
        )
    return payload


def _summary_payload(
    rows: list[dict[str, str]],
    *,
    device: torch.device,
    precision: str,
    elapsed_seconds: float,
    manifest_sha256: str | None,
    lpips_enabled: bool,
    bootstrap_samples: int,
    bootstrap_seed: int,
    model_name: str,
    checkpoint_sha256: str | None,
    memory_format: str,
    compile_mode: str,
) -> dict[str, object]:
    split_groups: dict[str, list[dict[str, str]]] = {}
    cluster_groups: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        split_groups.setdefault(row["split"], []).append(row)
        cluster_groups.setdefault(row["texture_cluster"], []).append(row)

    worst_count = max(1, math.ceil(len(rows) * 0.1))
    finite_psnr_rows = [row for row in rows if math.isfinite(float(row["psnr_db"]))]
    worst_rows = sorted(finite_psnr_rows, key=lambda row: float(row["psnr_db"]))[:worst_count]
    return {
        "schema_version": 1,
        "model": model_name,
        "evidence_label": (
            "labeled bicubic lower bound"
            if model_name == "bicubic"
            else f"labeled {model_name} checkpoint evaluation"
        ),
        "checkpoint_sha256": checkpoint_sha256,
        "device": str(device),
        "precision": precision,
        "execution": {
            "memory_format": memory_format,
            "compile_mode": compile_mode,
            "steady_state_timing": False,
            "note": "Elapsed time includes compilation and metric computation.",
        },
        "pair_count": len(rows),
        "split_counts": dict(sorted(Counter(row["split"] for row in rows).items())),
        "manifest_sha256": manifest_sha256,
        "elapsed_seconds": elapsed_seconds,
        "mean_milliseconds_per_image": elapsed_seconds * 1000.0 / len(rows),
        "metric_policy": {
            "prediction_scoring_boundary": "clamp to [0,1]",
            "target_scoring_boundary": "clamp to [0,1]",
            "psnr": {"data_range": 1.0},
            "ssim": SSIM_POLICY,
            "lpips": {**LPIPS_POLICY, "enabled": lpips_enabled},
            "bootstrap": {
                "samples": bootstrap_samples,
                "confidence": 0.95,
                "seed": bootstrap_seed,
            },
        },
        "aggregates": {
            "all": _aggregate(
                rows, bootstrap_samples=bootstrap_samples, seed=bootstrap_seed
            ),
            "by_split": {
                split: _aggregate(
                    group,
                    bootstrap_samples=bootstrap_samples,
                    seed=bootstrap_seed + index * 100,
                )
                for index, (split, group) in enumerate(sorted(split_groups.items()), start=1)
            },
            "by_texture_cluster": {
                cluster: _aggregate(
                    group,
                    bootstrap_samples=bootstrap_samples,
                    seed=bootstrap_seed + 1_000 + index * 100,
                )
                for index, (cluster, group) in enumerate(
                    sorted(cluster_groups.items()), start=1
                )
            },
            "worst_psnr_decile": _aggregate(
                worst_rows,
                bootstrap_samples=bootstrap_samples,
                seed=bootstrap_seed + 10_000,
            ),
        },
        "failure_counts": {"shape": 0, "non_finite": 0},
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.batch_size < 1:
            raise InputValidationError("--batch-size must be at least 1")
        if args.bootstrap_samples < 0:
            raise InputValidationError("--bootstrap-samples cannot be negative")
        if args.split and not args.manifest:
            raise InputValidationError("--split requires --manifest")

        csv_path = _artifact(args.per_image_csv, overwrite=args.overwrite)
        json_path = _artifact(args.summary_json, overwrite=args.overwrite)
        if csv_path == json_path:
            raise InputValidationError("Per-image CSV and summary JSON paths must be different")

        inputs = _index_by_stem(discover_npy_files(args.input_dir), "input")
        targets = _index_by_stem(discover_npy_files(args.target_dir), "target")
        if inputs.keys() != targets.keys():
            missing_targets = sorted(inputs.keys() - targets.keys())
            missing_inputs = sorted(targets.keys() - inputs.keys())
            raise InputValidationError(
                "Input/target stems do not match; "
                f"missing targets={missing_targets[:10]}, missing inputs={missing_inputs[:10]}"
            )

        manifest_labels: dict[str, dict[str, str]] = {}
        manifest_sha256 = None
        if args.manifest:
            manifest_labels, manifest_sha256 = _load_manifest_labels(args.manifest)
            unknown = inputs.keys() - manifest_labels.keys()
            if unknown:
                raise InputValidationError(
                    f"Manifest is missing discovered stem(s): {', '.join(sorted(unknown)[:10])}"
                )

        selected_splits = set(args.split or [])
        stems = [
            stem
            for stem in sorted(inputs)
            if not selected_splits
            or manifest_labels.get(stem, {}).get("split", "unassigned") in selected_splits
        ]
        if not stems:
            raise InputValidationError(
                f"No pairs selected for split filter: {', '.join(sorted(selected_splits))}"
            )

        device = resolve_device(args.device)
        precision = resolve_precision(args.precision, device)
        if args.compile_mode != "none" and device.type != "cuda":
            raise InputValidationError("--compile-mode requires CUDA")
        if args.memory_format == "channels_last" and device.type != "cuda":
            raise InputValidationError("--memory-format channels_last requires CUDA")
        checkpoint_sha256 = None
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
            checkpoint_sha256 = hashlib.sha256(
                args.weights.expanduser().resolve().read_bytes()
            ).hexdigest()
        model = model.to(device).eval()
        if args.memory_format == "channels_last":
            model = model.to(memory_format=torch.channels_last)
        if args.compile_mode != "none":
            model = torch.compile(
                model,
                mode=args.compile_mode,
                fullgraph=False,
                dynamic=False,
            )
        lpips_model = None if args.no_lpips else create_lpips_model(device)
        rows: list[dict[str, str]] = []
        started = time.perf_counter()

        with torch.inference_mode():
            for stem_batch in _chunks(stems, args.batch_size):
                input_arrays = [load_npy_image(inputs[stem].source) for stem in stem_batch]
                target_arrays = [load_npy_image(targets[stem].source) for stem in stem_batch]
                shapes = {array.shape for array in input_arrays}
                target_shapes = {array.shape for array in target_arrays}
                if len(shapes) != 1 or len(target_shapes) != 1:
                    raise InputValidationError(
                        "Mixed shapes occurred inside a metric batch; reduce --batch-size to 1"
                    )
                input_tensor = torch.from_numpy(np.stack(input_arrays))[:, None].to(device)
                if args.memory_format == "channels_last":
                    input_tensor = input_tensor.contiguous(memory_format=torch.channels_last)
                autocast_dtype = {
                    "bf16": torch.bfloat16,
                    "fp16": torch.float16,
                }.get(precision)
                autocast_context = (
                    torch.autocast(device_type="cuda", dtype=autocast_dtype)
                    if autocast_dtype is not None
                    else nullcontext()
                )
                with autocast_context:
                    prediction_tensor = model(input_tensor)
                expected_target_shape = tuple(prediction_tensor.shape[-2:])
                if any(array.shape != expected_target_shape for array in target_arrays):
                    raise InputValidationError(
                        f"Target shape must be 2x input; expected {expected_target_shape}"
                    )
                target_tensor = torch.from_numpy(np.stack(target_arrays))[:, None].to(device)
                if not torch.isfinite(prediction_tensor).all():
                    raise RuntimeError(f"{args.model} produced NaN or infinity")

                lpips_values: list[float | None]
                if lpips_model is None:
                    lpips_values = [None] * len(stem_batch)
                else:
                    lpips_values = [
                        float(value)
                        for value in lpips_distance(
                            lpips_model, prediction_tensor, target_tensor
                        ).cpu()
                    ]

                prediction_arrays = prediction_tensor.float().cpu().numpy()[:, 0]
                for stem, prediction, target, lpips_value in zip(
                    stem_batch,
                    prediction_arrays,
                    target_arrays,
                    lpips_values,
                    strict=True,
                ):
                    metrics = compute_image_metrics(
                        prediction, target, lpips_value=lpips_value
                    )
                    rows.append(
                        {
                            "stem": stem,
                            "split": manifest_labels.get(stem, {}).get(
                                "split", "all_labeled"
                            ),
                            "texture_cluster": manifest_labels.get(stem, {}).get(
                                "texture_cluster", "unassigned"
                            ),
                            "psnr_db": _number(metrics.psnr_db),
                            "ssim": _number(metrics.ssim),
                            "lpips_alex": _number(metrics.lpips_alex),
                            "sobel_l1": _number(metrics.sobel_l1),
                            "mean_intensity_bias": _number(metrics.mean_intensity_bias),
                            "pre_clamp_out_of_range_rate": _number(
                                metrics.pre_clamp_out_of_range_rate
                            ),
                            "prediction_shape": "x".join(map(str, prediction.shape)),
                            "target_shape": "x".join(map(str, target.shape)),
                        }
                    )

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - started

        from io import StringIO

        csv_buffer = StringIO(newline="")
        writer = csv.DictWriter(csv_buffer, fieldnames=CSV_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
        summary = _summary_payload(
            rows,
            device=device,
            precision=precision,
            elapsed_seconds=elapsed,
            manifest_sha256=manifest_sha256,
            lpips_enabled=not args.no_lpips,
            bootstrap_samples=args.bootstrap_samples,
            bootstrap_seed=args.bootstrap_seed,
            model_name=args.model,
            checkpoint_sha256=checkpoint_sha256,
            memory_format=args.memory_format,
            compile_mode=args.compile_mode,
        )
        _atomic_text(csv_path, csv_buffer.getvalue())
        _atomic_text(
            json_path,
            json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        )
    except (InputValidationError, OSError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    all_metrics = summary["aggregates"]["all"]
    psnr_mean = all_metrics["psnr_db"]["mean"]
    ssim_mean = all_metrics["ssim"]["mean"]
    lpips_mean = all_metrics["lpips_alex"]["mean"]
    print(
        f"Scored {len(rows)} pair(s) with {args.model} on {device}: "
        f"PSNR={psnr_mean:.4f} dB, SSIM={ssim_mean:.6f}, "
        f"LPIPS={lpips_mean:.6f}" if lpips_mean is not None else
        f"Scored {len(rows)} pair(s) with {args.model} on {device}: "
        f"PSNR={psnr_mean:.4f} dB, SSIM={ssim_mean:.6f}, LPIPS=disabled"
    )
    print(f"Per-image CSV: {csv_path}")
    print(f"Summary JSON: {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
