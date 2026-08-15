#!/usr/bin/env python
"""Cold external-domain controlled-degradation validation on Carinthia SEM images."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from semirestore.checkpoints import load_model_checkpoint  # noqa: E402
from semirestore.data import InputValidationError  # noqa: E402
from semirestore.metrics import (  # noqa: E402
    LPIPS_POLICY,
    SSIM_POLICY,
    compute_image_metrics,
    create_lpips_model,
    lpips_distance,
)

DATASET = {
    "name": "Carinthia SEM Defect Dataset",
    "doi": "10.5281/zenodo.10715190",
    "record_url": "https://zenodo.org/records/10715190",
    "data_url": "https://zenodo.org/records/10715190/files/data.zip?download=1",
    "data_zip_md5": "457011cf9063e5a49751f33ea468309d",
    "license": "CC BY 4.0",
}

SEVERITIES = {
    "downsample_only": {
        "gaussian_noise_std": 0.0,
        "speckle_std": 0.0,
        "bias_low": 0.0,
        "bias_high": 0.0,
    },
    "profile_low": {
        "gaussian_noise_std": 0.032383059530957825,
        "speckle_std": 0.04536565160023246,
        "bias_low": -0.000876969425291918,
        "bias_high": 0.0008910448643320024,
    },
    "profile_high": {
        "gaussian_noise_std": 0.08066570470161874,
        "speckle_std": 0.06322856237103856,
        "bias_low": -0.000876969425291918,
        "bias_high": 0.0008910448643320024,
    },
}

CSV_FIELDS = (
    "relative_path",
    "defect_group",
    "severity",
    "method",
    "psnr_db",
    "ssim",
    "lpips_alex",
    "sobel_l1",
    "mean_intensity_bias",
    "pre_clamp_out_of_range_rate",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image_root", type=Path)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--expected-count", type=int, default=4591)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--no-lpips", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _discover(root: Path) -> list[Path]:
    resolved = root.expanduser().resolve()
    if not resolved.is_dir():
        raise InputValidationError(f"Image root does not exist: {resolved}")
    suffixes = {".jpg", ".jpeg"}
    paths = sorted(
        path for path in resolved.rglob("*") if path.is_file() and path.suffix.lower() in suffixes
    )
    return paths


def _load_grayscale(path: Path) -> np.ndarray:
    try:
        with Image.open(path) as image:
            array = np.asarray(image.convert("L"), dtype=np.float32) / 255.0
    except (OSError, ValueError) as exc:
        raise InputValidationError(f"Could not decode {path}: {exc}") from exc
    if array.ndim != 2 or min(array.shape) < 12 or array.shape[0] % 2 or array.shape[1] % 2:
        raise InputValidationError(f"Expected an even grayscale image at least 12x12: {path}")
    return np.ascontiguousarray(array)


def _chunks(values: list[Path], size: int) -> Iterable[list[Path]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


def _seed(relative_path: str, severity: str, global_seed: int) -> int:
    digest = hashlib.sha256(
        f"{global_seed}|{severity}|{relative_path}".encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "little", signed=False)


def _degrade(
    targets: torch.Tensor,
    relative_paths: list[str],
    severity_name: str,
    global_seed: int,
) -> torch.Tensor:
    profile = SEVERITIES[severity_name]
    degraded = F.interpolate(
        targets,
        scale_factor=0.5,
        mode="bicubic",
        align_corners=False,
        antialias=True,
    ).cpu()
    gaussian_std = float(profile["gaussian_noise_std"])
    speckle_std = float(profile["speckle_std"])
    bias_low = float(profile["bias_low"])
    bias_high = float(profile["bias_high"])
    if not (gaussian_std or speckle_std or bias_low or bias_high):
        return degraded
    for index, relative_path in enumerate(relative_paths):
        rng = np.random.default_rng(_seed(relative_path, severity_name, global_seed))
        shape = tuple(degraded[index].shape)
        gaussian = torch.from_numpy(rng.standard_normal(shape).astype(np.float32))
        speckle = torch.from_numpy(rng.standard_normal(shape).astype(np.float32))
        bias = float(rng.uniform(bias_low, bias_high))
        degraded[index] = (
            degraded[index]
            + gaussian * gaussian_std
            + degraded[index] * speckle * speckle_std
            + bias
        )
    return degraded.contiguous()


def _finite_summary(values: list[float], samples: int, seed: int) -> dict[str, object]:
    array = np.asarray(values, dtype=np.float64)
    finite = array[np.isfinite(array)]
    if not finite.size:
        return {"mean": None, "median": None, "bootstrap_95_ci": [None, None]}
    rng = np.random.default_rng(seed)
    means = np.asarray(
        [rng.choice(finite, size=finite.size, replace=True).mean() for _ in range(samples)]
    ) if samples else np.asarray([], dtype=np.float64)
    return {
        "mean": float(finite.mean()),
        "median": float(np.median(finite)),
        "bootstrap_95_ci": (
            [float(value) for value in np.quantile(means, (0.025, 0.975))]
            if samples else [None, None]
        ),
        "finite_count": int(finite.size),
    }


def _aggregate(rows: list[dict[str, object]], samples: int, seed: int) -> dict[str, object]:
    output: dict[str, object] = {"count": len(rows)}
    for offset, metric in enumerate(
        ("psnr_db", "ssim", "lpips_alex", "sobel_l1", "mean_intensity_bias")
    ):
        values = [float(row[metric]) for row in rows if row[metric] is not None]
        output[metric] = _finite_summary(values, samples, seed + offset)
    return output


def _paired_summary(
    rows: list[dict[str, object]], samples: int, seed: int
) -> dict[str, object]:
    indexed = {
        (str(row["relative_path"]), str(row["severity"]), str(row["method"])): row
        for row in rows
    }
    output: dict[str, object] = {}
    for severity_index, severity in enumerate(SEVERITIES):
        deltas: dict[str, list[float]] = {"psnr_db": [], "ssim": [], "lpips_alex": []}
        wins = {metric: 0 for metric in deltas}
        count = 0
        paths = sorted(
            str(row["relative_path"])
            for row in rows
            if row["severity"] == severity and row["method"] == "bicubic"
        )
        for relative_path in paths:
            baseline = indexed[(relative_path, severity, "bicubic")]
            candidate = indexed[(relative_path, severity, "naf_sr")]
            count += 1
            for metric in deltas:
                if baseline[metric] is None or candidate[metric] is None:
                    continue
                delta = float(candidate[metric]) - float(baseline[metric])
                deltas[metric].append(delta)
                if (metric == "lpips_alex" and delta < 0) or (
                    metric != "lpips_alex" and delta > 0
                ):
                    wins[metric] += 1
        output[severity] = {
            "count": count,
            "candidate_minus_bicubic": {
                metric: _finite_summary(
                    values, samples, seed + severity_index * 100 + index
                )
                for index, (metric, values) in enumerate(deltas.items())
            },
            "candidate_win_rate": {
                metric: wins[metric] / len(values) if values else None
                for metric, values in deltas.items()
            },
        }
    return output


def _atomic_text(path: Path, text: str) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(text, encoding="utf-8", newline="")
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.batch_size < 1 or args.expected_count < 1 or args.bootstrap_samples < 0:
            raise InputValidationError("Invalid batch, count, or bootstrap setting")
        output_dir = args.output_dir.expanduser().resolve()
        if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
            raise InputValidationError(f"Output directory is not empty: {output_dir}")
        output_dir.mkdir(parents=True, exist_ok=True)
        image_root = args.image_root.expanduser().resolve()
        images = _discover(image_root)
        if len(images) != args.expected_count:
            raise InputValidationError(
                f"Expected {args.expected_count} JPEG images; discovered {len(images)}"
            )
        device = torch.device(args.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise InputValidationError("CUDA requested but unavailable")
        model, checkpoint = load_model_checkpoint(args.weights, map_location="cpu")
        if checkpoint.get("model_name") != "naf_sr":
            raise InputValidationError("External final validation requires a NAF-SR checkpoint")
        model = model.to(device).eval()
        lpips_model = None if args.no_lpips else create_lpips_model(device)
        rows: list[dict[str, object]] = []
        for severity in SEVERITIES:
            for batch_index, batch_paths in enumerate(_chunks(images, args.batch_size), start=1):
                arrays = [_load_grayscale(path) for path in batch_paths]
                shapes = {array.shape for array in arrays}
                if len(shapes) != 1:
                    raise InputValidationError("Mixed image shapes inside an external batch")
                relative_paths = [path.relative_to(image_root).as_posix() for path in batch_paths]
                targets_cpu = torch.from_numpy(np.stack(arrays))[:, None]
                degraded_cpu = _degrade(targets_cpu, relative_paths, severity, args.seed)
                targets = targets_cpu.to(device)
                degraded = degraded_cpu.to(device)
                with torch.inference_mode():
                    bicubic = F.interpolate(
                        degraded,
                        size=targets.shape[-2:],
                        mode="bicubic",
                        align_corners=False,
                        antialias=True,
                    )
                    restored = model(degraded)
                if restored.shape != targets.shape or not torch.isfinite(restored).all():
                    raise RuntimeError("Model returned an invalid external prediction")
                method_tensors = {"bicubic": bicubic, "naf_sr": restored}
                for method, prediction in method_tensors.items():
                    lpips_values: list[float | None]
                    if lpips_model is None:
                        lpips_values = [None] * len(batch_paths)
                    else:
                        lpips_values = [
                            float(value)
                            for value in lpips_distance(lpips_model, prediction, targets).cpu()
                        ]
                    prediction_arrays = prediction.float().cpu().numpy()[:, 0]
                    for path, relative_path, prediction_array, target_array, lpips_value in zip(
                        batch_paths,
                        relative_paths,
                        prediction_arrays,
                        arrays,
                        lpips_values,
                        strict=True,
                    ):
                        metrics = compute_image_metrics(
                            prediction_array, target_array, lpips_value=lpips_value
                        )
                        rows.append(
                            {
                                "relative_path": relative_path,
                                "defect_group": path.parent.name,
                                "severity": severity,
                                "method": method,
                                "psnr_db": metrics.psnr_db,
                                "ssim": metrics.ssim,
                                "lpips_alex": metrics.lpips_alex,
                                "sobel_l1": metrics.sobel_l1,
                                "mean_intensity_bias": metrics.mean_intensity_bias,
                                "pre_clamp_out_of_range_rate": metrics.pre_clamp_out_of_range_rate,
                            }
                        )
                if batch_index % 25 == 0:
                    print(f"{severity}: processed {min(batch_index * args.batch_size, len(images))}/{len(images)}")

        csv_path = output_dir / "per_image_metrics.csv"
        from io import StringIO

        buffer = StringIO(newline="")
        writer = csv.DictWriter(buffer, fieldnames=CSV_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
        _atomic_text(csv_path, buffer.getvalue())

        aggregates: dict[str, object] = {}
        for severity_index, severity in enumerate(SEVERITIES):
            aggregates[severity] = {}
            for method_index, method in enumerate(("bicubic", "naf_sr")):
                group = [
                    row for row in rows
                    if row["severity"] == severity and row["method"] == method
                ]
                aggregates[severity][method] = _aggregate(
                    group,
                    args.bootstrap_samples,
                    args.seed + severity_index * 1000 + method_index * 100,
                )

        summary = {
            "schema_version": 1,
            "evidence_label": "external-domain controlled-degradation validation",
            "claim_boundary": (
                "Carinthia supplies real semiconductor SEM reference images but no native aligned "
                "LR/HR pairs; degradations are deterministic controls and are not claimed as "
                "measured acquisition degradations."
            ),
            "dataset": {
                **DATASET,
                "image_count": len(images),
                "defect_group_counts": dict(sorted(Counter(path.parent.name for path in images).items())),
                "used_for_training_or_tuning": False,
            },
            "protocol": {
                "seed": args.seed,
                "scale": 2,
                "reference": "decoded grayscale JPEG normalized to [0,1]",
                "downsample": "PyTorch bicubic, align_corners=False, antialias=True",
                "noise": "deterministic per relative path; applied after downsampling; no clipping",
                "severities": SEVERITIES,
                "metric_policy": {
                    "psnr_data_range": 1.0,
                    "ssim": SSIM_POLICY,
                    "lpips": {**LPIPS_POLICY, "enabled": not args.no_lpips},
                },
                "bootstrap_samples": args.bootstrap_samples,
            },
            "checkpoint_sha256": hashlib.sha256(args.weights.read_bytes()).hexdigest(),
            "model_config": checkpoint["model_config"],
            "device": str(device),
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "aggregates": aggregates,
            "paired_comparison": _paired_summary(
                rows, args.bootstrap_samples, args.seed + 10_000
            ),
        }
        _atomic_text(
            output_dir / "summary.json",
            json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        )
    except (InputValidationError, OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print("External Carinthia controlled-degradation validation complete.")
    for severity in SEVERITIES:
        paired = summary["paired_comparison"][severity]
        print(
            f"{severity}: "
            f"delta PSNR={paired['candidate_minus_bicubic']['psnr_db']['mean']:+.4f} dB, "
            f"delta SSIM={paired['candidate_minus_bicubic']['ssim']['mean']:+.6f}, "
            f"delta LPIPS={paired['candidate_minus_bicubic']['lpips_alex']['mean']}"
        )
    print(f"Summary: {output_dir / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
