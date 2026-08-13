#!/usr/bin/env python
"""Measure synchronized model-only restoration latency and CUDA memory."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from semirestore.checkpoints import load_model_checkpoint  # noqa: E402
from semirestore.data import InputValidationError  # noqa: E402
from semirestore.inference import resolve_device, resolve_precision  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--model", choices=("edsr_lite", "naf_sr"), required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="fp32")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--height", type=int, default=128)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _context(device: torch.device, precision: str):
    if precision == "bf16":
        return torch.autocast(device_type=device.type, dtype=torch.bfloat16)
    return nullcontext()


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _percentile(samples: list[float], percentile: float) -> float:
    ordered = sorted(samples)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * percentile)))
    return ordered[index]


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if min(args.batch_size, args.height, args.width, args.iterations) < 1 or args.warmup < 0:
            raise InputValidationError("batch size, dimensions, and iterations must be positive; warmup >= 0")
        device = resolve_device(args.device)
        precision = resolve_precision(args.precision, device)
        model, payload = load_model_checkpoint(args.weights, map_location="cpu")
        if payload["model_name"] != args.model:
            raise InputValidationError(
                f"Checkpoint model is {payload['model_name']}, not requested {args.model}"
            )
        output = args.output.expanduser().resolve()
        if output.exists() and not args.overwrite:
            raise InputValidationError(f"Benchmark artifact already exists: {output}; use --overwrite")
        model = model.to(device).eval()
        inputs = torch.randn(args.batch_size, 1, args.height, args.width, device=device)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        with torch.inference_mode():
            for _ in range(args.warmup):
                with _context(device, precision):
                    _ = model(inputs)
            _synchronize(device)
            timings: list[float] = []
            for _ in range(args.iterations):
                _synchronize(device)
                started = time.perf_counter()
                with _context(device, precision):
                    prediction = model(inputs)
                _synchronize(device)
                timings.append((time.perf_counter() - started) * 1000.0)
        expected = (args.batch_size, 1, args.height * 2, args.width * 2)
        if tuple(prediction.shape) != expected:
            raise RuntimeError(f"Model returned {tuple(prediction.shape)}; expected {expected}")
        if not torch.isfinite(prediction).all():
            raise RuntimeError("Model produced non-finite output")
        median_ms = _percentile(timings, 0.5)
        payload_out = {
            "schema_version": 1,
            "scope": "model-only synthetic-tensor inference; excludes file I/O and checkpoint loading",
            "checkpoint": str(args.weights.expanduser().resolve()),
            "checkpoint_sha256": hashlib.sha256(args.weights.expanduser().resolve().read_bytes()).hexdigest(),
            "model": args.model,
            "model_config": payload["model_config"],
            "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
            "checkpoint_bytes": args.weights.expanduser().resolve().stat().st_size,
            "device": str(device),
            "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "precision": precision,
            "input_shape": list(inputs.shape),
            "output_shape": list(prediction.shape),
            "warmup_iterations": args.warmup,
            "timed_iterations": args.iterations,
            "latency_ms": {
                "mean": sum(timings) / len(timings),
                "median": median_ms,
                "p90": _percentile(timings, 0.90),
                "p95": _percentile(timings, 0.95),
            },
            "images_per_second": (args.batch_size * 1000.0) / median_ms,
            "peak_cuda_memory_bytes": (
                torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
            ),
            "environment": {
                "python": platform.python_version(),
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "cudnn": torch.backends.cudnn.version(),
            },
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
        temporary.write_text(json.dumps(payload_out, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temporary.replace(output)
    except (InputValidationError, OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(
        f"Benchmarked {args.model} on {device} ({precision}): "
        f"median={median_ms:.3f} ms, p95={payload_out['latency_ms']['p95']:.3f} ms, "
        f"{payload_out['images_per_second']:.2f} images/s."
    )
    print(f"Benchmark: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
