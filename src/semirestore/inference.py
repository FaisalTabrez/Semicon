"""Directory-to-directory restoration shared by the standalone CLI."""

from __future__ import annotations

import json
import os
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch import nn

from .data import (
    InputImage,
    InputValidationError,
    discover_npy_files,
    ensure_unique_output_paths,
    load_npy_image,
    validate_output_directory,
)


@dataclass(frozen=True)
class InferenceSummary:
    model: str
    device: str
    precision: str
    input_count: int
    elapsed_seconds: float
    mean_milliseconds_per_image: float
    checkpoint: str | None = None

    def to_dict(self) -> dict[str, str | int | float]:
        return asdict(self)


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise InputValidationError("CUDA was requested, but PyTorch cannot access a CUDA device")
    return torch.device(requested)


def resolve_precision(requested: str, device: torch.device) -> str:
    if requested == "auto":
        return "fp32"
    if requested in {"bf16", "fp16"} and device.type != "cuda":
        raise InputValidationError(
            f"{requested.upper()} inference is only enabled on CUDA after parity testing"
        )
    return requested


def _autocast_context(device: torch.device, precision: str):
    autocast_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(precision)
    if autocast_dtype is not None:
        return torch.autocast(device_type="cuda", dtype=autocast_dtype)
    return nullcontext()


def _atomic_save_npy(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("wb") as handle:
            np.save(handle, array, allow_pickle=False)
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _batches(items: list[InputImage], size: int) -> Iterable[list[InputImage]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def restore_directory(
    model: nn.Module,
    input_dir: str | Path,
    output_dir: str | Path,
    *,
    model_name: str,
    device: str = "auto",
    precision: str = "auto",
    batch_size: int = 1,
    overwrite: bool = False,
    checkpoint: str | None = None,
) -> InferenceSummary:
    """Restore every valid input, preserving its relative `.npy` path."""

    if batch_size < 1:
        raise InputValidationError(f"Batch size must be at least 1; received {batch_size}")

    inputs = discover_npy_files(input_dir)
    ensure_unique_output_paths(inputs)
    input_root = Path(input_dir).expanduser().resolve()
    requested_output = Path(output_dir).expanduser().resolve()
    if requested_output == input_root or requested_output.is_relative_to(input_root):
        raise InputValidationError(
            f"Output directory must not be the input directory or live inside it: {requested_output}"
        )
    output_root = validate_output_directory(output_dir, overwrite=overwrite)
    torch_device = resolve_device(device)
    resolved_precision = resolve_precision(precision, torch_device)

    model = model.to(torch_device)
    model.eval()
    started = time.perf_counter()

    with torch.inference_mode():
        for batch_items in _batches(inputs, batch_size):
            arrays = [load_npy_image(item.source) for item in batch_items]
            # Group within the requested chunk so mixed spatial sizes remain valid
            # without recursively rediscovering or duplicating inputs.
            shape_groups: dict[tuple[int, int], list[tuple[InputImage, np.ndarray]]] = {}
            for item, array in zip(batch_items, arrays, strict=True):
                shape_groups.setdefault(array.shape, []).append((item, array))

            for group in shape_groups.values():
                group_items = [item for item, _ in group]
                group_arrays = [array for _, array in group]
                batch = torch.from_numpy(np.stack(group_arrays, axis=0)).unsqueeze(1).to(torch_device)
                with _autocast_context(torch_device, resolved_precision):
                    predictions = model(batch)
                expected_shape = (
                    batch.shape[0],
                    1,
                    batch.shape[-2] * 2,
                    batch.shape[-1] * 2,
                )
                if tuple(predictions.shape) != expected_shape:
                    raise RuntimeError(
                        f"Model returned shape {tuple(predictions.shape)}; expected {expected_shape}"
                    )
                if not torch.isfinite(predictions).all():
                    raise RuntimeError("Model produced NaN or infinity")

                outputs = predictions.clamp(0.0, 1.0).float().cpu().numpy()[:, 0]
                for item, output in zip(group_items, outputs, strict=True):
                    _atomic_save_npy(
                        output_root / item.relative_path,
                        output.astype(np.float32, copy=False),
                    )

    if torch_device.type == "cuda":
        torch.cuda.synchronize(torch_device)
    elapsed = time.perf_counter() - started
    return InferenceSummary(
        model=model_name,
        device=str(torch_device),
        precision=resolved_precision,
        input_count=len(inputs),
        elapsed_seconds=elapsed,
        mean_milliseconds_per_image=(elapsed * 1000.0) / len(inputs),
        checkpoint=checkpoint,
    )


def write_report(path: str | Path, summary: InferenceSummary) -> None:
    report_path = Path(path).expanduser().resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(summary.to_dict(), indent=2) + "\n", encoding="utf-8")
