"""Self-describing PyTorch checkpoint helpers."""

from __future__ import annotations

import os
from pathlib import Path

import torch
from torch import nn

from .data import InputValidationError
from .models import create_model

CHECKPOINT_FORMAT_VERSION = 1


def atomic_torch_save(payload: dict[str, object], path: str | Path) -> Path:
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    try:
        torch.save(payload, temporary)
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return destination


def load_checkpoint_payload(path: str | Path, *, map_location: str | torch.device = "cpu"):
    checkpoint_path = Path(path).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise InputValidationError(f"Checkpoint does not exist: {checkpoint_path}")
    try:
        payload = torch.load(checkpoint_path, map_location=map_location, weights_only=True)
    except Exception as exc:
        raise InputValidationError(f"Could not load checkpoint {checkpoint_path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise InputValidationError(f"Checkpoint payload must be a dictionary: {checkpoint_path}")
    if payload.get("format_version") != CHECKPOINT_FORMAT_VERSION:
        raise InputValidationError(
            f"Unsupported checkpoint format version: {payload.get('format_version')}"
        )
    return payload


def load_model_checkpoint(
    path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> tuple[nn.Module, dict[str, object]]:
    payload = load_checkpoint_payload(path, map_location=map_location)
    model_name = payload.get("model_name")
    model_config = payload.get("model_config")
    state_dict = payload.get("model_state_dict")
    if not isinstance(model_name, str):
        raise InputValidationError("Checkpoint is missing a string model_name")
    if not isinstance(model_config, dict):
        raise InputValidationError("Checkpoint is missing model_config")
    if not isinstance(state_dict, dict):
        raise InputValidationError("Checkpoint is missing model_state_dict")
    try:
        model = create_model(model_name, model_config)
        model.load_state_dict(state_dict, strict=True)
    except (RuntimeError, TypeError, ValueError) as exc:
        raise InputValidationError(f"Checkpoint model metadata/state is invalid: {exc}") from exc
    return model, payload
