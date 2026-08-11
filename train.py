#!/usr/bin/env python
"""Script-first restoration training entry point for local and Colab runs."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import os
import platform
import random
import subprocess
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from semirestore.checkpoints import (  # noqa: E402
    CHECKPOINT_FORMAT_VERSION,
    atomic_torch_save,
    load_checkpoint_payload,
)
from semirestore.data import InputValidationError  # noqa: E402
from semirestore.inference import resolve_device  # noqa: E402
from semirestore.losses import CharbonnierLoss  # noqa: E402
from semirestore.models import create_model  # noqa: E402
from semirestore.training_data import (  # noqa: E402
    PairedNpyDataset,
    read_manifest_pairs,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a configured restoration baseline.")
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "baseline_edsr.yaml",
    )
    parser.add_argument("--manifest", type=Path, help="Override data.manifest")
    parser.add_argument("--dataset-root", type=Path, help="Override data.dataset_root")
    parser.add_argument("--run-dir", type=Path, help="Override output.run_dir")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument(
        "--resume",
        type=Path,
        help="Resume from a training_resume last.pt checkpoint",
    )
    parser.add_argument(
        "--stop-after-step",
        type=int,
        help="Stop at this absolute step while retaining the configured full schedule",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Enable deterministic PyTorch algorithms (intended for debug verification)",
    )
    parser.add_argument(
        "--overfit-samples",
        type=int,
        default=0,
        help="Train and validate on the first N training pairs as a pipeline proof",
    )
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace known run artifacts in an existing directory; nothing is deleted",
    )
    return parser


def _load_config(path: Path) -> dict[str, object]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise InputValidationError(f"Training config does not exist: {resolved}")
    try:
        payload = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise InputValidationError(f"Invalid YAML config {resolved}: {exc}") from exc
    if not isinstance(payload, dict):
        raise InputValidationError("Training config root must be a mapping")
    for section in ("model", "data", "training", "output"):
        if not isinstance(payload.get(section), dict):
            raise InputValidationError(f"Training config requires a '{section}' mapping")
    return payload


def _resolve_project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _apply_overrides(config: dict[str, object], args: argparse.Namespace) -> dict[str, object]:
    data = config["data"]
    training = config["training"]
    output = config["output"]
    assert isinstance(data, dict) and isinstance(training, dict) and isinstance(output, dict)
    if args.manifest is not None:
        data["manifest"] = str(args.manifest)
    if args.dataset_root is not None:
        data["dataset_root"] = str(args.dataset_root)
    if args.run_dir is not None:
        output["run_dir"] = str(args.run_dir)
    for argument, key in (
        (args.device, "device"),
        (args.max_steps, "max_steps"),
        (args.batch_size, "batch_size"),
        (args.num_workers, "num_workers"),
    ):
        if argument is not None:
            training[key] = argument
    if args.no_amp:
        training["amp"] = False
    if args.deterministic:
        training["deterministic"] = True
    return config


def _require_number(mapping: dict[str, object], key: str, kind: type, minimum=None):
    value = mapping.get(key)
    if not isinstance(value, kind) or isinstance(value, bool):
        raise InputValidationError(f"Configuration '{key}' must be {kind.__name__}")
    if minimum is not None and value < minimum:
        raise InputValidationError(f"Configuration '{key}' must be at least {minimum}")
    return value


def _prepare_run_dir(path: Path, *, overwrite: bool) -> Path:
    resolved = path.expanduser().resolve()
    if resolved.exists() and not resolved.is_dir():
        raise InputValidationError(f"Run path is not a directory: {resolved}")
    if resolved.exists() and any(resolved.iterdir()) and not overwrite:
        raise InputValidationError(
            f"Run directory is not empty: {resolved}. Use --overwrite to replace known artifacts."
        )
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def _seed_everything(seed: int, *, deterministic: bool) -> torch.Generator:
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    else:
        torch.use_deterministic_algorithms(False)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def _seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(PROJECT_ROOT), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _environment(device: torch.device, *, deterministic: bool) -> dict[str, object]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "pytorch": str(torch.__version__),
        "cuda_runtime": None if torch.version.cuda is None else str(torch.version.cuda),
        "cudnn": torch.backends.cudnn.version(),
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "deterministic_algorithms": deterministic,
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "git_commit": _git_commit(),
    }


def _create_ema_model(model: torch.nn.Module) -> torch.nn.Module:
    ema_model = copy.deepcopy(model).eval()
    ema_model.requires_grad_(False)
    return ema_model


@torch.no_grad()
def _update_ema(
    ema_model: torch.nn.Module, model: torch.nn.Module, *, decay: float
) -> None:
    ema_parameters = dict(ema_model.named_parameters())
    for name, parameter in model.named_parameters():
        ema_parameters[name].lerp_(parameter.detach(), 1.0 - decay)
    ema_buffers = dict(ema_model.named_buffers())
    for name, buffer in model.named_buffers():
        ema_buffers[name].copy_(buffer.detach())


def _optimizer_to(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device)


def _manifest_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _autocast(device: torch.device, enabled: bool):
    if enabled:
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def _validation_psnr(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    amp: bool,
) -> float:
    model.eval()
    scores: list[torch.Tensor] = []
    with torch.inference_mode():
        for degraded, target, _ in loader:
            degraded = degraded.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            with _autocast(device, amp):
                prediction = model(degraded)
            if prediction.shape != target.shape or not torch.isfinite(prediction).all():
                raise RuntimeError("Validation produced an invalid prediction")
            error = (prediction.float().clamp(0.0, 1.0) - target.float().clamp(0.0, 1.0))
            mse = error.square().flatten(1).mean(1).clamp_min(1e-12)
            scores.append(-10.0 * torch.log10(mse).cpu())
    return float(torch.cat(scores).mean().item())


def _scheduler_factor(step: int, *, max_steps: int, warmup_steps: int) -> float:
    if warmup_steps and step < warmup_steps:
        return (step + 1) / warmup_steps
    remaining = max(1, max_steps - warmup_steps)
    progress = min(1.0, max(0.0, (step - warmup_steps) / remaining))
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields = (
        "step",
        "epoch",
        "train_loss",
        "best_train_loss",
        "raw_val_psnr_db",
        "ema_val_psnr_db",
        "selected_val_psnr_db",
        "selected_weights",
        "lr",
    )
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _checkpoint_payload(
    *,
    model: torch.nn.Module,
    model_name: str,
    model_config: dict[str, object],
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.amp.GradScaler,
    ema_model: torch.nn.Module | None,
    dataloader_generator: torch.Generator,
    step: int,
    epoch: int,
    best_val_psnr: float,
    best_weights_source: str,
    best_model_state_dict: dict[str, torch.Tensor],
    config: dict[str, object],
    manifest_sha256: str,
    environment: dict[str, object],
    overfit_samples: int,
) -> dict[str, object]:
    data_config = config["data"]
    training_config = config["training"]
    assert isinstance(data_config, dict) and isinstance(training_config, dict)
    return {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "checkpoint_role": "training_resume",
        "model_name": model_name,
        "model_config": model_config,
        "model_state_dict": model.state_dict(),
        "ema_state_dict": None if ema_model is None else ema_model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "dataloader_generator_state": dataloader_generator.get_state(),
        "step": step,
        "epoch": epoch,
        "best_val_psnr_db": best_val_psnr,
        "best_weights_source": best_weights_source,
        "best_model_state_dict": best_model_state_dict,
        "planned_max_steps": training_config["max_steps"],
        "deterministic": bool(training_config.get("deterministic", False)),
        "ema": {
            "enabled": ema_model is not None,
            "decay": float(training_config.get("ema_decay", 0.999)),
        },
        "loss": {
            "name": "charbonnier",
            "epsilon": training_config["charbonnier_epsilon"],
        },
        "data": {
            "manifest_sha256": manifest_sha256,
            "train_split": data_config["train_split"],
            "val_split": data_config["val_split"],
            "overfit_samples": overfit_samples,
            "input_policy": "raw_float32_no_clip",
            "output_policy": "unbounded_during_training",
        },
        "environment": environment,
    }


def _validate_resume_payload(
    payload: dict[str, object],
    *,
    model_name: str,
    model_config: dict[str, object],
    manifest_sha256: str,
    max_steps: int,
    overfit_samples: int,
) -> None:
    if payload.get("checkpoint_role") != "training_resume":
        raise InputValidationError("--resume requires a training_resume last.pt checkpoint")
    if payload.get("model_name") != model_name or payload.get("model_config") != model_config:
        raise InputValidationError("Resume checkpoint model does not match the resolved config")
    checkpoint_data = payload.get("data")
    if not isinstance(checkpoint_data, dict):
        raise InputValidationError("Resume checkpoint is missing data provenance")
    if checkpoint_data.get("manifest_sha256") != manifest_sha256:
        raise InputValidationError("Resume checkpoint manifest hash does not match")
    if checkpoint_data.get("overfit_samples") != overfit_samples:
        raise InputValidationError("Resume checkpoint overfit sample count does not match")
    if payload.get("planned_max_steps") != max_steps:
        raise InputValidationError(
            "Resume checkpoint planned_max_steps does not match the resolved config"
        )
    required = (
        "model_state_dict",
        "optimizer_state_dict",
        "scheduler_state_dict",
        "scaler_state_dict",
        "dataloader_generator_state",
        "best_model_state_dict",
        "step",
        "epoch",
    )
    missing = [key for key in required if key not in payload]
    if missing:
        raise InputValidationError(
            "Resume checkpoint predates resumable-engine metadata: " + ", ".join(missing)
        )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config = _apply_overrides(_load_config(args.config), args)
        model_config = config["model"]
        data_config = config["data"]
        training_config = config["training"]
        output_config = config["output"]
        assert all(
            isinstance(section, dict)
            for section in (model_config, data_config, training_config, output_config)
        )
        model_name = model_config.get("name")
        if model_name not in {"edsr_lite", "naf_sr"}:
            raise InputValidationError("Training model.name must be edsr_lite or naf_sr")

        max_steps = _require_number(training_config, "max_steps", int, 1)
        batch_size = _require_number(training_config, "batch_size", int, 1)
        num_workers = _require_number(training_config, "num_workers", int, 0)
        validation_interval = _require_number(
            training_config, "validation_interval", int, 1
        )
        log_interval = _require_number(training_config, "log_interval", int, 1)
        seed = _require_number(training_config, "seed", int, 0)
        learning_rate = float(training_config.get("learning_rate", 2e-4))
        weight_decay = float(training_config.get("weight_decay", 0.0))
        grad_clip = float(training_config.get("gradient_clip_norm", 1.0))
        epsilon = float(training_config.get("charbonnier_epsilon", 1e-3))
        warmup_steps = int(training_config.get("warmup_steps", 0))
        deterministic = bool(training_config.get("deterministic", False))
        ema_enabled = bool(training_config.get("ema_enabled", True))
        ema_decay = float(training_config.get("ema_decay", 0.999))
        if learning_rate <= 0 or weight_decay < 0 or grad_clip <= 0 or warmup_steps < 0:
            raise InputValidationError("Invalid optimizer, clipping, or warmup configuration")
        if not 0.0 <= ema_decay < 1.0:
            raise InputValidationError("training.ema_decay must be in [0, 1)")
        if args.overfit_samples < 0:
            raise InputValidationError("--overfit-samples cannot be negative")
        target_step = max_steps if args.stop_after_step is None else args.stop_after_step
        if target_step < 1 or target_step > max_steps:
            raise InputValidationError("--stop-after-step must be between 1 and max_steps")

        manifest = _resolve_project_path(str(data_config["manifest"]))
        dataset_root = _resolve_project_path(str(data_config["dataset_root"]))
        run_dir = _prepare_run_dir(
            _resolve_project_path(str(output_config["run_dir"])), overwrite=args.overwrite
        )
        device = resolve_device(str(training_config.get("device", "auto")))
        amp = bool(training_config.get("amp", True)) and device.type == "cuda"
        generator = _seed_everything(seed, deterministic=deterministic)
        manifest_digest = _manifest_sha256(manifest)
        construction_config = {
            key: value for key, value in model_config.items() if key != "name"
        }
        resume_payload: dict[str, object] | None = None
        resume_path: Path | None = None
        if args.resume is not None:
            resume_path = args.resume.expanduser().resolve()
            resume_payload = load_checkpoint_payload(resume_path, map_location="cpu")
            _validate_resume_payload(
                resume_payload,
                model_name=str(model_name),
                model_config=construction_config,
                manifest_sha256=manifest_digest,
                max_steps=max_steps,
                overfit_samples=args.overfit_samples,
            )
            generator_state = resume_payload["dataloader_generator_state"]
            if not isinstance(generator_state, torch.Tensor):
                raise InputValidationError("Resume checkpoint has invalid DataLoader RNG state")
            generator.set_state(generator_state)

        train_pairs = read_manifest_pairs(
            manifest, dataset_root, split=str(data_config["train_split"])
        )
        if args.overfit_samples:
            if args.overfit_samples > len(train_pairs):
                raise InputValidationError(
                    f"Requested {args.overfit_samples} overfit samples, only {len(train_pairs)} exist"
                )
            train_pairs = train_pairs[: args.overfit_samples]
            val_pairs = train_pairs
        else:
            val_pairs = read_manifest_pairs(
                manifest, dataset_root, split=str(data_config["val_split"])
            )

        train_loader = DataLoader(
            PairedNpyDataset(train_pairs),
            batch_size=min(batch_size, len(train_pairs)),
            shuffle=True,
            num_workers=num_workers,
            pin_memory=device.type == "cuda",
            persistent_workers=num_workers > 0,
            worker_init_fn=_seed_worker,
            generator=generator,
        )
        val_loader = DataLoader(
            PairedNpyDataset(val_pairs),
            batch_size=min(batch_size, len(val_pairs)),
            shuffle=False,
            num_workers=num_workers,
            pin_memory=device.type == "cuda",
            persistent_workers=num_workers > 0,
            worker_init_fn=_seed_worker,
        )

        model = create_model(str(model_name), construction_config).to(device)
        ema_model = _create_ema_model(model) if ema_enabled else None
        loss_function = CharbonnierLoss(epsilon).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lambda step: _scheduler_factor(
                step, max_steps=max_steps, warmup_steps=warmup_steps
            ),
        )
        scaler = torch.amp.GradScaler(device.type, enabled=amp)
        environment = _environment(device, deterministic=deterministic)

        start_step = 0
        start_epoch = 0
        best_val_psnr = float("-inf")
        best_weights_source = "raw"
        best_model_state_dict: dict[str, torch.Tensor] | None = None
        if resume_payload is not None:
            model.load_state_dict(resume_payload["model_state_dict"], strict=True)
            optimizer.load_state_dict(resume_payload["optimizer_state_dict"])
            _optimizer_to(optimizer, device)
            scheduler.load_state_dict(resume_payload["scheduler_state_dict"])
            scaler.load_state_dict(resume_payload["scaler_state_dict"])
            checkpoint_ema = resume_payload.get("ema_state_dict")
            if ema_model is not None:
                if not isinstance(checkpoint_ema, dict):
                    raise InputValidationError("EMA-enabled resume checkpoint is missing EMA state")
                ema_model.load_state_dict(checkpoint_ema, strict=True)
            elif checkpoint_ema is not None:
                raise InputValidationError("Resume checkpoint EMA setting does not match config")
            start_step = int(resume_payload["step"])
            start_epoch = int(resume_payload["epoch"]) + 1
            best_val_psnr = float(resume_payload.get("best_val_psnr_db", float("-inf")))
            best_weights_source = str(resume_payload.get("best_weights_source", "raw"))
            checkpoint_best_state = resume_payload["best_model_state_dict"]
            if not isinstance(checkpoint_best_state, dict):
                raise InputValidationError("Resume checkpoint has invalid best model state")
            best_model_state_dict = checkpoint_best_state
            if target_step <= start_step:
                raise InputValidationError(
                    "--stop-after-step/max_steps must be greater than the resumed step"
                )
        resolved_record = {
            **config,
            "runtime": {
                "manifest": str(manifest),
                "dataset_root": str(dataset_root),
                "run_dir": str(run_dir),
                "overfit_samples": args.overfit_samples,
                "amp_enabled": amp,
                "deterministic": deterministic,
                "ema_enabled": ema_enabled,
                "ema_decay": ema_decay,
                "resume_from": None if resume_path is None else str(resume_path),
                "start_step": start_step,
                "target_step": target_step,
                "resume_data_policy": "fresh deterministic epoch from saved DataLoader RNG state",
            },
        }
        (run_dir / "resolved_config.yaml").write_text(
            yaml.safe_dump(resolved_record, sort_keys=True), encoding="utf-8"
        )
        (run_dir / "environment.json").write_text(
            json.dumps(environment, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

        parameter_count = sum(parameter.numel() for parameter in model.parameters())
        initial_raw_val_psnr = _validation_psnr(model, val_loader, device, amp=amp)
        initial_ema_val_psnr = (
            _validation_psnr(ema_model, val_loader, device, amp=amp)
            if ema_model is not None
            else None
        )
        print(
            f"Training {model_name} ({parameter_count:,} parameters) on {device}; "
            f"train={len(train_pairs)}, val={len(val_pairs)}, AMP={amp}, "
            f"EMA={ema_enabled}, deterministic={deterministic}."
        )
        print(f"Initial raw validation PSNR: {initial_raw_val_psnr:.4f} dB")
        if resume_path is not None:
            print(f"Resumed step {start_step} from {resume_path}")
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        history: list[dict[str, object]] = []
        best_train_loss = float("inf")
        initial_train_loss: float | None = None
        final_train_loss = float("nan")
        step = start_step
        epoch = start_epoch
        iterator = iter(train_loader)
        started = time.perf_counter()

        while step < target_step:
            try:
                degraded, target, _ = next(iterator)
            except StopIteration:
                epoch += 1
                iterator = iter(train_loader)
                degraded, target, _ = next(iterator)
            degraded = degraded.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            model.train()
            optimizer.zero_grad(set_to_none=True)
            with _autocast(device, amp):
                prediction = model(degraded)
                loss = loss_function(prediction, target)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite training loss at step {step + 1}")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            if ema_model is not None:
                _update_ema(ema_model, model, decay=ema_decay)

            step += 1
            final_train_loss = float(loss.detach().item())
            if initial_train_loss is None:
                initial_train_loss = final_train_loss
            best_train_loss = min(best_train_loss, final_train_loss)
            should_validate = step % validation_interval == 0 or step == target_step
            raw_val_psnr: float | None = None
            ema_val_psnr: float | None = None
            selected_val_psnr: float | None = None
            selected_weights = ""
            if should_validate:
                raw_val_psnr = _validation_psnr(model, val_loader, device, amp=amp)
                if ema_model is not None:
                    ema_val_psnr = _validation_psnr(
                        ema_model, val_loader, device, amp=amp
                    )
                candidates = [(raw_val_psnr, "raw")]
                if ema_val_psnr is not None:
                    candidates.append((ema_val_psnr, "ema"))
                selected_val_psnr, selected_weights = max(candidates, key=lambda item: item[0])
                is_new_best = selected_val_psnr > best_val_psnr
                if is_new_best:
                    best_val_psnr = selected_val_psnr
                    best_weights_source = selected_weights
                    selected_model = ema_model if selected_weights == "ema" else model
                    assert selected_model is not None
                    best_model_state_dict = {
                        key: value.detach().cpu().clone()
                        for key, value in selected_model.state_dict().items()
                    }
                assert best_model_state_dict is not None
                payload = _checkpoint_payload(
                    model=model,
                    model_name=str(model_name),
                    model_config=construction_config,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    ema_model=ema_model,
                    dataloader_generator=generator,
                    step=step,
                    epoch=epoch,
                    best_val_psnr=best_val_psnr,
                    best_weights_source=best_weights_source,
                    best_model_state_dict=best_model_state_dict,
                    config=config,
                    manifest_sha256=manifest_digest,
                    environment=environment,
                    overfit_samples=args.overfit_samples,
                )
                atomic_torch_save(payload, run_dir / "last.pt")
                best_payload = {
                    key: value
                    for key, value in payload.items()
                    if key
                    not in {
                        "optimizer_state_dict",
                        "scheduler_state_dict",
                        "scaler_state_dict",
                        "ema_state_dict",
                        "dataloader_generator_state",
                        "best_model_state_dict",
                    }
                }
                best_payload["model_state_dict"] = best_model_state_dict
                best_payload["checkpoint_role"] = "best_inference"
                best_payload["selected_weights"] = best_weights_source
                atomic_torch_save(best_payload, run_dir / "best.pt")

            history.append(
                {
                    "step": step,
                    "epoch": epoch,
                    "train_loss": final_train_loss,
                    "best_train_loss": best_train_loss,
                    "raw_val_psnr_db": "" if raw_val_psnr is None else raw_val_psnr,
                    "ema_val_psnr_db": "" if ema_val_psnr is None else ema_val_psnr,
                    "selected_val_psnr_db": (
                        "" if selected_val_psnr is None else selected_val_psnr
                    ),
                    "selected_weights": selected_weights,
                    "lr": optimizer.param_groups[0]["lr"],
                }
            )
            if step % log_interval == 0 or should_validate:
                validation_text = (
                    ""
                    if selected_val_psnr is None
                    else (
                        f", raw_psnr={raw_val_psnr:.4f}, "
                        f"ema_psnr={ema_val_psnr:.4f}, selected={selected_weights}"
                        if ema_val_psnr is not None
                        else f", raw_psnr={raw_val_psnr:.4f}, selected=raw"
                    )
                )
                print(
                    f"step={step}/{target_step} loss={final_train_loss:.6f} "
                    f"best_loss={best_train_loss:.6f}{validation_text}"
                )
            if step % log_interval == 0 or should_validate:
                _write_csv(run_dir / "history.csv", history)

        if device.type == "cuda":
            torch.cuda.synchronize(device)
            peak_cuda_memory_bytes: int | None = torch.cuda.max_memory_allocated(device)
        else:
            peak_cuda_memory_bytes = None
        elapsed = time.perf_counter() - started
        assert initial_train_loss is not None
        segment_steps = step - start_step
        summary = {
            "schema_version": 1,
            "model": model_name,
            "parameter_count": parameter_count,
            "step": step,
            "start_step": start_step,
            "target_step": target_step,
            "epoch": epoch,
            "train_pair_count": len(train_pairs),
            "val_pair_count": len(val_pairs),
            "overfit_samples": args.overfit_samples,
            "initial_train_loss": initial_train_loss,
            "final_train_loss": final_train_loss,
            "best_train_loss": best_train_loss,
            "loss_reduction_ratio": best_train_loss / initial_train_loss,
            "initial_raw_val_psnr_db": initial_raw_val_psnr,
            "initial_ema_val_psnr_db": initial_ema_val_psnr,
            "initial_val_psnr_db": initial_raw_val_psnr,
            "best_val_psnr_db": best_val_psnr,
            "best_weights_source": best_weights_source,
            "ema_enabled": ema_enabled,
            "ema_decay": ema_decay,
            "deterministic": deterministic,
            "resumed_from": None if resume_path is None else str(resume_path),
            "resume_data_policy": "fresh deterministic epoch from saved DataLoader RNG state",
            "elapsed_seconds": elapsed,
            "mean_step_milliseconds": elapsed * 1000.0 / segment_steps,
            "peak_cuda_memory_bytes": peak_cuda_memory_bytes,
            "manifest_sha256": manifest_digest,
            "environment": environment,
            "best_checkpoint": "best.pt",
            "last_checkpoint": "last.pt",
        }
        (run_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    except (InputValidationError, KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(
        f"Training complete: best loss={best_train_loss:.6f}, "
        f"best validation PSNR={best_val_psnr:.4f} dB, {elapsed:.1f}s."
    )
    print(f"Run artifacts: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
