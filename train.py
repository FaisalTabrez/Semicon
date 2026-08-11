#!/usr/bin/env python
"""Script-first EDSR-lite training entry point for local and Colab runs."""

from __future__ import annotations

import argparse
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
)
from semirestore.data import InputValidationError  # noqa: E402
from semirestore.inference import resolve_device  # noqa: E402
from semirestore.losses import CharbonnierLoss  # noqa: E402
from semirestore.models import EDSRLite  # noqa: E402
from semirestore.training_data import (  # noqa: E402
    PairedNpyDataset,
    read_manifest_pairs,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the EDSR-lite restoration baseline.")
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


def _seed_everything(seed: int) -> torch.Generator:
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


def _environment(device: torch.device) -> dict[str, object]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "pytorch": str(torch.__version__),
        "cuda_runtime": None if torch.version.cuda is None else str(torch.version.cuda),
        "cudnn": torch.backends.cudnn.version(),
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "git_commit": _git_commit(),
    }


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
    fields = ("step", "epoch", "train_loss", "best_train_loss", "val_psnr_db", "lr")
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
    model: EDSRLite,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    step: int,
    epoch: int,
    best_val_psnr: float,
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
        "model_name": "edsr_lite",
        "model_config": model.model_config(),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "step": step,
        "epoch": epoch,
        "best_val_psnr_db": best_val_psnr,
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
        if model_config.get("name") != "edsr_lite":
            raise InputValidationError("This training entry point currently requires edsr_lite")

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
        if learning_rate <= 0 or weight_decay < 0 or grad_clip <= 0 or warmup_steps < 0:
            raise InputValidationError("Invalid optimizer, clipping, or warmup configuration")
        if args.overfit_samples < 0:
            raise InputValidationError("--overfit-samples cannot be negative")

        manifest = _resolve_project_path(str(data_config["manifest"]))
        dataset_root = _resolve_project_path(str(data_config["dataset_root"]))
        run_dir = _prepare_run_dir(
            _resolve_project_path(str(output_config["run_dir"])), overwrite=args.overwrite
        )
        device = resolve_device(str(training_config.get("device", "auto")))
        amp = bool(training_config.get("amp", True)) and device.type == "cuda"
        generator = _seed_everything(seed)

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

        model = EDSRLite(
            width=int(model_config.get("width", 64)),
            num_blocks=int(model_config.get("num_blocks", 16)),
            residual_scale=float(model_config.get("residual_scale", 0.1)),
        ).to(device)
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
        environment = _environment(device)
        manifest_digest = _manifest_sha256(manifest)
        resolved_record = {
            **config,
            "runtime": {
                "manifest": str(manifest),
                "dataset_root": str(dataset_root),
                "run_dir": str(run_dir),
                "overfit_samples": args.overfit_samples,
                "amp_enabled": amp,
            },
        }
        (run_dir / "resolved_config.yaml").write_text(
            yaml.safe_dump(resolved_record, sort_keys=True), encoding="utf-8"
        )
        (run_dir / "environment.json").write_text(
            json.dumps(environment, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

        parameter_count = sum(parameter.numel() for parameter in model.parameters())
        initial_val_psnr = _validation_psnr(model, val_loader, device, amp=amp)
        print(
            f"Training EDSR-lite ({parameter_count:,} parameters) on {device}; "
            f"train={len(train_pairs)}, val={len(val_pairs)}, AMP={amp}."
        )
        print(f"Initial validation PSNR: {initial_val_psnr:.4f} dB")

        history: list[dict[str, object]] = []
        best_val_psnr = float("-inf")
        best_train_loss = float("inf")
        initial_train_loss: float | None = None
        final_train_loss = float("nan")
        step = 0
        epoch = 0
        iterator = iter(train_loader)
        started = time.perf_counter()

        while step < max_steps:
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

            step += 1
            final_train_loss = float(loss.detach().item())
            if initial_train_loss is None:
                initial_train_loss = final_train_loss
            best_train_loss = min(best_train_loss, final_train_loss)
            should_validate = step % validation_interval == 0 or step == max_steps
            val_psnr: float | None = None
            if should_validate:
                val_psnr = _validation_psnr(model, val_loader, device, amp=amp)
                payload = _checkpoint_payload(
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    step=step,
                    epoch=epoch,
                    best_val_psnr=max(best_val_psnr, val_psnr),
                    config=config,
                    manifest_sha256=manifest_digest,
                    environment=environment,
                    overfit_samples=args.overfit_samples,
                )
                atomic_torch_save(payload, run_dir / "last.pt")
                if val_psnr > best_val_psnr:
                    best_val_psnr = val_psnr
                    atomic_torch_save(payload, run_dir / "best.pt")

            history.append(
                {
                    "step": step,
                    "epoch": epoch,
                    "train_loss": final_train_loss,
                    "best_train_loss": best_train_loss,
                    "val_psnr_db": "" if val_psnr is None else val_psnr,
                    "lr": optimizer.param_groups[0]["lr"],
                }
            )
            if step % log_interval == 0 or should_validate:
                validation_text = "" if val_psnr is None else f", val_psnr={val_psnr:.4f}"
                print(
                    f"step={step}/{max_steps} loss={final_train_loss:.6f} "
                    f"best_loss={best_train_loss:.6f}{validation_text}"
                )
            if step % log_interval == 0 or should_validate:
                _write_csv(run_dir / "history.csv", history)

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - started
        assert initial_train_loss is not None
        summary = {
            "schema_version": 1,
            "model": "edsr_lite",
            "parameter_count": parameter_count,
            "step": step,
            "epoch": epoch,
            "train_pair_count": len(train_pairs),
            "val_pair_count": len(val_pairs),
            "overfit_samples": args.overfit_samples,
            "initial_train_loss": initial_train_loss,
            "final_train_loss": final_train_loss,
            "best_train_loss": best_train_loss,
            "loss_reduction_ratio": best_train_loss / initial_train_loss,
            "initial_val_psnr_db": initial_val_psnr,
            "best_val_psnr_db": best_val_psnr,
            "elapsed_seconds": elapsed,
            "mean_step_milliseconds": elapsed * 1000.0 / step,
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
