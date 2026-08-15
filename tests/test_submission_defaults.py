from __future__ import annotations

from pathlib import Path

import yaml

import evaluation
from scripts.generate_test_outputs import build_parser as build_generation_parser
from train import _load_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_evaluation_uses_calibrated_directory_batch_default() -> None:
    args = evaluation.build_parser().parse_args(["input", "output"])
    assert args.batch_size == 16
    assert args.model == "naf_sr"
    assert args.device == "auto"


def test_output_generator_matches_submission_batch_default() -> None:
    args = build_generation_parser().parse_args(["input", "output"])
    assert args.batch_size == 16
    assert args.expected_count == 400


def test_final_conditioned_config_is_complete_and_frozen() -> None:
    path = PROJECT_ROOT / "configs" / "final_conditioned.yaml"
    config = _load_config(path)
    assert config["model"]["statistics_conditioning"] is True
    assert config["training"]["synthetic_probability"] == 0.0
    assert config["training"]["max_steps"] == 5000
    assert config["training"]["batch_size"] == 16
    assert config["output"]["run_dir"] == "runs/final_conditioned"

    # Guard against accidental YAML values that cannot round-trip.
    assert yaml.safe_load(path.read_text(encoding="utf-8")) == config
