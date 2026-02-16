"""Entry point for running experiments.

Define experiment configs here and run them.

Usage:
    python -m experiments.run                          # runs baseline
    python -m experiments.run baseline
    python -m experiments.run group-lasso-channels-delay50
    python -m experiments.run --list                   # show available experiments
"""
import argparse

from src.resnet import resnet20
from experiments.config import ExperimentConfig
from experiments.sparsity import (
    group_lasso_channels,
    group_lasso_blocks,
    l1_kernel_sparsity,
    coupled_group_lasso,
)
from experiments.schedules import linear_warmup, cosine_anneal, constant  # noqa: F401
from experiments.trainer import run_experiment


# --- Example experiment configs ---

# Build model once to create coupled loss (needs architecture info)
_model_for_groups = resnet20()

EXPERIMENTS: dict[str, ExperimentConfig] = {}


def _register(cfg: ExperimentConfig) -> ExperimentConfig:
    EXPERIMENTS[cfg.name] = cfg
    return cfg


baseline = _register(ExperimentConfig(
    name="baseline",
    quantize=False,
))

codeq_paper = _register(ExperimentConfig(
    name="codeq-paper",
))

group_lasso_delay50 = _register(ExperimentConfig(
    name="group-lasso-channels-delay50",
    sparsity_fn=group_lasso_channels,
    sparsity_schedule=linear_warmup(delay=50, ramp=50),
))

block_lasso_cosine = _register(ExperimentConfig(
    name="block-lasso-bs4-cosine",
    sparsity_fn=group_lasso_blocks(block_size=4),
    sparsity_schedule=cosine_anneal(delay=30),
))

l1_kernel = _register(ExperimentConfig(
    name="l1-kernel-delay50",
    sparsity_fn=l1_kernel_sparsity,
    sparsity_schedule=linear_warmup(delay=50, ramp=50),
))

coupled = _register(ExperimentConfig(
    name="coupled-group-lasso",
    sparsity_fn=coupled_group_lasso(_model_for_groups),
    sparsity_schedule=constant(1.0),
))

coupled_delay50 = _register(ExperimentConfig(
    name="coupled-group-lasso-delay50",
    sparsity_fn=coupled_group_lasso(_model_for_groups),
    sparsity_schedule=linear_warmup(delay=50, ramp=50),
))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run CoDeQ experiments")
    parser.add_argument(
        "experiment",
        nargs="?",
        default="baseline",
        help="Experiment name (default: baseline)",
    )
    parser.add_argument(
        "--list", action="store_true",
        help="List available experiments and exit",
    )
    args = parser.parse_args()

    if args.list:
        print("Available experiments:")
        for name in EXPERIMENTS:
            print(f"  {name}")
        raise SystemExit(0)

    if args.experiment not in EXPERIMENTS:
        print(f"Unknown experiment: {args.experiment}")
        print(f"Available: {', '.join(EXPERIMENTS)}")
        raise SystemExit(1)

    run_experiment(EXPERIMENTS[args.experiment])
