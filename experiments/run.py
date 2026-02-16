"""Entry point for running experiments.

Define experiment configs here and run them.
"""
from resnet import resnet20
from experiments.config import ExperimentConfig
from experiments.sparsity import (
    group_lasso_channels,
    group_lasso_blocks,
    l1_kernel_sparsity,
    coupled_group_lasso,
)
from experiments.schedules import linear_warmup, cosine_anneal, constant
from experiments.trainer import run_experiment


# --- Example experiment configs ---

# Build model once to create coupled loss (needs architecture info)
_model_for_groups = resnet20()

baseline = ExperimentConfig(
    name="baseline-no-sparsity",
)

# Independent per-layer losses (ignore skip-connection entanglement)
group_lasso_delay50 = ExperimentConfig(
    name="group-lasso-channels-delay50",
    sparsity_fn=group_lasso_channels,
    sparsity_schedule=linear_warmup(delay=50, ramp=50),
)

block_lasso_cosine = ExperimentConfig(
    name="block-lasso-bs4-cosine",
    sparsity_fn=group_lasso_blocks(block_size=4),
    sparsity_schedule=cosine_anneal(delay=30),
)

l1_kernel = ExperimentConfig(
    name="l1-kernel-delay50",
    sparsity_fn=l1_kernel_sparsity,
    sparsity_schedule=linear_warmup(delay=50, ramp=50),
)

# Coupled loss (skip-connection aware — sparsity translates to structured savings)
coupled_delay50 = ExperimentConfig(
    name="coupled-group-lasso-delay50",
    sparsity_fn=coupled_group_lasso(_model_for_groups),
    sparsity_schedule=linear_warmup(delay=50, ramp=50),
)


if __name__ == "__main__":
    # Run a single experiment
    run_experiment(baseline)
