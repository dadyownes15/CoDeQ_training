"""Experiment configuration dataclass."""
from dataclasses import dataclass, field
from typing import Callable

import torch.nn as nn


@dataclass
class ExperimentConfig:
    """Configuration for a single pruning experiment.

    Training defaults match the CoDeQ paper (Wenshoj et al. 2025)
    settings for ResNet-20 on CIFAR-10.
    """
    # --- Experiment identity ---
    name: str

    # --- Model ---
    model: str = "resnet20"
    num_classes: int = 10

    # --- Dataset ---
    dataset: str = "cifar10"
    batch_size: int = 512
    num_workers: int = 2

    # --- Training ---
    epochs: int = 300
    lr: float = 1e-3
    weight_decay: float = 0.0
    lr_scheduler: str = "cosine"

    # --- Quantizer (DeadZone params) ---
    quantizer_kwargs: dict = field(default_factory=lambda: {
        'max_bits': 8,
        'init_bit_logit': 3.0,
        'init_deadzone_logit': 3.0,
        'learnable_bit': True,
        'learnable_deadzone': True,
    })
    exclude_layers: list[str] = field(default_factory=lambda: ['conv1', 'bn', 'linear'])
    lr_dz: float = 1e-3
    lr_bit: float = 1e-3
    weight_decay_dz: float = 0.01
    weight_decay_bit: float = 0.01

    # --- Structured sparsity ---
    sparsity_fn: Callable[[nn.Module], 'torch.Tensor'] | None = None
    sparsity_schedule: Callable[[int, int], float] = field(
        default_factory=lambda: lambda epoch, total: 1.0
    )

    # --- Device ---
    device: str = "cuda"

    # --- W&B ---
    wandb_project: str = "codeq"

    # --- Visualization ---
    viz_interval: int = 25  # log visualization images every N epochs
