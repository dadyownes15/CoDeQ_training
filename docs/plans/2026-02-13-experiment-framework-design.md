# Experiment Framework Design

## Goal

A modular experiment framework for pruning research with CoDeQ. Run ResNet-20 on CIFAR-10 with pluggable structured sparsity loss terms, configurable schedules (delay, warmup, ramp), and W&B logging. Evaluate both structured and unstructured BOPs compression.

## File Structure

```
experiments/
├── __init__.py
├── config.py          # ExperimentConfig dataclass
├── sparsity.py        # Structured sparsity loss functions
├── schedules.py       # Schedule factories → Callable[[int,int], float]
├── evaluator.py       # Evaluator: structured + unstructured BOPs, visualization
├── trainer.py         # Training loop + W&B logging
└── run.py             # Entry point
```

## Config (config.py)

```python
@dataclass
class ExperimentConfig:
    name: str

    # Model
    model: str = "resnet20"
    num_classes: int = 10

    # Dataset
    dataset: str = "cifar10"
    batch_size: int = 512
    num_workers: int = 2

    # Training (CoDeQ paper defaults for ResNet-20)
    epochs: int = 300
    lr: float = 1e-3
    weight_decay: float = 0.0
    lr_scheduler: str = "cosine"

    # Quantizer
    quantizer_kwargs: dict  # default: max_bits=8, init logits=3.0, learnable=True
    exclude_layers: list[str]  # default: ['conv1', 'bn', 'linear']
    lr_dz: float = 1e-3
    lr_bit: float = 1e-3
    weight_decay_dz: float = 0.01
    weight_decay_bit: float = 0.01

    # Structured sparsity
    sparsity_fn: Callable | None = None       # fn(model) → scalar loss
    sparsity_schedule: Callable = constant(1)  # fn(epoch, total) → coefficient

    device: str = "cuda"
```

## Sparsity Loss Functions (sparsity.py)

Signature: `fn(model: nn.Module) → torch.Tensor` (scalar)

Built-in:
- `group_lasso_channels(model)` — L1 of L2 filter norms (channel pruning)
- `group_lasso_blocks(block_size)` — factory returning fn; L2 norm of blocks of filters
- `l1_kernel_sparsity(model)` — L1 on individual kernel norms

Layer-specific weighting via closures.

## Schedules (schedules.py)

Signature: `factory(...) → Callable[[int, int], float]`

Built-in:
- `constant(value)` — always returns value
- `linear_warmup(delay, ramp)` — 0 for delay epochs, linear ramp over ramp epochs to 1.0
- `cosine_anneal(delay)` — 0 for delay epochs, then cosine 0→1

## Evaluator (evaluator.py)

```python
@dataclass
class EvalResult:
    structured_stats: ModelStats
    structured_bops_ratio: float
    structured_mac_ratio: float

    unstructured_stats: ModelStats
    unstructured_bops_ratio: float
    unstructured_mac_ratio: float

    test_accuracy: float
    test_loss: float
```

- Structured BOPs: existing null-channel tracking (metrics.py `_compute_effective_macs`)
- Unstructured BOPs: `nnz(w_valid) * m_h * m_w` per layer (new function in metrics.py)
- Both: `BOPs = MACs * b_w * b_a`, ratio vs dense baseline

Visualization (logged as wandb.Image):
- Channel liveness heatmap (rows=layers, cols=channels, black=dead)
- Sparsity map per layer (kernel-level zero/nonzero grid)
- Per-layer sparsity bar chart (structured vs unstructured)

## Trainer (trainer.py)

- AdamW optimizer with 3 param groups (base, dz, bit)
- Cosine annealing LR scheduler
- Loss: `ce_loss + schedule_coeff * sparsity_fn(model)`
- Evaluator runs every epoch
- W&B logs: train loss components, eval accuracy/loss, structured/unstructured BOPs ratios, per-layer bitwidths and pruning ratios, visualization images at intervals

## Training Settings (from CoDeQ paper)

| Setting | ResNet-20 CIFAR-10 |
|---------|-------------------|
| Optimizer | AdamW |
| Epochs | 300 |
| Batch size | 512 |
| LR (weights) | 1e-3 |
| LR (θ_dz, θ_bit) | 1e-3 |
| λ_dz (weight decay) | 0.01 |
| λ_bit (weight decay) | 0.01 |
| LR schedule | Cosine annealing |
| Init θ_dz, θ_bit | 3.0 |
| Bit range | 2–8 |
| Activations | 32-bit |
| Data augmentation | Random crop 32 pad 4, horizontal flip, CIFAR-10 normalize |

## Usage

```python
from experiments.config import ExperimentConfig
from experiments.sparsity import group_lasso_channels
from experiments.schedules import linear_warmup
from experiments.trainer import run_experiment

config = ExperimentConfig(
    name="group-lasso-delay50",
    sparsity_fn=group_lasso_channels,
    sparsity_schedule=linear_warmup(delay=50, ramp=50),
)
run_experiment(config)
```
