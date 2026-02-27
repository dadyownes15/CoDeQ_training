# Training Guide

This document describes how to use `run_training.py` with YAML config files for CoDeQ experiments.

## Quick Start

```bash
# Baseline ResNet-20 (no quantization)
python run_training.py --config configs/baseline_resnet20.yaml

# DeadZone QAT on ResNet-20 with group lasso
python run_training.py --config configs/deadzone_resnet20.yaml

# DeadZone QAT on MLP
python run_training.py --config configs/deadzone_mlp.yaml

# Override device for cluster
python run_training.py --config configs/deadzone_resnet20.yaml --device cuda

# Resume from checkpoint
python run_training.py --config configs/deadzone_resnet20.yaml --resume save_temp/checkpoint_run.th
```

## CLI Arguments

| Argument | Required | Description |
|----------|----------|-------------|
| `--config` | Yes | Path to YAML config file |
| `--device` | No | Override the `device` field in the YAML (useful for running same config locally vs cluster) |
| `--resume` | No | Path to checkpoint to resume training from |

## YAML Config Reference

### General

```yaml
model: resnet20          # resnet20, vit, or mlp
epochs: 300
batch_size: 128
img_size: 32             # 32 for CIFAR-10, 224 for ViT
device: mps              # cuda, mps, or cpu
wandb_project: "CoDeQ experiments"
workers: 4
use_compile: 0           # 1 to enable torch.compile
save_dir: save_temp
lr_scheduler: cosine     # cosine annealing with T_max=epochs
```

### Quantizer

Omit the `quantizer` block entirely for baseline (no QAT) training.

```yaml
quantizer:
  name: deadzone         # registry lookup (see Quantizer Registry below)
  kwargs:                # passed directly to quantizer constructor
    fixed_bit_val: 8
    max_bits: 8
    init_deadzone_logit: 3.0
    init_bit_logit: 3.0
    learnable_bit: true
    learnable_deadzone: true
  exclude_layers: [bn]   # skip layers whose name contains these strings
```

**Quantizer Registry:**

| Name | Class | Description |
|------|-------|-------------|
| `deadzone` | `DeadZoneLDZCompander` | Dead-zone quantizer with learnable bitwidth and dead-zone parameters (CoDeQ paper) |
| `uniform` | `UniformSymmetric` | Fixed-bitwidth uniform symmetric quantizer |

**DeadZone kwargs:**

| Kwarg | Default | Description |
|-------|---------|-------------|
| `fixed_bit_val` | 4 | Fixed bitwidth when `learnable_bit=false` |
| `max_bits` | 8 | Maximum bitwidth (range is 2 to max_bits) |
| `init_deadzone_logit` | 3.0 | Initial logit for dead-zone parameter |
| `init_bit_logit` | 3.0 | Initial logit for bitwidth parameter |
| `learnable_bit` | true | Whether bitwidth is learnable |
| `learnable_deadzone` | true | Whether dead-zone width is learnable |

**Uniform kwargs:**

| Kwarg | Default | Description |
|-------|---------|-------------|
| `bitwidth` | 8 | Fixed quantization bitwidth |

**Exclude layers by model:**
- ResNet-20: `[bn]` (skip batch norm)
- ViT: `[norm]` (skip layer norm)
- MLP: `[]` (quantize all layers)

### Optimizer

The optimizer supports a 3-way parameter group split for DeadZone QAT. When using `deadzone` quantizer, the model parameters are automatically split into:
- **base** — all standard model weights
- **dz** — dead-zone logit parameters (`logit_dz`), typically high weight decay to encourage sparsity
- **bit** — bitwidth logit parameters (`logit_bit`)

```yaml
optimizer:
  type: adamw            # adamw, adam, or sgd
  param_groups:
    base:
      lr: 0.001
      weight_decay: 0.0
    dz:                  # optional, only used when deadzone quantizer creates logit_dz params
      lr: 0.001
      weight_decay: 2.5
    bit:                 # optional, only used when deadzone quantizer creates logit_bit params
      lr: 0.001
      weight_decay: 0.0
```

For baseline (no QAT), only the `base` group is needed:

```yaml
optimizer:
  type: sgd
  param_groups:
    base:
      lr: 0.1
      momentum: 0.9
      weight_decay: 0.0001
```

### Loss Terms

The base loss is always `CrossEntropyLoss` (hardcoded in `train()`). The `loss_terms` list adds **additional regularization terms** on top:

```
total_loss = CE_loss + lambda_1 * loss_fn_1(model) + lambda_2 * loss_fn_2(model) + ...
```

Omit `loss_terms` entirely for pure CE training.

```yaml
loss_terms:
  - name: group_lasso   # registry lookup (see Loss Registry below)
    lambda: 1.0         # scalar multiplier for this term
    kwargs:              # passed to the loss function
      lambda_linear: 0.1
      lambda_conv: 0.1
```

**Loss Registry:**

| Name | Function | Description |
|------|----------|-------------|
| `group_lasso` | `group_lasso(model, lambda_linear, lambda_conv)` | Group L2/L1 structured sparsity. Computes L2 norm per column, then sums. Pushes entire input features/channels to zero. Works on both `nn.Linear` and `nn.Conv2d`. |

**Group lasso kwargs:**

| Kwarg | Default | Description |
|-------|---------|-------------|
| `lambda_linear` | 1e-4 | Regularization strength for `nn.Linear` layers |
| `lambda_conv` | 1e-4 | Regularization strength for `nn.Conv2d` layers |

Set either to `0.0` to disable for that layer type. For ViT, you'd typically set `lambda_conv: 0.0` since only the patch embedding is Conv2d.

## Models

| Name | Architecture | Input | Description |
|------|-------------|-------|-------------|
| `resnet20` | ResNet-20 | 32x32 | CIFAR-10 ResNet (option A shortcuts) |
| `vit` | ViT-Tiny-Patch16 | 224x224 | Vision Transformer from timm |
| `mlp` | 3072→120→84→10 | 32x32 | Simple 3-layer MLP with ReLU |

## Slurm

```bash
python ../run_training.py --config ../configs/deadzone_resnet20.yaml --device cuda
```

See `slurm/single_slurm.sh` for a full example.

## Extending

**Add a new quantizer:** Define the `nn.Module` class (must accept kwargs in constructor, implement `forward(w) -> w_hat`), then add it to `QUANTIZER_REGISTRY` in `run_training.py`.

**Add a new loss term:** Define `fn(model, **kwargs) -> scalar tensor` in `src/structured_loss.py`, then add it to `LOSS_REGISTRY`.

**Add a new model:** Add an `elif` branch in the model dispatch section of `main()` in `run_training.py`.

## Example Configs

See `configs/` for ready-to-use examples:
- `baseline_resnet20.yaml` — SGD, no QAT
- `deadzone_resnet20.yaml` — AdamW, DeadZone QAT, group lasso
- `deadzone_vit.yaml` — AdamW, DeadZone QAT for ViT
- `deadzone_mlp.yaml` — AdamW, DeadZone QAT for MLP
