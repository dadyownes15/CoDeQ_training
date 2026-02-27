# YAML-Driven QAT Configuration Design

## Goal

Replace argparse flags in `run_training.py` with a YAML config file system for configuring:
- Quantizer type and kwargs (registry-based, extensible)
- 3-way optimizer param groups (base/dz/bit with separate LR and weight decay)
- Structured loss terms (registry-based, in separate `src/structured_loss.py`)
- Slurm-compatible CLI with `--config` and optional `--device` override

## File Changes

```
configs/                         # NEW: experiment configs
├── baseline_resnet20.yaml       # No QAT baseline
├── deadzone_resnet20.yaml       # DeadZone QAT with group lasso
└── deadzone_vit.yaml            # DeadZone QAT for ViT

src/structured_loss.py           # NEW: loss functions + registry

run_training.py                  # MODIFIED: YAML loading, quantizer registry, loss injection
src/train.py                     # MODIFIED: accept optional loss_fns in train()
slurm/single_slurm.sh           # MODIFIED: use --config
```

## YAML Config Format

```yaml
# configs/deadzone_resnet20.yaml
model: resnet20
epochs: 300
batch_size: 128
img_size: 32
device: cuda
wandb_project: "CoDeQ experiments"
workers: 6
use_compile: 0

quantizer:
  name: deadzone                 # registry lookup
  kwargs:
    fixed_bit_val: 8
    max_bits: 8
    init_deadzone_logit: 3.0
    init_bit_logit: 3.0
    learnable_bit: true
    learnable_deadzone: true
  exclude_layers: [bn]

optimizer:
  type: adamw                    # adamw or sgd
  param_groups:
    base:
      lr: 0.001
      weight_decay: 0.0
    dz:
      lr: 0.001
      weight_decay: 2.5
    bit:
      lr: 0.001
      weight_decay: 0.0

loss_terms:
  - name: group_lasso            # registry lookup in src/structured_loss.py
    lambda: 0.1
    kwargs:
      lambda_linear: 0.1
      lambda_conv: 0.1

lr_scheduler: cosine             # cosine annealing with T_max=epochs
```

When `quantizer` key is absent or null, training runs without QAT (baseline).

## Quantizer Registry (`run_training.py`)

```python
from src.quantizer import DeadZoneLDZCompander
from src.utils_quantization import UniformSymmetric

QUANTIZER_REGISTRY = {
    "uniform": UniformSymmetric,
    "deadzone": DeadZoneLDZCompander,
}
```

## Structured Loss (`src/structured_loss.py`)

Contains loss functions with signature `fn(model, **kwargs) -> Tensor` and a registry dict.

Built-in:
- `group_lasso(model, lambda_linear, lambda_conv)` — L2-norm per column, L1 sum. Affects both `nn.Linear` and `nn.Conv2d`.

Registry:
```python
LOSS_REGISTRY = {
    "group_lasso": group_lasso,
}
```

## Changes to `src/train.py`

The `train()` function gets an optional `loss_fns` parameter:

```python
def train(train_loader, model, criterion, optimizer, epoch, device, loss_fns=None):
    ...
    loss = criterion(output, target)
    if loss_fns:
        for fn, weight, kwargs in loss_fns:
            loss = loss + weight * fn(model, **kwargs)
    loss.backward()
    ...
```

## Changes to `run_training.py`

1. Replace argparse with: `--config` (required), `--device` (optional override)
2. Load YAML with `yaml.safe_load`
3. Look up quantizer from `QUANTIZER_REGISTRY`, call `attach_weight_quantizers`
4. Build 3-way optimizer param groups by filtering `named_parameters` for `logit_dz` and `logit_bit`
5. Build loss_fns list from `loss_terms` config using `LOSS_REGISTRY`
6. Pass `loss_fns` to `train()`
7. Log full config dict to W&B

## Slurm Integration

```bash
python ../run_training.py --config ../configs/deadzone_resnet20.yaml --device cuda
```

`--device` overrides the YAML value so same config works locally (mps) and on cluster (cuda).

## Run Name Generation

Auto-generated from config: `{model}_{quantizer_name}{bit}b_e{epochs}_lr{lr}` or `{model}_baseline_e{epochs}_lr{lr}` when no quantizer.
