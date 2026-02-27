# YAML QAT Config Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace argparse flags in `run_training.py` with a YAML config system supporting pluggable quantizers, 3-way optimizer param groups, and structured loss terms via registries.

**Architecture:** `run_training.py` reads a YAML config file via `--config` arg. Quantizer classes and loss functions are looked up from simple registry dicts. The YAML specifies everything: model, training params, quantizer name+kwargs, optimizer param groups (base/dz/bit), and a list of loss terms with lambdas. `src/structured_loss.py` holds loss functions and their registry.

**Tech Stack:** PyTorch, PyYAML, wandb, timm

---

### Task 1: Create `src/structured_loss.py` with group lasso and registry

**Files:**
- Create: `src/structured_loss.py`

**Step 1: Write the loss function and registry**

```python
import torch
import torch.nn as nn


def group_lasso(model, lambda_linear=1e-4, lambda_conv=1e-4):
    """Group L2/L1 structured sparsity regularization.

    Computes L2 norm per column (input feature dimension), then sums (L1).
    Pushes entire input features/channels to zero for structured pruning.

    Works on both nn.Linear and nn.Conv2d layers.
    """
    reg = torch.tensor(0.0, device=next(model.parameters()).device)
    for m in model.modules():
        if isinstance(m, nn.Linear):
            W = m.weight  # [out_features, in_features]
            reg = reg + lambda_linear * torch.norm(W, dim=0).sum()
        elif isinstance(m, nn.Conv2d):
            W = m.weight  # [out_c, in_c, kH, kW]
            W_flat = W.view(W.size(0), -1)  # [out_c, in_c*kH*kW]
            reg = reg + lambda_conv * torch.norm(W_flat, dim=0).sum()
    return reg


LOSS_REGISTRY = {
    "group_lasso": group_lasso,
}
```

**Step 2: Commit**

```bash
git add src/structured_loss.py
git commit -m "feat: add structured loss module with group lasso and registry"
```

---

### Task 2: Modify `src/train.py` to accept optional loss functions

**Files:**
- Modify: `src/train.py:9` (the `train` function signature)

**Step 1: Add `loss_fns` parameter to `train()`**

Change the `train` function signature from:
```python
def train(train_loader, model, criterion, optimizer, epoch, device):
```
to:
```python
def train(train_loader, model, criterion, optimizer, epoch, device, loss_fns=None):
```

And in the training loop, after `loss = criterion(output, target)` (line 31), add the loss term injection before `optimizer.zero_grad()`:

```python
        # compute output
        output = model(input)
        loss = criterion(output, target)

        # add structured loss terms
        if loss_fns:
            for loss_fn, weight, kwargs in loss_fns:
                loss = loss + weight * loss_fn(model, **kwargs)

        # compute gradient and step optimizer
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

Note: The loss terms are added BEFORE `optimizer.zero_grad()` and `loss.backward()`. The existing code has `optimizer.zero_grad()` after `loss = criterion(...)` which is fine — we just insert the loss terms between computing CE loss and calling backward.

**Step 2: Commit**

```bash
git add src/train.py
git commit -m "feat: add optional loss_fns parameter to train()"
```

---

### Task 3: Create example YAML config files

**Files:**
- Create: `configs/baseline_resnet20.yaml`
- Create: `configs/deadzone_resnet20.yaml`
- Create: `configs/deadzone_vit.yaml`

**Step 1: Create configs directory and baseline config**

`configs/baseline_resnet20.yaml`:
```yaml
model: resnet20
epochs: 300
batch_size: 128
img_size: 32
device: mps
wandb_project: "CoDeQ experiments"
workers: 4
use_compile: 0
save_dir: save_temp

lr_scheduler: cosine

optimizer:
  type: sgd
  param_groups:
    base:
      lr: 0.1
      momentum: 0.9
      weight_decay: 0.0001
```

`configs/deadzone_resnet20.yaml`:
```yaml
model: resnet20
epochs: 300
batch_size: 128
img_size: 32
device: mps
wandb_project: "CoDeQ experiments"
workers: 4
use_compile: 0
save_dir: save_temp

lr_scheduler: cosine

quantizer:
  name: deadzone
  kwargs:
    fixed_bit_val: 8
    max_bits: 8
    init_deadzone_logit: 3.0
    init_bit_logit: 3.0
    learnable_bit: true
    learnable_deadzone: true
  exclude_layers: [bn]

optimizer:
  type: adamw
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
  - name: group_lasso
    lambda: 1.0
    kwargs:
      lambda_linear: 0.1
      lambda_conv: 0.1
```

`configs/deadzone_vit.yaml`:
```yaml
model: vit
epochs: 200
batch_size: 128
img_size: 224
device: mps
wandb_project: "CoDeQ experiments"
workers: 4
use_compile: 0
save_dir: save_temp

lr_scheduler: cosine

quantizer:
  name: deadzone
  kwargs:
    fixed_bit_val: 8
    max_bits: 8
    init_deadzone_logit: 3.0
    init_bit_logit: 3.0
    learnable_bit: true
    learnable_deadzone: true
  exclude_layers: [norm]

optimizer:
  type: adamw
  param_groups:
    base:
      lr: 0.0003
      weight_decay: 0.01
    dz:
      lr: 0.001
      weight_decay: 2.5
    bit:
      lr: 0.001
      weight_decay: 0.0

loss_terms:
  - name: group_lasso
    lambda: 1.0
    kwargs:
      lambda_linear: 0.1
      lambda_conv: 0.0
```

**Step 2: Commit**

```bash
git add configs/
git commit -m "feat: add example YAML experiment configs"
```

---

### Task 4: Rewrite `run_training.py` with YAML config loading

**Files:**
- Modify: `run_training.py` (full rewrite)

**Step 1: Rewrite the script**

Replace the entire `run_training.py` with the YAML-driven version. Key changes:

1. **Argparse**: Replace all flags with just `--config` (required) and `--device` (optional override)
2. **Registries**: Add `QUANTIZER_REGISTRY` and import `LOSS_REGISTRY`
3. **Config loading**: `yaml.safe_load()` the config file
4. **Model building**: Same model dispatch logic but reading from config dict
5. **Quantizer attachment**: Look up from registry, pass `kwargs` from config
6. **3-way optimizer**: Split `named_parameters()` into base/dz/bit groups using config LR/WD values
7. **Loss term building**: Build list of `(fn, weight, kwargs)` tuples from config
8. **Training loop**: Pass `loss_fns` to `train()`, log full config to wandb

```python
import argparse
import os
import yaml
from dotenv import load_dotenv
load_dotenv()
import wandb

import numpy as np
import timm
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import src.resnet as resnet

from src.utils_quantization import attach_weight_quantizers, toggle_quantization, UniformSymmetric
from src.quantizer import DeadZoneLDZCompander
from src.datasets import build_dataset_cifar
from src.train import train, validate, save_checkpoint
from src.structured_loss import LOSS_REGISTRY

# ── Registries ──────────────────────────────────────────────
QUANTIZER_REGISTRY = {
    "uniform": UniformSymmetric,
    "deadzone": DeadZoneLDZCompander,
}

# ── CLI ─────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description='CoDeQ Training')
parser.add_argument('--config', type=str, required=True, help='Path to YAML config file')
parser.add_argument('--device', type=str, default=None, help='Override device from config')
parser.add_argument('--resume', type=str, default='', help='Path to checkpoint to resume from')

best_prec1 = 0


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def build_run_name(cfg):
    q = cfg.get("quantizer")
    if q:
        bit = q["kwargs"].get("fixed_bit_val", q["kwargs"].get("max_bits", "?"))
        qat_tag = f"_{q['name']}{bit}b"
    else:
        qat_tag = "_baseline"
    return f"{cfg['model']}{qat_tag}_e{cfg['epochs']}_lr{cfg['optimizer']['param_groups']['base']['lr']}"


def build_optimizer(model, cfg):
    """Build optimizer with 3-way param group split (base / dz / bit)."""
    opt_cfg = cfg["optimizer"]
    groups_cfg = opt_cfg["param_groups"]

    base_params, dz_params, bit_params = [], [], []
    for name, param in model.named_parameters():
        if "logit_dz" in name:
            dz_params.append(param)
        elif "logit_bit" in name:
            bit_params.append(param)
        else:
            base_params.append(param)

    param_groups = [{"params": base_params, **groups_cfg["base"]}]
    if dz_params and "dz" in groups_cfg:
        param_groups.append({"params": dz_params, **groups_cfg["dz"]})
    if bit_params and "bit" in groups_cfg:
        param_groups.append({"params": bit_params, **groups_cfg["bit"]})

    opt_type = opt_cfg.get("type", "adamw").lower()
    if opt_type == "sgd":
        OptClass = torch.optim.SGD
    elif opt_type == "adamw":
        OptClass = torch.optim.AdamW
    elif opt_type == "adam":
        OptClass = torch.optim.Adam
    else:
        raise ValueError(f"Unknown optimizer type: {opt_type}")

    try:
        optimizer = OptClass(param_groups, fused=True)
    except TypeError:
        optimizer = OptClass(param_groups, foreach=True)

    return optimizer


def build_loss_fns(cfg):
    """Build list of (loss_fn, weight, kwargs) from config."""
    loss_fns = []
    for term in cfg.get("loss_terms", []):
        name = term["name"]
        if name not in LOSS_REGISTRY:
            raise ValueError(f"Unknown loss term: {name}. Available: {list(LOSS_REGISTRY.keys())}")
        fn = LOSS_REGISTRY[name]
        weight = term.get("lambda", 1.0)
        kwargs = term.get("kwargs", {})
        loss_fns.append((fn, weight, kwargs))
    return loss_fns or None


def main():
    global best_prec1
    args = parser.parse_args()
    cfg = load_config(args.config)

    # CLI overrides
    if args.device:
        cfg["device"] = args.device

    DEVICE = torch.device(cfg["device"])
    save_dir = cfg.get("save_dir", "save_temp")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # CUDA settings
    if DEVICE.type == "cuda":
        cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    run_name = build_run_name(cfg)
    wandb.init(project=cfg.get("wandb_project", "Uncategorized"), name=run_name, config=cfg)

    criterion = nn.CrossEntropyLoss().to(DEVICE)

    # ── Build model ─────────────────────────────────────────
    model_name = cfg["model"]
    if model_name == "resnet20":
        model = resnet.__dict__["resnet20"]()
    elif model_name == "vit":
        timm_name = "vit_tiny_patch16_224.augreg_in21k_ft_in1k"
        model = timm.create_model(timm_name, pretrained=False, num_classes=10)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    # ── Attach quantizer ────────────────────────────────────
    q_cfg = cfg.get("quantizer")
    if q_cfg:
        q_name = q_cfg["name"]
        if q_name not in QUANTIZER_REGISTRY:
            raise ValueError(f"Unknown quantizer: {q_name}. Available: {list(QUANTIZER_REGISTRY.keys())}")
        q_cls = QUANTIZER_REGISTRY[q_name]
        q_kwargs = q_cfg.get("kwargs", {})
        exclude = q_cfg.get("exclude_layers", [])
        attach_weight_quantizers(model, exclude_layers=exclude, quantizer=q_cls, quantizer_kwargs=q_kwargs, enabled=True)
        toggle_quantization(model, enabled=True)

    model.to(DEVICE)

    # ── Data ────────────────────────────────────────────────
    train_loader, val_loader = build_dataset_cifar(
        workers=cfg.get("workers", 4),
        batch_size=cfg.get("batch_size", 128),
        img_size=cfg.get("img_size", 32),
    )

    # ── Optimizer ───────────────────────────────────────────
    optimizer = build_optimizer(model, cfg)

    # ── LR Scheduler ────────────────────────────────────────
    epochs = cfg["epochs"]
    sched_type = cfg.get("lr_scheduler", "cosine")
    if sched_type == "cosine":
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    else:
        raise ValueError(f"Unknown lr_scheduler: {sched_type}")

    # ── Loss terms ──────────────────────────────────────────
    loss_fns = build_loss_fns(cfg)

    # ── Resume ──────────────────────────────────────────────
    start_epoch = 0
    if args.resume and os.path.isfile(args.resume):
        print(f"=> loading checkpoint '{args.resume}'")
        checkpoint = torch.load(args.resume, map_location=DEVICE)
        start_epoch = checkpoint.get("epoch", 0)
        best_prec1 = checkpoint.get("best_prec1", 0)
        state_dict_key = "model" if "model" in checkpoint else "state_dict"
        model.load_state_dict(checkpoint[state_dict_key])
        if "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
        if "scheduler" in checkpoint:
            lr_scheduler.load_state_dict(checkpoint["scheduler"])
        print(f"=> loaded checkpoint (epoch {start_epoch})")

    # ── Compile ─────────────────────────────────────────────
    if cfg.get("use_compile", 0):
        model = torch.compile(model, mode="max-autotune", fullgraph=False, dynamic=True)

    # ── Training loop ───────────────────────────────────────
    for epoch in range(start_epoch, epochs):
        print("current lr weights {:.5e}".format(optimizer.param_groups[0]["lr"]))

        loss, train_acc = train(train_loader, model, criterion, optimizer, epoch, DEVICE, loss_fns=loss_fns)
        lr_scheduler.step()

        prec1 = validate(val_loader, model, criterion, DEVICE)

        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)

        wandb.log({
            "epoch": epoch,
            "loss": loss,
            "train_acc": train_acc,
            "val_acc": prec1,
        })

        save_checkpoint({
            "epoch": epoch + 1,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": lr_scheduler.state_dict(),
            "best_prec1": best_prec1,
            "config": cfg,
        }, is_best, filename=os.path.join(save_dir, f"checkpoint_{run_name}.th"))


if __name__ == "__main__":
    main()
```

**Step 2: Commit**

```bash
git add run_training.py
git commit -m "feat: rewrite run_training.py with YAML config, quantizer registry, loss injection"
```

---

### Task 5: Update Slurm script and requirements

**Files:**
- Modify: `slurm/single_slurm.sh`
- Modify: `requirements.txt`

**Step 1: Update slurm script**

Replace the training command in `slurm/single_slurm.sh`:

From:
```bash
python ../train.py \
  --wandb "test runs" \
  --device "cuda" \
  --workers 6 \
  --use-compile 1 \
  --use-qat 0 \
  --quantizer-bit 8 \
  --model "vit" \
  --img-size 224 \
  --epochs 200 \
  --lr 0.0003 \
  --wd 0.01 \
  --batch-size 128
```

To:
```bash
python ../run_training.py --config ../configs/deadzone_vit.yaml --device cuda
```

**Step 2: Add pyyaml to requirements.txt**

Add `pyyaml` and `python-dotenv` (already used but not listed) to `requirements.txt`.

**Step 3: Commit**

```bash
git add slurm/single_slurm.sh requirements.txt
git commit -m "chore: update slurm script for YAML config, add pyyaml to requirements"
```

---

### Task 6: Add simple MLP model for CIFAR

**Files:**
- Modify: `run_training.py` (add MLP to model dispatch)

**Step 1: Add MLP class and model dispatch**

Add a simple `CifarMLP` class directly in `run_training.py`. Fixed 3-layer architecture (3072→120→84→10) matching a LeNet-style MLP for CIFAR:

```python
class CifarMLP(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3 * 32 * 32, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, num_classes),
        )

    def forward(self, x):
        return self.net(x.view(x.size(0), -1))
```

Add to model dispatch in `main()`:
```python
    elif model_name == "mlp":
        model = CifarMLP()
```

**Step 2: Create MLP config**

Create `configs/deadzone_mlp.yaml`:
```yaml
model: mlp
epochs: 100
batch_size: 128
img_size: 32
device: mps
wandb_project: "CoDeQ experiments"
workers: 4
use_compile: 0
save_dir: save_temp

lr_scheduler: cosine

quantizer:
  name: deadzone
  kwargs:
    fixed_bit_val: 8
    max_bits: 8
    init_deadzone_logit: 3.0
    init_bit_logit: 3.0
    learnable_bit: true
    learnable_deadzone: true
  exclude_layers: []

optimizer:
  type: adamw
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
  - name: group_lasso
    lambda: 1.0
    kwargs:
      lambda_linear: 0.1
      lambda_conv: 0.0
```

**Step 3: Commit**

```bash
git add run_training.py configs/deadzone_mlp.yaml
git commit -m "feat: add simple MLP model for CIFAR experiments"
```

---

### Task 7: Smoke test

**Step 1: Verify baseline config loading works**

Run:
```bash
python run_training.py --config configs/baseline_resnet20.yaml --device cpu 2>&1 | head -20
```

Expected: Should start training (or at least print the first epoch log line and W&B init). Ctrl-C after confirming it starts.

**Step 2: Verify deadzone QAT config works**

Run:
```bash
python run_training.py --config configs/deadzone_resnet20.yaml --device cpu 2>&1 | head -30
```

Expected: Should print "Attached weight quantizer to layer: ..." messages for each conv layer (not bn layers), then start training.

**Step 3: Verify MLP config works**

Run:
```bash
python run_training.py --config configs/deadzone_mlp.yaml --device cpu 2>&1 | head -30
```

Expected: Should attach quantizers to all linear layers and start training.

**Step 4: Commit any fixes if needed**
