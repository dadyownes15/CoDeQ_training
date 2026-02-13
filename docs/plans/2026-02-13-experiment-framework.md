# Experiment Framework Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a modular experiment framework for pruning research with pluggable structured sparsity losses, callable schedules, and W&B logging — evaluating both structured and unstructured BOPs compression.

**Architecture:** Separate modules for config, sparsity losses, schedules, evaluation, and training. The evaluator wraps `metrics.py` and adds unstructured BOPs. The trainer wires everything together with W&B. All experiment parameters live in a single `ExperimentConfig` dataclass.

**Tech Stack:** PyTorch, Weights & Biases, matplotlib, existing `metrics.py` + `src/quantizer.py` + `utils_quantization.py`

---

### Task 1: Add unstructured MACs to metrics.py

**Files:**
- Modify: `metrics.py` (add `_compute_unstructured_macs` function and `profile_unstructured` method)
- Test: `tests/test_bops.py` (add unstructured tests alongside existing structured tests)

**Step 1: Write the failing tests**

Add to `tests/test_bops.py`:

```python
from metrics import _compute_unstructured_macs


def test_unstructured_macs_dense():
    """All weights = 1 → nnz = c_out * c_in * k * k.

    SingleConv: (4, 2, 3, 3) all ones, m=4x4
    nnz = 4 * 2 * 3 * 3 = 72
    MACs = 72 * 4 * 4 = 1152 (same as structured when nothing is pruned)
    """
    model = SingleConv()
    nn.init.ones_(model.conv.weight)
    info = _get_layer_quant_info(model.conv)
    macs = _compute_unstructured_macs(info.weight, 4, 4)
    assert macs == 72 * 16  # 1152


def test_unstructured_macs_half_weights_zeroed():
    """Half the individual weights zeroed (scattered, not full channels).

    weight shape (4, 2, 3, 3) = 72 elements, zero out 36 → nnz=36
    MACs = 36 * 4 * 4 = 576
    """
    model = SingleConv()
    nn.init.ones_(model.conv.weight)
    with torch.no_grad():
        flat = model.conv.weight.flatten()
        flat[:36] = 0.0
    info = _get_layer_quant_info(model.conv)
    macs = _compute_unstructured_macs(info.weight, 4, 4)
    assert macs == 36 * 16  # 576


def test_unstructured_macs_all_zero():
    """All weights zero → 0 MACs."""
    model = SingleConv()
    nn.init.zeros_(model.conv.weight)
    info = _get_layer_quant_info(model.conv)
    macs = _compute_unstructured_macs(info.weight, 4, 4)
    assert macs == 0


def test_unstructured_vs_structured_gap():
    """Scattered zeros give more unstructured savings than structured.

    Zero out half the weights in each filter (no full channel dies).
    Structured: all channels alive → MACs = dense = 1152
    Unstructured: nnz=36 → MACs = 576
    """
    model = SingleConv()
    nn.init.ones_(model.conv.weight)
    with torch.no_grad():
        # Zero half of each filter's weights (scattered)
        for i in range(4):
            flat = model.conv.weight[i].flatten()
            flat[:flat.numel() // 2] = 0.0

    info = _get_layer_quant_info(model.conv)
    null_in = NullChannels.none(total=2)
    null_out = _compute_null_output_channels(info.weight, null_in)

    structured_macs = _compute_effective_macs(info.weight, null_in, null_out, 4, 4)
    unstructured_macs = _compute_unstructured_macs(info.weight, 4, 4)

    assert structured_macs == 1152  # no channels fully dead
    assert unstructured_macs == 576  # half the individual weights are zero
    assert unstructured_macs < structured_macs


def test_unstructured_macs_linear():
    """Linear(4->8), half weights zeroed.

    nnz = 16, m=1x1 → MACs = 16
    """
    model = SingleLinear()
    nn.init.ones_(model.fc.weight)
    with torch.no_grad():
        flat = model.fc.weight.flatten()
        flat[:16] = 0.0
    info = _get_layer_quant_info(model.fc)
    macs = _compute_unstructured_macs(info.weight, 1, 1)
    assert macs == 16
```

**Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_bops.py -v -k "unstructured"`
Expected: ImportError — `_compute_unstructured_macs` does not exist

**Step 3: Implement `_compute_unstructured_macs` in metrics.py**

Add after `_compute_effective_macs` (around line 198):

```python
def _compute_unstructured_macs(
    weight: torch.Tensor,
    m_h: int,
    m_w: int,
) -> int:
    """Compute MACs counting individual non-zero weights.

    Unlike the structured formula which operates at channel granularity,
    this counts every non-zero weight element as one MAC per spatial position:

        MACs_l = nnz(W_l) * m_h * m_w

    For Linear layers: m_h = m_w = 1.
    """
    nnz = (weight != 0).sum().item()
    return nnz * m_h * m_w
```

**Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_bops.py -v -k "unstructured"`
Expected: All 6 new tests PASS

**Step 5: Add `calculate_unstructured_stats` to `QuantizationComparison`**

Add method to the `QuantizationComparison` class:

```python
def calculate_unstructured_stats(self, model: nn.Module) -> ModelStats:
    """Profile model using unstructured (weight-level) sparsity for MACs."""
    model.eval()

    spatial = _get_spatial_dims(model, self.input_size)

    layer_info: dict[str, LayerQuantInfo] = {}
    module_map: dict[str, nn.Module] = {}
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            with torch.no_grad():
                layer_info[name] = _get_layer_quant_info(module)
            module_map[name] = module

    layers: list[LayerStats] = []
    total_macs = 0
    total_bops = 0.0

    for name, info in layer_info.items():
        if name not in spatial:
            continue

        m_h, m_w = spatial[name]
        macs = _compute_unstructured_macs(info.weight, m_h, m_w)
        bops = macs * info.bitwidth * self.b_a

        # For unstructured, pruning_ratio = fraction of zero weights
        total_weights = info.weight.numel()
        zero_weights = (info.weight == 0).sum().item()
        pruning_ratio = zero_weights / total_weights if total_weights > 0 else 0.0

        if isinstance(module_map[name], nn.Conv2d):
            k_size = (info.weight.shape[2], info.weight.shape[3])
        else:
            k_size = (1, 1)

        # Use empty NullChannels for unstructured (not applicable)
        c_in = info.weight.shape[1]
        c_out = info.weight.shape[0]

        layers.append(LayerStats(
            name=name,
            macs=macs,
            bops=bops,
            bitwidth=info.bitwidth,
            pruning_ratio=pruning_ratio,
            null_out=NullChannels.none(total=c_out),
            null_in=NullChannels.none(total=c_in),
            spatial=(m_h, m_w),
            kernel_size=k_size,
        ))

        total_macs += macs
        total_bops += bops

    return ModelStats(layers=layers, total_macs=total_macs, total_bops=total_bops)


def unstructured_compression_ratio(self, stats: ModelStats) -> float:
    """Unstructured BOPs compression ratio: baseline_BOPs / unstructured_BOPs."""
    assert self.baseline_stats is not None, "Call set_baseline() first"
    return self.baseline_stats.total_bops / stats.total_bops


def unstructured_mac_compression_ratio(self, stats: ModelStats) -> float:
    """Unstructured MAC compression ratio: baseline_MACs / unstructured_MACs."""
    assert self.baseline_stats is not None, "Call set_baseline() first"
    return self.baseline_stats.total_macs / stats.total_macs
```

**Step 6: Run full test suite**

Run: `python -m pytest tests/test_bops.py -v`
Expected: All tests PASS (old + new)

**Step 7: Commit**

```bash
git add metrics.py tests/test_bops.py
git commit -m "feat: add unstructured MACs/BOPs computation to metrics"
```

---

### Task 2: Create experiments/schedules.py

**Files:**
- Create: `experiments/__init__.py`
- Create: `experiments/schedules.py`
- Create: `tests/test_schedules.py`

**Step 1: Write the failing tests**

Create `tests/test_schedules.py`:

```python
"""Tests for sparsity coefficient schedule factories."""
import math
from experiments.schedules import constant, linear_warmup, cosine_anneal


def test_constant_always_returns_value():
    fn = constant(0.5)
    assert fn(0, 100) == 0.5
    assert fn(50, 100) == 0.5
    assert fn(99, 100) == 0.5


def test_constant_zero():
    fn = constant(0.0)
    assert fn(50, 100) == 0.0


def test_linear_warmup_delay_phase():
    """Zero during delay period."""
    fn = linear_warmup(delay=50, ramp=20)
    assert fn(0, 200) == 0.0
    assert fn(25, 200) == 0.0
    assert fn(49, 200) == 0.0


def test_linear_warmup_ramp_phase():
    """Linear increase from 0 to 1 during ramp period."""
    fn = linear_warmup(delay=50, ramp=20)
    assert fn(50, 200) == 0.0   # start of ramp
    assert fn(60, 200) == 0.5   # midpoint of ramp
    assert fn(70, 200) == 1.0   # end of ramp


def test_linear_warmup_after_ramp():
    """Stays at 1.0 after ramp completes."""
    fn = linear_warmup(delay=50, ramp=20)
    assert fn(71, 200) == 1.0
    assert fn(100, 200) == 1.0
    assert fn(199, 200) == 1.0


def test_linear_warmup_no_delay():
    """delay=0 → ramp starts immediately."""
    fn = linear_warmup(delay=0, ramp=10)
    assert fn(0, 100) == 0.0
    assert fn(5, 100) == 0.5
    assert fn(10, 100) == 1.0


def test_cosine_anneal_delay_phase():
    """Zero during delay period."""
    fn = cosine_anneal(delay=30)
    assert fn(0, 100) == 0.0
    assert fn(29, 100) == 0.0


def test_cosine_anneal_after_delay():
    """Cosine ramp from 0 → 1 over remaining epochs."""
    fn = cosine_anneal(delay=0)
    assert fn(0, 100) == 0.0
    assert abs(fn(50, 100) - 0.5) < 0.01  # midpoint ≈ 0.5
    assert abs(fn(99, 100) - 1.0) < 0.01  # end ≈ 1.0


def test_cosine_anneal_with_delay():
    fn = cosine_anneal(delay=50)
    assert fn(49, 100) == 0.0
    # After delay, 50 epochs of cosine ramp
    assert abs(fn(75, 100) - 0.5) < 0.01  # midpoint of ramp
    assert abs(fn(99, 100) - 1.0) < 0.02
```

**Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_schedules.py -v`
Expected: ModuleNotFoundError — `experiments.schedules` does not exist

**Step 3: Create `experiments/__init__.py` and `experiments/schedules.py`**

Create `experiments/__init__.py` (empty):

```python
```

Create `experiments/schedules.py`:

```python
"""Schedule factories for sparsity loss coefficients.

Each factory returns a Callable[[int, int], float] that maps
(current_epoch, total_epochs) → coefficient value.
"""
import math
from typing import Callable


def constant(value: float) -> Callable[[int, int], float]:
    """Always returns the given value."""
    def schedule(epoch: int, total_epochs: int) -> float:
        return value
    return schedule


def linear_warmup(delay: int, ramp: int) -> Callable[[int, int], float]:
    """Zero for `delay` epochs, then linear ramp from 0 to 1 over `ramp` epochs.

    Args:
        delay: Number of epochs with zero coefficient.
        ramp: Number of epochs to linearly ramp from 0 to 1.
    """
    def schedule(epoch: int, total_epochs: int) -> float:
        if epoch < delay:
            return 0.0
        elapsed = epoch - delay
        if elapsed >= ramp:
            return 1.0
        return elapsed / ramp
    return schedule


def cosine_anneal(delay: int = 0) -> Callable[[int, int], float]:
    """Zero for `delay` epochs, then cosine ramp from 0 to 1 over remaining epochs.

    Uses 0.5 * (1 - cos(pi * t)) where t goes from 0 to 1.

    Args:
        delay: Number of epochs with zero coefficient.
    """
    def schedule(epoch: int, total_epochs: int) -> float:
        if epoch < delay:
            return 0.0
        remaining = total_epochs - delay
        if remaining <= 0:
            return 1.0
        t = (epoch - delay) / remaining
        return 0.5 * (1.0 - math.cos(math.pi * t))
    return schedule
```

**Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_schedules.py -v`
Expected: All 10 tests PASS

**Step 5: Commit**

```bash
git add experiments/__init__.py experiments/schedules.py tests/test_schedules.py
git commit -m "feat: add schedule factories for sparsity coefficients"
```

---

### Task 3: Create experiments/sparsity.py

**Files:**
- Create: `experiments/sparsity.py`
- Create: `tests/test_sparsity.py`

**Step 1: Write the failing tests**

Create `tests/test_sparsity.py`:

```python
"""Tests for structured sparsity loss functions."""
import torch
import torch.nn as nn
from experiments.sparsity import group_lasso_channels, group_lasso_blocks, l1_kernel_sparsity


class TinyConvNet(nn.Module):
    """Minimal 2-layer conv net for testing."""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 4, 3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(4, 8, 3, padding=1, bias=False)

    def forward(self, x):
        return self.conv2(self.conv1(x))


def test_group_lasso_channels_zero_model():
    """All weights zero → loss = 0."""
    model = TinyConvNet()
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.zeros_(m.weight)
    loss = group_lasso_channels(model)
    assert loss.item() == 0.0


def test_group_lasso_channels_positive():
    """Non-zero weights → positive loss."""
    model = TinyConvNet()
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.ones_(m.weight)
    loss = group_lasso_channels(model)
    assert loss.item() > 0.0


def test_group_lasso_channels_gradient_flows():
    """Loss should have gradients w.r.t. weights."""
    model = TinyConvNet()
    loss = group_lasso_channels(model)
    loss.backward()
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            assert m.weight.grad is not None


def test_group_lasso_channels_value():
    """Verify exact value: L1 of L2 filter norms.

    conv1: (4, 3, 3, 3) all ones → each filter norm = sqrt(3*3*3) = sqrt(27)
    4 filters → sum = 4*sqrt(27)
    conv2: (8, 4, 3, 3) all ones → each filter norm = sqrt(4*3*3) = sqrt(36) = 6
    8 filters → sum = 8*6 = 48
    Total = 4*sqrt(27) + 48
    """
    model = TinyConvNet()
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.ones_(m.weight)
    loss = group_lasso_channels(model)
    expected = 4 * (27 ** 0.5) + 48.0
    assert abs(loss.item() - expected) < 1e-4


def test_group_lasso_blocks_zero_model():
    model = TinyConvNet()
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.zeros_(m.weight)
    fn = group_lasso_blocks(block_size=2)
    loss = fn(model)
    assert loss.item() == 0.0


def test_group_lasso_blocks_gradient_flows():
    model = TinyConvNet()
    fn = group_lasso_blocks(block_size=2)
    loss = fn(model)
    loss.backward()
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            assert m.weight.grad is not None


def test_l1_kernel_sparsity_zero_model():
    model = TinyConvNet()
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.zeros_(m.weight)
    loss = l1_kernel_sparsity(model)
    assert loss.item() == 0.0


def test_l1_kernel_sparsity_gradient_flows():
    model = TinyConvNet()
    loss = l1_kernel_sparsity(model)
    loss.backward()
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            assert m.weight.grad is not None


def test_l1_kernel_sparsity_value():
    """Verify exact value: sum of L1 norms of each kernel.

    conv1: (4, 3, 3, 3) all ones → each kernel L1 = 9, 4*3=12 kernels → 108
    conv2: (8, 4, 3, 3) all ones → each kernel L1 = 9, 8*4=32 kernels → 288
    Total = 396
    """
    model = TinyConvNet()
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.ones_(m.weight)
    loss = l1_kernel_sparsity(model)
    assert abs(loss.item() - 396.0) < 1e-4
```

**Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_sparsity.py -v`
Expected: ModuleNotFoundError — `experiments.sparsity` does not exist

**Step 3: Implement `experiments/sparsity.py`**

```python
"""Structured sparsity loss functions.

Each function has signature: fn(model: nn.Module) -> torch.Tensor (scalar loss).
For parameterized variants, a factory function returns the loss function.
"""
import torch
import torch.nn as nn
from typing import Callable


def group_lasso_channels(model: nn.Module) -> torch.Tensor:
    """Group lasso on output channels: L1 of L2 filter norms.

    Encourages entire output filters to go to zero (channel pruning).
    For a Conv2d with weight (c_out, c_in, k_h, k_w), computes:
        sum over c_out of ||W[c, :, :, :]||_2
    """
    loss = torch.tensor(0.0, device=next(model.parameters()).device)
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            # (c_out, c_in*k_h*k_w) → L2 norm per output filter
            filter_norms = m.weight.flatten(1).norm(p=2, dim=1)
            loss = loss + filter_norms.sum()
    return loss


def group_lasso_blocks(block_size: int = 4) -> Callable[[nn.Module], torch.Tensor]:
    """Factory: group lasso on blocks of output filters.

    Groups consecutive output channels into blocks and penalizes
    the L2 norm of each block.

    Args:
        block_size: Number of consecutive output channels per block.
    """
    def fn(model: nn.Module) -> torch.Tensor:
        loss = torch.tensor(0.0, device=next(model.parameters()).device)
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                w = m.weight  # (c_out, c_in, k_h, k_w)
                for i in range(0, w.shape[0], block_size):
                    block = w[i:i + block_size]
                    loss = loss + block.norm(p=2)
        return loss
    return fn


def l1_kernel_sparsity(model: nn.Module) -> torch.Tensor:
    """L1 norm of each individual kernel (finest granularity).

    For a Conv2d with weight (c_out, c_in, k_h, k_w), computes:
        sum over (c_out, c_in) of ||W[o, i, :, :]||_1
    """
    loss = torch.tensor(0.0, device=next(model.parameters()).device)
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            # (c_out * c_in, k_h * k_w) → L1 norm per kernel
            w = m.weight.view(m.weight.shape[0] * m.weight.shape[1], -1)
            kernel_norms = w.norm(p=1, dim=1)
            loss = loss + kernel_norms.sum()
    return loss
```

**Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_sparsity.py -v`
Expected: All 10 tests PASS

**Step 5: Commit**

```bash
git add experiments/sparsity.py tests/test_sparsity.py
git commit -m "feat: add structured sparsity loss functions"
```

---

### Task 4: Create experiments/config.py

**Files:**
- Create: `experiments/config.py`

**Step 1: Create the config dataclass**

```python
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
```

**Step 2: Commit**

```bash
git add experiments/config.py
git commit -m "feat: add ExperimentConfig dataclass"
```

---

### Task 5: Create experiments/evaluator.py

**Files:**
- Create: `experiments/evaluator.py`
- Create: `tests/test_evaluator.py`

**Step 1: Write the failing tests**

Create `tests/test_evaluator.py`:

```python
"""Tests for the experiment evaluator."""
import torch
import torch.nn as nn
from experiments.evaluator import Evaluator, EvalResult


class TinyConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 4, 3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(4, 8, 3, padding=1, bias=False)

    def forward(self, x):
        return self.conv2(self.conv1(x))


def test_evaluator_dense_baseline():
    """Dense model: structured and unstructured ratios should both be 1.0."""
    model = TinyConvNet()
    nn.init.ones_(model.conv1.weight)
    nn.init.ones_(model.conv2.weight)

    evaluator = Evaluator(model, input_size=(3, 4, 4))
    result = evaluator.evaluate(model)

    assert isinstance(result, EvalResult)
    assert abs(result.structured_bops_ratio - 1.0) < 1e-6
    assert abs(result.unstructured_bops_ratio - 1.0) < 1e-6


def test_evaluator_scattered_zeros():
    """Scattered zeros: unstructured ratio > structured ratio.

    Zero half the weights in each filter (no full channels die).
    Structured sees no savings; unstructured sees ~2x.
    """
    model = TinyConvNet()
    nn.init.ones_(model.conv1.weight)
    nn.init.ones_(model.conv2.weight)

    evaluator = Evaluator(model, input_size=(3, 4, 4))

    # Zero half weights scattered
    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                flat = m.weight.flatten()
                flat[:flat.numel() // 2] = 0.0

    result = evaluator.evaluate(model)

    # Structured: channels still alive → ratio close to 1
    # Unstructured: half weights zero → ratio close to 2
    assert result.unstructured_bops_ratio > result.structured_bops_ratio


def test_evaluator_full_channel_prune():
    """Full channel pruning: structured and unstructured both see savings."""
    model = TinyConvNet()
    nn.init.ones_(model.conv1.weight)
    nn.init.ones_(model.conv2.weight)

    evaluator = Evaluator(model, input_size=(3, 4, 4))

    with torch.no_grad():
        model.conv1.weight[2:, :, :, :] = 0.0  # kill channels 2,3

    result = evaluator.evaluate(model)

    assert result.structured_bops_ratio > 1.0
    assert result.unstructured_bops_ratio > 1.0
```

**Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_evaluator.py -v`
Expected: ModuleNotFoundError

**Step 3: Implement `experiments/evaluator.py`**

```python
"""Evaluator: structured + unstructured BOPs compression and visualization."""
import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from dataclasses import dataclass

from metrics import ModelStats, QuantizationComparison


@dataclass
class EvalResult:
    """Evaluation result with both structured and unstructured BOPs."""
    # Structured (channel-level null tracking)
    structured_stats: ModelStats
    structured_bops_ratio: float
    structured_mac_ratio: float

    # Unstructured (individual zero weights)
    unstructured_stats: ModelStats
    unstructured_bops_ratio: float
    unstructured_mac_ratio: float


class Evaluator:
    """Evaluates a model for both structured and unstructured compression.

    Usage:
        evaluator = Evaluator(baseline_model, input_size=(3, 32, 32))
        result = evaluator.evaluate(compressed_model)
    """

    def __init__(self, baseline_model: nn.Module, input_size: tuple = (3, 32, 32)):
        self.qc = QuantizationComparison(input_size=input_size)
        self.qc.set_baseline(baseline_model)

    def evaluate(self, model: nn.Module) -> EvalResult:
        """Compute structured and unstructured BOPs compression ratios."""
        structured_stats = self.qc.calculate_stats(model)
        unstructured_stats = self.qc.calculate_unstructured_stats(model)

        return EvalResult(
            structured_stats=structured_stats,
            structured_bops_ratio=self.qc.compression_ratio(structured_stats),
            structured_mac_ratio=self.qc.mac_compression_ratio(structured_stats),
            unstructured_stats=unstructured_stats,
            unstructured_bops_ratio=self.qc.unstructured_compression_ratio(unstructured_stats),
            unstructured_mac_ratio=self.qc.unstructured_mac_compression_ratio(unstructured_stats),
        )

    def plot_channel_liveness(self, model: nn.Module) -> plt.Figure:
        """Heatmap: rows=layers, cols=channels (padded), black=dead, white=alive."""
        layer_data = []
        layer_names = []
        max_channels = 0

        for name, m in model.named_modules():
            if isinstance(m, nn.Conv2d):
                w = m.weight.detach()
                c_out = w.shape[0]
                # Channel is dead if all weights in that output filter are zero
                alive = (w.flatten(1).norm(p=2, dim=1) > 0).float().cpu().numpy()
                layer_data.append(alive)
                layer_names.append(name)
                max_channels = max(max_channels, c_out)

        # Pad to same width (0.5 = gray for padding)
        grid = np.full((len(layer_data), max_channels), 0.5)
        for i, data in enumerate(layer_data):
            grid[i, :len(data)] = data

        fig, ax = plt.subplots(figsize=(max(8, max_channels * 0.15), max(4, len(layer_data) * 0.4)))
        ax.imshow(grid, cmap='gray', interpolation='nearest', vmin=0, vmax=1, aspect='auto')
        ax.set_yticks(range(len(layer_names)))
        ax.set_yticklabels(layer_names, fontsize=7)
        ax.set_xlabel('Output Channel')
        ax.set_title('Channel Liveness (white=alive, black=dead, gray=padding)')
        fig.tight_layout()
        return fig

    def plot_sparsity_map(self, model: nn.Module, layer_name: str) -> plt.Figure:
        """Per-kernel zero/nonzero grid for a single layer.

        Rows = output channels, Cols = input channels.
        Each cell is a k x k binary kernel (white=nonzero, black=zero).
        """
        modules = dict(model.named_modules())
        layer = modules[layer_name]
        w = layer.weight.detach().cpu().numpy()
        out_ch, in_ch, k_h, k_w = w.shape

        padding = 1
        grid_h = out_ch * (k_h + padding) + padding
        grid_w = in_ch * (k_w + padding) + padding
        grid = np.ones((grid_h, grid_w)) * 0.5

        for o in range(out_ch):
            for i in range(in_ch):
                kernel = w[o, i, :, :]
                binary = (kernel != 0).astype(float)
                r = padding + o * (k_h + padding)
                c = padding + i * (k_w + padding)
                grid[r:r + k_h, c:c + k_w] = binary

        fig, ax = plt.subplots(figsize=(max(6, in_ch * 0.5), max(6, out_ch * 0.5)))
        ax.imshow(grid, cmap='gray', interpolation='nearest', vmin=0, vmax=1)
        ax.set_title(f'Sparsity Map — {layer_name}\n(rows=out_ch, cols=in_ch, black=zero)')
        ax.axis('off')
        fig.tight_layout()
        return fig

    def plot_layer_sparsity_bars(self, result: EvalResult) -> plt.Figure:
        """Bar chart: structured vs unstructured pruning ratio per layer."""
        names = [l.name for l in result.structured_stats.layers]
        structured = [l.pruning_ratio for l in result.structured_stats.layers]
        unstructured = [l.pruning_ratio for l in result.unstructured_stats.layers]

        x = np.arange(len(names))
        width = 0.35

        fig, ax = plt.subplots(figsize=(max(8, len(names) * 0.6), 5))
        ax.bar(x - width / 2, structured, width, label='Structured (channel)', color='steelblue')
        ax.bar(x + width / 2, unstructured, width, label='Unstructured (weight)', color='coral')
        ax.set_ylabel('Pruning Ratio')
        ax.set_title('Per-Layer Sparsity: Structured vs Unstructured')
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=45, ha='right', fontsize=7)
        ax.legend()
        ax.set_ylim(0, 1)
        fig.tight_layout()
        return fig
```

**Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_evaluator.py -v`
Expected: All 3 tests PASS

**Step 5: Commit**

```bash
git add experiments/evaluator.py tests/test_evaluator.py
git commit -m "feat: add evaluator with structured/unstructured BOPs and visualization"
```

---

### Task 6: Create experiments/trainer.py

**Files:**
- Create: `experiments/trainer.py`

This is the integration module — the training loop. Due to W&B and GPU dependencies, we don't unit-test this directly. Instead, we verify it works end-to-end in Task 8.

**Step 1: Implement the trainer**

```python
"""Training loop with W&B logging."""
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import wandb
from dataclasses import asdict

from resnet import resnet20, resnet32, resnet44, resnet56, resnet110
from src.quantizer import DeadZoneLDZCompander
from utils_quantization import attach_weight_quantizers, toggle_quantization
from experiments.config import ExperimentConfig
from experiments.evaluator import Evaluator


MODEL_REGISTRY = {
    'resnet20': resnet20,
    'resnet32': resnet32,
    'resnet44': resnet44,
    'resnet56': resnet56,
    'resnet110': resnet110,
}

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)


def _build_model(config: ExperimentConfig) -> nn.Module:
    model_fn = MODEL_REGISTRY[config.model]
    return model_fn()


def _build_dataloaders(config: ExperimentConfig):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

    train_set = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train)
    test_set = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=config.batch_size, shuffle=True,
        num_workers=config.num_workers, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=config.batch_size, shuffle=False,
        num_workers=config.num_workers, pin_memory=True)

    return train_loader, test_loader


def _split_param_groups(model: nn.Module):
    base_params, dz_params, bit_params = [], [], []
    for name, param in model.named_parameters():
        if 'logit_dz' in name:
            dz_params.append(param)
        elif 'logit_bit' in name:
            bit_params.append(param)
        else:
            base_params.append(param)
    return base_params, dz_params, bit_params


@torch.no_grad()
def _evaluate(model: nn.Module, test_loader, device: str):
    model.eval()
    correct = 0
    total = 0
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()

    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        total_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    accuracy = correct / total
    avg_loss = total_loss / total
    return accuracy, avg_loss


def run_experiment(config: ExperimentConfig):
    """Run a full training experiment with W&B logging.

    Args:
        config: Experiment configuration.
    """
    # --- W&B init ---
    wandb.init(
        project=config.wandb_project,
        name=config.name,
        config={k: v for k, v in asdict(config).items()
                if k not in ('sparsity_fn', 'sparsity_schedule')},
    )

    device = config.device

    # --- Model ---
    model = _build_model(config)

    # --- Baseline evaluation (before quantization) ---
    evaluator = Evaluator(model, input_size=(3, 32, 32))

    # --- Attach quantizers ---
    attach_weight_quantizers(
        model=model,
        exclude_layers=config.exclude_layers,
        quantizer=DeadZoneLDZCompander,
        quantizer_kwargs=config.quantizer_kwargs,
        enabled=True,
    )
    model.to(device)

    # --- Optimizer ---
    base_params, dz_params, bit_params = _split_param_groups(model)
    optimizer = optim.AdamW([
        {'params': base_params, 'lr': config.lr, 'weight_decay': config.weight_decay},
        {'params': dz_params, 'lr': config.lr_dz, 'weight_decay': config.weight_decay_dz},
        {'params': bit_params, 'lr': config.lr_bit, 'weight_decay': config.weight_decay_bit},
    ])

    # --- LR scheduler (cosine on all param groups) ---
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)

    # --- Data ---
    train_loader, test_loader = _build_dataloaders(config)

    # --- Training loop ---
    criterion = nn.CrossEntropyLoss()

    for epoch in range(config.epochs):
        model.train()
        toggle_quantization(model, enabled=True)

        coeff = config.sparsity_schedule(epoch, config.epochs)
        running_ce = 0.0
        running_sp = 0.0
        num_batches = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            ce_loss = criterion(outputs, labels)

            if config.sparsity_fn is not None:
                sp_loss = config.sparsity_fn(model)
                total_loss = ce_loss + coeff * sp_loss
            else:
                sp_loss = torch.tensor(0.0)
                total_loss = ce_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            running_ce += ce_loss.item()
            running_sp += sp_loss.item()
            num_batches += 1

        scheduler.step()

        # --- Eval ---
        test_acc, test_loss = _evaluate(model, test_loader, device)

        # --- Compression metrics ---
        result = evaluator.evaluate(model)

        # --- W&B logging ---
        log_dict = {
            'epoch': epoch,
            'train/ce_loss': running_ce / num_batches,
            'train/sparsity_loss': running_sp / num_batches,
            'train/sparsity_coeff': coeff,
            'eval/accuracy': test_acc,
            'eval/loss': test_loss,
            'compression/structured_bops_ratio': result.structured_bops_ratio,
            'compression/structured_mac_ratio': result.structured_mac_ratio,
            'compression/unstructured_bops_ratio': result.unstructured_bops_ratio,
            'compression/unstructured_mac_ratio': result.unstructured_mac_ratio,
        }

        # Per-layer bitwidth and sparsity
        for layer_stat in result.structured_stats.layers:
            log_dict[f'layers/{layer_stat.name}/bitwidth'] = layer_stat.bitwidth
            log_dict[f'layers/{layer_stat.name}/structured_pruning'] = layer_stat.pruning_ratio
        for layer_stat in result.unstructured_stats.layers:
            log_dict[f'layers/{layer_stat.name}/unstructured_pruning'] = layer_stat.pruning_ratio

        # Visualization images at intervals
        if epoch % config.viz_interval == 0 or epoch == config.epochs - 1:
            fig_liveness = evaluator.plot_channel_liveness(model)
            log_dict['viz/channel_liveness'] = wandb.Image(fig_liveness)
            plt.close(fig_liveness)

            fig_bars = evaluator.plot_layer_sparsity_bars(result)
            log_dict['viz/sparsity_bars'] = wandb.Image(fig_bars)
            plt.close(fig_bars)

        wandb.log(log_dict)

        print(
            f"Epoch {epoch+1}/{config.epochs} | "
            f"CE: {running_ce / num_batches:.4f} | "
            f"Acc: {test_acc:.4f} | "
            f"Struct BOPs: {result.structured_bops_ratio:.2f}x | "
            f"Unstruct BOPs: {result.unstructured_bops_ratio:.2f}x"
        )

    wandb.finish()
    return model
```

**Step 2: Add missing import at top of trainer.py**

Note: the `plt` import is needed for closing figures. Add `import matplotlib.pyplot as plt` after the other imports.

**Step 3: Commit**

```bash
git add experiments/trainer.py
git commit -m "feat: add training loop with W&B logging and compression tracking"
```

---

### Task 7: Create experiments/run.py

**Files:**
- Create: `experiments/run.py`

**Step 1: Create the entry point**

```python
"""Entry point for running experiments.

Define experiment configs here and run them.
"""
from experiments.config import ExperimentConfig
from experiments.sparsity import group_lasso_channels, group_lasso_blocks, l1_kernel_sparsity
from experiments.schedules import linear_warmup, cosine_anneal, constant
from experiments.trainer import run_experiment


# --- Example experiment configs ---

baseline = ExperimentConfig(
    name="baseline-no-sparsity",
)

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


if __name__ == "__main__":
    # Run a single experiment
    run_experiment(baseline)
```

**Step 2: Commit**

```bash
git add experiments/run.py
git commit -m "feat: add experiment runner entry point with example configs"
```

---

### Task 8: Update requirements.txt and smoke test

**Files:**
- Modify: `requirements.txt`

**Step 1: Update requirements.txt**

Add `wandb` to the requirements:

```
torch
torchvision
numpy
matplotlib
wandb
```

**Step 2: Install wandb**

Run: `pip install wandb`

**Step 3: Run full test suite**

Run: `python -m pytest tests/ -v`
Expected: All tests PASS

**Step 4: Smoke test the import chain**

Run: `python -c "from experiments.config import ExperimentConfig; from experiments.sparsity import group_lasso_channels; from experiments.schedules import linear_warmup; from experiments.evaluator import Evaluator; print('All imports OK')"`
Expected: `All imports OK`

**Step 5: Commit**

```bash
git add requirements.txt
git commit -m "chore: add wandb dependency"
```
