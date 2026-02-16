# BOPs Calculator Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement `QuantizationComparison` in `metrics.py` that computes per-layer MACs, BOPs, and compression ratios between two models, with correct null-channel tracking through skip connections.

**Architecture:** Walk Conv2d/Linear layers to extract quantized weights and bitwidths via `FakeQuantParametrization` / `DeadZoneLDZCompander`. Track null (all-zero) channels through the network, computing the intersection at residual adds. Use forward-pass hooks for spatial output dimensions. b_a = 32 globally (activations not quantized, matching CoDeQ paper).

**Tech Stack:** PyTorch, `src/quantizer.py` (DeadZoneLDZCompander), `utils_quantization.py` (FakeQuantParametrization)

---

## Key Formulas

Per-layer:
```
effective_MACs_l = nnz_weights(W_l, null_input_channels_l) * m_h * m_w
BOPs_l = effective_MACs_l * b_w_l * 32
```

Where `nnz_weights(W_l, null_in)` counts non-zero weights in `W_l` after masking out columns corresponding to null input channels. For Linear layers, `m_h = m_w = 1`.

Null channel tracking at residual add:
```
null_channels_after_add = null_channels_residual_branch ∩ null_channels_shortcut
```

Compression ratio:
```
ratio = total_BOPs_baseline / total_BOPs_compressed
```

## Key Design Decision: Null Channel Tracking at Skip Connections

A channel after a residual add is only null if it is null in **both** the residual branch output and the shortcut path. This matters because:

- **Identity shortcut** (`Sequential()`): null channels = same as block input null channels (identity preserves zeros)
- **LambdaLayer shortcut** (option A, CIFAR ResNet): `F.pad(x[:, :, ::2, ::2], (0,0,0,0, planes//4, planes//4), "constant", 0)` — the first `planes//4` and last `planes//4` channels are always null (zero-padded); middle channels inherit null status from block input

Example: if `layer1.0.conv2` weights are all zero, ALL its output channels are null. But the identity shortcut passes the block input through unchanged (no null channels). The intersection is empty — so `layer1.1.conv1` sees `p_in = 0` (no null input channels). The shortcut "restores" the channels.

---

## Task 1: Layer quantization info extraction

**Files:**
- Modify: `metrics.py`

**Step 1: Write `_get_layer_quant_info` helper**

Extracts bitwidth and quantized weight tensor from a single module by reading the `FakeQuantParametrization` wrapper.

```python
import torch
import torch.nn as nn


def _get_layer_quant_info(module):
    """Extract (bitwidth, quantized_weight) from a module.

    If the module has a FakeQuantParametrization on its weight, reads b_w from
    the DeadZoneLDZCompander and gets the quantized weight by accessing
    module.weight (which triggers the parametrization forward pass).

    If not parametrized, returns (32.0, module.weight).
    """
    if hasattr(module, 'parametrizations') and hasattr(module.parametrizations, 'weight'):
        fq = module.parametrizations.weight[0]  # FakeQuantParametrization
        b_w = fq.quantizer.get_bitwidth().item()
        w_quant = module.weight  # triggers quantizer forward
    else:
        b_w = 32.0
        w_quant = module.weight
    return b_w, w_quant.detach()
```

**Step 2: Write `_get_spatial_dims` using forward hooks**

Registers temporary forward hooks on all Conv2d/Linear modules, runs a single forward pass with a dummy tensor, captures output spatial dimensions, then removes hooks.

```python
def _get_spatial_dims(model, input_size, device='cpu'):
    """Run one forward pass and return {layer_name: (m_h, m_w)} for Conv2d/Linear layers."""
    spatial = {}
    hooks = []

    def make_hook(name):
        def hook(module, inp, out):
            if isinstance(module, nn.Conv2d):
                spatial[name] = (out.shape[2], out.shape[3])
            elif isinstance(module, nn.Linear):
                spatial[name] = (1, 1)
        return hook

    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            hooks.append(module.register_forward_hook(make_hook(name)))

    with torch.no_grad():
        dummy = torch.zeros(1, *input_size, device=device)
        model(dummy)

    for h in hooks:
        h.remove()

    return spatial
```

**Step 3: Verify manually**

Run in notebook:
```python
from metrics import _get_layer_quant_info, _get_spatial_dims
spatial = _get_spatial_dims(compressed_model, (3, 32, 32))
for name, dims in spatial.items():
    print(f"{name}: {dims}")
```

Expected: `conv1: (32, 32)`, `layer1.0.conv1: (32, 32)`, ..., `layer2.0.conv1: (16, 16)`, ..., `layer3.0.conv1: (8, 8)`, `linear: (1, 1)`.

**Step 4: Commit**

```bash
git add metrics.py
git commit -m "feat(metrics): add layer quant info and spatial dim extraction helpers"
```

---

## Task 2: Null channel computation primitives

**Files:**
- Modify: `metrics.py`

**Step 1: Write `_compute_null_output_channels`**

Given a quantized weight tensor and a set of null input channel indices, determines which output channels are null (produce zero output regardless of input). An output channel is null when all its weights for non-null input channels are zero.

```python
def _compute_null_output_channels(weight, null_input_channels):
    """Return set of output channel indices that are null.

    An output channel is null if all weights connecting it to non-null
    input channels are zero.

    Args:
        weight: quantized weight tensor, shape (c_out, c_in, ...) for Conv2d
                or (out_features, in_features) for Linear
        null_input_channels: set of input channel indices known to be null
    """
    c_out, c_in = weight.shape[0], weight.shape[1]

    # Mask for valid (non-null) input channels
    valid_mask = torch.ones(c_in, dtype=torch.bool, device=weight.device)
    if null_input_channels:
        valid_mask[list(null_input_channels)] = False

    # Check each output channel: are all weights for valid inputs zero?
    w_valid = weight[:, valid_mask]                # (c_out, valid_c_in, ...)
    per_out_nnz = (w_valid != 0).flatten(1).sum(dim=1)  # (c_out,)

    return set(torch.where(per_out_nnz == 0)[0].tolist())
```

**Step 2: Write `_compute_effective_macs`**

Counts non-zero weights after masking out null input channels, then multiplies by spatial output size.

```python
def _compute_effective_macs(weight, null_input_channels, m_h, m_w):
    """Compute effective MACs for a layer, excluding null input channels.

    Each non-zero weight element contributes m_h * m_w MACs (for Conv2d)
    or 1 MAC (for Linear, where m_h = m_w = 1).

    Args:
        weight: quantized weight tensor
        null_input_channels: set of null input channel indices
        m_h, m_w: spatial output dimensions
    """
    c_in = weight.shape[1]

    valid_mask = torch.ones(c_in, dtype=torch.bool, device=weight.device)
    if null_input_channels:
        valid_mask[list(null_input_channels)] = False

    w_valid = weight[:, valid_mask]
    nnz = (w_valid != 0).sum().item()

    return nnz * m_h * m_w
```

**Step 3: Verify manually**

```python
# A layer with all zeros should have 0 MACs
w_zero = torch.zeros(16, 16, 3, 3)
assert _compute_effective_macs(w_zero, set(), 32, 32) == 0
assert _compute_null_output_channels(w_zero, set()) == set(range(16))

# A dense layer with no null inputs
w_dense = torch.ones(16, 16, 3, 3)
assert _compute_effective_macs(w_dense, set(), 32, 32) == 16*16*3*3 * 32*32
assert _compute_null_output_channels(w_dense, set()) == set()

# Null input channels should be excluded
assert _compute_effective_macs(w_dense, {0, 1, 2, 3}, 32, 32) == 16*12*3*3 * 32*32
```

**Step 4: Commit**

```bash
git add metrics.py
git commit -m "feat(metrics): add null channel and effective MACs computation"
```

---

## Task 3: Null channel tracking through CIFAR ResNet

**Files:**
- Modify: `metrics.py`

This is the critical algorithm. It walks the CIFAR ResNet structure (conv1 -> layer1/2/3 -> linear) and tracks which channels are null at each point, correctly handling the intersection at residual adds.

**Step 1: Write `_track_null_channels_resnet`**

```python
from resnet import LambdaLayer


def _track_null_channels_resnet(model, layer_info):
    """Track null channels through a CIFAR ResNet (resnet.py architecture).

    Walks conv1 -> layer1 -> layer2 -> layer3 -> linear, tracking which
    channels are null (all-zero) at each point. At residual adds, null
    channels = intersection of residual branch and shortcut path.

    Args:
        model: ResNet model instance (from resnet.py)
        layer_info: dict {layer_name: (bitwidth, quantized_weight)} from
                    _get_layer_quant_info for each Conv2d/Linear

    Returns:
        dict {layer_name: set of null input channel indices} for each
        Conv2d/Linear layer
    """
    null_input_map = {}

    # conv1: input is RGB image, no null channels
    current_null = set()
    null_input_map['conv1'] = current_null.copy()

    # After conv1: which output channels are null?
    _, w_conv1 = layer_info['conv1']
    current_null = _compute_null_output_channels(w_conv1, current_null)

    # Walk layer1, layer2, layer3
    for layer_group_name in ['layer1', 'layer2', 'layer3']:
        layer_group = getattr(model, layer_group_name)

        for block_idx in range(len(layer_group)):
            prefix = f'{layer_group_name}.{block_idx}'
            block = layer_group[block_idx]

            # Save block input null channels (for shortcut path)
            block_input_null = current_null.copy()

            # --- Residual branch ---

            # conv1 in block
            conv1_name = f'{prefix}.conv1'
            null_input_map[conv1_name] = current_null.copy()
            _, w = layer_info[conv1_name]
            current_null = _compute_null_output_channels(w, current_null)

            # conv2 in block
            conv2_name = f'{prefix}.conv2'
            null_input_map[conv2_name] = current_null.copy()
            _, w = layer_info[conv2_name]
            residual_null = _compute_null_output_channels(w, current_null)

            # --- Shortcut path ---
            shortcut = block.shortcut

            if isinstance(shortcut, LambdaLayer):
                # Option A: F.pad(x[:, :, ::2, ::2], (0,0,0,0, pad, pad), "constant", 0)
                # Expands channels: first `pad` are zero, middle are from input, last `pad` are zero
                c_out_block = w.shape[0]  # output channels of this block (from conv2)
                c_in_block = layer_info[f'{prefix}.conv1'][1].shape[1]  # input channels of block
                pad = c_out_block // 4

                shortcut_null = set()
                # First `pad` channels: always zero (padding)
                shortcut_null.update(range(0, pad))
                # Middle channels: inherit null status from block input
                for ch_idx in range(c_in_block):
                    if ch_idx in block_input_null:
                        shortcut_null.add(ch_idx + pad)
                # Last `pad` channels: always zero (padding)
                shortcut_null.update(range(pad + c_in_block, c_out_block))

            elif isinstance(shortcut, nn.Sequential) and len(shortcut) == 0:
                # Identity shortcut: null channels pass through unchanged
                shortcut_null = block_input_null

            else:
                # Unknown shortcut type: conservatively assume no null channels
                shortcut_null = set()

            # --- Residual add: intersection ---
            current_null = residual_null & shortcut_null

    # Linear layer
    null_input_map['linear'] = current_null.copy()

    return null_input_map
```

**Step 2: Verify with the skip-connection test from the notebook**

Zero out `layer1.0.conv2` weights and confirm that `layer1.1.conv1` still sees `p_in = 0` because the identity shortcut restores channels:

```python
null_map = _track_null_channels_resnet(compressed_model, layer_info)

# layer1.0.conv2 output: all channels null (weights zeroed)
# shortcut: identity, passes block input (no null channels)
# intersection: empty -> layer1.1.conv1 sees no null input channels
assert len(null_map['layer1.1.conv1']) == 0
```

**Step 3: Commit**

```bash
git add metrics.py
git commit -m "feat(metrics): null channel tracking through ResNet skip connections"
```

---

## Task 4: QuantizationComparison class

**Files:**
- Modify: `metrics.py`

Ties everything together into the main class.

**Step 1: Write `_LayerStats` dataclass and `_profile_model` helper**

```python
from dataclasses import dataclass


@dataclass
class LayerStats:
    name: str
    macs: int
    bops: float
    bitwidth: float
    pruning_ratio: float         # P_l: fraction of zero weights overall
    null_out_channels: int       # p_out count
    null_in_channels: int        # p_in count
    total_out_channels: int
    total_in_channels: int
    spatial: tuple               # (m_h, m_w)
    kernel_size: tuple           # (k_h, k_w) or (1, 1) for Linear


@dataclass
class ModelStats:
    layers: list                 # list of LayerStats
    total_macs: int
    total_bops: float
```

**Step 2: Write `QuantizationComparison` class**

```python
class QuantizationComparison:
    def __init__(self, input_size=(3, 32, 32), b_a=32):
        """
        Args:
            input_size: CHW tuple for the model input (no batch dim)
            b_a: activation bitwidth (32 for CoDeQ, activations not quantized)
        """
        self.input_size = input_size
        self.b_a = b_a
        self.baseline_stats = None

    def set_baseline(self, model):
        """Profile the baseline model and store its stats."""
        self.baseline_stats = self._profile_model(model)

    def calculate_stats(self, model):
        """Profile a model and compute compression ratio vs baseline.

        Args:
            model: nn.Module to profile (may or may not have quantizers)

        Returns:
            ModelStats with per-layer and total MACs/BOPs

        Raises:
            AssertionError if baseline is set and architectures don't match
        """
        stats = self._profile_model(model)

        if self.baseline_stats is not None:
            self._assert_compatible(self.baseline_stats, stats)

        return stats

    def compression_ratio(self, compressed_stats):
        """Compute BOPs compression ratio: baseline_BOPs / compressed_BOPs."""
        assert self.baseline_stats is not None, "Call set_baseline() first"
        return self.baseline_stats.total_bops / compressed_stats.total_bops

    def mac_compression_ratio(self, compressed_stats):
        """Compute MAC compression ratio: baseline_MACs / compressed_MACs."""
        assert self.baseline_stats is not None, "Call set_baseline() first"
        return self.baseline_stats.total_macs / compressed_stats.total_macs

    def _profile_model(self, model):
        model.eval()

        # 1. Spatial dims via forward pass
        spatial = _get_spatial_dims(model, self.input_size)

        # 2. Per-layer quant info
        layer_info = {}
        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                with torch.no_grad():
                    layer_info[name] = _get_layer_quant_info(module)

        # 3. Null channel tracking
        null_input_map = _track_null_channels_resnet(model, layer_info)

        # 4. Per-layer stats
        layers = []
        total_macs = 0
        total_bops = 0.0

        for name, (b_w, w_quant) in layer_info.items():
            if name not in spatial:
                continue

            m_h, m_w = spatial[name]
            null_in = null_input_map.get(name, set())

            macs = _compute_effective_macs(w_quant, null_in, m_h, m_w)
            bops = macs * b_w * self.b_a

            null_out = _compute_null_output_channels(w_quant, null_in)

            c_out, c_in = w_quant.shape[0], w_quant.shape[1]
            p_l = (w_quant == 0).float().mean().item()

            if isinstance(dict(model.named_modules())[name], nn.Conv2d):
                k_size = (w_quant.shape[2], w_quant.shape[3])
            else:
                k_size = (1, 1)

            layers.append(LayerStats(
                name=name,
                macs=macs,
                bops=bops,
                bitwidth=b_w,
                pruning_ratio=p_l,
                null_out_channels=len(null_out),
                null_in_channels=len(null_in),
                total_out_channels=c_out,
                total_in_channels=c_in,
                spatial=(m_h, m_w),
                kernel_size=k_size,
            ))

            total_macs += macs
            total_bops += bops

        return ModelStats(layers=layers, total_macs=total_macs, total_bops=total_bops)

    @staticmethod
    def _assert_compatible(baseline, compressed):
        bl = {l.name for l in baseline.layers}
        cl = {l.name for l in compressed.layers}
        assert bl == cl, (
            f"Architecture mismatch. "
            f"In baseline only: {bl - cl}, in compressed only: {cl - bl}"
        )
```

**Step 3: Commit**

```bash
git add metrics.py
git commit -m "feat(metrics): QuantizationComparison class with BOPs and compression ratio"
```

---

## Task 5: Print / comparison utilities

**Files:**
- Modify: `metrics.py`

Match the output format shown in the notebook (cell 9 and cell 11).

**Step 1: Write `print_model_stats`**

```python
def print_model_stats(stats):
    """Print per-layer stats table matching notebook format."""
    header = (f"{'Layer':<45} {'Shape':>20} {'b_w':>5} {'p_out':>6} {'p_in':>6} "
              f"{'P_l':>6} {'MACs':>12} {'BOPs':>14}")
    print(header)
    print("-" * len(header))

    for l in stats.layers:
        shape_str = (f"{l.total_out_channels}x{l.total_in_channels}x"
                     f"{l.kernel_size[0]}x{l.kernel_size[1]} -> "
                     f"{l.spatial[0]}x{l.spatial[1]}")
        p_out = l.null_out_channels / l.total_out_channels if l.total_out_channels else 0
        p_in = l.null_in_channels / l.total_in_channels if l.total_in_channels else 0

        print(f"{l.name:<45} {shape_str:>20} {l.bitwidth:>5.1f} {p_out:>6.3f} {p_in:>6.3f} "
              f"{l.pruning_ratio:>6.3f} {l.macs:>12} {l.bops:>14.0f}")

    print("-" * len(header))
    print(f"{'TOTAL':<45} {'':>20} {'':>5} {'':>6} {'':>6} "
          f"{'':>6} {stats.total_macs:>12} {stats.total_bops:>14.0f}")
```

**Step 2: Write `print_comparison`**

```python
def print_comparison(baseline_stats, compressed_stats):
    """Print side-by-side comparison of baseline vs compressed."""
    header = (f"{'Layer':<45} {'MACs(base)':>12} {'MACs(comp)':>12} "
              f"{'BOPs(base)':>14} {'BOPs(comp)':>14}")
    print(header)
    print("-" * len(header))

    bl_map = {l.name: l for l in baseline_stats.layers}
    for l in compressed_stats.layers:
        bl = bl_map[l.name]
        print(f"{l.name:<45} {bl.macs:>12} {l.macs:>12} "
              f"{bl.bops:>14.0f} {l.bops:>14.0f}")

    print("-" * len(header))
    print(f"{'TOTAL':<45} {baseline_stats.total_macs:>12} {compressed_stats.total_macs:>12} "
          f"{baseline_stats.total_bops:>14.0f} {compressed_stats.total_bops:>14.0f}")
    print()

    if baseline_stats.total_macs > 0 and compressed_stats.total_macs > 0:
        print(f"MAC compression ratio: {baseline_stats.total_macs / compressed_stats.total_macs:.4f}x")
    if baseline_stats.total_bops > 0 and compressed_stats.total_bops > 0:
        print(f"BOP compression ratio: {baseline_stats.total_bops / compressed_stats.total_bops:.4f}x")
```

**Step 3: Commit**

```bash
git add metrics.py
git commit -m "feat(metrics): add print_model_stats and print_comparison utilities"
```

---

## Task 6: Simple toy-network BOPs tests

**Files:**
- Create: `tests/test_bops.py`

Hand-verifiable tests using tiny networks where we can compute expected MACs/BOPs on paper. These do NOT use the ResNet null-channel tracker — they test the core primitives (`_compute_effective_macs`, `_compute_null_output_channels`) and a simple sequential profiling path.

**Step 1: Write `test_single_conv_dense` — one Conv2d, all weights = 1**

A single 3x3 conv: 2 input channels, 4 output channels, input 4x4.

```
Weight shape: (4, 2, 3, 3)
All weights = 1.0 (dense, no sparsity)
Output spatial: floor((4 + 2*1 - 3)/1) + 1 = 4  →  4x4

Expected MACs = nnz * m_h * m_w
             = (4 * 2 * 3 * 3) * 4 * 4
             = 72 * 16 = 1152

b_w = 32 (no quantizer), b_a = 32
Expected BOPs = 1152 * 32 * 32 = 1,179,648
```

```python
import torch
import torch.nn as nn
from metrics import (
    _get_layer_quant_info,
    _get_spatial_dims,
    _compute_effective_macs,
    _compute_null_output_channels,
)


class SingleConv(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 4, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        return self.conv(x)


def test_single_conv_dense():
    model = SingleConv()
    nn.init.ones_(model.conv.weight)

    b_w, w = _get_layer_quant_info(model.conv)
    spatial = _get_spatial_dims(model, (2, 4, 4))

    assert b_w == 32.0
    assert spatial['conv'] == (4, 4)

    macs = _compute_effective_macs(w, set(), 4, 4)
    assert macs == 4 * 2 * 3 * 3 * 4 * 4  # 1152

    bops = macs * b_w * 32
    assert bops == 1_179_648

    null_out = _compute_null_output_channels(w, set())
    assert null_out == set()  # no null channels, all weights are 1
```

**Step 2: Write `test_single_conv_half_sparse` — half the weights zeroed**

Same architecture, but zero out 2 of 4 output channels entirely.

```
Weight shape: (4, 2, 3, 3)
Channels 0,1: all weights = 1.0
Channels 2,3: all weights = 0.0

Expected null_out = {2, 3}
Expected nnz = 2 * 2 * 3 * 3 = 36  (only channels 0,1 contribute)
Expected MACs = 36 * 4 * 4 = 576
Expected BOPs = 576 * 32 * 32 = 589,824
```

```python
def test_single_conv_half_sparse():
    model = SingleConv()
    nn.init.ones_(model.conv.weight)
    with torch.no_grad():
        model.conv.weight[2:, :, :, :] = 0.0  # zero out channels 2,3

    b_w, w = _get_layer_quant_info(model.conv)
    macs = _compute_effective_macs(w, set(), 4, 4)

    assert macs == 2 * 2 * 3 * 3 * 4 * 4  # 576

    null_out = _compute_null_output_channels(w, set())
    assert null_out == {2, 3}
```

**Step 3: Write `test_two_conv_null_propagation` — null channels propagate**

Two sequential convs. Zero out all output channels of conv1 → conv2 should see all input channels as null → 0 effective MACs.

```python
class TwoConvSeq(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(2, 4, kernel_size=3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(4, 8, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        return self.conv2(self.conv1(x))


def test_two_conv_null_propagation():
    model = TwoConvSeq()
    nn.init.ones_(model.conv1.weight)
    nn.init.ones_(model.conv2.weight)

    # Zero out ALL of conv1 → all 4 output channels are null
    with torch.no_grad():
        model.conv1.weight.zero_()

    _, w1 = _get_layer_quant_info(model.conv1)
    _, w2 = _get_layer_quant_info(model.conv2)

    null_after_conv1 = _compute_null_output_channels(w1, set())
    assert null_after_conv1 == {0, 1, 2, 3}  # all 4 channels null

    # conv2 receives 4 null input channels → even though its weights are 1,
    # all input channels are null so effective MACs = 0
    macs_conv2 = _compute_effective_macs(w2, null_after_conv1, 4, 4)
    assert macs_conv2 == 0

    null_after_conv2 = _compute_null_output_channels(w2, null_after_conv1)
    assert null_after_conv2 == set(range(8))  # all 8 output channels null
```

**Step 4: Write `test_two_conv_partial_null` — some null input channels reduce MACs**

Zero out 2 of 4 output channels of conv1. Conv2 should only count MACs from the 2 non-null input channels.

```
conv1: (4, 2, 3, 3), channels 2,3 zeroed → null_out = {2, 3}
conv2: (8, 4, 3, 3), all weights = 1.0, null_in = {2, 3}

conv2 effective weights: 8 output * 2 valid input * 3 * 3 = 144 non-zero
Expected MACs = 144 * 4 * 4 = 2304
(vs dense: 8 * 4 * 3 * 3 * 16 = 4608 — exactly half)
```

```python
def test_two_conv_partial_null():
    model = TwoConvSeq()
    nn.init.ones_(model.conv1.weight)
    nn.init.ones_(model.conv2.weight)

    with torch.no_grad():
        model.conv1.weight[2:, :, :, :] = 0.0  # null channels 2,3

    _, w1 = _get_layer_quant_info(model.conv1)
    _, w2 = _get_layer_quant_info(model.conv2)

    null_after_conv1 = _compute_null_output_channels(w1, set())
    assert null_after_conv1 == {2, 3}

    macs_conv2 = _compute_effective_macs(w2, null_after_conv1, 4, 4)
    assert macs_conv2 == 8 * 2 * 3 * 3 * 4 * 4  # 2304 (half of dense)
```

**Step 5: Write `test_linear_layer` — verify Linear MACs (m_h = m_w = 1)**

```
Linear: in_features=4, out_features=8, all weights = 1.0
Expected MACs = nnz * 1 * 1 = 4 * 8 = 32
Expected BOPs = 32 * 32 * 32 = 32,768
```

```python
class SingleLinear(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 8, bias=False)

    def forward(self, x):
        return self.fc(x)


def test_linear_layer():
    model = SingleLinear()
    nn.init.ones_(model.fc.weight)

    b_w, w = _get_layer_quant_info(model.fc)
    spatial = _get_spatial_dims(model, (4,))

    assert spatial['fc'] == (1, 1)
    assert b_w == 32.0

    macs = _compute_effective_macs(w, set(), 1, 1)
    assert macs == 4 * 8  # 32

    bops = macs * b_w * 32
    assert bops == 32_768
```

**Step 6: Write `test_compression_ratio_simple` — verify ratio calculation**

Two identical SingleConv models. Baseline is dense (b_w=32). "Compressed" has half its weights zeroed.

```
Baseline MACs  = 72 * 16 = 1152,  BOPs = 1152 * 32 * 32 = 1,179,648
Compressed MACs = 36 * 16 = 576,  BOPs = 576 * 32 * 32 = 589,824

MAC ratio = 1152 / 576 = 2.0x
BOP ratio = 1,179,648 / 589,824 = 2.0x
(Same ratio because both are b_w=32 — the ratio only diverges with mixed bitwidths)
```

```python
from metrics import QuantizationComparison


def test_compression_ratio_simple():
    """Verify compression ratio with a trivial model.

    Note: QuantizationComparison._profile_model calls
    _track_null_channels_resnet which expects a CIFAR ResNet structure.
    For these toy networks we test the primitives directly, not the
    full class. This test is a placeholder — the class is tested with
    real ResNets in the notebook (Task 7).
    """
    # Test the math directly
    baseline_bops = 1152 * 32 * 32  # dense, FP32
    compressed_bops = 576 * 32 * 32  # half weights zeroed, still FP32

    ratio = baseline_bops / compressed_bops
    assert ratio == 2.0
```

**Step 7: Write `test_skip_identity_restores_channels` — identity shortcut prevents null propagation**

Uses a minimal CIFAR ResNet (1 block per group) from `resnet.py`. Zero out `layer1.0.conv2` — the identity shortcut should restore all channels so the next layer sees no null inputs.

```
Architecture: ResNet(BasicBlock, [1, 1, 1])
  conv1: (16, 3, 3, 3)        → 32x32
  layer1.0.conv1: (16, 16, 3, 3) → 32x32
  layer1.0.conv2: (16, 16, 3, 3) → 32x32  ← ZEROED
  layer1.0.shortcut: identity (Sequential())
  layer2.0.conv1: (32, 16, 3, 3) → 16x16  ← should see 0 null input channels

After layer1.0.conv2 (zeroed): residual_null = {0..15} (all 16 channels)
Shortcut is identity: shortcut_null = {} (conv1 output was dense)
Intersection: {} ∩ {0..15} = {}
→ layer2.0.conv1 null_in = {} (0 null channels)
```

```python
from resnet import ResNet, BasicBlock
from metrics import _get_layer_quant_info, _track_null_channels_resnet


def test_skip_identity_restores_channels():
    """Identity shortcut restores channels even when residual branch is fully zeroed."""
    model = ResNet(BasicBlock, [1, 1, 1])  # minimal: 1 block per group

    # Collect layer info
    layer_info = {}
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            with torch.no_grad():
                layer_info[name] = _get_layer_quant_info(module)

    # Zero out layer1.0.conv2
    with torch.no_grad():
        model.layer1[0].conv2.weight.zero_()
        layer_info['layer1.0.conv2'] = _get_layer_quant_info(model.layer1[0].conv2)

    null_map = _track_null_channels_resnet(model, layer_info)

    # layer1.0.conv2 should see some null input (from conv1 output analysis)
    # but the KEY check: after the residual add, identity shortcut restores channels
    assert len(null_map['layer2.0.conv1']) == 0, (
        "Identity shortcut should restore all channels — "
        f"got {len(null_map['layer2.0.conv1'])} null inputs"
    )
```

**Step 8: Write `test_skip_lambda_padded_channels_are_null` — LambdaLayer shortcut null padding**

At `layer2.0`, the shortcut is a LambdaLayer that pads 16→32 channels. Zero out `layer2.0.conv2` — the padded channels (0..7 and 24..31) should remain null since they're zero in both paths, but the middle 16 channels (from the shortcut) should not be null.

```
layer2.0 shortcut: LambdaLayer, 16 → 32 channels
  pad = 32 // 4 = 8
  Shortcut output: [8 zero-padded | 16 from input | 8 zero-padded]

layer2.0.conv2 ZEROED: residual_null = {0..31} (all 32 channels)
shortcut_null = {0..7} ∪ {24..31} = 16 channels  (padded ones)
  (channels 8..23 are from block input, which is dense → not null)

Intersection: {0..7, 24..31} ∩ {0..31} = {0..7, 24..31} = 16 null channels

→ layer2.1.conv1 (or next layer) sees 16 null input channels out of 32
```

```python
def test_skip_lambda_padded_channels_are_null():
    """LambdaLayer shortcut: only the zero-padded channels remain null after intersection."""
    model = ResNet(BasicBlock, [1, 1, 1])

    layer_info = {}
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            with torch.no_grad():
                layer_info[name] = _get_layer_quant_info(module)

    # Zero out layer2.0.conv2 (this block has LambdaLayer shortcut)
    with torch.no_grad():
        model.layer2[0].conv2.weight.zero_()
        layer_info['layer2.0.conv2'] = _get_layer_quant_info(model.layer2[0].conv2)

    null_map = _track_null_channels_resnet(model, layer_info)

    # Next layer after layer2.0 is layer3.0.conv1 (since [1,1,1] → 1 block per group)
    null_in_next = null_map['layer3.0.conv1']

    # Expect 16 null channels: the 8 front-padded + 8 back-padded from LambdaLayer
    assert len(null_in_next) == 16, (
        f"Expected 16 null channels (padded), got {len(null_in_next)}"
    )
    # Specifically channels 0..7 and 24..31
    expected_null = set(range(0, 8)) | set(range(24, 32))
    assert null_in_next == expected_null, (
        f"Expected null channels {expected_null}, got {null_in_next}"
    )
```

**Step 9: Write `test_skip_both_paths_null_intersection` — null in both paths**

If the block input already has null channels, AND the residual branch produces null channels, only the intersection should be null after the add.

```
Setup: zero out conv1 channels 0..7 (of 16) → 8 null channels enter layer1.0
  layer1.0.conv1 sees 8 null inputs (channels 0..7)
  layer1.0.conv2 also zeroed completely → residual_null = {0..15}
  identity shortcut: shortcut_null = {0..7} (inherited from block input)

  Intersection: {0..15} ∩ {0..7} = {0..7}
  → next layer sees 8 null input channels (NOT 0, NOT 16)
```

```python
def test_skip_both_paths_null_intersection():
    """When both paths have null channels, only the intersection propagates."""
    model = ResNet(BasicBlock, [1, 1, 1])

    # Zero out half of conv1's output channels (channels 0..7)
    with torch.no_grad():
        model.conv1.weight[:8, :, :, :] = 0.0

    # Zero out all of layer1.0.conv2
    with torch.no_grad():
        model.layer1[0].conv2.weight.zero_()

    layer_info = {}
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            with torch.no_grad():
                layer_info[name] = _get_layer_quant_info(module)

    null_map = _track_null_channels_resnet(model, layer_info)

    # After conv1: channels 0..7 are null
    null_after_conv1 = _compute_null_output_channels(layer_info['conv1'][1], set())
    assert null_after_conv1 == set(range(8))

    # After layer1.0 residual add:
    #   residual branch (conv2 zeroed): all 16 null
    #   identity shortcut: channels 0..7 null (from conv1)
    #   intersection: {0..7}
    null_in_next = null_map['layer2.0.conv1']
    assert null_in_next == set(range(8)), (
        f"Expected null channels {{0..7}} (intersection), got {null_in_next}"
    )
```

**Step 10: Write `test_skip_dense_shortcut_clears_all_null` — dense shortcut overrides**

If the shortcut path has no null channels (dense input, identity shortcut), then even if the residual branch is completely null, the intersection is empty.

```python
def test_skip_dense_shortcut_clears_all_null():
    """A dense identity shortcut means intersection with any residual null is empty."""
    model = ResNet(BasicBlock, [1, 1, 1])
    # conv1 is dense (random init), so block input has no null channels

    # Zero only layer1.0.conv2 (residual branch all null)
    with torch.no_grad():
        model.layer1[0].conv2.weight.zero_()

    layer_info = {}
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            with torch.no_grad():
                layer_info[name] = _get_layer_quant_info(module)

    null_map = _track_null_channels_resnet(model, layer_info)

    # conv1 is dense → block input has no null channels
    # identity shortcut: shortcut_null = {} (empty)
    # residual: all null
    # intersection: {} → no null channels
    assert len(null_map['layer2.0.conv1']) == 0
```

**Step 11: Run all tests**

```bash
python -m pytest tests/test_bops.py -v
```

Expected: all 10 tests pass.

**Step 12: Commit**

```bash
git add tests/test_bops.py
git commit -m "test(metrics): add hand-verifiable toy-network BOPs and skip-connection tests"
```

---

## Task 7: Validate with real ResNet in notebook

**Files:**
- Modify: `test_compression.ipynb`

These tests exercise the full pipeline including `_track_null_channels_resnet` and `QuantizationComparison` against the actual CIFAR ResNet20 architecture.

**Step 1: Update notebook imports and run end-to-end**

Update cell 1 to use new class name:
```python
from metrics import QuantizationComparison, print_model_stats, print_comparison
```

Update cell 5:
```python
INPUT_SIZE = (3, 32, 32)
qc = QuantizationComparison(input_size=INPUT_SIZE)
qc.set_baseline(baseline_model)
compressed_stats = qc.calculate_stats(compressed_model)
baseline_stats = qc.baseline_stats
```

**Step 2: Verify expected outputs**

- All baseline layers should have `b_w = 32.0`, `p_out = 0`, `p_in = 0`, `P_l = 0`
- Compressed layers should show learned bitwidths and sparsity
- `conv1` spatial should be `(32, 32)`, `layer2.0.conv1` should be `(16, 16)`, etc.
- BOP compression ratio should be > 1 (compressed uses fewer BOPs)

**Step 3: Run skip-connection stress test**

Zero out `layer1.0.conv2` weights and verify:
```python
with torch.no_grad():
    compressed_model.layer1[0].conv2.parametrizations.weight.original.zero_()

pruned_stats = qc.calculate_stats(compressed_model)
conv2 = next(l for l in pruned_stats.layers if l.name == "layer1.0.conv2")
next_conv = next(l for l in pruned_stats.layers if l.name == "layer1.1.conv1")

assert conv2.macs == 0, "Zeroed layer should have 0 MACs"
assert conv2.null_out_channels == conv2.total_out_channels, "All output channels should be null"
assert next_conv.null_in_channels == 0, "Identity shortcut should restore all channels"
```

Expected: `layer1.0.conv2` has 0 MACs, but `layer1.1.conv1` has full MACs because the identity shortcut restores all channels (intersection of {all channels} and {} = {}).

**Step 4: Run LambdaLayer shortcut test**

Zero out `layer2.0.conv2` (which has a LambdaLayer shortcut at `layer2.0`):
```python
with torch.no_grad():
    compressed_model.layer2[0].conv2.parametrizations.weight.original.zero_()

pruned_stats2 = qc.calculate_stats(compressed_model)
conv2 = next(l for l in pruned_stats2.layers if l.name == "layer2.0.conv2")
next_conv = next(l for l in pruned_stats2.layers if l.name == "layer2.1.conv1")

# layer2.0 has LambdaLayer shortcut: 16->32 channels
# pad = 32//4 = 8, so channels 0..7 and 24..31 are zero-padded (always null)
# channels 8..23 inherit from block input (not null)
# residual branch: all 32 channels null (conv2 zeroed)
# intersection: channels 0..7 and 24..31 (the padded ones)
assert next_conv.null_in_channels == 16, "Should have 16 null input channels (the zero-padded ones)"
print(f"layer2.1.conv1 null_in = {next_conv.null_in_channels} (expected 16)")
```

**Step 5: Commit**

```bash
git add metrics.py test_compression.ipynb
git commit -m "feat(metrics): complete BOPs calculator with skip-connection-aware null channel tracking"
```
