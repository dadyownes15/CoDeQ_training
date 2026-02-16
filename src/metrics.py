import torch
import torch.nn as nn
from dataclasses import dataclass, field
from src.resnet import LambdaLayer


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class LayerQuantInfo:
    """Quantization state of a single layer's weights."""
    bitwidth: float              # b_w: weight bitwidth (32.0 if not quantized)
    weight: torch.Tensor         # quantized weight tensor (detached)


@dataclass
class NullChannels:
    """Tracks which channels in a tensor are guaranteed to be all-zero.

    A channel is "null" when the upstream computation guarantees it is always
    zero regardless of the network input.  For example, if a Conv2d layer has
    output channel 5 with all-zero weights, its output on channel 5 is always
    zero — so the *next* layer's input channel 5 is null.

    At a residual add, a channel is null only if it is null in BOTH the
    residual branch and the shortcut (set intersection).
    """
    indices: set[int] = field(default_factory=set)   # channel indices that are null
    total: int = 0                                    # total number of channels

    @property
    def count(self) -> int:
        return len(self.indices)

    @property
    def ratio(self) -> float:
        """Fraction of channels that are null (0.0 to 1.0)."""
        return self.count / self.total if self.total > 0 else 0.0

    def intersect(self, other: 'NullChannels') -> 'NullChannels':
        """Channels that are null in BOTH self and other (residual add)."""
        assert self.total == other.total, (
            f"Cannot intersect: {self.total} vs {other.total} channels"
        )
        return NullChannels(
            indices=self.indices & other.indices,
            total=self.total,
        )

    def copy(self) -> 'NullChannels':
        return NullChannels(indices=self.indices.copy(), total=self.total)

    @staticmethod
    def none(total: int) -> 'NullChannels':
        """No null channels (e.g. the RGB input to the network)."""
        return NullChannels(indices=set(), total=total)

    @staticmethod
    def all(total: int) -> 'NullChannels':
        """All channels are null."""
        return NullChannels(indices=set(range(total)), total=total)


@dataclass
class LayerStats:
    """Per-layer MACs, BOPs, and sparsity statistics."""
    name: str
    macs: int
    bops: float
    bitwidth: float
    pruning_ratio: float         # fraction of zero weights in this layer
    null_out: NullChannels       # which output channels are always zero
    null_in: NullChannels        # which input channels are always zero
    spatial: tuple               # (m_h, m_w) output feature map size
    kernel_size: tuple           # (k_h, k_w) or (1, 1) for Linear


@dataclass
class ModelStats:
    """Aggregate statistics for a full model."""
    layers: list[LayerStats]
    total_macs: int
    total_bops: float


# ---------------------------------------------------------------------------
# Layer quantization info extraction
# ---------------------------------------------------------------------------

def _get_layer_quant_info(module) -> LayerQuantInfo:
    """Extract quantization info from a module.

    If the module has a FakeQuantParametrization on its weight, reads b_w from
    the DeadZoneLDZCompander and gets the quantized weight by accessing
    module.weight (which triggers the parametrization forward pass).

    If not parametrized, returns bitwidth=32 with the raw weight.
    """
    if hasattr(module, 'parametrizations') and hasattr(module.parametrizations, 'weight'):
        fq = module.parametrizations.weight[0]  # FakeQuantParametrization
        b_w = fq.quantizer.get_bitwidth().item()
        w_quant = module.weight  # triggers quantizer forward
    else:
        b_w = 32.0
        w_quant = module.weight
    return LayerQuantInfo(bitwidth=b_w, weight=w_quant.detach())


def _get_spatial_dims(model, input_size, device='cpu') -> dict[str, tuple[int, int]]:
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


# ---------------------------------------------------------------------------
# Null channel computation
# ---------------------------------------------------------------------------

def _compute_null_output_channels(weight: torch.Tensor, null_in: NullChannels) -> NullChannels:
    """Determine which output channels are null given the weight tensor and null inputs.

    An output channel is null if all its weights connecting to *non-null*
    input channels are zero.  Weights connecting to null input channels are
    irrelevant — the input is guaranteed zero regardless.

    Example:
        weight shape = (c_out=8, c_in=4, 3, 3)
        null_in.indices = {2, 3}   ← input channels 2 and 3 are always zero

        We only look at weight[:, [0,1], :, :] (the non-null inputs).
        If output channel 5 has weight[5, [0,1], :, :] == 0, then
        output channel 5 is null.
    """
    c_out, c_in = weight.shape[0], weight.shape[1]

    valid_mask = torch.ones(c_in, dtype=torch.bool, device=weight.device)
    if null_in.indices:
        valid_mask[list(null_in.indices)] = False

    w_valid = weight[:, valid_mask]
    per_out_nnz = (w_valid != 0).flatten(1).sum(dim=1)  # (c_out,)

    null_indices = set(torch.where(per_out_nnz == 0)[0].tolist())
    return NullChannels(indices=null_indices, total=c_out)


def _compute_effective_macs(
    weight: torch.Tensor,
    null_in: NullChannels,
    null_out: NullChannels,
    m_h: int,
    m_w: int,
) -> int:
    """Compute MACs using the structured (channel-level) pruning formula.

    MACs_l = (1 - p_{l-1}) * c_{l-1} * (1 - p_l) * c_l * k_h * k_w * m_h * m_w

    Where p_l is the fraction of fully-null output channels, and p_{l-1}
    is the fraction of null input channels.  Non-null channels contribute
    the full kernel size (k_h * k_w) per channel pair.

    For Linear layers: k_h = k_w = 1, m_h = m_w = 1.
    """
    c_out, c_in = weight.shape[0], weight.shape[1]

    effective_c_in = c_in - null_in.count
    effective_c_out = c_out - null_out.count

    if weight.ndim == 4:
        k_h, k_w = weight.shape[2], weight.shape[3]
    else:
        k_h, k_w = 1, 1

    return effective_c_in * effective_c_out * k_h * k_w * m_h * m_w


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


# ---------------------------------------------------------------------------
# Null channel tracking through CIFAR ResNet
# ---------------------------------------------------------------------------

def _track_null_channels_resnet(
    model,
    layer_info: dict[str, LayerQuantInfo],
) -> dict[str, NullChannels]:
    """Track null channels through a CIFAR ResNet (resnet.py architecture).

    Walks conv1 → layer1 → layer2 → layer3 → linear, tracking which
    channels are null at each point.

    At each residual add the null channels from the residual branch and
    the shortcut are *intersected*: a channel is only null after the add
    if it is null in both paths.

    Returns:
        dict mapping each Conv2d/Linear layer name to the NullChannels
        describing its input.
    """
    null_input_map: dict[str, NullChannels] = {}

    # conv1 input: RGB image, no null channels
    c_in_conv1 = layer_info['conv1'].weight.shape[1]  # 3
    current_null = NullChannels.none(total=c_in_conv1)
    null_input_map['conv1'] = current_null.copy()

    # After conv1
    current_null = _compute_null_output_channels(layer_info['conv1'].weight, current_null)

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
            w = layer_info[conv1_name].weight
            current_null = _compute_null_output_channels(w, current_null)

            # conv2 in block
            conv2_name = f'{prefix}.conv2'
            null_input_map[conv2_name] = current_null.copy()
            w = layer_info[conv2_name].weight
            residual_null = _compute_null_output_channels(w, current_null)

            # --- Shortcut path ---
            shortcut = block.shortcut

            if isinstance(shortcut, LambdaLayer):
                # Option A: F.pad(x[:, :, ::2, ::2], (0,0,0,0, pad, pad), "constant", 0)
                # Expands channels: [pad zeros | c_in from input | pad zeros]
                c_out_block = w.shape[0]
                c_in_block = layer_info[f'{prefix}.conv1'].weight.shape[1]
                pad = c_out_block // 4

                shortcut_indices = set()
                # First `pad` channels: always zero (padding)
                shortcut_indices.update(range(0, pad))
                # Middle channels: inherit null status from block input
                for ch_idx in range(c_in_block):
                    if ch_idx in block_input_null.indices:
                        shortcut_indices.add(ch_idx + pad)
                # Last `pad` channels: always zero (padding)
                shortcut_indices.update(range(pad + c_in_block, c_out_block))

                shortcut_null = NullChannels(indices=shortcut_indices, total=c_out_block)

            elif isinstance(shortcut, nn.Sequential) and len(shortcut) == 0:
                # Identity shortcut: null channels pass through unchanged
                shortcut_null = block_input_null

            else:
                # Unknown shortcut type: conservatively assume no null channels
                shortcut_null = NullChannels.none(total=residual_null.total)

            # --- Residual add: intersection ---
            current_null = residual_null.intersect(shortcut_null)

    # Linear layer
    null_input_map['linear'] = current_null.copy()

    return null_input_map


# ---------------------------------------------------------------------------
# Null channel tracking through sequential models
# ---------------------------------------------------------------------------

def _track_null_channels_sequential(
    model,
    layer_info: dict[str, LayerQuantInfo],
) -> dict[str, NullChannels]:
    """Track null channels through a sequential (no skip connections) model.

    Walks layers in module order, propagating null output channels of each
    layer as null input channels of the next.

    At a conv → linear boundary (flatten), null conv output channels are
    expanded: each null channel maps to features_per_channel null input
    features in the linear layer.
    """
    null_input_map: dict[str, NullChannels] = {}
    current_null: NullChannels | None = None

    for name, module in model.named_modules():
        if name not in layer_info:
            continue

        info = layer_info[name]
        c_in = info.weight.shape[1]

        if current_null is None:
            # First layer: no null inputs
            current_null = NullChannels.none(total=c_in)
        elif current_null.total != c_in:
            # Dimension change (e.g. conv→linear flatten)
            if c_in % current_null.total == 0:
                features_per_ch = c_in // current_null.total
                expanded = set()
                for ch in current_null.indices:
                    for i in range(features_per_ch):
                        expanded.add(ch * features_per_ch + i)
                current_null = NullChannels(indices=expanded, total=c_in)
            else:
                current_null = NullChannels.none(total=c_in)

        null_input_map[name] = current_null.copy()
        current_null = _compute_null_output_channels(info.weight, current_null)

    return null_input_map


# ---------------------------------------------------------------------------
# QuantizationComparison class
# ---------------------------------------------------------------------------

class QuantizationComparison:
    """Compare MACs, BOPs, and compression ratio between two models.

    Usage:
        qc = QuantizationComparison(input_size=(3, 32, 32))
        qc.set_baseline(baseline_model)
        stats = qc.calculate_stats(compressed_model)
        print(qc.compression_ratio(stats))
    """

    def __init__(self, input_size=(3, 32, 32), b_a=32):
        """
        Args:
            input_size: CHW tuple for the model input (no batch dim)
            b_a: activation bitwidth (32 for CoDeQ — activations not quantized)
        """
        self.input_size = input_size
        self.b_a = b_a
        self.baseline_stats: ModelStats | None = None

    def set_baseline(self, model: nn.Module) -> ModelStats:
        """Profile the baseline model and store its stats."""
        self.baseline_stats = self._profile_model(model)
        return self.baseline_stats

    def calculate_stats(self, model: nn.Module) -> ModelStats:
        """Profile a model. Asserts architecture matches baseline if one is set."""
        stats = self._profile_model(model)
        if self.baseline_stats is not None:
            self._assert_compatible(self.baseline_stats, stats)
        return stats

    def compression_ratio(self, compressed_stats: ModelStats) -> float:
        """BOPs compression ratio: baseline_BOPs / compressed_BOPs."""
        assert self.baseline_stats is not None, "Call set_baseline() first"
        return self.baseline_stats.total_bops / compressed_stats.total_bops

    def mac_compression_ratio(self, compressed_stats: ModelStats) -> float:
        """MAC compression ratio: baseline_MACs / compressed_MACs."""
        assert self.baseline_stats is not None, "Call set_baseline() first"
        return self.baseline_stats.total_macs / compressed_stats.total_macs

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

    def _profile_model(self, model: nn.Module) -> ModelStats:
        model.eval()

        # 1. Spatial dims via forward pass
        spatial = _get_spatial_dims(model, self.input_size)

        # 2. Per-layer quant info
        layer_info: dict[str, LayerQuantInfo] = {}
        module_map: dict[str, nn.Module] = {}
        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                with torch.no_grad():
                    layer_info[name] = _get_layer_quant_info(module)
                module_map[name] = module

        # 3. Null channel tracking (auto-detect architecture)
        if hasattr(model, 'layer1') and hasattr(model, 'layer2') and hasattr(model, 'layer3'):
            null_input_map = _track_null_channels_resnet(model, layer_info)
        else:
            null_input_map = _track_null_channels_sequential(model, layer_info)

        # 4. Per-layer stats
        layers: list[LayerStats] = []
        total_macs = 0
        total_bops = 0.0

        for name, info in layer_info.items():
            if name not in spatial:
                continue

            m_h, m_w = spatial[name]
            null_in = null_input_map.get(name, NullChannels.none(total=info.weight.shape[1]))
            null_out = _compute_null_output_channels(info.weight, null_in)

            macs = _compute_effective_macs(info.weight, null_in, null_out, m_h, m_w)
            bops = macs * info.bitwidth * self.b_a

            if isinstance(module_map[name], nn.Conv2d):
                k_size = (info.weight.shape[2], info.weight.shape[3])
            else:
                k_size = (1, 1)

            layers.append(LayerStats(
                name=name,
                macs=macs,
                bops=bops,
                bitwidth=info.bitwidth,
                pruning_ratio=null_out.ratio,
                null_out=null_out,
                null_in=null_in,
                spatial=(m_h, m_w),
                kernel_size=k_size,
            ))

            total_macs += macs
            total_bops += bops

        return ModelStats(layers=layers, total_macs=total_macs, total_bops=total_bops)

    @staticmethod
    def _assert_compatible(baseline: ModelStats, compressed: ModelStats):
        bl = {l.name for l in baseline.layers}
        cl = {l.name for l in compressed.layers}
        assert bl == cl, (
            f"Architecture mismatch. "
            f"In baseline only: {bl - cl}, in compressed only: {cl - bl}"
        )


# ---------------------------------------------------------------------------
# Print utilities
# ---------------------------------------------------------------------------

def print_model_stats(stats: ModelStats):
    """Print per-layer stats showing the structured formula terms.

    MACs_l = (1-p_{l-1})*c_{l-1} * (1-p_l)*c_l * k_h*k_w * m_h*m_w
    BOPs_l = MACs_l * b_w * b_a
    """
    header = (
        f"{'Layer':<25} "
        f"{'c_in':>5} {'p_in':>5} {'c_out':>5} {'p_out':>5} "
        f"{'k':>3} {'m_h':>4} {'m_w':>4} "
        f"{'b_w':>4} "
        f"{'MACs':>12} {'BOPs':>14}"
    )
    print(header)
    print("-" * len(header))

    for l in stats.layers:
        k = f"{l.kernel_size[0]}x{l.kernel_size[1]}" if l.kernel_size != (1, 1) else "1"
        print(
            f"{l.name:<25} "
            f"{l.null_in.total:>5} {l.null_in.ratio:>5.2f} "
            f"{l.null_out.total:>5} {l.null_out.ratio:>5.2f} "
            f"{k:>3} {l.spatial[0]:>4} {l.spatial[1]:>4} "
            f"{l.bitwidth:>4.0f} "
            f"{l.macs:>12} {l.bops:>14.0f}"
        )

    print("-" * len(header))
    print(
        f"{'TOTAL':<25} "
        f"{'':>5} {'':>5} {'':>5} {'':>5} "
        f"{'':>3} {'':>4} {'':>4} "
        f"{'':>4} "
        f"{stats.total_macs:>12} {stats.total_bops:>14.0f}"
    )


def print_comparison(baseline_stats: ModelStats, compressed_stats: ModelStats):
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
