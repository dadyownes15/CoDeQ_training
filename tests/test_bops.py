"""Hand-verifiable BOPs tests using toy networks.

Every expected value can be computed on paper using the structured formula:

    MACs_l = (1 - p_{l-1}) * c_{l-1} * (1 - p_l) * c_l * k_h * k_w * m_h * m_w
    BOPs_l = MACs_l * b_w * b_a

Where p_l = fraction of fully-null output channels (structured pruning ratio).
"""
import torch
import torch.nn as nn

from metrics import (
    NullChannels,
    _get_layer_quant_info,
    _get_spatial_dims,
    _compute_effective_macs,
    _compute_null_output_channels,
    _compute_unstructured_macs,
    _track_null_channels_resnet,
)
from resnet import ResNet, BasicBlock


# ---------------------------------------------------------------------------
# Helpers: tiny toy networks
# ---------------------------------------------------------------------------

class SingleConv(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 4, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        return self.conv(x)


class TwoConvSeq(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(2, 4, kernel_size=3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(4, 8, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        return self.conv2(self.conv1(x))


class SingleLinear(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 8, bias=False)

    def forward(self, x):
        return self.fc(x)


# ---------------------------------------------------------------------------
# Primitive tests
# ---------------------------------------------------------------------------

def test_single_conv_dense():
    """One Conv2d, all weights = 1.  No pruned channels.

    c_in=2, c_out=4, k=3x3, m=4x4, p_in=0, p_out=0
    MACs = 2 * 4 * 3 * 3 * 4 * 4 = 1152
    BOPs = 1152 * 32 * 32 = 1,179,648
    """
    model = SingleConv()
    nn.init.ones_(model.conv.weight)

    info = _get_layer_quant_info(model.conv)
    spatial = _get_spatial_dims(model, (2, 4, 4))

    assert info.bitwidth == 32.0
    assert spatial['conv'] == (4, 4)

    null_in = NullChannels.none(total=2)
    null_out = _compute_null_output_channels(info.weight, null_in)
    assert null_out.indices == set()

    macs = _compute_effective_macs(info.weight, null_in, null_out, 4, 4)
    assert macs == 2 * 4 * 3 * 3 * 4 * 4  # 1152

    bops = macs * info.bitwidth * 32
    assert bops == 1_179_648


def test_single_conv_half_channels_pruned():
    """Output channels 2,3 fully zeroed → p_out = 0.5.

    effective_c_out = 4 - 2 = 2
    MACs = 2 * 2 * 3 * 3 * 4 * 4 = 576  (half of dense)
    """
    model = SingleConv()
    nn.init.ones_(model.conv.weight)
    with torch.no_grad():
        model.conv.weight[2:, :, :, :] = 0.0

    info = _get_layer_quant_info(model.conv)
    null_in = NullChannels.none(total=2)
    null_out = _compute_null_output_channels(info.weight, null_in)

    assert null_out.indices == {2, 3}
    assert null_out.ratio == 0.5

    macs = _compute_effective_macs(info.weight, null_in, null_out, 4, 4)
    assert macs == 2 * 2 * 3 * 3 * 4 * 4  # 576


def test_two_conv_null_propagation():
    """conv1 fully zeroed → all output channels null → conv2 has 0 effective input channels.

    conv2: effective_c_in = 0 → MACs = 0
    """
    model = TwoConvSeq()
    nn.init.ones_(model.conv1.weight)
    nn.init.ones_(model.conv2.weight)

    with torch.no_grad():
        model.conv1.weight.zero_()

    info1 = _get_layer_quant_info(model.conv1)
    info2 = _get_layer_quant_info(model.conv2)

    null_after_conv1 = _compute_null_output_channels(info1.weight, NullChannels.none(total=2))
    assert null_after_conv1.indices == {0, 1, 2, 3}

    null_out_conv2 = _compute_null_output_channels(info2.weight, null_after_conv1)
    assert null_out_conv2.indices == set(range(8))

    macs_conv2 = _compute_effective_macs(info2.weight, null_after_conv1, null_out_conv2, 4, 4)
    assert macs_conv2 == 0


def test_two_conv_partial_null():
    """conv1 channels 2,3 pruned → conv2 has 2 null inputs.

    conv2: c_in=4, null_in={2,3}, effective_c_in=2, c_out=8, p_out=0
    MACs = 2 * 8 * 3 * 3 * 4 * 4 = 2304  (half of dense 4608)
    """
    model = TwoConvSeq()
    nn.init.ones_(model.conv1.weight)
    nn.init.ones_(model.conv2.weight)

    with torch.no_grad():
        model.conv1.weight[2:, :, :, :] = 0.0

    info1 = _get_layer_quant_info(model.conv1)
    info2 = _get_layer_quant_info(model.conv2)

    null_after_conv1 = _compute_null_output_channels(info1.weight, NullChannels.none(total=2))
    assert null_after_conv1.indices == {2, 3}

    null_out_conv2 = _compute_null_output_channels(info2.weight, null_after_conv1)
    macs_conv2 = _compute_effective_macs(info2.weight, null_after_conv1, null_out_conv2, 4, 4)
    assert macs_conv2 == 2 * 8 * 3 * 3 * 4 * 4  # 2304


def test_linear_layer():
    """Linear(4 -> 8), all weights = 1.  k=1, m=1x1.

    MACs = 4 * 8 * 1 * 1 * 1 * 1 = 32
    BOPs = 32 * 32 * 32 = 32,768
    """
    model = SingleLinear()
    nn.init.ones_(model.fc.weight)

    info = _get_layer_quant_info(model.fc)
    spatial = _get_spatial_dims(model, (4,))

    assert spatial['fc'] == (1, 1)
    assert info.bitwidth == 32.0

    null_in = NullChannels.none(total=4)
    null_out = _compute_null_output_channels(info.weight, null_in)
    macs = _compute_effective_macs(info.weight, null_in, null_out, 1, 1)
    assert macs == 4 * 8  # 32

    bops = macs * info.bitwidth * 32
    assert bops == 32_768


def test_compression_ratio_arithmetic():
    """Half channels pruned at same bitwidth → 2x BOP ratio."""
    baseline_bops = 2 * 4 * 9 * 16 * 32 * 32   # dense FP32
    compressed_bops = 2 * 2 * 9 * 16 * 32 * 32  # half output channels pruned

    ratio = baseline_bops / compressed_bops
    assert ratio == 2.0


# ---------------------------------------------------------------------------
# Skip-connection tests (using minimal ResNet)
# ---------------------------------------------------------------------------

def _build_minimal_resnet_layer_info(model):
    """Helper: collect LayerQuantInfo for all Conv2d/Linear in a ResNet."""
    layer_info = {}
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            with torch.no_grad():
                layer_info[name] = _get_layer_quant_info(module)
    return layer_info


def test_skip_identity_restores_channels():
    """Identity shortcut restores channels even when residual branch is fully zeroed.

    layer1.0.conv2 zeroed → residual_null = {0..15}
    Identity shortcut: shortcut_null = {} (conv1 output was dense)
    Intersection: {} → layer2.0.conv1 sees 0 null inputs
    """
    model = ResNet(BasicBlock, [1, 1, 1])

    with torch.no_grad():
        model.layer1[0].conv2.weight.zero_()

    layer_info = _build_minimal_resnet_layer_info(model)
    null_map = _track_null_channels_resnet(model, layer_info)

    assert null_map['layer2.0.conv1'].count == 0, (
        "Identity shortcut should restore all channels — "
        f"got {null_map['layer2.0.conv1'].count} null inputs"
    )


def test_skip_lambda_padded_channels_are_null():
    """LambdaLayer shortcut: only the zero-padded channels remain null.

    layer2.0 has LambdaLayer shortcut: 16 -> 32 channels
      pad = 32 // 4 = 8
      Shortcut: [8 zeros | 16 from input | 8 zeros]

    layer2.0.conv2 zeroed → residual_null = {0..31}
    shortcut_null = {0..7, 24..31} (padded channels)
    Intersection: {0..7, 24..31} = 16 null channels
    """
    model = ResNet(BasicBlock, [1, 1, 1])

    with torch.no_grad():
        model.layer2[0].conv2.weight.zero_()

    layer_info = _build_minimal_resnet_layer_info(model)
    null_map = _track_null_channels_resnet(model, layer_info)

    null_next = null_map['layer3.0.conv1']
    expected_null = set(range(0, 8)) | set(range(24, 32))

    assert null_next.count == 16, f"Expected 16 null, got {null_next.count}"
    assert null_next.indices == expected_null


def test_skip_both_paths_null_intersection():
    """When both paths have null channels, only the intersection propagates.

    conv1 channels 0..7 zeroed → block input has 8 null channels
    layer1.0.conv2 fully zeroed → residual_null = {0..15}
    Identity shortcut: shortcut_null = {0..7}
    Intersection: {0..7}
    """
    model = ResNet(BasicBlock, [1, 1, 1])

    with torch.no_grad():
        model.conv1.weight[:8, :, :, :] = 0.0
        model.layer1[0].conv2.weight.zero_()

    layer_info = _build_minimal_resnet_layer_info(model)
    null_map = _track_null_channels_resnet(model, layer_info)

    null_in_next = null_map['layer2.0.conv1']
    assert null_in_next.indices == set(range(8))


def test_skip_dense_shortcut_clears_all_null():
    """A dense identity shortcut means no null channels survive the add.

    conv1 is dense → block input has no null channels
    layer1.0.conv2 fully zeroed → residual_null = {0..15}
    Identity shortcut: shortcut_null = {} (dense)
    Intersection: {}
    """
    model = ResNet(BasicBlock, [1, 1, 1])

    with torch.no_grad():
        model.layer1[0].conv2.weight.zero_()

    layer_info = _build_minimal_resnet_layer_info(model)
    null_map = _track_null_channels_resnet(model, layer_info)

    assert null_map['layer2.0.conv1'].count == 0


# ---------------------------------------------------------------------------
# Unstructured MACs tests
# ---------------------------------------------------------------------------

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
