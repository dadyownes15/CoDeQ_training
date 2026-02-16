"""Tests for structured sparsity loss functions.

Tests cover three independent per-layer losses (group_lasso_channels,
group_lasso_blocks, l1_kernel_sparsity) and the skip-connection-aware
coupled_group_lasso that penalizes entangled channel groups jointly.
"""
import torch
import torch.nn as nn
from experiments.sparsity import (
    group_lasso_channels,
    group_lasso_blocks,
    l1_kernel_sparsity,
    coupled_group_lasso,
)
from resnet import ResNet, BasicBlock


class TinyConvNet(nn.Module):
    """Minimal 2-layer conv net for testing."""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 4, 3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(4, 8, 3, padding=1, bias=False)

    def forward(self, x):
        return self.conv2(self.conv1(x))


# ---------------------------------------------------------------------------
# Independent per-layer losses
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Coupled group lasso (skip-connection aware)
# ---------------------------------------------------------------------------

def test_coupled_group_lasso_gradient_flows():
    """Loss should have gradients w.r.t. all conv weights in a ResNet."""
    model = ResNet(BasicBlock, [1, 1, 1])
    fn = coupled_group_lasso(model)
    loss = fn(model)
    loss.backward()
    for name, m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            assert m.weight.grad is not None, f"No gradient for {name}"


def test_coupled_group_lasso_zero_model():
    """All weights zero → loss = 0."""
    model = ResNet(BasicBlock, [1, 1, 1])
    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.zero_()
    fn = coupled_group_lasso(model)
    loss = fn(model)
    assert loss.item() == 0.0


def test_coupled_group_lasso_positive():
    """Non-zero weights → positive loss."""
    model = ResNet(BasicBlock, [1, 1, 1])
    fn = coupled_group_lasso(model)
    loss = fn(model)
    assert loss.item() > 0.0


def test_coupled_penalizes_same_channel_across_block():
    """Coupled loss for channel c should include weights from both conv2
    output channel c AND the conv1 input channel c of the next layer.

    In a ResNet with identity shortcut, channel c at the block output
    is entangled: conv2 output c feeds the add, and the shortcut passes
    channel c from the block input unchanged. So pruning channel c
    requires zeroing conv2[c, :, :, :] AND all downstream layers that
    consume channel c.

    The coupled loss groups these together, so it should produce a
    DIFFERENT value than summing independent per-layer norms.
    """
    model = ResNet(BasicBlock, [2, 1, 1])
    fn_coupled = coupled_group_lasso(model)
    loss_coupled = fn_coupled(model)

    # Independent loss (for comparison — should differ)
    loss_independent = group_lasso_channels(model)

    # They should differ because the coupled loss groups weights
    # across layers into joint norms rather than summing separate norms
    assert loss_coupled.item() != loss_independent.item()


def test_coupled_handles_lambda_shortcut():
    """At stride boundaries (LambdaLayer shortcut), the padded channels
    are always zero on the shortcut side, so the loss should still work.
    """
    model = ResNet(BasicBlock, [1, 1, 1])  # has LambdaLayer at layer2, layer3
    fn = coupled_group_lasso(model)
    loss = fn(model)
    assert loss.item() > 0.0
    loss.backward()  # should not error
