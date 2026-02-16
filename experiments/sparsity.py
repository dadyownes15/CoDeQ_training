"""Structured sparsity loss functions.

Two categories:

1. **Independent per-layer losses**: penalize each Conv2d layer's weights
   independently. Simple but ignore skip-connection entanglement — sparsity
   in conv2 output channels may not translate to structured speedup if the
   identity shortcut keeps those channels alive.

2. **Coupled losses**: group weights that are entangled via residual
   connections and penalize each group jointly. A channel can only be
   structurally pruned if ALL weights contributing to it (across the
   residual and shortcut paths) are zero.

All functions have signature: fn(model: nn.Module) -> torch.Tensor (scalar).
For parameterized/architecture-aware variants, a factory returns the fn.
"""
import torch
import torch.nn as nn
from typing import Callable

from src.resnet import BasicBlock, LambdaLayer


# ---------------------------------------------------------------------------
# Independent per-layer losses (baseline / unstructured comparison)
# ---------------------------------------------------------------------------

def group_lasso_channels(model: nn.Module) -> torch.Tensor:
    """Group lasso on output channels: L1 of L2 filter norms.

    Encourages entire output filters to go to zero (channel pruning).
    For a Conv2d with weight (c_out, c_in, k_h, k_w), computes:
        sum over c_out of ||W[c, :, :, :]||_2

    NOTE: This ignores skip-connection entanglement. Use coupled_group_lasso
    for losses that respect residual structure.
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


# ---------------------------------------------------------------------------
# Coupled group lasso (skip-connection aware)
# ---------------------------------------------------------------------------

def _build_channel_groups(model: nn.Module) -> list[list[tuple[str, str, int]]]:
    """Walk a CIFAR ResNet and build coupled channel groups.

    In a ResNet BasicBlock with identity shortcut:
        y = F(x) + x

    Output channel c of the block is alive if EITHER:
    - conv2 output channel c is non-zero, OR
    - input channel c (from shortcut) is non-zero

    So to prune channel c structurally, we need to zero:
    - conv2 output filter c (kills the residual path for channel c)
    - conv1 input channel c of the NEXT layer (stops consuming channel c)
    - AND the channel must also be dead on the shortcut side

    For identity shortcuts within a layer group (same spatial size),
    channels flow unchanged through the shortcut. We group:
    - conv2[block_i].weight[c, :, :, :] (output filter c of residual)
    - conv1[block_{i+1}].weight[:, c, :, :] (input channel c of next block)
    across ALL blocks in the same layer group into one coupled group.

    At LambdaLayer boundaries (stride changes), the channel mapping changes
    due to zero-padding. The padded channels are always zero on the shortcut
    side, so they are NOT entangled — only the middle channels are coupled
    to the previous layer group.

    Returns:
        List of channel groups. Each group is a list of (layer_name, direction, channel_idx)
        tuples identifying which output filter / input channel slices belong together.
    """
    # Validate: only layer1-3 supported (CIFAR ResNets)
    SUPPORTED_LAYERS = ['layer1', 'layer2', 'layer3']
    for i in range(4, 20):
        if hasattr(model, f'layer{i}'):
            raise ValueError(
                f"Model has 'layer{i}' — coupled_group_lasso only supports "
                f"CIFAR ResNets with {SUPPORTED_LAYERS}. "
                f"For deeper architectures (e.g. ImageNet ResNets), extend _build_channel_groups."
            )

    groups = []

    for layer_group_name in SUPPORTED_LAYERS:
        layer_group = getattr(model, layer_group_name, None)
        if layer_group is None:
            continue

        num_blocks = len(layer_group)
        num_channels = layer_group[0].conv2.weight.shape[0]

        for c in range(num_channels):
            group = []

            for block_idx in range(num_blocks):
                prefix = f'{layer_group_name}.{block_idx}'

                # conv2 output filter c (residual path produces channel c)
                group.append((f'{prefix}.conv2', 'out', c))

                # conv1 input channel c (residual path consumes channel c)
                # Only for blocks after the first — block 0's input comes
                # from the previous layer group or conv1
                if block_idx > 0:
                    group.append((f'{prefix}.conv1', 'in', c))

            groups.append(group)

    return groups


def coupled_group_lasso(model: nn.Module) -> Callable[[nn.Module], torch.Tensor]:
    """Factory: skip-connection-aware group lasso for CIFAR ResNets.

    Builds coupled channel groups at construction time by walking the model
    architecture, then returns a loss function that penalizes the joint
    L2 norm of each group.

    For each channel group g, concatenates all weight slices belonging to
    the group into one vector and takes its L2 norm:

        L = sum_g ||w_g||_2

    where w_g = concat(conv2.weight[c,:,:,:], conv1_next.weight[:,c,:,:], ...)

    This ensures gradients drive ALL entangled weights toward zero together,
    so structured pruning actually translates to speedup.

    Args:
        model: The ResNet model (used to build groups at factory time).

    Returns:
        Loss function fn(model) -> scalar tensor.
    """
    groups = _build_channel_groups(model)

    # Also add independent penalty for layers not covered by groups:
    # conv1 (stem) and linear (classifier)
    uncoupled_layers = set()
    for name, m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            uncoupled_layers.add(name)
    for group in groups:
        for layer_name, _, _ in group:
            uncoupled_layers.discard(layer_name)

    def fn(model: nn.Module) -> torch.Tensor:
        device = next(model.parameters()).device
        loss = torch.tensor(0.0, device=device)
        module_map_live = {name: m for name, m in model.named_modules()}

        # Coupled groups: joint L2 norm per channel group
        for group in groups:
            parts = []
            for layer_name, direction, c in group:
                m = module_map_live[layer_name]
                if direction == 'out':
                    # Output filter c: shape (c_in, k_h, k_w) → flatten
                    parts.append(m.weight[c].flatten())
                else:  # 'in'
                    # Input channel c: shape (c_out, k_h, k_w) → flatten
                    parts.append(m.weight[:, c].flatten())
            group_vec = torch.cat(parts)
            loss = loss + group_vec.norm(p=2)

        # Uncoupled layers: standard per-channel group lasso
        for layer_name in uncoupled_layers:
            m = module_map_live[layer_name]
            if isinstance(m, nn.Conv2d):
                filter_norms = m.weight.flatten(1).norm(p=2, dim=1)
                loss = loss + filter_norms.sum()

        return loss

    return fn
