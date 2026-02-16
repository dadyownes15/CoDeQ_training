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
