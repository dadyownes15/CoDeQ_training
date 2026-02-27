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



"""

The idea is as follows:

Calculate the structured sparsity of model by:

1.



"""