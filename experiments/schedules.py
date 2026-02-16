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
