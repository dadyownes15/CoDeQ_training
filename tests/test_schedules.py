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
