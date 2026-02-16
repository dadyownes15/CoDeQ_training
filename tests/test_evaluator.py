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

    # Zero half weights scattered WITHIN each filter (no full channels die)
    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                for i in range(m.weight.shape[0]):
                    flat = m.weight[i].flatten()
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
