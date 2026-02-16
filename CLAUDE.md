# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

CoDeQ (Compression with Dead-Zone Quantizer) is a PyTorch implementation of the paper "CoDeQ: End-to-End Joint Model Compression with Dead-Zone Quantizer for High-Sparsity and Low-Precision Networks" (Wenshoj et al. 2025). It provides differentiable weight quantization with learnable bitwidth and dead-zone parameters for neural network compression via quantization-aware training (QAT).

## Commands

```bash
# Setup
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt   # torch, torchvision, numpy, matplotlib

# Run the demo (ResNet18 QAT on FakeData with sparsity visualization)
python main.py
```

There are no tests or linting configured.

## Architecture

The core library lives in two files under `src/`:

- **`src/quantizer.py`** — `DeadZoneLDZCompander(nn.Module)`: the dead-zone quantizer. Has two learnable parameters (`logit_dz` for dead-zone width, `logit_bit` for bitwidth) mapped through `tanh` to bounded ranges. The forward pass: compute absmax (99th percentile via custom `torch_quantile`), derive step size and dead-zone, then quantize weights using `RoundSTE`/`ClampSTE`/`ReluSTE` (straight-through estimator autograd functions) so gradients flow through the non-differentiable rounding.

- **`utils_quantization.py`** (root level) — `FakeQuantParametrization`: wraps a quantizer as a `torch.nn.utils.parametrize` parametrization. `attach_weight_quantizers()` walks `model.named_modules()`, skips layers matching `exclude_layers` strings, and registers the parametrization on `weight`. `toggle_quantization()` enables/disables quantization at inference vs training time.

**Note:** The README imports from `src.utils_quantization` but the actual file is at root level (`utils_quantization.py`). The working import path used in `main.py` is `from src.utils_quantization import ...` — this only works if there's a copy/symlink in `src/` or the file has been moved. The git status shows the `src/` copy was deleted. This discrepancy may need resolving.

**Other files:**
- **`resnet.py`** — CIFAR-10 ResNet variants (ResNet20–1202) from Yerlan Idelbayev's implementation. Uses option A (padding-based) shortcuts. Separate from torchvision's ResNet used in `main.py`.
- **`main.py`** — End-to-end demo: attaches quantizers to torchvision ResNet18, trains on FakeData, visualizes per-kernel sparsity maps.

## Key Design Patterns

- Quantization is applied via **PyTorch parametrizations** (`torch.nn.utils.parametrize`), not by modifying weights directly. Access `module.weight` to get the quantized weight; the original (latent) weight lives in `module.parametrizations.weight.original`.
- Optimizer param groups must be split three ways: base model weights, dead-zone logits (`logit_dz`), and bitwidth logits (`logit_bit`), each with different learning rates and weight decay. Dead-zone logits typically use high LR (~0.5) and high weight decay (~0.4) to encourage sparsity.
- All quantization operations use **straight-through estimators** (STE) for gradient propagation through rounding/clamping.
