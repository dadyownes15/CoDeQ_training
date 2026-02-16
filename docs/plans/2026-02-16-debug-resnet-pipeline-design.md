# Debug ResNet-20 Pipeline Validation

**Date:** 2026-02-16

## Goal

Create a self-contained debug script that validates the full CoDeQ training pipeline end-to-end: quantization-aware training with learnable bitwidth/dead-zone, coupled structured sparsity loss, and compression evaluation. Must provide both quantitative (console) and visual (plots) confirmation that everything works correctly.

## Design

### Script: `experiments/debug_run.py`

A single file with no W&B dependency. Trains ResNet-20 on CIFAR-10 for 20 epochs with coupled group lasso sparsity, then saves diagnostic plots.

### Training Setup

- **Model**: `resnet20()` from `resnet.py` (~0.27M params, trains fast)
- **Dataset**: CIFAR-10, standard augmentation (random crop, horizontal flip, normalize)
- **Epochs**: 20 (enough to see accuracy climb and sparsity emerge)
- **Batch size**: 512
- **Quantizer**: `DeadZoneLDZCompander` with paper defaults (`max_bits=8`, `init_bit_logit=3.0`, `init_deadzone_logit=3.0`, both learnable)
- **Excluded layers**: `['conv1', 'bn', 'linear']`
- **Optimizer**: AdamW, 3 param groups:
  - Base weights: lr=1e-3, weight_decay=0
  - Dead-zone logits (`logit_dz`): lr=1e-3, weight_decay=0.01
  - Bitwidth logits (`logit_bit`): lr=1e-3, weight_decay=0.01
- **Sparsity**: `coupled_group_lasso` with `linear_warmup(delay=5, ramp=5)`
- **Device**: Auto-detect (MPS > CUDA > CPU)

### Per-Epoch Console Output

One line per epoch:
```
Epoch  1/20 | Loss: 2.31 | Acc: 10.2% | Unstr: 0.0% | Struct: 0.0% | Bits: 7.9
Epoch  2/20 | Loss: 1.85 | Acc: 28.4% | Unstr: 1.2% | Struct: 0.0% | Bits: 7.6
...
```

Metrics per epoch:
- **Loss**: Train loss (CE + weighted sparsity), averaged over batches
- **Acc**: Test accuracy on CIFAR-10 test set
- **Unstr**: Unstructured sparsity — % of quantized weights that are exactly zero
- **Struct**: Structured sparsity — % of output channels entirely dead
- **Bits**: Mean learned bitwidth across quantized layers

### Saved Plots (to `debug_output/`)

1. **`accuracy_curve.png`** — Test accuracy vs epoch. Confirms model learns despite quantization.

2. **`sparsity_curves.png`** — Unstructured sparsity %, structured sparsity %, and sparsity loss coefficient vs epoch (coefficient on right y-axis). Shows sparsity emerging as penalty ramps.

3. **`channel_liveness.png`** — Heatmap from `Evaluator.plot_channel_liveness()`. Rows=layers, cols=channels. Black=dead, white=alive. Visual fingerprint of pruning pattern.

4. **`per_layer_channels.png`** — Stacked bar chart: for each quantized conv layer, total channels split into alive (blue) and dead (gray). Each bar annotated with learned bitwidth. Per-layer structural compression summary.

### Computing Sparsity Metrics

- **Unstructured sparsity**: Iterate quantized conv layers, compute `(weight == 0).float().mean()` on the quantized (parametrized) weights.
- **Structured sparsity**: Use `Evaluator.evaluate()` to get `EvalResult`, extract per-layer null channel counts from `structured_stats.layers`.
- **Average bitwidth**: Iterate quantizer parametrizations, call `quantizer.get_bitwidth()`, average.

### What "Working Correctly" Looks Like

After 20 epochs we expect:
- **Accuracy**: Should reach ~60-75% (won't match full 300-epoch training but should clearly climb from ~10%)
- **Unstructured sparsity**: Should grow from 0% to some nonzero amount as dead-zone/quantization take effect
- **Structured sparsity**: May be small (channels take longer to fully die) but should be nonzero if sparsity loss is working
- **Bitwidths**: Should decrease from ~8 toward lower values as weight_decay on logit_bit pushes them down
- **Channel liveness**: Should show some black cells (dead channels), concentrated in layers where the sparsity penalty is strongest

### Dependencies

Uses existing project infrastructure:
- `resnet.py` — `resnet20()`
- `src/quantizer.py` — `DeadZoneLDZCompander`
- `utils_quantization.py` — `attach_weight_quantizers`, `toggle_quantization`
- `experiments/evaluator.py` — `Evaluator`, `EvalResult`
- `experiments/sparsity.py` — `coupled_group_lasso`
- `experiments/schedules.py` — `linear_warmup`
- `metrics.py` — `ModelStats`, `QuantizationComparison`
