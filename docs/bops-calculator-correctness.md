# Structured BOPs Calculator — Why Unstructured Sparsity Doesn't Count

## Formula

```
MACs_l  = (1 - p_in) * c_in  *  (1 - p_out) * c_out  *  k_h * k_w  *  m_h * m_w
BOPs_l  = MACs_l  *  b_w  *  b_a
```

`p_in` / `p_out` = fraction of **fully-null channels** (every weight in that channel is zero).
A channel with even one non-zero weight is **not null** and contributes full cost.

---

## Three examples: 2-channel networks

All three share the same architecture:

```
input (2ch) ──┬── conv1 (2→2, 3x3) ── conv2 (2→2, 3x3) ──┬── ADD ── output
              └──────────── identity shortcut ──────────────┘
```

`c_in=2, c_out=2, k=3x3, m=4x4, b_w=32, b_a=32`

Each output channel has `c_in * k_h * k_w = 2 * 3 * 3 = 18` weights.

---

### Example 1: Unstructured sparsity (~89% zeros scattered across channels)

```
conv1 weights (per output channel, flattened):
  ch 0: [0 0 0 0 1 0 0 0 0 | 1 0 0 0 0 0 0 0 0]   2/18 non-zero
  ch 1: [0 0 0 0 0 0 0 0 1 | 0 0 0 1 0 0 0 0 0]   2/18 non-zero

conv2 weights:
  ch 0: [0 1 0 0 0 0 0 0 0 | 0 0 0 0 0 0 1 0 0]   2/18 non-zero
  ch 1: [0 0 0 0 0 1 0 0 0 | 1 0 0 0 0 0 0 0 0]   2/18 non-zero
```

Both channels have 89% zeros, but **neither channel is fully null** (each has at least one non-zero weight).

```
Layer   null_in   null_out   p_in   p_out    MACs     BOPs
conv1      {}        {}      0.00   0.00      576    589,824
conv2      {}        {}      0.00   0.00      576    589,824
─────────────────────────────────────────────────────────────
TOTAL                                        1152  1,179,648
```

**Result: identical to dense.** The 89% zero weights are invisible to structured BOPs.

---

### Example 2: Dense (no sparsity)

```
conv1 weights:
  ch 0: [1 1 1 1 1 1 1 1 1 | 1 1 1 1 1 1 1 1 1]  18/18 non-zero
  ch 1: [1 1 1 1 1 1 1 1 1 | 1 1 1 1 1 1 1 1 1]  18/18 non-zero

conv2 weights:
  ch 0: [1 1 1 1 1 1 1 1 1 | 1 1 1 1 1 1 1 1 1]  18/18 non-zero
  ch 1: [1 1 1 1 1 1 1 1 1 | 1 1 1 1 1 1 1 1 1]  18/18 non-zero
```

```
Layer   null_in   null_out   p_in   p_out    MACs     BOPs
conv1      {}        {}      0.00   0.00      576    589,824
conv2      {}        {}      0.00   0.00      576    589,824
─────────────────────────────────────────────────────────────
TOTAL                                        1152  1,179,648
```

---

### Example 3: Structured sparsity (conv2 output channel 0 fully zeroed)

```
conv1 weights:
  ch 0: [1 1 1 1 1 1 1 1 1 | 1 1 1 1 1 1 1 1 1]  18/18 non-zero
  ch 1: [1 1 1 1 1 1 1 1 1 | 1 1 1 1 1 1 1 1 1]  18/18 non-zero

conv2 weights:
  ch 0: [0 0 0 0 0 0 0 0 0 | 0 0 0 0 0 0 0 0 0]   0/18 non-zero  ← FULLY NULL
  ch 1: [1 1 1 1 1 1 1 1 1 | 1 1 1 1 1 1 1 1 1]  18/18 non-zero
```

```
Layer   null_in   null_out   p_in   p_out    MACs     BOPs
conv1      {}        {}      0.00   0.00      576    589,824
conv2      {}       {0}      0.00   0.50      288    294,912
─────────────────────────────────────────────────────────────
TOTAL                                         864    884,736
```

conv2 loses half its output channels, halving its MACs.

---

### Skip connection effect (all three examples)

```
              residual_null    shortcut_null    after ADD (intersection)
Example 1:        {}                {}                  {}
Example 2:        {}                {}                  {}
Example 3:       {0}                {}                  {}
```

In all cases the identity shortcut has `shortcut_null = {}` (input is dense),
so the intersection is always `{}`. The next layer sees `p_in = 0.00`.

Even in Example 3 where conv2 nulls channel 0, the shortcut **restores** it.

---

## Summary

| Example | Description | Total MACs | Total BOPs | vs Dense |
|---------|-------------|------------|------------|----------|
| 1 | Unstructured 89% sparse | **1152** | **1,179,648** | **1.0x** |
| 2 | Dense (no sparsity) | **1152** | **1,179,648** | **1.0x** |
| 3 | Structured (1 channel zeroed) | **864** | **884,736** | **1.33x** |

Examples 1 and 2 are **identical** in structured BOPs.
Only Example 3, with a fully-zeroed channel, reduces computation.

This is by design: structured BOPs measures the cost of **channel-level** operations.
Scattering zeros across a channel does not remove the channel from the computation graph.

---

## Test suite (10/10 passed)

```
tests/test_bops.py::test_single_conv_dense                   PASSED
tests/test_bops.py::test_single_conv_half_channels_pruned    PASSED
tests/test_bops.py::test_two_conv_null_propagation            PASSED
tests/test_bops.py::test_two_conv_partial_null                PASSED
tests/test_bops.py::test_linear_layer                         PASSED
tests/test_bops.py::test_compression_ratio_arithmetic         PASSED
tests/test_bops.py::test_skip_identity_restores_channels      PASSED
tests/test_bops.py::test_skip_lambda_padded_channels_are_null PASSED
tests/test_bops.py::test_skip_both_paths_null_intersection    PASSED
tests/test_bops.py::test_skip_dense_shortcut_clears_all_null  PASSED
```
