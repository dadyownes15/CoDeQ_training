# Training Parameters

This document describes all available parameters for the `train.py` training script for CoDeQ.

## Quick Start

```bash
# Basic training (ResNet20 on CIFAR-10)
python train.py

# With quantization-aware training (QAT)
python train.py --use-qat 1

# Custom model and learning rate
python train.py --model vit --lr 0.01 --epochs 200
```

## Parameter Reference

All parameters are optional and have sensible defaults. They can be combined in any way.

### Model Architecture

| Parameter | Default | Type | Description |
|-----------|---------|------|-------------|
| `--model` | `resnet20` | `str` (choice) | Model architecture to train. Choices: `resnet20`, `vit`, `mlp` |

**Notes:**
- `resnet20`: CIFAR-10 ResNet with 20 layers (defined in `resnet.py`)
- `vit`: Vision Transformer (Tiny ViT, pretrained on ImageNet-21K)
- `mlp`: Not yet implemented

### Training Dynamics

| Parameter | Default | Type | Description |
|-----------|---------|------|-------------|
| `--epochs` | `400` | `int` | Total number of training epochs to run |
| `--batch-size` or `-b` | `128` | `int` | Mini-batch size for training and validation |
| `--workers` or `-j` | `4` | `int` | Number of data loading workers (threads) for data loader |
| `--start-epoch` | `0` | `int` | Manual epoch number for resuming training (used with `--resume`) |

### Optimization

| Parameter | Default | Type | Description |
|-----------|---------|------|-------------|
| `--lr` or `--learning-rate` | `0.1` | `float` | Initial learning rate for the optimizer |
| `--momentum` | `0.9` | `float` | Momentum for SGD optimizer (ResNet20) |
| `--weight-decay` or `--wd` | `1e-4` | `float` | L2 weight decay regularization coefficient |

**Notes:**
- ResNet20 uses SGD optimizer with momentum
- ViT uses AdamW optimizer (momentum parameter is ignored)
- Learning rate is decayed using cosine annealing over the full training epoch range

### Data & Input

| Parameter | Default | Type | Description |
|-----------|---------|------|-------------|
| `--img-size` or `--input-size` | `32` | `int` | Input image resolution (width and height). Use 32 for CIFAR-10, 224 for ViT |

**Notes:**
- Must match the model's expected input size
- ViT is trained on ImageNet (224x224), but can work with 32x32 for CIFAR-10

### Quantization

| Parameter | Default | Type | Description |
|-----------|---------|------|-------------|
| `--use-qat` | `0` | `int` (binary) | Enable quantization-aware training (QAT). Set to `1` to enable |
| `--quantizer-bit` | `8` | `int` | Bit-width for the quantizer (e.g., 8 for 8-bit quantization) |

**Notes:**
- When `--use-qat 1`, the training script attaches fake quantizers to model weights
- ResNet20 quantizes all layers except batch normalization (`bn`)
- ViT quantizes all layers except normalization layers (`norm`)
- Quantizer is `UniformSymmetric` with the specified bit-width

### Checkpointing & Resuming

| Parameter | Default | Type | Description |
|-----------|---------|------|-------------|
| `--resume` | `''` (none) | `str` | Path to checkpoint file to resume training from |
| `--save-dir` | `save_temp` | `str` | Directory where checkpoints and models will be saved |

**Notes:**
- If checkpoint exists, training resumes from that epoch with restored optimizer and scheduler state
- Checkpoints are saved every epoch with the format: `checkpoint_{wandb_run_name}.th`
- Each checkpoint contains: model state, optimizer state, scheduler state, epoch number, and best accuracy

### Hardware & Performance

| Parameter | Default | Type | Description |
|-----------|---------|------|-------------|
| `--device` | `mps` | `str` (choice) | Computing device. Choices: `cuda` (NVIDIA), `mps` (Apple Silicon), `cpu` |
| `--use-compile` | `0` | `int` (binary) | Enable PyTorch 2.x compilation for performance optimization. Set to `1` to enable |

**Notes:**
- `mps`: Metal Performance Shaders (macOS with Apple Silicon)
- `cuda`: NVIDIA GPUs
- `cpu`: CPU-only training (very slow)
- When `--use-compile 1`, uses "max-autotune" mode with full-graph disabled

### Logging & Monitoring

| Parameter | Default | Type | Description |
|-----------|---------|------|-------------|
| `--wandb` | `Uncategorized` | `str` | Weights & Biases project name for experiment tracking |

**Notes:**
- Requires `.env` file with `WANDB_API_KEY` set
- Logs are saved with key: `epoch`, `loss`, `train_acc`, `val_acc`
- Each run is automatically named and checkpoints are tagged with the run name

## Example Commands

```bash
# ResNet20 basic training
python train.py --epochs 200 --batch-size 256

# ResNet20 with QAT (8-bit quantization)
python train.py --use-qat 1 --quantizer-bit 8 --epochs 300

# ViT with higher learning rate for fine-tuning
python train.py --model vit --lr 0.001 --img-size 224 --epochs 100

# Resume from checkpoint with modified learning rate
python train.py --resume save_temp/checkpoint_run_123.th --lr 0.01 --epochs 100

# GPU training with compilation enabled
python train.py --device cuda --use-compile 1 --epochs 400

# Custom directory and W&B project
python train.py --save-dir my_experiments --wandb my_project_name
```

## Environment Requirements

- **`.env` file** (optional): Must contain `WANDB_API_KEY` if using W&B logging
- **CUDA** (optional): Required if `--device cuda` is used
- **PyTorch 2.0+** (optional): Required for `--use-compile 1`

## Notes

- All defaults are tuned for CIFAR-10 training on ResNet20
- The script automatically adjusts optimizer and scheduler based on the selected model
- Checkpoints are saved every epoch; only the best checkpoint is marked separately
- Training automatically adjusts CUDA settings (tf32, cudnn.benchmark) when using NVIDIA GPUs
