"""Debug training script for validating the CoDeQ pipeline.

Trains ResNet-20 on CIFAR-10 for 20 epochs with coupled group lasso,
prints per-epoch metrics, and saves diagnostic plots to debug_output/.

Usage:
    python -m experiments.debug_run
"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from resnet import resnet20
from src.quantizer import DeadZoneLDZCompander
from utils_quantization import attach_weight_quantizers, toggle_quantization
from experiments.evaluator import Evaluator
from experiments.sparsity import coupled_group_lasso
from experiments.schedules import linear_warmup

OUTPUT_DIR = "debug_output"
EPOCHS = 20
BATCH_SIZE = 512
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)


def _get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _compute_unstructured_sparsity(model: nn.Module) -> float:
    """Fraction of quantized weights that are exactly zero."""
    total = 0
    zeros = 0
    for m in model.modules():
        if isinstance(m, nn.Conv2d) and hasattr(m, 'parametrizations'):
            w = m.weight.detach().cpu()
            total += w.numel()
            zeros += (w == 0).sum().item()
    if total == 0:
        return 0.0
    return zeros / total


def _compute_avg_bitwidth(model: nn.Module) -> float:
    """Mean learned bitwidth across all quantized layers."""
    bitwidths = []
    for m in model.modules():
        if hasattr(m, 'parametrizations') and hasattr(m.parametrizations, 'weight'):
            fq = m.parametrizations.weight[0]
            bitwidths.append(fq.quantizer.get_bitwidth().item())
    if not bitwidths:
        return 0.0
    return sum(bitwidths) / len(bitwidths)


def _compute_structured_sparsity(model: nn.Module) -> float:
    """Fraction of output channels that are entirely dead across quantized conv layers."""
    total_channels = 0
    dead_channels = 0
    for m in model.modules():
        if isinstance(m, nn.Conv2d) and hasattr(m, 'parametrizations'):
            w = m.weight.detach().cpu()
            c_out = w.shape[0]
            alive = w.flatten(1).norm(p=2, dim=1) > 0
            total_channels += c_out
            dead_channels += (c_out - alive.sum().item())
    if total_channels == 0:
        return 0.0
    return dead_channels / total_channels


@torch.no_grad()
def _evaluate(model: nn.Module, test_loader, device: str) -> tuple[float, float]:
    model.eval()
    correct = 0
    total = 0
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        total_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    return correct / total, total_loss / total


def plot_accuracy_curve(history: dict, output_dir: str):
    """Plot 1: Test accuracy vs epoch."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(history['epoch'], [a * 100 for a in history['accuracy']], 'b-o', markersize=4)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Test Accuracy (%)')
    ax.set_title('Test Accuracy Over Training')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'accuracy_curve.png'), dpi=150)
    plt.close(fig)


def plot_sparsity_curves(history: dict, output_dir: str):
    """Plot 2: Unstructured and structured sparsity + loss coefficient vs epoch."""
    fig, ax1 = plt.subplots(figsize=(8, 5))

    ax1.plot(history['epoch'], [s * 100 for s in history['unstr_sparsity']],
             'r-o', markersize=4, label='Unstructured Sparsity')
    ax1.plot(history['epoch'], [s * 100 for s in history['struct_sparsity']],
             'b-s', markersize=4, label='Structured Sparsity')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Sparsity (%)')
    ax1.set_title('Sparsity Over Training')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(history['epoch'], history['coeff'], 'g--', alpha=0.6, label='Sparsity Coeff')
    ax2.set_ylabel('Sparsity Loss Coefficient', color='green')
    ax2.legend(loc='lower right')

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'sparsity_curves.png'), dpi=150)
    plt.close(fig)


def plot_per_layer_channels(model: nn.Module, output_dir: str):
    """Plot 4: Stacked bar chart — alive vs dead channels per layer, annotated with bitwidth."""
    layer_names = []
    alive_counts = []
    dead_counts = []
    bitwidths = []

    for name, m in model.named_modules():
        if isinstance(m, nn.Conv2d) and hasattr(m, 'parametrizations'):
            w = m.weight.detach()
            c_out = w.shape[0]
            alive = (w.flatten(1).norm(p=2, dim=1) > 0).sum().item()
            dead = c_out - alive
            bw = m.parametrizations.weight[0].quantizer.get_bitwidth().item()

            layer_names.append(name)
            alive_counts.append(alive)
            dead_counts.append(dead)
            bitwidths.append(bw)

    x = np.arange(len(layer_names))
    fig, ax = plt.subplots(figsize=(max(8, len(layer_names) * 0.8), 6))
    ax.bar(x, alive_counts, label='Alive', color='steelblue')
    ax.bar(x, dead_counts, bottom=alive_counts, label='Dead', color='lightgray')

    # Annotate with bitwidth
    for i, bw in enumerate(bitwidths):
        total = alive_counts[i] + dead_counts[i]
        ax.text(i, total + 0.5, f'{bw:.0f}b', ha='center', va='bottom', fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(layer_names, rotation=45, ha='right', fontsize=7)
    ax.set_ylabel('Channels')
    ax.set_title('Per-Layer Channel Status (annotated with learned bitwidth)')
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'per_layer_channels.png'), dpi=150)
    plt.close(fig)


def main():
    device = _get_device()
    print(f"Using device: {device}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --- Data ---
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    train_set = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train)
    test_set = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=(device == "cuda"))
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=(device == "cuda"))

    # --- Model ---
    model = resnet20()

    # --- Baseline snapshot (before quantization) ---
    evaluator = Evaluator(model, input_size=(3, 32, 32))

    # --- Attach quantizers ---
    attach_weight_quantizers(
        model=model,
        exclude_layers=['conv1', 'bn', 'linear'],
        quantizer=DeadZoneLDZCompander,
        quantizer_kwargs={
            'max_bits': 8,
            'init_bit_logit': 3.0,
            'init_deadzone_logit': 3.0,
            'learnable_bit': True,
            'learnable_deadzone': True,
        },
        enabled=True,
    )
    model.to(device)

    # --- Sparsity loss ---
    sparsity_fn = coupled_group_lasso(model)
    sparsity_schedule = linear_warmup(delay=5, ramp=5)

    # --- Optimizer (3 param groups) ---
    base_params, dz_params, bit_params = [], [], []
    for name, param in model.named_parameters():
        if 'logit_dz' in name:
            dz_params.append(param)
        elif 'logit_bit' in name:
            bit_params.append(param)
        else:
            base_params.append(param)

    optimizer = optim.AdamW([
        {'params': base_params, 'lr': 1e-3, 'weight_decay': 0.0},
        {'params': dz_params, 'lr': 1e-3, 'weight_decay': 0.01},
        {'params': bit_params, 'lr': 1e-3, 'weight_decay': 0.01},
    ])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = nn.CrossEntropyLoss()

    # --- History for plots ---
    history = {
        'epoch': [], 'accuracy': [], 'loss': [],
        'unstr_sparsity': [], 'struct_sparsity': [],
        'avg_bits': [], 'coeff': [],
    }

    # --- Training loop ---
    print(f"\n{'Epoch':>7} | {'Loss':>8} | {'Acc':>7} | {'Unstr':>7} | {'Struct':>7} | {'Bits':>5}")
    print("-" * 60)

    for epoch in range(EPOCHS):
        model.train()
        toggle_quantization(model, enabled=True)

        coeff = sparsity_schedule(epoch, EPOCHS)
        running_loss = 0.0
        num_batches = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            ce_loss = criterion(outputs, labels)
            sp_loss = sparsity_fn(model)
            total_loss = ce_loss + coeff * sp_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            running_loss += total_loss.item()
            num_batches += 1

        scheduler.step()

        # --- Evaluate ---
        test_acc, test_loss = _evaluate(model, test_loader, device)
        unstr = _compute_unstructured_sparsity(model)
        struct = _compute_structured_sparsity(model)
        avg_bits = _compute_avg_bitwidth(model)
        avg_loss = running_loss / num_batches

        # Record history
        history['epoch'].append(epoch + 1)
        history['accuracy'].append(test_acc)
        history['loss'].append(avg_loss)
        history['unstr_sparsity'].append(unstr)
        history['struct_sparsity'].append(struct)
        history['avg_bits'].append(avg_bits)
        history['coeff'].append(coeff)

        print(
            f"  {epoch+1:>3}/{EPOCHS} | "
            f"{avg_loss:>8.4f} | "
            f"{test_acc*100:>5.1f}% | "
            f"{unstr*100:>5.1f}% | "
            f"{struct*100:>5.1f}% | "
            f"{avg_bits:>5.1f}"
        )

    # --- Final evaluation with Evaluator ---
    print("\n--- Final Compression Metrics ---")
    model.cpu()  # Evaluator needs model on CPU for spatial dim profiling
    result = evaluator.evaluate(model)
    print(f"Structured BOPs compression:   {result.structured_bops_ratio:.2f}x")
    print(f"Unstructured BOPs compression: {result.unstructured_bops_ratio:.2f}x")
    print(f"Structured MAC compression:    {result.structured_mac_ratio:.2f}x")
    print(f"Unstructured MAC compression:  {result.unstructured_mac_ratio:.2f}x")

    # --- Save plots ---
    print(f"\nSaving diagnostic plots to {OUTPUT_DIR}/")
    plot_accuracy_curve(history, OUTPUT_DIR)
    plot_sparsity_curves(history, OUTPUT_DIR)

    # Channel liveness from evaluator
    fig_liveness = evaluator.plot_channel_liveness(model)
    fig_liveness.savefig(os.path.join(OUTPUT_DIR, 'channel_liveness.png'), dpi=150)
    plt.close(fig_liveness)

    plot_per_layer_channels(model, OUTPUT_DIR)

    print("Done! Check debug_output/ for plots.")


if __name__ == '__main__':
    main()
