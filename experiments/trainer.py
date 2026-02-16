"""Training loop with W&B logging."""
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import wandb
import matplotlib.pyplot as plt
from dataclasses import asdict

from src.resnet import resnet20, resnet32, resnet44, resnet56, resnet110
from src.quantizer import DeadZoneLDZCompander
from src.utils_quantization import attach_weight_quantizers, toggle_quantization
from experiments.config import ExperimentConfig
from experiments.evaluator import Evaluator


MODEL_REGISTRY = {
    'resnet20': resnet20,
    'resnet32': resnet32,
    'resnet44': resnet44,
    'resnet56': resnet56,
    'resnet110': resnet110,
}

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)


def _build_model(config: ExperimentConfig) -> nn.Module:
    model_fn = MODEL_REGISTRY[config.model]
    return model_fn()


def _build_dataloaders(config: ExperimentConfig):
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
        train_set, batch_size=config.batch_size, shuffle=True,
        num_workers=config.num_workers, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=config.batch_size, shuffle=False,
        num_workers=config.num_workers, pin_memory=True)

    return train_loader, test_loader


def _split_param_groups(model: nn.Module):
    base_params, dz_params, bit_params = [], [], []
    for name, param in model.named_parameters():
        if 'logit_dz' in name:
            dz_params.append(param)
        elif 'logit_bit' in name:
            bit_params.append(param)
        else:
            base_params.append(param)
    return base_params, dz_params, bit_params


@torch.no_grad()
def _evaluate(model: nn.Module, test_loader, device: str):
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

    accuracy = correct / total
    avg_loss = total_loss / total
    return accuracy, avg_loss


def run_experiment(config: ExperimentConfig):
    """Run a full training experiment with W&B logging.

    Args:
        config: Experiment configuration.
    """
    # --- W&B init ---
    wandb.init(
        project=config.wandb_project,
        name=config.name,
        config={k: v for k, v in asdict(config).items()
                if k not in ('sparsity_fn', 'sparsity_schedule')},
    )

    device = config.device

    # --- Model ---
    model = _build_model(config)

    # --- Baseline evaluation (before quantization) ---
    evaluator = Evaluator(model, input_size=(3, 32, 32))

    # --- Attach quantizers (skip for baseline) ---
    if config.quantize:
        attach_weight_quantizers(
            model=model,
            exclude_layers=config.exclude_layers,
            quantizer=DeadZoneLDZCompander,
            quantizer_kwargs=config.quantizer_kwargs,
            enabled=True,
        )
    model.to(device)

    # --- Optimizer (SGD, matching original ResNet regime) ---
    if config.quantize:
        base_params, dz_params, bit_params = _split_param_groups(model)
        param_groups = [
            {'params': base_params, 'lr': config.lr, 'weight_decay': config.weight_decay},
            {'params': dz_params, 'lr': config.lr_dz, 'weight_decay': config.weight_decay_dz},
            {'params': bit_params, 'lr': config.lr_bit, 'weight_decay': config.weight_decay_bit},
        ]
    else:
        param_groups = [
            {'params': list(model.parameters()), 'lr': config.lr, 'weight_decay': config.weight_decay},
        ]
    optimizer = optim.SGD(param_groups, momentum=config.momentum)

    # --- LR scheduler (cosine on all param groups) ---
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)

    # --- Data ---
    train_loader, test_loader = _build_dataloaders(config)

    # --- Training loop ---
    criterion = nn.CrossEntropyLoss()

    for epoch in range(config.epochs):
        model.train()
        if config.quantize:
            toggle_quantization(model, enabled=True)

        coeff = config.sparsity_schedule(epoch, config.epochs)
        running_ce = 0.0
        running_sp = 0.0
        num_batches = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            ce_loss = criterion(outputs, labels)

            if config.sparsity_fn is not None:
                sp_loss = config.sparsity_fn(model)
                total_loss = ce_loss + coeff * sp_loss
            else:
                sp_loss = torch.tensor(0.0)
                total_loss = ce_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            running_ce += ce_loss.item()
            running_sp += sp_loss.item()
            num_batches += 1

        scheduler.step()

        # --- Eval ---
        test_acc, test_loss = _evaluate(model, test_loader, device)

        # --- Compression metrics ---
        result = evaluator.evaluate(model)

        # --- W&B logging ---
        log_dict = {
            'epoch': epoch,
            'train/ce_loss': running_ce / num_batches,
            'train/sparsity_loss': running_sp / num_batches,
            'train/sparsity_coeff': coeff,
            'eval/accuracy': test_acc,
            'eval/loss': test_loss,
            'compression/structured_bops_ratio': result.structured_bops_ratio,
            'compression/structured_mac_ratio': result.structured_mac_ratio,
            'compression/unstructured_bops_ratio': result.unstructured_bops_ratio,
            'compression/unstructured_mac_ratio': result.unstructured_mac_ratio,
        }

        # Per-layer bitwidth and sparsity
        for layer_stat in result.structured_stats.layers:
            log_dict[f'layers/{layer_stat.name}/bitwidth'] = layer_stat.bitwidth
            log_dict[f'layers/{layer_stat.name}/structured_pruning'] = layer_stat.pruning_ratio
        for layer_stat in result.unstructured_stats.layers:
            log_dict[f'layers/{layer_stat.name}/unstructured_pruning'] = layer_stat.pruning_ratio

        # Visualization images at intervals
        if epoch % config.viz_interval == 0 or epoch == config.epochs - 1:
            fig_liveness = evaluator.plot_channel_liveness(model)
            log_dict['viz/channel_liveness'] = wandb.Image(fig_liveness)
            plt.close(fig_liveness)

            fig_bars = evaluator.plot_layer_sparsity_bars(result)
            log_dict['viz/sparsity_bars'] = wandb.Image(fig_bars)
            plt.close(fig_bars)

        wandb.log(log_dict)

        print(
            f"Epoch {epoch+1}/{config.epochs} | "
            f"CE: {running_ce / num_batches:.4f} | "
            f"Acc: {test_acc:.4f} | "
            f"Struct BOPs: {result.structured_bops_ratio:.2f}x | "
            f"Unstruct BOPs: {result.unstructured_bops_ratio:.2f}x"
        )

    wandb.finish()
    return model
