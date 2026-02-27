import argparse
import os
import yaml
from dotenv import load_dotenv
load_dotenv()
import wandb

import timm
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import src.resnet as resnet

from src.utils_quantization import attach_weight_quantizers, toggle_quantization, UniformSymmetric
from src.quantizer import DeadZoneLDZCompander
from src.datasets import build_dataset_cifar
from src.train import train, validate, save_checkpoint
from src.structured_loss import LOSS_REGISTRY
from src.neural_networks import CifarMLP

# ── Registries ──────────────────────────────────────────────
QUANTIZER_REGISTRY = {
    "uniform": UniformSymmetric,
    "deadzone": DeadZoneLDZCompander,
}

# ── CLI ─────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description='CoDeQ Training')
parser.add_argument('--config', type=str, required=True, help='Path to YAML config file')
parser.add_argument('--device', type=str, default=None, help='Override device from config')
parser.add_argument('--resume', type=str, default='', help='Path to checkpoint to resume from')

best_prec1 = 0


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def build_run_name(cfg):
    q = cfg.get("quantizer")
    if q:
        bit = q["kwargs"].get("fixed_bit_val", q["kwargs"].get("max_bits", "?"))
        qat_tag = f"_{q['name']}{bit}b"
    else:
        qat_tag = "_baseline"
    return f"{cfg['model']}{qat_tag}_e{cfg['epochs']}_lr{cfg['optimizer']['param_groups']['base']['lr']}"


def build_optimizer(model, cfg):
    """Build optimizer with 3-way param group split (base / dz / bit)."""
    opt_cfg = cfg["optimizer"]
    groups_cfg = opt_cfg["param_groups"]

    base_params, dz_params, bit_params = [], [], []
    for name, param in model.named_parameters():
        if "logit_dz" in name:
            dz_params.append(param)
        elif "logit_bit" in name:
            bit_params.append(param)
        else:
            base_params.append(param)

    param_groups = [{"params": base_params, **groups_cfg["base"]}]
    if dz_params and "dz" in groups_cfg:
        param_groups.append({"params": dz_params, **groups_cfg["dz"]})
    if bit_params and "bit" in groups_cfg:
        param_groups.append({"params": bit_params, **groups_cfg["bit"]})

    opt_type = opt_cfg.get("type", "adamw").lower()
    if opt_type == "sgd":
        OptClass = torch.optim.SGD
    elif opt_type == "adamw":
        OptClass = torch.optim.AdamW
    elif opt_type == "adam":
        OptClass = torch.optim.Adam
    else:
        raise ValueError(f"Unknown optimizer type: {opt_type}")

    try:
        optimizer = OptClass(param_groups, fused=True)
    except TypeError:
        optimizer = OptClass(param_groups, foreach=True)

    return optimizer


def build_loss_fns(cfg):
    """Build list of (loss_fn, weight, kwargs) from config."""
    loss_fns = []
    for term in cfg.get("loss_terms", []):
        name = term["name"]
        if name not in LOSS_REGISTRY:
            raise ValueError(f"Unknown loss term: {name}. Available: {list(LOSS_REGISTRY.keys())}")
        fn = LOSS_REGISTRY[name]
        weight = term.get("lambda", 1.0)
        kwargs = term.get("kwargs", {})
        loss_fns.append((fn, weight, kwargs))
    return loss_fns or None


def main():
    global best_prec1
    args = parser.parse_args()
    cfg = load_config(args.config)

    # CLI overrides
    if args.device:
        cfg["device"] = args.device

    DEVICE = torch.device(cfg["device"])
    save_dir = cfg.get("save_dir", "save_temp")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # CUDA settings
    if DEVICE.type == "cuda":
        cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    run_name = build_run_name(cfg)
    wandb.init(project=cfg.get("wandb_project", "Uncategorized"), name=run_name, config=cfg)

    criterion = nn.CrossEntropyLoss().to(DEVICE)

    # ── Build model ─────────────────────────────────────────
    model_name = cfg["model"]
    if model_name == "resnet20":
        model = resnet.__dict__["resnet20"]()
    elif model_name == "vit":
        timm_name = "vit_tiny_patch16_224.augreg_in21k_ft_in1k"
        model = timm.create_model(timm_name, pretrained=False, num_classes=10)
    elif model_name == "mlp":
        model = CifarMLP()
    else:
        raise ValueError(f"Unknown model: {model_name}")

    # ── Attach quantizer ────────────────────────────────────
    q_cfg = cfg.get("quantizer")
    if q_cfg:
        q_name = q_cfg["name"]
        if q_name not in QUANTIZER_REGISTRY:
            raise ValueError(f"Unknown quantizer: {q_name}. Available: {list(QUANTIZER_REGISTRY.keys())}")
        q_cls = QUANTIZER_REGISTRY[q_name]
        q_kwargs = q_cfg.get("kwargs", {})
        exclude = q_cfg.get("exclude_layers", [])
        attach_weight_quantizers(model, exclude_layers=exclude, quantizer=q_cls, quantizer_kwargs=q_kwargs, enabled=True)
        toggle_quantization(model, enabled=True)

    model.to(DEVICE)

    # ── Data ────────────────────────────────────────────────
    train_loader, val_loader = build_dataset_cifar(
        workers=cfg.get("workers", 4),
        batch_size=cfg.get("batch_size", 128),
        img_size=cfg.get("img_size", 32),
    )

    # ── Optimizer ───────────────────────────────────────────
    optimizer = build_optimizer(model, cfg)

    # ── LR Scheduler ────────────────────────────────────────
    epochs = cfg["epochs"]
    sched_type = cfg.get("lr_scheduler", "cosine")
    if sched_type == "cosine":
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    else:
        raise ValueError(f"Unknown lr_scheduler: {sched_type}")

    # ── Loss terms ──────────────────────────────────────────
    loss_fns = build_loss_fns(cfg)

    # ── Resume ──────────────────────────────────────────────
    start_epoch = 0
    if args.resume and os.path.isfile(args.resume):
        print(f"=> loading checkpoint '{args.resume}'")
        checkpoint = torch.load(args.resume, map_location=DEVICE)
        start_epoch = checkpoint.get("epoch", 0)
        best_prec1 = checkpoint.get("best_prec1", 0)
        state_dict_key = "model" if "model" in checkpoint else "state_dict"
        model.load_state_dict(checkpoint[state_dict_key])
        if "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
        if "scheduler" in checkpoint:
            lr_scheduler.load_state_dict(checkpoint["scheduler"])
        print(f"=> loaded checkpoint (epoch {start_epoch})")

    # ── Compile ─────────────────────────────────────────────
    if cfg.get("use_compile", 0):
        model = torch.compile(model, mode="max-autotune", fullgraph=False, dynamic=True)

    # ── Training loop ───────────────────────────────────────
    for epoch in range(start_epoch, epochs):
        print("current lr weights {:.5e}".format(optimizer.param_groups[0]["lr"]))

        loss, train_acc = train(train_loader, model, criterion, optimizer, epoch, DEVICE, loss_fns=loss_fns)
        lr_scheduler.step()

        prec1 = validate(val_loader, model, criterion, DEVICE)

        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)

        wandb.log({
            "epoch": epoch,
            "loss": loss,
            "train_acc": train_acc,
            "val_acc": prec1,
        })

        save_checkpoint({
            "epoch": epoch + 1,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": lr_scheduler.state_dict(),
            "best_prec1": best_prec1,
            "config": cfg,
        }, is_best, filename=os.path.join(save_dir, f"checkpoint_{run_name}.th"))


if __name__ == "__main__":
    main()
