import argparse
import os
from dotenv import load_dotenv
load_dotenv()
import wandb

import numpy as np
import timm
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import src.resnet as resnet
from torchvision.models import resnet18, resnet50, ResNet18_Weights, ResNet50_Weights

from src.utils_quantization import *
from src.quantizer import *
from src.datasets import *
from src.train import *

parser = argparse.ArgumentParser(description='EXPERIMENT MOMO')
parser.add_argument('--model', metavar='MODEL', choices=['resnet20', 'vit', 'mlp'], default='resnet20')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N', help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=400, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=128, type=int, metavar='N', help='mini-batch size (default: 128)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float, metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--img-size', '--input-size', default=32, type=int, metavar='N', help='input image size (default: 32 for CIFAR-10, use 224 for ViT)')
parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('--save-dir', dest='save_dir', help='The directory used to save the trained models', default='save_temp', type=str)
parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
parser.add_argument('--wandb', type=str, default="Uncategorized", help='Name for the Weights & Biases project')
parser.add_argument('--device', type=str, default='mps', choices=['cuda', 'mps', 'cpu'], help='Device to use: cuda, mps, or cpu (default: mps)')
parser.add_argument('--use-compile', default=0, type=int, metavar='N', help='Use torch.compile to optimize the model (PyTorch 2.x)')
parser.add_argument('--use-qat', default=0, type=int, metavar='N', help='Run trianing and evaluation with fake quantizers')
parser.add_argument('--quantizer-bit', default=8, type=int, metavar='N', help='Set bit for quantizer')

best_prec1 = 0


def main():
    global args, best_prec1
    args = parser.parse_args()
    
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    
    DEVICE = torch.device(args.device)
    
    # Cuda settings
    if DEVICE.type == "cuda":
        cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            print('No matmul high')
            pass
    
    # Run name
    qat_tag = f"_qat{args.quantizer_bit}b" if args.use_qat else ""
    run_name = f"{args.model}{qat_tag}_e{args.epochs}_lr{args.lr}"

    # WAND
    wandb.init(project=args.wandb, name=run_name, config=vars(args))

    # Build model, criterion, optim, data loader
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.to(DEVICE)
    
    # RESNET20
    if args.model == "resnet20":
        model = resnet.__dict__["resnet20"]()
        
        # Attach quantizer and set QAT
        if args.use_qat:
            attach_weight_quantizers(model, exclude_layers=['bn'], quantizer=UniformSymmetric, quantizer_kwargs=dict(bitwidth=args.quantizer_bit), enabled=True)
            toggle_quantization(model, enabled=True)
        
        model.to(DEVICE)
        train_loader, val_loader = build_dataset_cifar(workers=args.workers, batch_size=args.batch_size, img_size=args.img_size)
        
        SGD_args = dict(lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        try:
            optimizer = torch.optim.SGD(model.parameters(), **SGD_args, fused=True)
        except TypeError:
            optimizer = torch.optim.SGD(model.parameters(), **SGD_args, foreach=True)
    # TINY VIT
    elif args.model == "vit":
        timm_name = 'vit_tiny_patch16_224.augreg_in21k_ft_in1k'
        model = timm.create_model(timm_name, pretrained=False, num_classes=10)

        # Attach quantizer and set QAT
        if args.use_qat:
            attach_weight_quantizers(model, exclude_layers=['norm'], quantizer=UniformSymmetric, quantizer_kwargs=dict(bitwidth=args.quantizer_bit), enabled=True)
            toggle_quantization(model, enabled=True)
            
        model.to(DEVICE)
        train_loader, val_loader = build_dataset_cifar(workers=args.workers, batch_size=args.batch_size, img_size=args.img_size)
        
        ADAM_args = dict(lr=args.lr, weight_decay=args.weight_decay)
        try:
            optimizer = torch.optim.AdamW(model.parameters(), **ADAM_args, fused=True)
        except TypeError:
            optimizer = torch.optim.AdamW(model.parameters(), **ADAM_args, foreach=True)
    # MLP
    elif args.model == "mlp":
        pass
        # todo: implement an mlp
        # model = build_mlp(input_dim=3*32*32, num_layers=4, hidden_dim=256, num_classes=10)
        # model.to(DEVICE)
        # train_loader, val_loader = build_dataset_cifar(workers=args.workers, batch_size=args.batch_size, img_size=args.img_size)
        # 
        # ADAM_args = dict(lr=args.lr, weight_decay=args.weight_decay)
        # try:
        #     optimizer = torch.optim.AdamW(model.parameters(), **ADAM_args, fused=True)
        # except TypeError:
        #     optimizer = torch.optim.AdamW(model.parameters(), **ADAM_args, foreach=True)
        
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print(f"=> loading checkpoint '{args.resume}'")
            checkpoint = torch.load(args.resume, map_location=DEVICE)
            args.start_epoch = checkpoint.get('epoch', 0)
            best_prec1 = checkpoint.get('best_prec1', 0)
            state_dict_key = 'model' if 'model' in checkpoint else 'state_dict'
            model.load_state_dict(checkpoint[state_dict_key])
            if 'optimizer' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer'])
            if 'scheduler' in checkpoint:
                lr_scheduler.load_state_dict(checkpoint['scheduler'])
            print(f"=> loaded checkpoint (epoch {args.start_epoch})")
        else:
            print(f"=> no checkpoint found at '{args.resume}'")

    # Compile model
    if args.use_compile:
        model = torch.compile(model, mode="max-autotune", fullgraph=False, dynamic=True)
    
    for epoch in range(args.start_epoch, args.epochs):
        # Train 
        print('current lr weights {:.5e}'.format(optimizer.param_groups[0]['lr']))

        loss, train_acc = train(train_loader, model, criterion, optimizer, epoch, DEVICE)
        lr_scheduler.step()

        # Validate
        prec1 = validate(val_loader, model, criterion, DEVICE)

        # Remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)

        # Logging
        if True:
            log_dict = {
                "epoch": epoch,
                "loss": loss,
                "train_acc": train_acc,
                "val_acc": prec1,
            }
            wandb.log(log_dict)

        # last checkpoint every epoch (optional: keep only last/best to save space)
        save_checkpoint({
            'epoch': epoch + 1,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': lr_scheduler.state_dict(),
            'best_prec1': best_prec1,
            'args': vars(args),
        }, is_best, filename=os.path.join(args.save_dir, f'checkpoint_{run_name}.th'))


if __name__ == '__main__':
    main()
