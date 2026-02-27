import numpy as np
import torch
import torch.nn as nn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets



def build_dataset_cifar(workers, batch_size, img_size):
    normalize = transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2470, 0.2435, 0.2616],
    )
    
    # TRAIN
    train_transform_list = [
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
    ]

    if img_size != 32:
        train_transform_list.append(transforms.Resize((img_size, img_size)))

    train_transform_list.extend([
        transforms.RandAugment(num_ops=2, magnitude=9),
        transforms.ToTensor(),
        normalize,
    ])
    train_transform = transforms.Compose(train_transform_list)

    # VAL
    val_transform_list = []
    if img_size != 32:
        val_transform_list.append(transforms.Resize((img_size, img_size)))

    val_transform_list.extend([
        transforms.ToTensor(),
        normalize,
    ])
    val_transform = transforms.Compose(val_transform_list)

    # DATALOADERS
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(
            root='./data',
            train=True,
            transform=train_transform,
            download=True,
        ),
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=True,
    )

    val_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(
            root='./data',
            train=False,
            transform=val_transform,
        ),
        batch_size=128,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
    )

    return train_loader, val_loader
