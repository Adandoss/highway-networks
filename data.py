"""CIFAR-10 held as tensors on the training device, so batches need no DataLoader workers."""
from pathlib import Path

import torch
import torch.nn.functional as F
from torchvision.datasets import CIFAR10


def load_split(train):
    dataset = CIFAR10(Path(__file__).parent / "data", train=train, download=True)
    images = torch.from_numpy(dataset.data).permute(0, 3, 1, 2).float()
    # Global contrast normalization, image by image (follow-up, Section 3.3.2).
    mean, std = images.mean((1, 2, 3), keepdim=True), images.std((1, 2, 3), keepdim=True)
    return (images - mean) / std.clamp_min(1.0), torch.tensor(dataset.targets)


def load_cifar10(device, val_size=0):
    """Return {split: (images, labels)}. With val_size > 0 the last training images become a validation
    split and the test set is left out, so a learning-rate search cannot touch it."""
    x, y = load_split(train=True)
    if val_size:
        splits = {"train": (x[:-val_size], y[:-val_size]), "val": (x[-val_size:], y[-val_size:])}
    else:
        splits = {"train": (x, y), "test": load_split(train=False)}
    return {name: (images.to(device), labels.to(device)) for name, (images, labels) in splits.items()}


def augment(images, pad=4):
    """Small translations and mirroring (follow-up, Section 3.3.2), different for every image in the batch."""
    n, _, height, width = images.shape
    padded = F.pad(images, (pad, pad, pad, pad))
    top, left = torch.randint(0, 2 * pad + 1, (2, n), device=images.device)
    rows = (top[:, None] + torch.arange(height, device=images.device))[:, :, None]
    cols = (left[:, None] + torch.arange(width, device=images.device))[:, None, :]
    cropped = padded[torch.arange(n, device=images.device)[:, None, None], :, rows, cols].permute(0, 3, 1, 2)
    mirror = torch.rand(n, device=images.device) < 0.5
    cropped[mirror] = cropped[mirror].flip(-1)
    return cropped


def batches(images, labels, batch_size, augmented=False):
    """Shuffled mini-batches for one epoch."""
    for index in torch.randperm(len(images), device=images.device).split(batch_size):
        yield (augment(images[index]) if augmented else images[index]), labels[index]
