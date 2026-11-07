"""CIFAR-10 held as tensors on the training device, so batches need no DataLoader workers."""
from pathlib import Path

import torch
import torch.nn.functional as F
from torchvision.datasets import CIFAR10

DATA = Path(__file__).parent / "data"


def load_split(train):
    """The training or test images, normalized, and their labels."""
    dataset = CIFAR10(DATA, train=train, download=True)
    images = torch.from_numpy(dataset.data).permute(0, 3, 1, 2).float()
    # Global contrast normalization, image by image (follow-up, Section 3.3.2). Pixels run from 0 to 255, so the
    # floor of 1 on the standard deviation only holds back nearly blank images.
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
    """Random translations by up to `pad` pixels and mirroring (follow-up, Section 3.3.2), different for every image."""
    n, _, height, width = images.shape
    device = images.device
    padded = F.pad(images, (pad, pad, pad, pad))
    top, left = torch.randint(0, 2 * pad + 1, (2, n, 1, 1), device=device)  # each image's crop corner
    rows = top + torch.arange(height, device=device)[:, None]  # (n, height, 1)
    cols = left + torch.arange(width, device=device)  # (n, 1, width)
    batch = torch.arange(n, device=device)[:, None, None]  # (n, 1, 1)
    # The three index tensors broadcast to (n, height, width); indexing moves the channel dimension to the end.
    cropped = padded[batch, :, rows, cols].permute(0, 3, 1, 2)
    mirror = torch.rand(n, device=device) < 0.5
    cropped[mirror] = cropped[mirror].flip(-1)
    return cropped


def batches(images, labels, batch_size, augmented=False):
    """Shuffled mini-batches for one epoch."""
    for index in torch.randperm(len(images), device=images.device).split(batch_size):
        yield (augment(images[index]) if augmented else images[index]), labels[index]
