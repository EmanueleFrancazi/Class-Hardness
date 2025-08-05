"""Dataset utilities for loading and preprocessing data."""

from typing import Tuple, Dict

import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms


class StandardizeTransform:
    """Standardize inputs using dataset statistics and optional shift."""

    def __init__(self, mean: torch.Tensor, std: torch.Tensor, shift: float = 0.0):
        self.mean = mean
        self.std = std
        self.shift = shift

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        # ``Normalize`` from torchvision subtracts mean and divides by std; here
        # we additionally allow shifting the centered data by a constant value.
        return (x - self.mean[..., None, None]) / self.std[..., None, None] + self.shift


def compute_mean_std(dataset) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute channel-wise mean and std of a dataset."""
    loader = DataLoader(dataset, batch_size=256, shuffle=False)
    mean = 0.0
    var = 0.0
    total = 0
    for x, _ in loader:
        c = x.size(0)
        mean += x.mean(dim=[0, 2, 3]) * c
        var += x.var(dim=[0, 2, 3], unbiased=False) * c
        total += c
    mean /= total
    std = torch.sqrt(var / total)
    return mean, std


def get_torchvision_datasets(name: str, shift: float) -> Tuple[TensorDataset, TensorDataset, Tuple[int, int, int], int]:
    """Load a torchvision dataset and apply standardization with optional shift."""
    name = name.lower()
    if name == 'mnist':
        dataset_cls = datasets.MNIST
    elif name == 'cifar10':
        dataset_cls = datasets.CIFAR10
    elif name == 'cifar100':
        dataset_cls = datasets.CIFAR100
    else:
        raise ValueError(f"Unknown dataset {name}")

    # First pass: load with ``ToTensor`` only to compute statistics
    base_transform = transforms.ToTensor()
    train_dataset = dataset_cls(root='data', train=True, download=True, transform=base_transform)
    test_dataset = dataset_cls(root='data', train=False, download=True, transform=base_transform)

    mean, std = compute_mean_std(train_dataset)

    # Update transforms to include standardization and shifting
    standardize = transforms.Compose([
        transforms.ToTensor(),
        StandardizeTransform(mean, std, shift),
    ])
    train_dataset.transform = standardize
    test_dataset.transform = standardize

    input_shape = train_dataset[0][0].shape
    num_classes = len(train_dataset.classes)
    return train_dataset, test_dataset, input_shape, num_classes


def get_gaussian_datasets(cfg: Dict) -> Tuple[TensorDataset, TensorDataset, Tuple[int, int, int], int]:
    """Generate synthetic Gaussian blobs according to the provided configuration."""
    gcfg = cfg['gaussian']
    c, h, w = gcfg['channels'], gcfg['height'], gcfg['width']
    num_classes = gcfg['num_classes']
    mean, std = gcfg.get('mean', 0.0), gcfg.get('std', 1.0)
    train_n = gcfg['train_samples_per_class']
    test_n = gcfg['test_samples_per_class']

    # Generate data for each class
    train_x = mean + std * torch.randn(num_classes * train_n, c, h, w)
    test_x = mean + std * torch.randn(num_classes * test_n, c, h, w)
    train_y = torch.arange(num_classes).repeat_interleave(train_n)
    test_y = torch.arange(num_classes).repeat_interleave(test_n)

    train_dataset = TensorDataset(train_x, train_y)
    test_dataset = TensorDataset(test_x, test_y)

    # Standardize the synthetic data
    dmean, dstd = compute_mean_std(train_dataset)
    transform = StandardizeTransform(dmean, dstd, cfg.get('shift', 0.0))
    train_dataset = TensorDataset(transform(train_x), train_y)
    test_dataset = TensorDataset(transform(test_x), test_y)

    input_shape = (c, h, w)
    return train_dataset, test_dataset, input_shape, num_classes


def get_dataloaders(cfg: Dict):
    """Return training and test dataloaders based on configuration settings."""
    ds_cfg = cfg['dataset']
    name = ds_cfg['name'].lower()
    shift = ds_cfg.get('shift', 0.0)

    if name == 'gaussian':
        train_dataset, test_dataset, input_shape, num_classes = get_gaussian_datasets(ds_cfg)
    else:
        train_dataset, test_dataset, input_shape, num_classes = get_torchvision_datasets(name, shift)

    batch_size = cfg['training']['batch_size']
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader, train_dataset, input_shape, num_classes

