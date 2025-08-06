"""Dataset utilities for loading and preprocessing data.

This module centralizes dataset handling for the simulation framework. It
supports torchvision datasets as well as synthetic Gaussian data and exposes
helpers for standardization and class filtering.
"""

from typing import Tuple, Dict

import numpy as np
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

def _subset_torchvision_dataset(dataset, class_map: Dict[int, int]):
    """Return a subset of ``dataset`` keeping only classes in ``class_map``.

    Parameters
    ----------
    dataset: torchvision Dataset
        Dataset instance whose ``data`` and ``targets`` attributes will be
        filtered **in place**.
    class_map: Dict[int, int]
        Mapping from original class labels to new labels. Classes not present
        in the keys are dropped. The values assign the new label index.
    """

    if not class_map:
        return dataset

    # Convert targets to a tensor for easier masking regardless of storage type
    targets = torch.as_tensor(dataset.targets)
    mask = torch.zeros_like(targets, dtype=torch.bool)
    for old in class_map.keys():
        mask |= targets == int(old)
    indices = mask.nonzero(as_tuple=False).squeeze()

    # Subset the underlying data array (numpy or tensor)
    if isinstance(dataset.data, np.ndarray):
        dataset.data = dataset.data[indices.numpy()]
    else:  # torch.Tensor
        dataset.data = dataset.data[indices]

    # Remap labels to the new values
    targets = targets[indices]
    for old, new in class_map.items():
        targets[targets == int(old)] = int(new)
    dataset.targets = targets.tolist()

    # Optionally shrink the ``classes`` attribute for better introspection
    if hasattr(dataset, "classes"):
        dataset.classes = [dataset.classes[int(old)] for old in class_map.keys()]
    return dataset


def get_torchvision_datasets(cfg: Dict) -> Tuple[TensorDataset, TensorDataset, Tuple[int, int, int], int]:
    """Load a torchvision dataset and apply standardization and class mapping.

    The configuration dictionary ``cfg`` should contain ``name`` (mnist,
    cifar10, cifar100), optional ``shift`` for pixel shifting, and an optional
    ``class_map`` dict specifying which classes to keep and the new label for
    each of them.
    """

    name = cfg['name'].lower()
    shift = cfg.get('shift', 0.0)
    class_map = cfg.get('class_map', {})

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

    # Apply class filtering before computing statistics so that normalization
    # is based solely on the selected subset.
    train_dataset = _subset_torchvision_dataset(train_dataset, class_map)
    test_dataset = _subset_torchvision_dataset(test_dataset, class_map)
    
    mean, std = compute_mean_std(train_dataset)

    # Update transforms to include standardization and shifting
    standardize = transforms.Compose([
        transforms.ToTensor(),
        StandardizeTransform(mean, std, shift),
    ])
    train_dataset.transform = standardize
    test_dataset.transform = standardize

    input_shape = train_dataset[0][0].shape
    num_classes = len(class_map) if class_map else len(train_dataset.classes)

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

    # Optionally subset and relabel classes according to ``class_map``
    class_map = cfg.get('class_map', {})
    if class_map:
        train_mask = torch.zeros_like(train_y, dtype=torch.bool)
        test_mask = torch.zeros_like(test_y, dtype=torch.bool)
        for old in class_map.keys():
            train_mask |= train_y == int(old)
            test_mask |= test_y == int(old)
        train_x, train_y = train_x[train_mask], train_y[train_mask]
        test_x, test_y = test_x[test_mask], test_y[test_mask]
        for old, new in class_map.items():
            train_y[train_y == int(old)] = int(new)
            test_y[test_y == int(old)] = int(new)
        num_classes = len(class_map)

    train_dataset = TensorDataset(train_x, train_y)
    test_dataset = TensorDataset(test_x, test_y)

    # Standardize the synthetic data using statistics from the filtered data
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
    if name == 'gaussian':
        train_dataset, test_dataset, input_shape, num_classes = get_gaussian_datasets(ds_cfg)
    else:
        train_dataset, test_dataset, input_shape, num_classes = get_torchvision_datasets(ds_cfg)

    batch_size = cfg['training']['batch_size']
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader, train_dataset, input_shape, num_classes

