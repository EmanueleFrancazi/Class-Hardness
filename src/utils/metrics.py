"""Metric computation utilities used during evaluation."""

import torch
from torch.utils.data import DataLoader
from typing import Tuple


def compute_overlap(class_means: torch.Tensor) -> torch.Tensor:
    """Compute cosine overlap matrix between class mean vectors."""
    normed = class_means / class_means.norm(dim=1, keepdim=True)
    return normed @ normed.t()


def compute_static_overlap(dataset, device: torch.device, num_classes: int) -> torch.Tensor:
    """Compute class overlaps using raw input features before training.

    For each class we average its data points to obtain a mean representation
    and then compute pairwise cosine overlaps between these means.
    """
    loader = DataLoader(dataset, batch_size=256, shuffle=False)
    first_sample = dataset[0][0]
    feat_dim = first_sample.numel()
    sums = torch.zeros(num_classes, feat_dim, device=device)
    counts = torch.zeros(num_classes, device=device)
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device).view(x.size(0), -1)
            y = y.to(device)
            for c in range(num_classes):
                mask = y == c
                if mask.any():
                    # Accumulate feature sums and counts for each class
                    sums[c] += x[mask].sum(dim=0)
                    counts[c] += mask.sum().item()
    means = sums / counts.unsqueeze(1)
    return compute_overlap(means).cpu()


def evaluate(model, data_loader: DataLoader, device: torch.device, num_classes: int, criterion):
    """Evaluate a model, returning various accuracy, loss and overlap metrics."""
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total = 0
    class_loss = torch.zeros(num_classes, device=device)
    class_correct = torch.zeros(num_classes, device=device)
    class_count = torch.zeros(num_classes, device=device)
    feature_sums = None
    with torch.no_grad():
        for x, y in data_loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            preds = logits.argmax(dim=1)

            # Update global loss and accuracy statistics
            total_loss += loss.item() * x.size(0)
            total_correct += (preds == y).sum().item()
            total += x.size(0)

            # Per-class loss and accuracy
            for c in range(num_classes):
                mask = y == c
                if mask.any():
                    class_loss[c] += criterion(logits[mask], y[mask]).item() * mask.sum().item()
                    class_correct[c] += (preds[mask] == c).sum().item()
                    class_count[c] += mask.sum().item()

            # Accumulate hidden features for dynamic overlap computation
            feats = model.get_features(x)
            if feature_sums is None:
                feature_sums = torch.zeros(num_classes, feats.size(1), device=device)
            for c in range(num_classes):
                mask = y == c
                if mask.any():
                    feature_sums[c] += feats[mask].sum(dim=0)

    # Compute overlaps of hidden representations using class means
    class_means = feature_sums / class_count.unsqueeze(1)
    overlaps = compute_overlap(class_means)
    metrics = {
        'loss': total_loss / total,
        'acc': total_correct / total,
        'loss_per_class': (class_loss / class_count).cpu(),
        'acc_per_class': (class_correct / class_count).cpu(),
        'overlap': overlaps.cpu(),
    }
    return metrics
