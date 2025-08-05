"""Utilities for initializing neural network weights."""

import torch.nn as nn
import torch.nn.init as init


def initialize_weights(model: nn.Module, method: str):
    """Initialize ``Linear`` and ``Conv2d`` layers using the selected method."""
    for m in model.modules():
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            # Apply the requested initialization scheme to the weights
            if method == 'kaiming_normal':
                init.kaiming_normal_(m.weight)
            elif method == 'kaiming_uniform':
                init.kaiming_uniform_(m.weight)
            else:
                raise ValueError(f"Unknown initialization {method}")

            # Biases are set to zero for simplicity
            if m.bias is not None:
                init.zeros_(m.bias)
