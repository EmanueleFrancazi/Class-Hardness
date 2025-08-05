import torch.nn as nn
import torch.nn.init as init


def initialize_weights(model: nn.Module, method: str):
    for m in model.modules():
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            if method == 'kaiming_normal':
                init.kaiming_normal_(m.weight)
            elif method == 'kaiming_uniform':
                init.kaiming_uniform_(m.weight)
            else:
                raise ValueError(f"Unknown initialization {method}")
            if m.bias is not None:
                init.zeros_(m.bias)
