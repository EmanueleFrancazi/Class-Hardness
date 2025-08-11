"""Customizable ResNet-50 backbone."""

from typing import Dict, Tuple, Type

import torch

import torch.nn as nn
from torchvision.models.resnet import ResNet, Bottleneck



def _get_global_pool(name: str) -> nn.Module:
    name = (name or "avg").lower()
    if name in {"avg", "adaptive_avg"}:
        return nn.AdaptiveAvgPool2d((1, 1))
    if name in {"max", "adaptive_max"}:
        return nn.AdaptiveMaxPool2d((1, 1))
    if name in {"none", "identity"}:
        return nn.Identity()
    raise ValueError(f"Unknown pooling: {name}")


def _activation_spec(name: str) -> Tuple[Type[nn.Module], Dict]:
    """Return the activation class and kwargs for ``name``."""
    name = (name or "relu").lower()
    if name == "relu":
        return nn.ReLU, {"inplace": True}
    if name == "gelu":
        return nn.GELU, {}
    if name == "silu":
        return nn.SiLU, {"inplace": True}
    if name == "leaky_relu":
        return nn.LeakyReLU, {"inplace": True, "negative_slope": 0.01}
    if name == "tanh":
        return nn.Tanh, {}
    raise ValueError(f"Unknown activation: {name}")


def _replace_relu_with_factory(module: nn.Module, act_cls: Type[nn.Module], act_kwargs: Dict):
    """Recursively replace ``nn.ReLU`` modules with ``act_cls`` instances."""
    for name, child in list(module.named_children()):
        if isinstance(child, nn.ReLU):
            setattr(module, name, act_cls(**act_kwargs))
        else:
            _replace_relu_with_factory(child, act_cls, act_kwargs)


class ResNetCustom(ResNet):
    """ResNet50 with pluggable activation and global pooling.

    Parameters
    ----------
    num_classes: int
        Number of output classes.
    activation: str, optional
        ``'relu'``, ``'tanh'``, ``'gelu'``, ``'silu'`` or ``'leaky_relu'``.

    global_pool: str, optional
        ``'avg'``, ``'max'`` or ``'none'`` for the global pooling layer.
    use_cifar_stem: bool, optional
        If ``True``, replace the initial stem with a 3×3 stride-1 convolution and
        remove the maxpool, which is more suitable for 32×32 images.
    """

    def __init__(self, num_classes: int, activation: str = "relu",
                 global_pool: str = "avg", use_cifar_stem: bool = True):
        super().__init__(block=Bottleneck, layers=[3, 4, 6, 3], num_classes=num_classes)

        # Swap activations throughout the network
        act_cls, act_kwargs = _activation_spec(activation)
        self.relu = act_cls(**act_kwargs)
        _replace_relu_with_factory(self, act_cls, act_kwargs)


        # Replace global pooling
        self.avgpool = _get_global_pool(global_pool)

        # Optionally adapt the stem for CIFAR-style inputs
        if use_cifar_stem:
            inplanes = 64
            self.conv1 = nn.Conv2d(3, inplanes, kernel_size=3, stride=1, padding=1, bias=False)
            self.maxpool = nn.Identity()

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return flattened features from the penultimate layer.

        The method mirrors :func:`ResNet.forward` up to (but excluding) the
        final fully connected layer, providing representations suitable for
        overlap computations.
        """

        # Stem
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # Residual stages
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # Global pooling and flatten
        x = self.avgpool(x)
        return torch.flatten(x, 1)


if __name__ == "__main__":
    import torch

    model = ResNetCustom(num_classes=10, activation="tanh", global_pool="avg", use_cifar_stem=True)
    x = torch.randn(2, 3, 32, 32)
    y = model(x)
    assert y.shape == (2, 10)


