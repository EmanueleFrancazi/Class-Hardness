"""ResNet-50 architecture adapted for the simulation framework."""

import torch.nn as nn
from torchvision.models import resnet50


class ResNet50(nn.Module):
    """ResNet-50 with feature extraction helper.

    Parameters
    ----------
    num_classes: int
        Number of output classes for the final classifier layer.
    small_input: bool, optional
        If ``True`` (default), modify the initial convolution and remove the
        first max-pooling layer so that the network can process small images
        such as those from CIFAR datasets (32×32).
    """

    def __init__(self, num_classes: int, small_input: bool = True):
        super().__init__()
        # Build the standard ResNet-50 from torchvision without pretrained weights
        base = resnet50(weights=None)
        if small_input:
            # Adjust first conv layer and remove maxpool for small images
            base.conv1 = nn.Conv2d(
                base.conv1.in_channels,
                base.conv1.out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            )
            base.maxpool = nn.Identity()
        # ``children()`` returns all modules; we drop the final fully connected layer
        self.feature = nn.Sequential(*list(base.children())[:-1])
        self.classifier = nn.Linear(base.fc.in_features, num_classes)

    def forward(self, x):
        """Compute class logits for a batch of images."""
        features = self.feature(x)
        features = features.view(features.size(0), -1)
        logits = self.classifier(features)
        return logits

    def get_features(self, x):
        """Return flattened features from the penultimate layer."""
        features = self.feature(x)
        return features.view(features.size(0), -1)
