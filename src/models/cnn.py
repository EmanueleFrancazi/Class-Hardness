"""Simple configurable convolutional neural network (CNN).

This module defines a minimal CNN architecture whose depth, width, activation
function and pooling strategy are driven by the configuration file. Similar to
the MLP implementation, the network exposes a ``get_features`` helper used for
representation analysis.
"""
import torch.nn as nn


class SimpleCNN(nn.Module):
    """CNN composed of stacked ``Conv2d`` layers followed by pooling."""

    def __init__(self, in_channels: int, input_height: int, input_width: int, num_classes: int,
                 depth: int, width: int, activation: str, pooling: str):
        super().__init__()
        # Choose activation and pooling modules based on config strings
        act = nn.ReLU if activation.lower() == 'relu' else nn.Tanh
        pool = nn.MaxPool2d if pooling.lower() == 'max' else nn.AvgPool2d

        layers = []
        channels = in_channels
        for _ in range(depth):
            # Convolution -> activation -> pooling forms one block
            layers.append(nn.Conv2d(channels, width, kernel_size=3, padding=1))
            layers.append(act())
            layers.append(pool(2))
            channels = width  # update channel count for next iteration

        # Sequential feature extractor composed of the stacked blocks above
        self.feature = nn.Sequential(*layers)

        # Determine size of flattened features after the convolutional blocks
        h = input_height // (2 ** depth)
        w = input_width // (2 ** depth)

        # Final linear classifier operating on flattened feature maps
        self.classifier = nn.Linear(channels * h * w, num_classes)

    def forward(self, x):
        """Compute class logits for a batch of images."""
        features = self.feature(x)
        logits = self.classifier(features.view(x.size(0), -1))
        return logits

    def get_features(self, x):
        """Return flattened features from the last convolutional layer."""
        features = self.feature(x)
        return features.view(x.size(0), -1)
