"""Definition of a simple multi-layer perceptron (MLP) model.

The module exposes a configurable MLP where depth, width and activation
function are all provided by a configuration file. The network is split into
two parts:

* ``feature``: stack of fully connected layers with activation in between
* ``classifier``: final layer mapping features to the desired number of classes

The ``get_features`` method allows external modules to extract the hidden
representation of a batch without passing through the classifier, which is
useful for measuring representation overlaps.
"""

import torch.nn as nn


class MLP(nn.Module):
    """Configurable MLP supporting variable depth, width and activation."""

    def __init__(self, input_dim: int, num_classes: int, depth: int, width: int, activation: str):
        super().__init__()
        # Select activation class based on provided name
        act = nn.ReLU if activation.lower() == 'relu' else nn.Tanh

        layers = []
        in_dim = input_dim
        for _ in range(depth):
            # Each iteration adds one linear layer followed by the activation
            layers.append(nn.Linear(in_dim, width))
            layers.append(act())
            in_dim = width  # output dimension becomes input for next layer

        # Sequential container holding the feature extractor part of the network
        self.feature = nn.Sequential(*layers)
        # Final linear layer mapping features to class logits
        self.classifier = nn.Linear(in_dim, num_classes)

    def forward(self, x):
        """Run the MLP on a batch of images and return class logits."""
        # Flatten input images to 2D [batch, features]
        x = x.view(x.size(0), -1)
        features = self.feature(x)
        logits = self.classifier(features)
        return logits

    def get_features(self, x):
        """Return hidden representations before the classifier layer."""
        x = x.view(x.size(0), -1)
        features = self.feature(x)
        return features
