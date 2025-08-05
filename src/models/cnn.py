import math
import torch.nn as nn


class SimpleCNN(nn.Module):
    def __init__(self, in_channels: int, input_height: int, input_width: int, num_classes: int,
                 depth: int, width: int, activation: str, pooling: str):
        super().__init__()
        act = nn.ReLU if activation.lower() == 'relu' else nn.Tanh
        pool = nn.MaxPool2d if pooling.lower() == 'max' else nn.AvgPool2d
        layers = []
        channels = in_channels
        for _ in range(depth):
            layers.append(nn.Conv2d(channels, width, kernel_size=3, padding=1))
            layers.append(act())
            layers.append(pool(2))
            channels = width
        self.feature = nn.Sequential(*layers)
        h = input_height // (2 ** depth)
        w = input_width // (2 ** depth)
        self.classifier = nn.Linear(channels * h * w, num_classes)

    def forward(self, x):
        features = self.feature(x)
        logits = self.classifier(features.view(x.size(0), -1))
        return logits

    def get_features(self, x):
        features = self.feature(x)
        return features.view(x.size(0), -1)
