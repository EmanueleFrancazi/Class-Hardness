import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, input_dim: int, num_classes: int, depth: int, width: int, activation: str):
        super().__init__()
        act = nn.ReLU if activation.lower() == 'relu' else nn.Tanh
        layers = []
        in_dim = input_dim
        for _ in range(depth):
            layers.append(nn.Linear(in_dim, width))
            layers.append(act())
            in_dim = width
        self.feature = nn.Sequential(*layers)
        self.classifier = nn.Linear(in_dim, num_classes)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        features = self.feature(x)
        logits = self.classifier(features)
        return logits

    def get_features(self, x):
        x = x.view(x.size(0), -1)
        features = self.feature(x)
        return features
