"""
SimpleCNN model architecture for MNIST Federated Learning.
"""

import torch
import torch.nn as nn


class SimpleCNN(nn.Module):
    """
    Convolutional Neural Network for MNIST classification.
    Architecture:
      - Conv2d(1, 32, 3) + ReLU + MaxPool2d(2)
      - Conv2d(32, 64, 3) + ReLU + MaxPool2d(2)
      - Flatten (1600)
      - Linear(1600, 128) named fc1 + ReLU
      - Linear(128, 10) named fc2
    The fc2 layer has (128 * 10) + 10 = 1290 parameters.
    """

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2)

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features=1600, out_features=128)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(in_features=128, out_features=num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)
        return x


def get_fc2_num_params(model: SimpleCNN) -> int:
    """Return total number of parameters in the fc2 layer (weight + bias)."""
    return model.fc2.weight.numel() + (model.fc2.bias.numel() if model.fc2.bias is not None else 0)
