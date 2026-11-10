"""
Tests for SimpleCNN model architecture and parameter counts.
"""

import torch
import pytest
from src.model import SimpleCNN, get_fc2_num_params


def test_simple_cnn_forward():
    """Verify SimpleCNN forward pass output shape is (batch_size, 10)."""
    model = SimpleCNN(num_classes=10)
    batch_size = 16
    x = torch.randn(batch_size, 1, 28, 28)
    out = model(x)
    assert out.shape == (batch_size, 10), f"Expected shape ({batch_size}, 10), got {out.shape}"


def test_fc2_parameter_count():
    """Verify fc2 layer contains exactly 1,290 parameters (128*10 weight + 10 bias)."""
    model = SimpleCNN(num_classes=10)
    fc2_params = get_fc2_num_params(model)
    expected_params = (128 * 10) + 10  # 1280 + 10 = 1290
    assert fc2_params == 1290, f"Expected 1,290 fc2 parameters, got {fc2_params}"
    assert model.fc2.weight.shape == (10, 128)
    assert model.fc2.bias.shape == (10,)
