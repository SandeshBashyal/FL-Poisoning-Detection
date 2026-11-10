"""
Tests for Client local training functionality.
"""

import pytest
import torch
from torch.utils.data import TensorDataset
from src.model import SimpleCNN
from src.client import Client


def test_client_local_training():
    """Verify Client.train updates weights and returns finite training loss."""
    # Synthetic small dataset
    x = torch.randn(64, 1, 28, 28)
    y = torch.randint(0, 10, (64,))
    dataset = TensorDataset(x, y)

    client = Client(
        client_id=0,
        dataset=dataset,
        batch_size=32,
        local_epochs=1,
        lr=0.01,
        momentum=0.9,
        weight_decay=0.0001,
        device=torch.device("cpu")
    )

    initial_model = SimpleCNN()
    initial_state = initial_model.state_dict()

    new_state, loss, num_samples = client.train(initial_state)

    assert num_samples == 64
    assert isinstance(loss, float)
    assert not torch.isnan(torch.tensor(loss))
    assert loss > 0.0

    # Ensure weights actually changed during training
    weight_diff = (new_state["fc2.weight"] - initial_state["fc2.weight"]).abs().sum().item()
    assert weight_diff > 0.0, "Client training did not update weights!"
