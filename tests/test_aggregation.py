"""
Tests for mathematical sample-weighted FedAvg aggregation.
"""

import pytest
import torch
from src.aggregation import federated_averaging


def test_federated_averaging_weighted_math():
    """Verify sample-weighted FedAvg aggregates parameters with exact mathematical precision."""
    # Client 0 has 100 samples with param values 2.0
    # Client 1 has 300 samples with param values 6.0
    # Total samples = 400
    # Expected: (100/400)*2.0 + (300/400)*6.0 = 0.5 + 4.5 = 5.0
    state_0 = {"w": torch.tensor([2.0, 4.0]), "b": torch.tensor([10.0])}
    state_1 = {"w": torch.tensor([6.0, 8.0]), "b": torch.tensor([2.0])}

    client_models = {0: state_0, 1: state_1}
    client_samples = {0: 100, 1: 300}
    selected = [0, 1]

    agg = federated_averaging(client_models, client_samples, selected)

    # For w[0]: 0.25 * 2.0 + 0.75 * 6.0 = 5.0
    # For w[1]: 0.25 * 4.0 + 0.75 * 8.0 = 1.0 + 6.0 = 7.0
    # For b[0]: 0.25 * 10.0 + 0.75 * 2.0 = 2.5 + 1.5 = 4.0
    assert torch.allclose(agg["w"], torch.tensor([5.0, 7.0]))
    assert torch.allclose(agg["b"], torch.tensor([4.0]))


def test_federated_averaging_subset_selection():
    """Verify FedAvg works correctly on selected subsets of clients."""
    state_0 = {"w": torch.tensor([10.0])}
    state_1 = {"w": torch.tensor([20.0])}
    state_2 = {"w": torch.tensor([100.0])}  # Malicious outlier

    client_models = {0: state_0, 1: state_1, 2: state_2}
    client_samples = {0: 50, 1: 50, 2: 100}

    # Only aggregate honest clients 0 and 1
    agg = federated_averaging(client_models, client_samples, selected_clients=[0, 1])
    assert torch.allclose(agg["w"], torch.tensor([15.0]))


def test_federated_averaging_empty_clients_raises():
    """Verify ValueError is raised if selected client list is empty."""
    with pytest.raises(ValueError):
        federated_averaging({0: {}}, {0: 10}, [])
