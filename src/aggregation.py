"""
Mathematical implementation of Federated Averaging (FedAvg).
"""

from typing import Dict, List
import copy
import torch


def federated_averaging(
    client_models: dict[int, dict[str, torch.Tensor]],
    client_sample_sizes: dict[int, int],
    selected_clients: list[int]
) -> dict[str, torch.Tensor]:
    """
    Perform mathematical sample-weighted FedAvg over selected clients.
    w_global = sum_{k in selected} (n_k / total_n) * w_k

    Args:
        client_models: Dict mapping client_id -> model state_dict.
        client_sample_sizes: Dict mapping client_id -> number of local training samples.
        selected_clients: List of client IDs whose models will be aggregated.

    Returns:
        dict[str, torch.Tensor]: Aggregated state dict for the global model.
    """
    if not selected_clients:
        raise ValueError("Cannot aggregate empty list of selected clients.")

    # Calculate total samples among selected clients
    total_samples = sum(client_sample_sizes[cid] for cid in selected_clients)
    if total_samples <= 0:
        raise ValueError(f"Total samples among selected clients must be positive, got {total_samples}")

    # Use first selected client to initialize structure
    first_cid = selected_clients[0]
    first_state = client_models[first_cid]
    aggregated_state: dict[str, torch.Tensor] = {}

    for key, tensor in first_state.items():
        weight = client_sample_sizes[first_cid] / total_samples
        aggregated_state[key] = tensor.detach().clone().float() * weight

    # Accumulate the rest of selected clients
    for cid in selected_clients[1:]:
        weight = client_sample_sizes[cid] / total_samples
        c_state = client_models[cid]
        for key in aggregated_state.keys():
            aggregated_state[key] += c_state[key].detach().clone().float() * weight

    # Match original parameter dtypes if necessary (e.g. integer buffers)
    for key, tensor in first_state.items():
        if tensor.dtype in (torch.int64, torch.int32, torch.bool):
            aggregated_state[key] = aggregated_state[key].to(tensor.dtype)

    return aggregated_state
