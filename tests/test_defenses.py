"""
Tests for layer-wise cosine similarity defense, anomaly detection, and fallback safeguard.
"""

import pytest
import numpy as np
import torch
from src.model import SimpleCNN
from src.defenses import (
    extract_layer_update_vector,
    compute_pairwise_cosine_distance_matrix,
    compute_anomaly_scores,
    CosineSimilarityDefense
)


def test_extract_layer_update_vector_fc2_dimension():
    """Verify extracted update vector has exactly 1,290 dimensions for fc2."""
    model_client = SimpleCNN()
    model_global = SimpleCNN()

    client_state = model_client.state_dict()
    global_state = model_global.state_dict()

    update_vec = extract_layer_update_vector(client_state, global_state, target_layer_prefix="fc2")
    assert update_vec.shape == (1290,), f"Expected shape (1290,), got {update_vec.shape}"


def test_cosine_distance_matrix_properties():
    """Verify cosine distance matrix has zero diagonal, is symmetric, and handles zero vectors."""
    # 3 vectors: v0 and v1 identical direction, v2 orthogonal, v3 zero vector
    v0 = torch.tensor([1.0, 0.0, 0.0])
    v1 = torch.tensor([2.0, 0.0, 0.0])  # Same direction as v0 -> dist = 0
    v2 = torch.tensor([0.0, 1.0, 0.0])  # Orthogonal -> dist = 1.0
    v3 = torch.tensor([0.0, 0.0, 0.0])  # Zero vector -> safeguard

    vectors = torch.stack([v0, v1, v2, v3])
    dist_matrix = compute_pairwise_cosine_distance_matrix(vectors, eps=1e-8)

    assert dist_matrix.shape == (4, 4)
    # Diagonal must be 0
    for i in range(4):
        assert np.isclose(dist_matrix[i, i], 0.0)

    # Identical direction: distance between 0 and 1 is 0
    assert np.isclose(dist_matrix[0, 1], 0.0, atol=1e-5)
    assert np.isclose(dist_matrix[1, 0], 0.0, atol=1e-5)

    # Orthogonal: distance between 0 and 2 is 1.0
    assert np.isclose(dist_matrix[0, 2], 1.0, atol=1e-5)

    # Symmetric
    assert np.allclose(dist_matrix, dist_matrix.T, atol=1e-5)


def test_compute_anomaly_scores():
    """Verify anomaly scores exclude diagonal entries."""
    # Matrix where client 0 is far from everyone (distance 1.0), while clients 1 and 2 are close (dist 0.2)
    dist_matrix = np.array([
        [0.0, 1.0, 1.0],
        [1.0, 0.0, 0.2],
        [1.0, 0.2, 0.0]
    ])
    scores = compute_anomaly_scores(dist_matrix)
    # Client 0 score: (1.0 + 1.0)/2 = 1.0
    # Client 1 score: (1.0 + 0.2)/2 = 0.6
    # Client 2 score: (1.0 + 0.2)/2 = 0.6
    assert np.isclose(scores[0], 1.0)
    assert np.isclose(scores[1], 0.6)
    assert np.isclose(scores[2], 0.6)


def test_cosine_similarity_defense_filtering():
    """Verify defense correctly detects anomalous outlier clients."""
    global_model = SimpleCNN()
    global_state = global_model.state_dict()

    # Create 10 clients: 8 honest with similar small updates, 2 malicious with huge opposite updates
    client_updates = {}
    for i in range(10):
        c_state = {k: v.clone() for k, v in global_state.items()}
        if i in (2, 6):  # Malicious
            c_state["fc2.weight"] = c_state["fc2.weight"] - 5.0
            c_state["fc2.bias"] = c_state["fc2.bias"] - 5.0
        else:  # Honest
            c_state["fc2.weight"] = c_state["fc2.weight"] + 0.1 * (i + 1)
            c_state["fc2.bias"] = c_state["fc2.bias"] + 0.1 * (i + 1)
        client_updates[i] = c_state

    defense = CosineSimilarityDefense(target_layer="fc2", multiplier=1.5, min_accepted_clients=3)
    result = defense.filter_clients(client_updates, global_state)

    # Check structure
    assert "accepted_clients" in result
    assert "rejected_clients" in result
    assert "anomaly_scores" in result
    assert "threshold" in result
    assert len(result["anomaly_scores"]) == 10

    # Malicious clients should have high anomaly scores and be rejected
    assert 2 in result["rejected_clients"]
    assert 6 in result["rejected_clients"]


def test_minimum_accepted_client_safeguard():
    """Verify fallback retains the 3 lowest-scoring clients if threshold rejects too many."""
    global_model = SimpleCNN()
    global_state = global_model.state_dict()

    # Construct scenario where an extremely high multiplier or extreme divergence would reject almost everyone
    client_updates = {}
    for i in range(10):
        c_state = {k: v.clone() for k, v in global_state.items()}
        c_state["fc2.weight"] = c_state["fc2.weight"] + torch.randn_like(c_state["fc2.weight"]) * (i + 1) * 10
        client_updates[i] = c_state

    # Set extreme negative multiplier so threshold is tiny, which would reject all
    defense = CosineSimilarityDefense(target_layer="fc2", multiplier=-10.0, min_accepted_clients=3)
    result = defense.filter_clients(client_updates, global_state)

    assert result["fallback_triggered"] is True
    assert len(result["accepted_clients"]) == 3
    assert len(result["rejected_clients"]) == 7
