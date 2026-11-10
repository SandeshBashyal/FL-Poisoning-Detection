"""
Layer-wise cosine similarity filtering defense for Federated Learning.
"""

import logging
from typing import Dict, List, Tuple
import numpy as np
import torch

logger = logging.getLogger(__name__)


def extract_layer_update_vector(
    client_state_dict: dict[str, torch.Tensor],
    global_state_dict: dict[str, torch.Tensor],
    target_layer_prefix: str = "fc2"
) -> torch.Tensor:
    """
    Extract weight and bias update for target layer (e.g. fc2) and flatten into a 1D vector.
    For SimpleCNN fc2:
      fc2.weight: (10, 128) -> 1280 params
      fc2.bias:   (10,)     -> 10 params
      Total: 1290 params.
    """
    weight_key = f"{target_layer_prefix}.weight"
    bias_key = f"{target_layer_prefix}.bias"

    delta_w = (client_state_dict[weight_key].detach().cpu() - global_state_dict[weight_key].detach().cpu()).flatten()
    delta_b = (client_state_dict[bias_key].detach().cpu() - global_state_dict[bias_key].detach().cpu()).flatten()

    vector = torch.cat([delta_w, delta_b], dim=0)
    return vector


def compute_pairwise_cosine_distance_matrix(
    update_vectors: torch.Tensor,
    eps: float = 1e-12
) -> np.ndarray:
    """
    Calculate pairwise cosine distance matrix between client update vectors.
    Cosine distance D_ij = 1 - (u_i . u_j) / (||u_i|| * ||u_j||).

    Args:
        update_vectors: Tensor of shape (num_clients, vector_dim).
        eps: Small epsilon for numerical stability.

    Returns:
        np.ndarray of shape (num_clients, num_clients)
    """
    num_clients = update_vectors.shape[0]
    # Normalize vectors
    norms = torch.norm(update_vectors, p=2, dim=1, keepdim=True)
    # Safeguard against zero-norm
    safe_norms = torch.clamp(norms, min=eps)
    normalized_vectors = update_vectors / safe_norms

    # Cosine similarity matrix: S_ij in [-1, 1]
    cos_sim = torch.mm(normalized_vectors, normalized_vectors.t())
    # Numerical clamp
    cos_sim = torch.clamp(cos_sim, -1.0, 1.0)

    # Cosine distance: D_ij = 1 - S_ij
    cos_dist = 1.0 - cos_sim
    # Force diagonal to 0
    cos_dist.fill_diagonal_(0.0)

    return cos_dist.numpy()


def compute_anomaly_scores(distance_matrix: np.ndarray) -> np.ndarray:
    """
    Compute each client's anomaly score: average distance to other clients excluding itself (diagonal).
    """
    num_clients = distance_matrix.shape[0]
    if num_clients <= 1:
        return np.zeros(num_clients)

    # Sum rows and divide by (num_clients - 1), since diagonal is 0
    scores = np.sum(distance_matrix, axis=1) / (num_clients - 1)
    return scores


class CosineSimilarityDefense:
    """
    Layer-wise Cosine Similarity Filtering Defense.
    Filters out anomalous client updates whose average cosine distance exceeds
    mean(scores) + multiplier * std(scores).
    Enforces minimum accepted clients safeguard.
    """

    def __init__(
        self,
        target_layer: str = "fc2",
        multiplier: float = 1.5,
        min_accepted_clients: int = 3,
        eps: float = 1e-12
    ) -> None:
        self.target_layer = target_layer
        self.multiplier = multiplier
        self.min_accepted_clients = min_accepted_clients
        self.eps = eps

    def filter_clients(
        self,
        client_updates: dict[int, dict[str, torch.Tensor]],
        global_state_dict: dict[str, torch.Tensor]
    ) -> dict:
        """
        Execute cosine similarity filtering defense.

        Args:
            client_updates: Dict mapping client_id -> client_state_dict.
            global_state_dict: Global model state dict before aggregation.

        Returns:
            Dict containing:
                'accepted_clients': list of accepted client IDs,
                'rejected_clients': list of rejected client IDs,
                'anomaly_scores': dict of client_id -> score,
                'threshold': float threshold value,
                'distance_matrix': 2D numpy array,
                'client_order': list of client IDs matching matrix rows/cols,
                'fallback_triggered': bool indicating if min client fallback occurred
        """
        client_ids = sorted(list(client_updates.keys()))
        num_clients = len(client_ids)

        # 1. Extract layer update vectors
        vectors = []
        for cid in client_ids:
            vec = extract_layer_update_vector(client_updates[cid], global_state_dict, self.target_layer)
            vectors.append(vec)
        vectors_tensor = torch.stack(vectors, dim=0)

        # 2. Pairwise cosine distance matrix
        dist_matrix = compute_pairwise_cosine_distance_matrix(vectors_tensor, eps=self.eps)

        # 3. Anomaly scores (average distance to all other clients)
        scores_arr = compute_anomaly_scores(dist_matrix)
        scores_dict = {cid: float(score) for cid, score in zip(client_ids, scores_arr)}

        # 4. Anomaly threshold: mean + multiplier * std
        mean_score = float(np.mean(scores_arr))
        std_score = float(np.std(scores_arr))
        threshold = mean_score + self.multiplier * std_score

        # 5. Filter clients
        accepted = [cid for cid, s in scores_dict.items() if s <= threshold]
        rejected = [cid for cid, s in scores_dict.items() if s > threshold]

        fallback_triggered = False
        # 6. Minimum accepted-client safeguard
        if len(accepted) < self.min_accepted_clients:
            fallback_triggered = True
            logger.warning(
                f"Defense accepted {len(accepted)} clients, below minimum {self.min_accepted_clients}. "
                f"Falling back to retaining the {self.min_accepted_clients} lowest-scoring clients."
            )
            # Sort all clients by score ascending and take top min_accepted_clients
            sorted_by_score = sorted(client_ids, key=lambda c: scores_dict[c])
            accepted = sorted_by_score[:self.min_accepted_clients]
            rejected = sorted_by_score[self.min_accepted_clients:]

        print("\n--- Cosine Defense Diagnostics ---")
        print(f"Mean score: {mean_score:.6f}")
        print(f"Std score: {std_score:.6f}")
        print(f"Threshold: {threshold:.6f}")

        for cid in client_ids:
            status = "REJECTED" if cid in rejected else "ACCEPTED"
            print(
                f"Client {cid}: "
                f"score={scores_dict[cid]:.6f} | {status}"
            )

        return {
            "accepted_clients": accepted,
            "rejected_clients": rejected,
            "anomaly_scores": scores_dict,
            "threshold": threshold,
            "mean_score": mean_score,
            "std_score": std_score,
            "distance_matrix": dist_matrix,
            "client_order": client_ids,
            "fallback_triggered": fallback_triggered
        }
