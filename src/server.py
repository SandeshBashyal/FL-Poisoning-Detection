"""
Federated Learning Server managing communication rounds, defense filtering, and aggregation.
"""

import copy
import logging
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

from src.model import SimpleCNN
from src.client import Client
from src.defenses import CosineSimilarityDefense
from src.aggregation import federated_averaging
from src.evaluation import evaluate_model_on_test_data, compute_detection_metrics

logger = logging.getLogger(__name__)


class Server:
    """
    Federated Learning Server coordinating clients across communication rounds.
    Supports Clean FedAvg, Attacked FedAvg, Cosine Defense, and Honest Oracle.
    """

    def __init__(
        self,
        test_dataset: Dataset,
        clients: dict[int, Client],
        defense: Optional[CosineSimilarityDefense] = None,
        malicious_client_ids: list[int] = None,
        device: torch.device = None,
        initial_state_dict: dict[str, torch.Tensor] = None
    ) -> None:
        self.test_dataset = test_dataset
        self.clients = clients
        self.all_client_ids = sorted(list(clients.keys()))
        self.defense = defense
        self.malicious_client_ids = set(malicious_client_ids or [2, 6])
        self.device = device or torch.device("cpu")

        # Global model
        self.global_model = SimpleCNN().to(self.device)
        if initial_state_dict is not None:
            self.global_model.load_state_dict(initial_state_dict)

        self.last_defense_result = None

    def run_round(
        self,
        round_idx: int,
        mode: str = "clean_fedavg"
    ) -> dict:
        """
        Execute one communication round of federated learning.

        Modes:
          - 'clean_fedavg': No attack, all clients aggregated.
          - 'attacked_fedavg': Label flipping on malicious clients, all clients aggregated.
          - 'cosine_defense': Label flipping on malicious clients, cosine similarity defense filters updates.
          - 'honest_only_oracle': Only honest clients are aggregated.

        Returns:
            Dictionary of metrics for the communication round.
        """
        # 1. Distribute global model to all clients
        global_state = copy.deepcopy(self.global_model.state_dict())
        global_state_cpu = {k: v.detach().cpu().clone() for k, v in global_state.items()}

        # 2. Local training on all participating clients
        client_updates = {}
        client_losses = {}
        client_samples = {}

        for cid in self.all_client_ids:
            client = self.clients[cid]
            c_state, c_loss, c_num_samples = client.train(global_state_cpu)
            client_updates[cid] = c_state
            client_losses[cid] = c_loss
            client_samples[cid] = c_num_samples

        # 3. Determine accepted and rejected clients
        defense_info = None
        if mode == "cosine_defense":
            if self.defense is None:
                raise ValueError("Cosine defense instance must be provided for 'cosine_defense' mode.")
            defense_result = self.defense.filter_clients(client_updates, global_state_cpu)
            accepted = defense_result["accepted_clients"]
            rejected = defense_result["rejected_clients"]
            self.last_defense_result = defense_result
            defense_info = defense_result
            det_metrics = compute_detection_metrics(rejected, self.malicious_client_ids, self.all_client_ids)
        elif mode == "honest_only_oracle":
            accepted = [cid for cid in self.all_client_ids if cid not in self.malicious_client_ids]
            rejected = [cid for cid in self.all_client_ids if cid in self.malicious_client_ids]
            # Oracle benchmark: ground-truth identity exclusion reference
            det_metrics = compute_detection_metrics(rejected, self.malicious_client_ids, self.all_client_ids)
        elif mode in ("clean_fedavg", "attacked_fedavg"):
            accepted = list(self.all_client_ids)
            rejected = []
            # No detector running in baseline FedAvg; detection metrics are Not Applicable (N/A)
            det_metrics = {
                "tpr": None,
                "fpr": None,
                "tp": None,
                "fp": None,
                "tn": None,
                "fn": None
            }
        else:
            raise ValueError(f"Unknown mode: {mode}")

        # 4. Aggregate updates using sample-weighted FedAvg
        new_global_state = federated_averaging(client_updates, client_samples, accepted)
        # Move back to server device
        new_global_state_dev = {k: v.to(self.device) for k, v in new_global_state.items()}
        self.global_model.load_state_dict(new_global_state_dev)

        # 5. Global model evaluation on clean test set
        eval_metrics = evaluate_model_on_test_data(
            self.global_model,
            self.test_dataset,
            batch_size=128,
            device=self.device
        )

        # Compute average training loss over accepted clients
        tot_accepted_samples = sum(client_samples[cid] for cid in accepted)
        train_loss = sum(client_losses[cid] * client_samples[cid] for cid in accepted) / max(1, tot_accepted_samples)

        round_metrics = {
            "round": round_idx,
            "mode": mode,
            "cda": eval_metrics["cda"],
            "asr": eval_metrics["asr"],
            "class1_accuracy": eval_metrics["class1_accuracy"],
            "class7_accuracy": eval_metrics["class7_accuracy"],
            "train_loss": float(train_loss),
            "test_loss": eval_metrics["test_loss"],
            "accepted_count": len(accepted),
            "rejected_count": len(rejected),
            "accepted_clients": str(accepted),
            "rejected_clients": str(rejected),
            "tpr": det_metrics["tpr"],
            "fpr": det_metrics["fpr"],
            "tp": det_metrics["tp"],
            "fp": det_metrics["fp"],
            "tn": det_metrics["tn"],
            "fn": det_metrics["fn"]
        }

        if defense_info is not None:
            round_metrics["anomaly_scores"] = defense_info["anomaly_scores"]
            round_metrics["threshold"] = defense_info["threshold"]
            round_metrics["distance_matrix"] = defense_info["distance_matrix"]

        return round_metrics
