"""
Targeted label-flipping attack implementation for Federated Learning.
"""

from typing import Optional
from src.dataset import ClientDataset


class TargetedLabelFlippingAttack:
    """
    Targeted Label-Flipping Attack:
    Flips a source class label to a target class label for malicious clients only.
    By default:
      - Source label: 1
      - Target label: 7
      - Flip ratio: 1.0 (100% of local source label samples)
      - Malicious clients: IDs 2 and 6
    """

    def __init__(
        self,
        malicious_client_ids: list[int] = None,
        source_label: int = 1,
        target_label: int = 7,
        flip_ratio: float = 1.0
    ) -> None:
        self.malicious_client_ids = set(malicious_client_ids or [2, 6])
        self.source_label = source_label
        self.target_label = target_label
        self.flip_ratio = flip_ratio
        self.flipped_counts: dict[int, int] = {}

    def is_malicious(self, client_id: int) -> bool:
        """Check if client ID is configured as malicious."""
        return client_id in self.malicious_client_ids

    def create_client_dataset(
        self,
        base_dataset,
        client_id: int,
        indices: list[int],
        enable_attack: bool = True,
        seed: int = 42
    ) -> ClientDataset:
        """
        Create a ClientDataset for the given client.
        If enable_attack is True and the client is in malicious_client_ids,
        apply targeted label-flipping without modifying base_dataset.
        """
        if enable_attack and self.is_malicious(client_id):
            flip_map = {self.source_label: self.target_label}
            dataset = ClientDataset(
                base_dataset=base_dataset,
                indices=indices,
                label_flip_map=flip_map,
                flip_ratio=self.flip_ratio,
                seed=seed + client_id
            )
            self.flipped_counts[client_id] = dataset.flipped_count
            return dataset
        else:
            dataset = ClientDataset(
                base_dataset=base_dataset,
                indices=indices,
                label_flip_map={},
                flip_ratio=0.0,
                seed=seed + client_id
            )
            self.flipped_counts[client_id] = 0
            return dataset

    def get_flipped_count(self, client_id: int) -> int:
        """Return the count of flipped samples for a client."""
        return self.flipped_counts.get(client_id, 0)
