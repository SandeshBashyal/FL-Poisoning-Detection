"""
Tests for targeted label-flipping attack.
"""

import pytest
import numpy as np
import torch
from torch.utils.data import Dataset
from src.attacks import TargetedLabelFlippingAttack


class DummyDataset(Dataset):
    def __init__(self, labels: list[int]):
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return torch.randn(1, 28, 28), self.labels[idx]


def test_targeted_label_flipping_malicious_only():
    """Verify only malicious clients have labels flipped, while honest do not."""
    attack = TargetedLabelFlippingAttack(
        malicious_client_ids=[2, 6],
        source_label=1,
        target_label=7,
        flip_ratio=1.0
    )

    dummy_data = [1, 1, 2, 3, 1, 7, 0, 1]
    dataset = DummyDataset(dummy_data)
    indices = list(range(len(dummy_data)))

    # Honest client 0
    honest_ds = attack.create_client_dataset(dataset, client_id=0, indices=indices, enable_attack=True)
    honest_labels = [honest_ds[i][1] for i in range(len(honest_ds))]
    assert honest_labels == dummy_data, "Honest client had labels flipped!"
    assert attack.get_flipped_count(0) == 0

    # Malicious client 2
    malicious_ds = attack.create_client_dataset(dataset, client_id=2, indices=indices, enable_attack=True)
    malicious_labels = [malicious_ds[i][1] for i in range(len(malicious_ds))]
    expected_malicious = [7, 7, 2, 3, 7, 7, 0, 7]
    assert malicious_labels == expected_malicious, f"Expected {expected_malicious}, got {malicious_labels}"
    assert attack.get_flipped_count(2) == 4

    # When attack is disabled (clean mode), malicious client should have clean labels
    disabled_ds = attack.create_client_dataset(dataset, client_id=2, indices=indices, enable_attack=False)
    disabled_labels = [disabled_ds[i][1] for i in range(len(disabled_ds))]
    assert disabled_labels == dummy_data
