"""
Tests for Dirichlet Non-IID dataset partitioning and ClientDataset.
"""

import pytest
import numpy as np
import torch
from torch.utils.data import Dataset
from src.dataset import dirichlet_non_iid_partition, compute_client_class_distribution, ClientDataset


class MockMNISTDataset(Dataset):
    """Synthetic dataset mirroring MNIST targets distribution for quick tests."""
    def __init__(self, num_samples: int = 15000):
        self.targets = torch.tensor(np.repeat(np.arange(10), num_samples // 10), dtype=torch.long)
        self.data = torch.randn(len(self.targets), 1, 28, 28)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return self.data[idx], int(self.targets[idx].item())


def test_dirichlet_partitioning_constraints():
    """Verify partition satisfies: 10 clients, no duplicates, all samples assigned, min samples."""
    dataset = MockMNISTDataset(num_samples=15000)
    num_clients = 10
    min_samples = 1000
    malicious_ids = (2, 6)

    partitions = dirichlet_non_iid_partition(
        train_dataset=dataset,
        num_clients=num_clients,
        alpha=0.5,
        min_samples_per_client=min_samples,
        malicious_ids=malicious_ids,
        min_class1_malicious=50,
        seed=42
    )

    assert len(partitions) == num_clients

    # Verify all samples assigned exactly once with no duplicates
    all_indices = []
    for cid, idxs in partitions.items():
        assert len(idxs) >= min_samples, f"Client {cid} has {len(idxs)} samples, below minimum {min_samples}"
        all_indices.extend(idxs)

    assert len(all_indices) == len(dataset)
    assert len(set(all_indices)) == len(dataset), "Found duplicate sample indices across clients!"

    # Verify malicious clients have sufficient class-1 samples
    targets_np = dataset.targets.numpy()
    for m_id in malicious_ids:
        class1_count = np.sum(targets_np[partitions[m_id]] == 1)
        assert class1_count >= 50, f"Malicious client {m_id} has {class1_count} class-1 samples, below minimum"


def test_compute_client_class_distribution():
    """Verify class distribution matrix dimensions."""
    dataset = MockMNISTDataset(num_samples=20000)
    partitions = dirichlet_non_iid_partition(
        train_dataset=dataset,
        num_clients=10,
        alpha=0.5,
        min_samples_per_client=1000,
        malicious_ids=(2, 6),
        min_class1_malicious=50,
        seed=42
    )
    dist_matrix = compute_client_class_distribution(dataset, partitions, num_classes=10)
    assert dist_matrix.shape == (10, 10)
    assert dist_matrix.sum() == len(dataset)


def test_client_dataset_label_flipping_immutability():
    """Verify ClientDataset flips labels without mutating base dataset."""
    dataset = MockMNISTDataset(num_samples=1000)
    orig_labels = dataset.targets.clone().numpy()

    # Create client dataset with 1 -> 7 flip
    indices = list(range(len(dataset)))
    client_ds = ClientDataset(
        base_dataset=dataset,
        indices=indices,
        label_flip_map={1: 7},
        flip_ratio=1.0,
        seed=42
    )

    # Check that in client_ds, all class 1 became 7
    for i in range(len(client_ds)):
        _, y = client_ds[i]
        orig_y = int(orig_labels[i])
        if orig_y == 1:
            assert y == 7
        else:
            assert y == orig_y

    # Check base dataset was NOT mutated
    assert np.array_equal(dataset.targets.numpy(), orig_labels)
