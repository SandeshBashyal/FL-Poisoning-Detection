"""
Dataset loading, class-wise Dirichlet Non-IID partitioning, and distribution visualization.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms


def get_mnist_datasets(data_dir: str = "./data"):
    """
    Download and return MNIST train and test datasets normalized with standard MNIST mean and std.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_dataset = datasets.MNIST(root=data_dir, train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root=data_dir, train=False, download=True, transform=transform)
    return train_dataset, test_dataset


def dirichlet_non_iid_partition(
    train_dataset: Dataset,
    num_clients: int = 10,
    alpha: float = 0.5,
    min_samples_per_client: int = 1000,
    malicious_ids: tuple = (2, 6),
    min_class1_malicious: int = 100,
    seed: int = 42
) -> dict[int, list[int]]:
    """
    Partition MNIST training dataset across clients using class-wise Dirichlet distribution.

    Requirements satisfied:
    - Partition training data among num_clients.
    - Every sample is assigned exactly once.
    - No duplicate indices across or within clients.
    - Minimum min_samples_per_client per client.
    - Sufficient class-1 samples for malicious clients.
    """
    if isinstance(train_dataset.targets, torch.Tensor):
        targets = train_dataset.targets.cpu().numpy()
    else:
        targets = np.array(train_dataset.targets)
    num_samples = len(targets)
    classes = np.unique(targets)
    num_classes = len(classes)

    attempt = 0
    max_attempts = 1000

    while attempt < max_attempts:
        attempt_rng = np.random.RandomState(seed + attempt * 100)
        # Class-wise indices
        class_indices = {c: np.where(targets == c)[0].copy() for c in classes}
        for c in classes:
            attempt_rng.shuffle(class_indices[c])

        client_indices = {i: [] for i in range(num_clients)}

        # Sample Dirichlet distribution for each class
        for c in classes:
            c_indices = class_indices[c]
            num_c_samples = len(c_indices)

            # Dirichlet distribution across clients
            proportions = attempt_rng.dirichlet(np.repeat(alpha, num_clients))
            # Normalize to avoid rounding errors
            proportions = proportions / proportions.sum()

            # Split counts
            counts = (proportions * num_c_samples).astype(int)
            remainder = num_c_samples - counts.sum()
            # Distribute remainder to largest proportion clients
            sorted_indices = np.argsort(proportions)[::-1]
            for i in range(remainder):
                counts[sorted_indices[i]] += 1

            # Assign slices to clients
            start = 0
            for client_id in range(num_clients):
                count = counts[client_id]
                if count > 0:
                    client_indices[client_id].extend(c_indices[start:start + count].tolist())
                    start += count

        # Verify minimum samples constraint
        client_lens = [len(client_indices[i]) for i in range(num_clients)]
        if min(client_lens) < min_samples_per_client:
            attempt += 1
            continue

        # Verify malicious clients have sufficient class-1 samples
        has_enough_class1 = True
        for m_id in malicious_ids:
            m_labels = targets[client_indices[m_id]]
            class1_count = np.sum(m_labels == 1)
            if class1_count < min_class1_malicious:
                has_enough_class1 = False
                break

        if not has_enough_class1:
            attempt += 1
            continue

        # Verify all samples assigned exactly once
        all_assigned = []
        for i in range(num_clients):
            all_assigned.extend(client_indices[i])
        if len(all_assigned) == num_samples and len(set(all_assigned)) == num_samples:
            return client_indices

        attempt += 1

    raise RuntimeError(f"Could not find valid Dirichlet partition satisfying constraints within {max_attempts} attempts.")


def compute_client_class_distribution(train_dataset: Dataset, client_partitions: dict[int, list[int]], num_classes: int = 10) -> np.ndarray:
    """
    Compute a 2D matrix of shape (num_clients, num_classes) containing sample counts.
    """
    if isinstance(train_dataset.targets, torch.Tensor):
        targets = train_dataset.targets.cpu().numpy()
    else:
        targets = np.array(train_dataset.targets)
    num_clients = len(client_partitions)
    dist_matrix = np.zeros((num_clients, num_classes), dtype=int)
    for client_id, indices in client_partitions.items():
        client_targets = targets[indices]
        for c in range(num_classes):
            dist_matrix[client_id, c] = int(np.sum(client_targets == c))
    return dist_matrix


def visualize_client_distribution(
    dist_matrix: np.ndarray,
    save_path: str = "./results/figures/client_class_distribution.png"
) -> None:
    """
    Visualize client class distribution as a heatmap and save the figure.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.figure(figsize=(10, 6))
    sns.heatmap(
        dist_matrix,
        annot=True,
        fmt="d",
        cmap="YlGnBu",
        cbar=True,
        xticklabels=[f"Class {i}" for i in range(dist_matrix.shape[1])],
        yticklabels=[f"Client {i}" for i in range(dist_matrix.shape[0])]
    )
    plt.title("Client Data Distribution (Dirichlet α=0.5 Non-IID)", fontsize=14, fontweight="bold", pad=12)
    plt.xlabel("Digit Class", fontsize=12)
    plt.ylabel("Client ID", fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def save_partitions(partitions: dict[int, list[int]], save_path: str) -> None:
    """Save partitions to JSON file."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        # Convert keys to string for JSON serialization
        json.dump({str(k): v for k, v in partitions.items()}, f)


def load_partitions(load_path: str) -> dict[int, list[int]]:
    """Load partitions from JSON file."""
    with open(load_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {int(k): v for k, v in data.items()}


class ClientDataset(Dataset):
    """
    Client dataset pointing to subsets of the global training dataset.
    Supports optional targeted label-flipping without modifying the original dataset.
    """

    def __init__(
        self,
        base_dataset: Dataset,
        indices: list[int],
        label_flip_map: dict[int, int] = None,
        flip_ratio: float = 1.0,
        seed: int = 42
    ) -> None:
        self.base_dataset = base_dataset
        self.indices = list(indices)
        self.label_flip_map = label_flip_map or {}
        self.flip_ratio = flip_ratio

        # Precompute flipped labels for this client without mutating base_dataset
        self.client_labels = []
        self.flipped_count = 0
        rng = np.random.RandomState(seed)

        has_targets_attr = hasattr(self.base_dataset, "targets")
        for idx in self.indices:
            if has_targets_attr:
                orig_label = self.base_dataset.targets[idx]
            else:
                _, orig_label = self.base_dataset[idx]
            # Ensure integer
            if isinstance(orig_label, torch.Tensor):
                orig_label = orig_label.item()
            orig_label = int(orig_label)

            if orig_label in self.label_flip_map:
                if rng.uniform(0, 1) < self.flip_ratio:
                    new_label = self.label_flip_map[orig_label]
                    self.client_labels.append(new_label)
                    self.flipped_count += 1
                else:
                    self.client_labels.append(orig_label)
            else:
                self.client_labels.append(orig_label)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        data_idx = self.indices[idx]
        image, _ = self.base_dataset[data_idx]
        target = self.client_labels[idx]
        return image, target
