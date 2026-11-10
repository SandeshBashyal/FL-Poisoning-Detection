"""
Federated Learning Client implementation.
"""

import copy
from typing import Tuple, Dict
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from src.model import SimpleCNN


class Client:
    """
    Federated Learning Client.
    Executes local training given global model weights.
    """

    def __init__(
        self,
        client_id: int,
        dataset: Dataset,
        batch_size: int = 64,
        local_epochs: int = 1,
        lr: float = 0.01,
        momentum: float = 0.9,
        weight_decay: float = 0.0001,
        device: torch.device = None
    ) -> None:
        self.client_id = client_id
        self.dataset = dataset
        self.num_samples = len(dataset)
        self.batch_size = batch_size
        self.local_epochs = local_epochs
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.device = device or torch.device("cpu")

        self.dataloader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=False
        )
        self.criterion = nn.CrossEntropyLoss()

    def train(self, global_state_dict: dict[str, torch.Tensor]) -> tuple[dict[str, torch.Tensor], float, int]:
        """
        Train local model starting from the global model state dict.

        Args:
            global_state_dict: State dict distributed by the server.

        Returns:
            Tuple of (updated_state_dict, average_training_loss, num_samples).
        """
        model = SimpleCNN().to(self.device)
        model.load_state_dict(global_state_dict)
        model.train()

        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=self.lr,
            momentum=self.momentum,
            weight_decay=self.weight_decay
        )

        total_loss = 0.0
        total_batches = 0

        for _ in range(self.local_epochs):
            for images, targets in self.dataloader:
                images = images.to(self.device)
                targets = targets.to(self.device)

                optimizer.zero_grad()
                outputs = model(images)
                loss = self.criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                total_batches += 1

        avg_loss = total_loss / max(1, total_batches)

        # Move state dict to CPU before returning to conserve GPU memory
        cpu_state_dict = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        return cpu_state_dict, avg_loss, self.num_samples
