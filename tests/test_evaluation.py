"""
Tests for evaluation metrics: CDA, ASR, TPR, and FPR.
"""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset
from src.evaluation import evaluate_model_on_test_data, compute_detection_metrics


class MockClassifier(nn.Module):
    """Classifier that returns predetermined logits."""
    def __init__(self, pred_class: int):
        super().__init__()
        self.pred_class = pred_class

    def forward(self, x):
        batch_size = x.size(0)
        logits = torch.zeros(batch_size, 10)
        logits[:, self.pred_class] = 10.0
        return logits


def test_evaluate_model_cda_and_asr():
    """Verify CDA and ASR calculation logic on synthetic data."""
    # Create dataset: 50 samples of class 1, 50 samples of class 7
    x = torch.randn(100, 1, 28, 28)
    y = torch.tensor([1] * 50 + [7] * 50, dtype=torch.long)
    dataset = TensorDataset(x, y)

    # Model always predicts class 7
    # For class 1: all 50 predicted as 7 -> ASR = 50/50 = 1.0; class1_acc = 0/50 = 0.0
    # For class 7: all 50 predicted as 7 -> class7_acc = 50/50 = 1.0
    # Overall correct: 50 out of 100 -> CDA = 0.50
    model = MockClassifier(pred_class=7)
    metrics = evaluate_model_on_test_data(model, dataset, batch_size=50, source_label=1, target_label=7)

    assert metrics["cda"] == 0.50
    assert metrics["asr"] == 1.00
    assert metrics["class1_accuracy"] == 0.00
    assert metrics["class7_accuracy"] == 1.00


def test_compute_detection_metrics_tpr_fpr():
    """Verify TPR and FPR calculation logic."""
    all_clients = list(range(10))
    malicious = [2, 6]  # 2 malicious, 8 honest

    # Case 1: Perfect detection (both 2 and 6 rejected, no honest rejected)
    det1 = compute_detection_metrics(rejected_clients=[2, 6], malicious_client_ids=malicious, all_client_ids=all_clients)
    assert det1["tpr"] == 1.0
    assert det1["fpr"] == 0.0
    assert det1["tp"] == 2
    assert det1["fp"] == 0

    # Case 2: Partial detection (only client 2 rejected, client 0 falsely rejected)
    det2 = compute_detection_metrics(rejected_clients=[2, 0], malicious_client_ids=malicious, all_client_ids=all_clients)
    assert det2["tpr"] == 0.5  # 1/2
    assert det2["fpr"] == 1 / 8  # 0.125
    assert det2["tp"] == 1
    assert det2["fp"] == 1

    # Case 3: No clients rejected
    det3 = compute_detection_metrics(rejected_clients=[], malicious_client_ids=malicious, all_client_ids=all_clients)
    assert det3["tpr"] == 0.0
    assert det3["fpr"] == 0.0
