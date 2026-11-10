"""
Evaluation metrics calculation: CDA, ASR, class accuracies, TPR, and FPR.
"""

from typing import Dict, List, Set, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from src.model import SimpleCNN


def evaluate_model_on_test_data(
    model: nn.Module,
    test_dataset: Dataset,
    batch_size: int = 128,
    source_label: int = 1,
    target_label: int = 7,
    device: torch.device = None
) -> dict[str, float]:
    """
    Evaluate global model on clean test data.

    Calculates:
      - Clean Data Accuracy (CDA): accuracy across all test samples.
      - Attack Success Rate (ASR): fraction of clean class-1 images predicted as class-7.
      - Class-1 Accuracy: fraction of clean class-1 images predicted as class-1.
      - Class-7 Accuracy: fraction of clean class-7 images predicted as class-7.
      - Test Loss: average cross-entropy loss over the test set.
    """
    device = device or torch.device("cpu")
    model.eval()
    dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    total_samples = 0
    total_correct = 0

    class1_total = 0
    class1_pred_7 = 0
    class1_pred_1 = 0

    class7_total = 0
    class7_pred_7 = 0

    with torch.no_grad():
        for images, targets in dataloader:
            images = images.to(device)
            targets = targets.to(device)
            outputs = model(images)
            loss = criterion(outputs, targets)

            preds = outputs.argmax(dim=1)

            total_loss += loss.item() * len(targets)
            total_samples += len(targets)
            total_correct += (preds == targets).sum().item()

            # Class 1 metrics
            mask_1 = (targets == source_label)
            class1_total += mask_1.sum().item()
            class1_pred_7 += ((preds == target_label) & mask_1).sum().item()
            class1_pred_1 += ((preds == source_label) & mask_1).sum().item()

            # Class 7 metrics
            mask_7 = (targets == target_label)
            class7_total += mask_7.sum().item()
            class7_pred_7 += ((preds == target_label) & mask_7).sum().item()

    cda = (total_correct / total_samples) if total_samples > 0 else 0.0
    asr = (class1_pred_7 / class1_total) if class1_total > 0 else 0.0
    class1_acc = (class1_pred_1 / class1_total) if class1_total > 0 else 0.0
    class7_acc = (class7_pred_7 / class7_total) if class7_total > 0 else 0.0
    avg_loss = (total_loss / total_samples) if total_samples > 0 else 0.0

    return {
        "cda": float(cda),
        "asr": float(asr),
        "class1_accuracy": float(class1_acc),
        "class7_accuracy": float(class7_acc),
        "test_loss": float(avg_loss)
    }


def compute_detection_metrics(
    rejected_clients: list[int],
    malicious_client_ids: set[int] | list[int],
    all_client_ids: list[int]
) -> dict[str, float]:
    """
    Calculate detection performance metrics:
      - True Positive Rate (TPR): fraction of malicious clients correctly rejected.
      - False Positive Rate (FPR): fraction of honest clients incorrectly rejected.
      - True Positives (TP), False Positives (FP), True Negatives (TN), False Negatives (FN).
    """
    malicious_set = set(malicious_client_ids)
    all_set = set(all_client_ids)
    honest_set = all_set - malicious_set
    rejected_set = set(rejected_clients)

    tp = len(rejected_set & malicious_set)
    fp = len(rejected_set & honest_set)
    fn = len(malicious_set - rejected_set)
    tn = len(honest_set - rejected_set)

    tpr = (tp / len(malicious_set)) if len(malicious_set) > 0 else 0.0
    fpr = (fp / len(honest_set)) if len(honest_set) > 0 else 0.0

    return {
        "tpr": float(tpr),
        "fpr": float(fpr),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn
    }
