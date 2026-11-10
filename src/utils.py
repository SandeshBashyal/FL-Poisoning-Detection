"""
Utility functions for Federated Learning Poisoning Detection project.
"""

import os
import random
import yaml
import numpy as np
import pandas as pd
import torch


def set_seed(seed: int = 42) -> None:
    """Set random seed across all libraries for deterministic execution."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_config(config_path: str) -> dict:
    """Load configuration from a YAML file."""
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def save_config(config: dict, config_path: str) -> None:
    """Save configuration to a YAML file."""
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, default_flow_style=False, sort_keys=False)


def get_device(preference: str = "cuda") -> torch.device:
    """Get torch device according to availability and preference."""
    if preference == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def ensure_dir(path: str) -> str:
    """Ensure that directory exists and return the path."""
    os.makedirs(path, exist_ok=True)
    return path


def save_checkpoint(state_dict: dict, path: str, metadata: dict = None) -> None:
    """Save model checkpoint with optional metadata."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "state_dict": state_dict,
        "metadata": metadata or {}
    }
    torch.save(payload, path)


def load_checkpoint(path: str, map_location: str = "cpu") -> dict:
    """Load model checkpoint and metadata."""
    return torch.load(path, map_location=map_location, weights_only=True)


def save_metrics_to_csv(records: list[dict], filepath: str) -> None:
    """Save list of metric records to CSV."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df = pd.DataFrame(records)
    df.to_csv(filepath, index=False)
