"""
Visualization module for generating required figures at 300 DPI.
"""

import os
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def plot_clean_accuracy_comparison(
    df: pd.DataFrame,
    save_path: str = "./results/figures/clean_accuracy_comparison.png"
) -> None:
    """
    Plot Clean Data Accuracy (CDA) vs communication rounds across all modes.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.figure(figsize=(9, 6))

    mode_styles = {
        "clean_fedavg": {"label": "Clean FedAvg (No Attack)", "color": "#2ca02c", "linestyle": "-", "marker": "o"},
        "attacked_fedavg": {"label": "Attacked FedAvg (Label Flipping, No Defense)", "color": "#d62728", "linestyle": "--", "marker": "s"},
        "cosine_defense": {"label": "Cosine Defense (Ours)", "color": "#1f77b4", "linestyle": "-", "marker": "^"},
        "honest_only_oracle": {"label": "Honest Oracle (Ideal Baseline)", "color": "#9467bd", "linestyle": ":", "marker": "D"}
    }

    for mode in df["mode"].unique():
        mode_df = df[df["mode"] == mode].sort_values("round")
        style = mode_styles.get(mode, {"label": mode, "color": "black", "linestyle": "-", "marker": "o"})
        plt.plot(
            mode_df["round"],
            mode_df["cda"] * 100,
            label=style["label"],
            color=style["color"],
            linestyle=style["linestyle"],
            marker=style["marker"],
            linewidth=2,
            markersize=6
        )

    plt.title("Clean Data Accuracy (CDA) across Communication Rounds", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Communication Round", fontsize=11)
    plt.ylabel("Accuracy on Clean Test Set (%)", fontsize=11)
    plt.xticks(sorted(df["round"].unique()))
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(fontsize=10, loc="lower right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_attack_success_rate_comparison(
    df: pd.DataFrame,
    save_path: str = "./results/figures/attack_success_rate_comparison.png"
) -> None:
    """
    Plot Attack Success Rate (ASR) vs communication rounds for Attacked FedAvg vs Cosine Defense.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.figure(figsize=(9, 6))

    mode_styles = {
        "attacked_fedavg": {"label": "Attacked FedAvg (No Defense)", "color": "#d62728", "linestyle": "-", "marker": "s"},
        "cosine_defense": {"label": "Cosine Defense (Ours)", "color": "#1f77b4", "linestyle": "-", "marker": "^"},
        "clean_fedavg": {"label": "Clean FedAvg (Baseline)", "color": "#2ca02c", "linestyle": ":", "marker": "o"},
        "honest_only_oracle": {"label": "Honest Oracle (Baseline)", "color": "#9467bd", "linestyle": "--", "marker": "D"}
    }

    # Focus primarily on attacked and defense, but include others for context
    for mode in ["attacked_fedavg", "cosine_defense", "clean_fedavg", "honest_only_oracle"]:
        if mode in df["mode"].values:
            mode_df = df[df["mode"] == mode].sort_values("round")
            style = mode_styles[mode]
            plt.plot(
                mode_df["round"],
                mode_df["asr"] * 100,
                label=style["label"],
                color=style["color"],
                linestyle=style["linestyle"],
                marker=style["marker"],
                linewidth=2,
                markersize=6
            )

    plt.title("Targeted Attack Success Rate (1 → 7) across Rounds", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Communication Round", fontsize=11)
    plt.ylabel("Attack Success Rate (%)", fontsize=11)
    plt.xticks(sorted(df["round"].unique()))
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(fontsize=10, loc="center right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_cosine_distance_heatmap(
    distance_matrix: np.ndarray,
    malicious_ids: set[int] | list[int] = (2, 6),
    save_path: str = "./results/figures/cosine_distance_heatmap.png"
) -> None:
    """
    Plot pairwise cosine distance matrix for all clients.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.figure(figsize=(9, 7))

    num_clients = distance_matrix.shape[0]
    labels = [
        f"Client {i} (M)" if i in malicious_ids else f"Client {i}"
        for i in range(num_clients)
    ]

    sns.heatmap(
        distance_matrix,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        xticklabels=labels,
        yticklabels=labels,
        cbar_kws={"label": "Cosine Distance"}
    )
    plt.title("Pairwise Cosine Distance Matrix (Defense Layer fc2)", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Client", fontsize=11)
    plt.ylabel("Client", fontsize=11)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_client_anomaly_scores(
    anomaly_scores: dict[int, float],
    threshold: float,
    malicious_ids: set[int] | list[int] = (2, 6),
    round_idx: Optional[int] = None,
    save_path: str = "./results/figures/client_anomaly_scores.png"
) -> None:
    """
    Plot bar chart of client anomaly scores vs detection threshold.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.figure(figsize=(10, 6))

    client_ids = sorted(list(anomaly_scores.keys()))
    scores = [anomaly_scores[cid] for cid in client_ids]
    colors = ["#d62728" if cid in malicious_ids else "#1f77b4" for cid in client_ids]

    bars = plt.bar(client_ids, scores, color=colors, alpha=0.85, edgecolor="black", width=0.6)

    # Threshold line
    plt.axhline(
        y=threshold,
        color="#e6550d",
        linestyle="--",
        linewidth=2.5,
        label=f"Detection Threshold ({threshold:.3f})"
    )

    # Custom legend for Honest vs Malicious
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#1f77b4", edgecolor="black", label="Honest Client"),
        Patch(facecolor="#d62728", edgecolor="black", label="Malicious Client (GT)"),
        plt.Line2D([0], [0], color="#e6550d", linestyle="--", linewidth=2.5, label=f"Threshold (μ + 1.5σ: {threshold:.3f})")
    ]

    title_suffix = f" (Round {round_idx})" if round_idx is not None else ""
    plt.title(f"Client Layer-Wise Cosine Anomaly Scores{title_suffix}", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Client ID", fontsize=11)
    plt.ylabel("Anomaly Score (Mean Cosine Distance)", fontsize=11)
    plt.xticks(client_ids)
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.legend(handles=legend_elements, fontsize=10, loc="upper right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_detection_performance(
    df: pd.DataFrame,
    save_path: str = "./results/figures/detection_performance.png"
) -> None:
    """
    Plot True Positive Rate (TPR) and False Positive Rate (FPR) across rounds for Cosine Defense,
    with Oracle reference benchmark clearly separated.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.figure(figsize=(9, 5.5))

    defense_df = df[df["mode"] == "cosine_defense"].sort_values("round")
    if defense_df.empty:
        defense_df = df.sort_values("round")

    # Filter out any non-numeric / NA rows
    valid_defense = defense_df[defense_df["tpr"].notna() & defense_df["fpr"].notna()]
    if not valid_defense.empty:
        plt.plot(
            valid_defense["round"],
            valid_defense["tpr"] * 100,
            label="Cosine Defense TPR (Malicious Detected)",
            color="#2ca02c",
            linestyle="-",
            marker="o",
            linewidth=2.5,
            markersize=7
        )
        plt.plot(
            valid_defense["round"],
            valid_defense["fpr"] * 100,
            label="Cosine Defense FPR (Honest Falsely Rejected)",
            color="#d62728",
            linestyle="--",
            marker="s",
            linewidth=2.5,
            markersize=7
        )

    # Clearly separated Oracle Reference (Ground-Truth Identity Exclusion)
    oracle_df = df[df["mode"] == "honest_only_oracle"].sort_values("round")
    if not oracle_df.empty and oracle_df["tpr"].notna().any():
        plt.axhline(
            y=100,
            color="#2ca02c",
            linestyle=":",
            alpha=0.6,
            linewidth=1.8,
            label="Oracle TPR Ref (Ideal 100%)"
        )
        plt.axhline(
            y=0,
            color="#d62728",
            linestyle=":",
            alpha=0.6,
            linewidth=1.8,
            label="Oracle FPR Ref (Ideal 0%)"
        )

    plt.title("Detection Performance: Layer-Wise Cosine Similarity Defense", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Communication Round", fontsize=11)
    plt.ylabel("Rate (%)", fontsize=11)
    plt.ylim(-5, 105)
    if not defense_df.empty:
        plt.xticks(sorted(defense_df["round"].unique()))
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(fontsize=9, loc="center right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
