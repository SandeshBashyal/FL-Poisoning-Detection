"""
End-to-end experiment runner for Federated Learning Poisoning Detection.
Runs the four experimental modes:
  1. clean_fedavg: no attack, no defense
  2. attacked_fedavg: label flipping, no defense
  3. cosine_defense: label flipping with cosine filtering defense
  4. honest_only_oracle: only honest clients participate/contribute
"""

import os
import sys
import copy
import argparse
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils import set_seed, load_config, get_device, save_checkpoint, save_metrics_to_csv, ensure_dir
from src.model import SimpleCNN
from src.dataset import (
    get_mnist_datasets,
    dirichlet_non_iid_partition,
    compute_client_class_distribution,
    visualize_client_distribution,
    save_partitions
)
from src.attacks import TargetedLabelFlippingAttack
from src.defenses import CosineSimilarityDefense
from src.client import Client
from src.server import Server
from src.visualization import (
    plot_clean_accuracy_comparison,
    plot_attack_success_rate_comparison,
    plot_cosine_distance_heatmap,
    plot_client_anomaly_scores,
    plot_detection_performance
)


def run_experiment_mode(
    mode: str,
    config: dict,
    train_dataset,
    test_dataset,
    partitions: dict[int, list[int]],
    initial_state_dict: dict[str, torch.Tensor],
    device: torch.device
) -> tuple[list[dict], dict]:
    """
    Run one mode of the federated learning simulation.
    """
    seed = config["project"]["random_seed"]
    set_seed(seed)

    fl_cfg = config["federated_learning"]
    att_cfg = config["attack"]
    def_cfg = config["defense"]

    num_rounds = fl_cfg["num_rounds"]
    malicious_ids = att_cfg["malicious_client_ids"]

    # Configure attack
    # Clean mode has no attack
    enable_attack = (mode in ("attacked_fedavg", "cosine_defense"))
    attack = TargetedLabelFlippingAttack(
        malicious_client_ids=malicious_ids,
        source_label=att_cfg["source_label"],
        target_label=att_cfg["target_label"],
        flip_ratio=att_cfg["flip_ratio"]
    )

    # Instantiate clients
    clients = {}
    for cid, indices in partitions.items():
        client_ds = attack.create_client_dataset(
            base_dataset=train_dataset,
            client_id=cid,
            indices=indices,
            enable_attack=enable_attack,
            seed=seed
        )
        clients[cid] = Client(
            client_id=cid,
            dataset=client_ds,
            batch_size=fl_cfg["batch_size"],
            local_epochs=fl_cfg["local_epochs"],
            lr=fl_cfg["optimizer"]["lr"],
            momentum=fl_cfg["optimizer"]["momentum"],
            weight_decay=fl_cfg["optimizer"]["weight_decay"],
            device=device
        )

    # Configure defense
    defense = None
    if mode == "cosine_defense":
        defense = CosineSimilarityDefense(
            target_layer=def_cfg["target_layer"],
            multiplier=def_cfg["multiplier"],
            min_accepted_clients=def_cfg["min_accepted_clients"],
            eps=float(def_cfg["eps"])
        )

    # Instantiate server
    server = Server(
        test_dataset=test_dataset,
        clients=clients,
        defense=defense,
        malicious_client_ids=malicious_ids,
        device=device,
        initial_state_dict=initial_state_dict
    )

    metrics_history = []
    last_defense_state = {}

    print(f"\n==========================================")
    print(f" Starting Experiment: {mode.upper()} ({num_rounds} rounds)")
    print(f"==========================================")

    pbar = tqdm(range(1, num_rounds + 1), desc=f"Mode: {mode}")
    for r in pbar:
        round_metrics = server.run_round(round_idx=r, mode=mode)
        metrics_history.append(round_metrics)

        tpr_val = round_metrics["tpr"]
        fpr_val = round_metrics["fpr"]
        tpr_str = f"{tpr_val*100:.0f}%" if (tpr_val is not None and pd.notna(tpr_val)) else "N/A"
        fpr_str = f"{fpr_val*100:.0f}%" if (fpr_val is not None and pd.notna(fpr_val)) else "N/A"

        pbar.set_postfix({
            "CDA": f"{round_metrics['cda']*100:.2f}%",
            "ASR": f"{round_metrics['asr']*100:.2f}%",
            "TPR": tpr_str,
            "FPR": fpr_str,
            "Accepted": round_metrics['accepted_count']
        })

        if "distance_matrix" in round_metrics:
            last_defense_state = {
                "distance_matrix": round_metrics["distance_matrix"],
                "anomaly_scores": round_metrics["anomaly_scores"],
                "threshold": round_metrics["threshold"],
                "round": r
            }

    # Save final model checkpoint
    ckpt_path = os.path.join(config["paths"]["checkpoints_dir"], f"{mode}_final.pt")
    save_checkpoint(
        server.global_model.state_dict(),
        ckpt_path,
        metadata={"mode": mode, "rounds": num_rounds, "final_cda": metrics_history[-1]["cda"]}
    )

    return metrics_history, last_defense_state


def main():
    parser = argparse.ArgumentParser(description="Federated Learning Poisoning Detection Runner")
    parser.add_argument("--config", type=str, default="configs/experiment.yaml", help="Path to config YAML")
    parser.add_argument("--modes", nargs="+", default=["clean_fedavg", "attacked_fedavg", "cosine_defense", "honest_only_oracle"],
                        help="Modes to run")
    parser.add_argument("--rounds", type=int, default=None, help="Override communication rounds")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.rounds is not None:
        config["federated_learning"]["num_rounds"] = args.rounds

    seed = config["project"]["random_seed"]
    set_seed(seed)

    device = get_device(config["project"].get("device", "cuda"))
    print(f"Using device: {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")

    # Ensure output directories exist
    ensure_dir(config["paths"]["results_dir"])
    ensure_dir(config["paths"]["figures_dir"])
    ensure_dir(config["paths"]["metrics_dir"])
    ensure_dir(config["paths"]["checkpoints_dir"])

    # 1. Prepare Datasets & Partitions
    print("\n--- Step 3: Downloading & Partitioning MNIST Data ---")
    train_dataset, test_dataset = get_mnist_datasets(config["dataset"]["data_dir"])
    partitions = dirichlet_non_iid_partition(
        train_dataset=train_dataset,
        num_clients=config["dataset"]["num_clients"],
        alpha=config["dataset"]["dirichlet_alpha"],
        min_samples_per_client=config["dataset"]["min_samples_per_client"],
        malicious_ids=tuple(config["attack"]["malicious_client_ids"]),
        seed=seed
    )

    # Save partitions
    partition_file = os.path.join(config["paths"]["results_dir"], "partitions.json")
    save_partitions(partitions, partition_file)
    print(f"Partitions saved to {partition_file}")

    # Visualize distribution heatmap
    dist_matrix = compute_client_class_distribution(train_dataset, partitions)
    heatmap_path = os.path.join(config["paths"]["figures_dir"], "client_class_distribution.png")
    visualize_client_distribution(dist_matrix, heatmap_path)
    print(f"Client class distribution heatmap saved to {heatmap_path}")

    # 2. Initialize identical initial weights for fair comparison across modes
    init_model = SimpleCNN()
    initial_state_dict = copy.deepcopy(init_model.state_dict())
    init_ckpt = os.path.join(config["paths"]["checkpoints_dir"], "initial_model.pt")
    save_checkpoint(initial_state_dict, init_ckpt, metadata={"info": "Initial weights for all experiments"})

    # 3. Run Experiments across modes
    all_metrics = []
    saved_defense_state = None

    for mode in args.modes:
        history, def_state = run_experiment_mode(
            mode=mode,
            config=config,
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            partitions=partitions,
            initial_state_dict=initial_state_dict,
            device=device
        )
        all_metrics.extend(history)
        if mode == "cosine_defense" and def_state:
            saved_defense_state = def_state

        # Save per-mode CSV
        mode_csv = os.path.join(config["paths"]["metrics_dir"], f"{mode}_metrics.csv")
        save_metrics_to_csv(history, mode_csv)
        print(f"Saved mode metrics to {mode_csv}")

    # 4. Save combined metrics CSV
    all_df = pd.DataFrame(all_metrics)
    # Exclude complex matrix/dict objects from main CSV to keep it clean
    cols_to_drop = [c for c in ["distance_matrix", "anomaly_scores"] if c in all_df.columns]
    clean_csv_df = all_df.drop(columns=cols_to_drop)
    all_csv = os.path.join(config["paths"]["metrics_dir"], "all_experiments_metrics.csv")
    clean_csv_df.to_csv(all_csv, index=False)
    print(f"\nAll experiments metrics saved to {all_csv}")

    # 5. Generate STEP 10 Required Graphs (300 DPI)
    print("\n--- Step 10: Generating Publication Figures (300 DPI) ---")
    figures_dir = config["paths"]["figures_dir"]

    # Graph 1: Clean Data Accuracy Comparison
    plot_clean_accuracy_comparison(clean_csv_df, os.path.join(figures_dir, "clean_accuracy_comparison.png"))
    print("Graph 1 generated: clean_accuracy_comparison.png")

    # Graph 2: Attack Success Rate Comparison
    plot_attack_success_rate_comparison(clean_csv_df, os.path.join(figures_dir, "attack_success_rate_comparison.png"))
    print("Graph 2 generated: attack_success_rate_comparison.png")

    # Graph 3 & 4: Cosine Distance Heatmap and Client Anomaly Scores
    if saved_defense_state:
        plot_cosine_distance_heatmap(
            saved_defense_state["distance_matrix"],
            malicious_ids=config["attack"]["malicious_client_ids"],
            save_path=os.path.join(figures_dir, "cosine_distance_heatmap.png")
        )
        print("Graph 3 generated: cosine_distance_heatmap.png")

        plot_client_anomaly_scores(
            saved_defense_state["anomaly_scores"],
            threshold=saved_defense_state["threshold"],
            malicious_ids=config["attack"]["malicious_client_ids"],
            round_idx=saved_defense_state.get("round", None),
            save_path=os.path.join(figures_dir, "client_anomaly_scores.png")
        )
        print("Graph 4 generated: client_anomaly_scores.png")

    # Graph 5: Detection Performance (TPR & FPR)
    plot_detection_performance(clean_csv_df, os.path.join(figures_dir, "detection_performance.png"))
    print("Graph 5 generated: detection_performance.png")

    # Summary Table
    print("\n" + "="*90)
    print(f"{'EXPERIMENTAL SUMMARY (Round 15)':^90}")
    print("="*90)
    print(f"{'Mode':<26} | {'CDA (%)':<10} | {'ASR (%)':<10} | {'Class 1 (%)':<12} | {'TPR (%)':<10} | {'FPR (%)':<10}")
    print("-"*90)
    for mode in args.modes:
        sub = clean_csv_df[clean_csv_df["mode"] == mode].sort_values("round")
        if not sub.empty:
            last = sub.iloc[-1]
            tpr_val = last["tpr"]
            fpr_val = last["fpr"]
            if pd.notna(tpr_val):
                tpr_str = f"{tpr_val*100:.1f}" + (" (oracle)" if mode == "honest_only_oracle" else "")
            else:
                tpr_str = "N/A"
            if pd.notna(fpr_val):
                fpr_str = f"{fpr_val*100:.1f}" + (" (oracle)" if mode == "honest_only_oracle" else "")
            else:
                fpr_str = "N/A"
            mode_display = mode + (" (oracle ref)" if mode == "honest_only_oracle" else "")
            print(f"{mode_display:<26} | {last['cda']*100:<10.2f} | {last['asr']*100:<10.2f} | {last['class1_accuracy']*100:<12.2f} | {tpr_str:<10} | {fpr_str:<10}")
    print("="*90 + "\n")


if __name__ == "__main__":
    main()
