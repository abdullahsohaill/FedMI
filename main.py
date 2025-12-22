import torch
import numpy as np
import random
import json
import os

from config import Config
from models import get_model
# from dataset import get_cifar10, partition_iid, get_dataloader, get_test_dataloader
from dataset import get_dataset, partition_iid, get_dataloader, get_test_dataloader 
from train import analyze_iou
from fed import run_federated_training


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


import argparse

def main():
    parser = argparse.ArgumentParser(description="FedMI Training")
    parser.add_argument("--train_mode", type=str, choices=["sparse", "dense"], help="Training mode")
    parser.add_argument("--device", type=str, help="Device to use (cuda/cpu)")
    parser.add_argument("--seed", type=int, help="Random seed")
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint if available")
    args = parser.parse_args()

    config = Config()
    
    # Override config with CLI args
    if args.train_mode:
        config.train_mode = args.train_mode
    if args.device:
        config.device = args.device
    if args.seed:
        config.seed = args.seed
    if args.resume:
        config.resume = True

    set_seed(config.seed)
    print(f"Using device: {config.device}")
    print(f"Dataset: {config.dataset_name}")

    # --- CHANGED: Use generic get_dataset ---
    trainset, testset = get_dataset(config)

    testloader = get_test_dataloader(testset, config)
    # MNIST has a .classes attribute too, usually ['0 - zero', '1 - one', etc]
    class_names = trainset.classes 

    # # Load data
    # trainset, testset = get_cifar10(config)
    # testloader = get_test_dataloader(testset, config)
    # class_names = trainset.classes

    # Partition data for FL (IID)
    print("Partitioning data for IID Federated Learning...")
    client_indices = partition_iid(trainset, config.num_clients)
    client_dataloaders = [
        get_dataloader(trainset, indices, config) for indices in client_indices
    ]

    # Initialize global model
    global_model = get_model(config)


    if not os.path.exists(config.checkpoint_dir):
        os.makedirs(config.checkpoint_dir)
    torch.save(global_model.state_dict(), os.path.join(config.checkpoint_dir, "initialization.pt"))
    print("Saved model initialization to checkpoints/initialization.pt")

    # Federated Training (Phase 1 & 2 Integrated)
    # Includes local sparse training and per-round circuit discovery
    global_model, all_circuits = run_federated_training(
        global_model, client_dataloaders, testloader, config, class_names
    )

    # Phase 3: IoU Analysis on Final Global Model
    print("\n=== Phase 3: IoU Analysis (Final Global Model) ===")
    
    # Extract global circuits from the last round
    final_round_key = f"round_{config.num_rounds}"
    if final_round_key in all_circuits:
        # Dictionary: client_id -> { class_name -> indices }
        final_global_circuits = all_circuits[final_round_key]["clients_global_model"]
        
        for client_id, class_circuits in final_global_circuits.items():
            print(f"\n--- Analysis for Global Model on {client_id} Data ---")
            
            # Convert list indices back to sets for analyze_iou
            formatted_circuits = {}
            for class_name, layers in class_circuits.items():
                formatted_circuits[class_name] = {
                    layer: set(indices) for layer, indices in layers.items()
                }
            
            analyze_iou(formatted_circuits)
    else:
        print("Could not find final round circuits for analysis.")


if __name__ == "__main__":
    main()
