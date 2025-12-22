# In analysis.py

import json
import itertools
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict

def calculate_iou(circuit1: List[int], circuit2: List[int]) -> float:
    """Calculates the Intersection over Union (IoU) for two circuits."""
    set1 = set(circuit1)
    set2 = set(circuit2)
    
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    
    if union == 0:
        return 1.0  # If both circuits are empty, they are identical.
        
    return intersection / union

def analyze_circuits(filepath: str):
    """
    Loads circuit data, calculates Inter-Client Similarity for each round,
    and plots the evolution of circuits over time.
    """
    try:
        with open(filepath, 'r') as f:
            all_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: The file '{filepath}' was not found.")
        print("Please make sure you have run main.py successfully first.")
        return

    # Dynamically get metadata from the first round's data
    first_round_key = sorted(all_data.keys())[0]
    global_circuits_sample = all_data[first_round_key]["clients_global_model"]
    
    client_keys = sorted(global_circuits_sample.keys())
    if len(client_keys) < 2:
        print("Error: Need at least two clients to perform a comparative analysis.")
        return
        
    class_names = sorted(global_circuits_sample[client_keys[0]].keys())
    layer_names = sorted(global_circuits_sample[client_keys[0]][class_names[0]].keys())
    round_keys = sorted(all_data.keys())
    
    # --- Data Structure to hold results for plotting ---
    # Example: plot_data['airplane']['conv1'] = [0.5, 0.6, 0.7] # Avg IoU for rounds 1, 2, 3
    plot_data: Dict[str, Dict[str, List[float]]] = {
        cls: {layer: [] for layer in layer_names} for cls in class_names
    }

    print("--- Inter-Client Circuit Similarity (IoU) Analysis ---")
    
    # --- Main Analysis Loop ---
    for round_key in round_keys:
        print(f"\nAnalyzing {round_key.replace('_', ' ').capitalize()}...")
        global_circuits_this_round = all_data[round_key]["clients_global_model"]

        for class_name in class_names:
            for layer_name in layer_names:
                
                iou_scores_for_round = []
                
                # Get all unique pairs of clients (e.g., (c0, c1), (c0, c2), (c1, c2))
                for client1_key, client2_key in itertools.combinations(client_keys, 2):
                    
                    circuit1 = global_circuits_this_round[client1_key][class_name][layer_name]
                    circuit2 = global_circuits_this_round[client2_key][class_name][layer_name]
                    
                    iou = calculate_iou(circuit1, circuit2)
                    iou_scores_for_round.append(iou)

                # Calculate the average IoU for this specific class, layer, and round
                avg_iou = np.mean(iou_scores_for_round) if iou_scores_for_round else 0
                
                # Store the result for plotting
                plot_data[class_name][layer_name].append(avg_iou)
                
                print(f"  {class_name:<12} | {layer_name:<8} | Avg. IoU: {avg_iou:.4f}")

    # --- Plotting the Results ---
    num_classes = len(class_names)
    fig, axes = plt.subplots(num_classes, 1, figsize=(10, 5 * num_classes), sharex=True)
    if num_classes == 1:
        axes = [axes] # Make it iterable if there's only one subplot

    fig.suptitle('Inter-Client Circuit Similarity (IoU) Across Rounds (IID)', fontsize=16)

    for i, class_name in enumerate(class_names):
        ax = axes[i]
        for layer_name in layer_names:
            ax.plot(
                range(1, len(round_keys) + 1), 
                plot_data[class_name][layer_name], 
                marker='o', 
                linestyle='-', 
                label=layer_name
            )
        
        ax.set_title(f'Class: "{class_name}"')
        ax.set_ylabel('Average IoU')
        ax.set_ylim(0, 1.05)
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend()

    axes[-1].set_xlabel('Federated Round')
    plt.xticks(range(1, len(round_keys) + 1))
    plt.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust layout to make space for suptitle
    plt.show()


if __name__ == "__main__":
    # The script will look for the JSON file in the same directory
    circuits_filepath = "circuits_per_round.json"
    analyze_circuits(circuits_filepath)