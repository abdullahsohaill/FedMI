import os
import json
import subprocess

# =======================================================
#  CONFIGURATION FOR THE AUTOMATED EXPERIMENT SUITE
# =======================================================

# --- Experiment 1: Long Run ---
LONG_RUN_CONFIG = {
    "name": "3_class_long_run",
    "num_rounds": 50,
    "target_sparsity": 0.995,
    "client_map": {0: [0, 1, 2], 1: [3, 4, 5], 2: [6, 7, 8]}
}

# --- Experiment 2: Sparsity Ablation ---
SPARSITY_ABLATION_CONFIG = {
    "name": "sparsity_ablation",
    "num_rounds": 10,
    "sparsity_levels": [0.95, 0.97, 0.99, 0.995],
    "client_map": {0: [0, 1], 1: [2, 3], 2: [4, 5]}
}

# =======================================================

def run_command(command):
    """Helper to run a shell command and print its output."""
    print(f"\nExecuting: {command}")
    # Using shell=True for simplicity with chained commands
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in process.stdout:
        print(line, end='')
    process.wait()
    if process.returncode != 0:
        raise Exception(f"Command failed with exit code {process.returncode}")

def main():
    print("=============================================")
    print("  STARTING FEDMI AUTOMATED EXPERIMENT SUITE  ")
    print("=============================================")

    # --- 1. RUN THE LONG-RUN EXPERIMENT ---
    print("\n\n--- [1/2] RUNNING EXPERIMENT: 3-CLASS LONG RUN (50 ROUNDS) ---")
    
    config = LONG_RUN_CONFIG
    checkpoint_dir = f"./checkpoints/{config['name']}"
    client_map_json = json.dumps(config['client_map'])

    # A. Run Training
    run_command(
        f"python run_non-iid.py "
        f"--num_rounds {config['num_rounds']} "
        f"--checkpoint_dir \"{checkpoint_dir}\" "
        f"--client_map_json '{client_map_json}'"  # Use single quotes for JSON
    )
    
    # B. Run ALL Analysis Scripts
    print(f"\n--- Analyzing Results for {config['name']} ---")
    run_command(f"python visualize-non-iid/visualize_cross_accuracy.py --dir \"{checkpoint_dir}\"")
    run_command(f"python visualize-non-iid/analysis_controlled_noniid.py --dir \"{checkpoint_dir}\"")
    run_command(f"python visualize-non-iid/generate_heatmap.py --dir \"{checkpoint_dir}\" --round round_{config['num_rounds']}")
    run_command(f"python visualize-non-iid/analyze_controlled_noniid_graphs.py --dir \"{checkpoint_dir}\" --round round_{config['num_rounds']}")

    # C. Archive
    print(f"\n--- Archiving Results for {config['name']} ---")
    os.makedirs(f"exps/{config['name']}", exist_ok=True)
    os.rename(checkpoint_dir, f"exps/{config['name']}/data")

    # --- 2. RUN THE SPARSITY ABLATION STUDY ---
    print("\n\n--- [2/2] RUNNING EXPERIMENT: SPARSITY ABLATION STUDY ---")
    
    base_config = SPARSITY_ABLATION_CONFIG
    client_map_json = json.dumps(base_config['client_map'])

    for sparsity in base_config['sparsity_levels']:
        run_name = f"sparsity_{sparsity}"
        print(f"\n--- Running Sparsity Level: {sparsity} ---")
        checkpoint_dir = f"./checkpoints/{run_name}"

        # A. Run Training
        run_command(
            f"python run_non-iid.py "
            f"--num_rounds {base_config['num_rounds']} "
            f"--target_sparsity {sparsity} "
            f"--checkpoint_dir \"{checkpoint_dir}\" "
            f"--client_map_json '{client_map_json}'"
        )
        
        # B. Run ALL Analysis Scripts
        print(f"\n--- Analyzing Sparsity: {sparsity} ---")
        run_command(f"python visualize-non-iid/visualize_cross_accuracy.py --dir \"{checkpoint_dir}\"")
        run_command(f"python visualize-non-iid/analysis_controlled_noniid.py --dir \"{checkpoint_dir}\"")
        run_command(f"python visualize-non-iid/generate_heatmap.py --dir \"{checkpoint_dir}\" --round round_{base_config['num_rounds']}")
        run_command(f"python visualize-non-iid/analyze_controlled_noniid_graphs.py --dir \"{checkpoint_dir}\" --round round_{base_config['num_rounds']}")
        
        # C. Archive
        print(f"\n--- Archiving Sparsity: {sparsity} ---")
        os.makedirs(f"exps/{base_config['name']}/{run_name}", exist_ok=True)
        os.rename(checkpoint_dir, f"exps/{base_config['name']}/{run_name}/data")

    print("\n\n=============================================")
    print("      ALL EXPERIMENTS COMPLETE! ✅")
    print("=============================================")

if __name__ == "__main__":
    main()